use anyhow::Context;
use candle_core::{
    backend::BackendStorage,
    cuda::cudarc::driver::{
        self,
        sys::{
            CUgraph, CUgraphDebugDot_flags, CUgraphExec, CUgraphInstantiate_flags, CUgraphNode,
            CUstream, CUstreamCaptureMode,
        },
        DevicePtr, DeviceSlice,
    },
    cuda::CudaDType,
    quantized::{GgmlDType, QMatMul, QTensor},
    CudaStorage, DType, Device, Storage, Tensor,
};
use candle_nn::Module;
use half::{bf16, f16};
use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    ffi::CString,
    marker::PhantomData,
    mem::MaybeUninit,
    path::Path,
    process::Command,
    ptr,
};

use crate::{CudaTensorExtension, KernelLaunchParams, Node, NodeData, COPY2D_FINGERPRINT};

/// `count` elements of one dtype, from `src`'s storage at `src_o` to `dst`'s at `dst_o`.
///
/// # Safety
/// Same contract as [`copy_inplace`]: `dst`'s storage is written through a shared reference, so
/// it must not be aliased mutably elsewhere.
unsafe fn memcpy_dtod<T: CudaDType>(
    src: &CudaStorage,
    src_o: usize,
    dst: &CudaStorage,
    dst_o: usize,
    count: usize,
) -> anyhow::Result<()> {
    let src = src.as_cuda_slice::<T>()?;
    let dst = dst.as_cuda_slice::<T>()?;
    anyhow::ensure!(
        src_o + count <= src.len() && dst_o + count <= dst.len(),
        "copy_inplace: {count} elements at offsets {src_o}/{dst_o} do not fit in storages of \
         {}/{} elements",
        src.len(),
        dst.len()
    );
    let elem = std::mem::size_of::<T>();
    driver::result::memcpy_dtod_sync(
        *dst.device_ptr() + (dst_o * elem) as u64,
        *src.device_ptr() + (src_o * elem) as u64,
        count * elem,
    )?;
    Ok(())
}

/// Copy a tensor inplace from src to dst. This can be used to implement `GraphInput`.
///
/// # Safety
/// It must be ensured that the storage of src can be cast to &mut. So no aliasing across threads.
pub unsafe fn copy_inplace(src: &Tensor, dst: &Tensor, device: &Device) -> anyhow::Result<()> {
    // The copy below is a flat memcpy, so a non-contiguous `src` (most commonly a broadcast
    // one, e.g. `Tensor::full`, whose storage holds a single element regardless of its shape)
    // has no run of `elem_count()` elements to copy from in the first place. `GraphInput for
    // HashMap` already rejects non-contiguous inputs for this reason; check here too so every
    // caller gets the same guarantee, and a clearer message than the storage-bounds check.
    anyhow::ensure!(
        src.is_contiguous(),
        "copy_inplace: src must be contiguous (got shape {:?}); \
         call `.contiguous()` on it first",
        src.shape()
    );
    anyhow::ensure!(
        dst.is_contiguous(),
        "copy_inplace: dst must be contiguous (got shape {:?})",
        dst.shape()
    );
    anyhow::ensure!(
        src.shape() == dst.shape(),
        "copy_inplace: shape mismatch, src {:?} <> dst {:?}",
        src.shape(),
        dst.shape()
    );

    // `CudaStorage` is the whole allocation, not this tensor's view of it, so the offsets and
    // the element count have to come from the layouts. A contiguous *view* -- the
    // `positions.narrow(0, i, 1)` that `Cache::append_at` is meant to be driven with -- shares
    // its storage with the buffer it was carved from, so copying from the storage's base
    // pointer for the storage's full length would read the wrong element and write past the end
    // of a destination whose own storage is smaller.
    let (src_storage, src_layout) = src.storage_and_layout();
    let (dst_storage, dst_layout) = dst.storage_and_layout();
    let src_o = src_layout.start_offset();
    let dst_o = dst_layout.start_offset();
    // Both shapes were checked equal above, so either count works.
    let count = src_layout.shape().elem_count();

    match (&*src_storage, &*dst_storage) {
        (Storage::Cuda(src), Storage::Cuda(tgt)) => {
            // What we are really doing:

            // unsafe fn cast_to_mut<T>(r: &T) -> &mut T {
            //     // Cast immutable reference to mutable reference
            //     #[allow(invalid_reference_casting)]
            //     &mut *(r as *const T as *mut T)
            // }
            // let dst = unsafe { cast_to_mut(tgt.as_cuda_slice::<bf16>()?) };
            // cu_device.dtod_copy(src, dst)?;

            anyhow::ensure!(src.dtype() == tgt.dtype(), "DTypes must match!");

            match src.dtype() {
                DType::BF16 => memcpy_dtod::<bf16>(src, src_o, tgt, dst_o, count)?,
                DType::F16 => memcpy_dtod::<f16>(src, src_o, tgt, dst_o, count)?,
                DType::F32 => memcpy_dtod::<f32>(src, src_o, tgt, dst_o, count)?,
                DType::F64 => memcpy_dtod::<f64>(src, src_o, tgt, dst_o, count)?,
                DType::I64 => memcpy_dtod::<i64>(src, src_o, tgt, dst_o, count)?,
                DType::U32 => memcpy_dtod::<u32>(src, src_o, tgt, dst_o, count)?,
                DType::U8 => memcpy_dtod::<u8>(src, src_o, tgt, dst_o, count)?,
            }
            device.synchronize()?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

pub enum GraphDumpFormat {
    Svg,
    Png,
    Dot,
}

pub enum GraphDumpVerbosity {
    Clean,
    Verbose,
}

pub trait GraphInput {
    fn to_inputs(&self) -> HashMap<String, Tensor>;
    fn load_inputs_inplace(&self, input: Self, device: &Device) -> anyhow::Result<()>;
}

impl GraphInput for HashMap<&'static str, Tensor> {
    fn to_inputs(&self) -> HashMap<String, Tensor> {
        self.iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }
    fn load_inputs_inplace(&self, input: Self, device: &Device) -> anyhow::Result<()> {
        let mut added = HashSet::new();
        for (name, input) in &input {
            if !added.insert(name) {
                panic!("Got duplicate inputs {name}");
            }
            if !input.is_contiguous() {
                panic!("Input {name} is not contiguous");
            }
            if let Some(inp_ref) = self.get(name) {
                unsafe { copy_inplace(input, inp_ref, device)? };
            } else {
                panic!("Graph has no input {name}");
            }
        }
        if added.len() != input.len() {
            panic!(
                "Some inputs were not provided: expected {:?}, got {added:?}",
                input.keys().collect::<Vec<_>>()
            );
        }
        Ok(())
    }
}

pub struct Graph<T: GraphInput> {
    graph: CUgraph,
    exec: CUgraphExec,
    stream: CUstream,
    device: Device,
    input: T,
    /// Tensors the capture baked in as raw device addresses but that the graph does not
    /// otherwise own. Nothing on the replay path would notice the Rust side dropping one -- the
    /// replay keeps writing to the address either way -- so a dropped tensor turns every later
    /// replay into a write to freed memory. Candle tensors are refcounted, so holding a clone
    /// keeps the allocation alive exactly as long as the graph can replay into it. `input` is
    /// already covered; this is for everything the captured closure touched by reference, such
    /// as a `Cache`'s backing buffer.
    _retained: Vec<Tensor>,
    ran_graph: Cell<bool>,
    // CUgraph is not thread safe!
    _marker: PhantomData<*const ()>,
}

impl<T: GraphInput> Graph<T> {
    /// Initialize a CUDA graph, executing the closure to capture a graph.
    ///
    /// The input tensors provided must all be contiguous.
    ///
    /// Anything the closure writes to besides `input` -- a KV cache, say -- must outlive the
    /// graph, since the capture keeps only its device address. Use
    /// [`new_retaining`](Self::new_retaining) to hand that responsibility to the graph.
    pub fn new(
        from_code: impl FnOnce(&T) -> anyhow::Result<()>,
        device: &Device,
        input: T,
    ) -> anyhow::Result<Self> {
        Self::new_retaining(from_code, device, input, Vec::new())
    }

    /// Like [`new`](Self::new), but keeps `retained` alive for as long as the graph can replay.
    ///
    /// A capture records device addresses, not tensors. Whatever the closure wrote to that is
    /// not part of `input` is still owned by the caller, and dropping it frees the allocation
    /// the graph goes on replaying into -- silently, because there is no Rust code on the replay
    /// path to notice. Passing those tensors here makes the graph hold its own reference, so the
    /// caller is free to drop theirs.
    ///
    /// For a cache, that is [`Cache::all_data`](crate::cache::Cache::all_data) after
    /// [`Cache::reserve`](crate::cache::Cache::reserve) -- see the `decode_replay` example.
    pub fn new_retaining(
        from_code: impl FnOnce(&T) -> anyhow::Result<()>,
        device: &Device,
        input: T,
        retained: Vec<Tensor>,
    ) -> anyhow::Result<Self> {
        let cu_device = match &device {
            Device::Cuda(dev) => dev,
            _ => anyhow::bail!("Must have CUDA device."),
        };

        let cu_stream = cu_device.cu_stream();

        // Initialize all ptx files
        // `load_ptx` cannot be called while capturing the stream so we need this to happen
        // beforehand.
        {
            // Fill
            let x = Tensor::zeros((128, 128), DType::F32, device)?;

            // Affine
            let _ = x.affine(1., 0.5)?;

            // Binary
            let _ = x.mul(&x)?;

            // Cast
            let _ = x.to_dtype(DType::BF16)?;

            // Conv2d
            {
                let ws = Tensor::zeros((3, 3, 4, 4), DType::F32, device)?;
                let conv_xs = Tensor::zeros((1, 3, 48, 48), DType::F32, device)?;
                let _ = conv_xs.conv2d(&ws, 0, 1, 1, 1)?;
            }

            // Indexing
            {
                let indices = Tensor::new(vec![0u32, 2, 4], device)?;
                let _ = x.index_select(&indices, 0)?;
            }

            // FUSED_RMS_NORM
            // TODO

            // FUSED_ROPE
            // TODO

            // Quantized
            {
                let q = QMatMul::from_qtensor(QTensor::quantize(&x, GgmlDType::Q8_0)?)?;
                let _ = q.forward(&x)?;
            }

            // Reduce
            let _ = candle_nn::ops::softmax_last_dim(&x)?;

            // Sort
            let _ = x.sort_last_dim(true)?;

            // Ternary
            let _ = x.to_dtype(DType::U32)?.where_cond(
                &Tensor::new(0f32, device)?.broadcast_as(x.shape())?,
                &Tensor::new(1f32, device)?.broadcast_as(x.shape())?,
            )?;

            // Unary
            let _ = x.neg()?;

            // UPDATE_KV (this crate's own copy2d/copy2d_dynoffset kernels, used by
            // Cache::append/append_at via CudaTensorExtension) -- must be warmed up here
            // too, for the same reason as the built-ins above: this may be the first time
            // this process has touched this particular CudaDevice, and load_ptx can't
            // happen once capture starts. Without this, a closure whose first-ever call
            // to slice_set_fingerprinted[_at] happens during capture silently captures a
            // kernel node that doesn't write correctly on replay.
            //
            // Every dtype, not just F32: `get_or_load_func` keys its module cache on the
            // *function* name, and the launch path picks a different one per dtype
            // (`copy2d_dynoffset_bf16` vs `..._f32`), so warming one leaves the others cold.
            // A bf16 or f16 KV cache -- the usual case for decode -- would otherwise hit its
            // first load inside the capture.
            for dtype in [
                DType::BF16,
                DType::F16,
                DType::F32,
                DType::F64,
                DType::I64,
                DType::U32,
                DType::U8,
            ] {
                // Deliberately not propagated: the f16 and bf16 kernels are compiled behind
                // `__CUDA_ARCH__` guards, so on an older device they are absent from the PTX
                // and this fails. That is not a reason to refuse to build a graph that never
                // uses them -- and one that does will raise the same error at its own call,
                // outside the capture.
                let _ = (|| -> anyhow::Result<()> {
                    let dst = Tensor::zeros((1, 1), dtype, device)?;
                    let src = Tensor::zeros((1, 1), dtype, device)?;
                    dst.slice_set_fingerprinted(&src, 0, 0)?;
                    let position = Tensor::new(&[0u32], device)?;
                    dst.slice_set_fingerprinted_at(&src, 0, 0, &position)?;
                    Ok(())
                })();
            }

            device.synchronize()?;
        }

        let mut cu_graph: CUgraph = unsafe {
            let mut cu_graph = MaybeUninit::uninit();
            driver::sys::lib()
                .cuGraphCreate(cu_graph.as_mut_ptr(), 0)
                .result()?;
            cu_graph.assume_init()
        };

        unsafe {
            driver::sys::lib()
                .cuStreamBeginCapture_v2(
                    *cu_stream,
                    CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                )
                .result()?
        }

        from_code(&input)?;

        /////  END CAPTURE AND WRITE TO THE GRAPH
        unsafe {
            driver::sys::lib()
                .cuStreamEndCapture(*cu_stream, &mut cu_graph as *mut _)
                .result()?;
        }

        /////  CREATING THE GRAPH EXECUTOR
        let cu_graph_e: CUgraphExec = unsafe {
            let mut cu_graph_e = MaybeUninit::uninit();
            // https://github.com/pytorch/pytorch/blob/c7b0d4b148cf2e4e68f14193549945e1639bff40/aten/src/ATen/cuda/CUDAGraph.cpp#L166-L176
            driver::sys::lib()
                .cuGraphInstantiateWithFlags(
                    cu_graph_e.as_mut_ptr(),
                    cu_graph,
                    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
                        as u64,
                )
                .result()?;
            cu_graph_e.assume_init()
        };

        for (name, input) in &input.to_inputs() {
            if !input.is_contiguous() {
                anyhow::bail!("Input {name} is not contiguous");
            }
        }

        Ok(Self {
            graph: cu_graph,
            exec: cu_graph_e,
            stream: *cu_stream,
            device: device.clone(),
            input,
            _retained: retained,
            _marker: PhantomData,
            ran_graph: Cell::new(false),
        })
    }

    /// Execute the graph with the provided inputs.
    ///
    /// All inputs are copied, so it may be detrimental to performance if large
    /// inputs are provided.
    ///
    /// # Panics
    /// - The inputs provided here must match the inputs provided upon construction.
    /// - The inputs provided here must all be continuous
    pub fn replay(&self, input: T) -> anyhow::Result<()> {
        self.input.load_inputs_inplace(input, &self.device)?;
        unsafe {
            driver::sys::lib()
                .cuGraphLaunch(self.exec, self.stream)
                .result()?
        }
        self.ran_graph.set(true);
        self.device.synchronize()?;
        Ok(())
    }

    /// Requires that you have installed the [graphviz](https://graphviz.org/download/) library.
    /// Writes the graph to the specified path.
    pub fn output_dot<P: AsRef<Path>>(
        &self,
        out: P,
        format: GraphDumpFormat,
        verbosity: GraphDumpVerbosity,
    ) -> anyhow::Result<()> {
        let tmp = if let GraphDumpFormat::Dot = format {
            out.as_ref().to_string_lossy().trim().to_string()
        } else {
            format!("{}.dot", out.as_ref().display())
        };
        let cstr = unsafe { CString::from_vec_unchecked(tmp.as_bytes().to_vec()) };
        let verbosity = match verbosity {
            GraphDumpVerbosity::Verbose => {
                CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE as u32
            }
            GraphDumpVerbosity::Clean => 0,
        };
        unsafe { driver::sys::lib().cuGraphDebugDotPrint(self.graph, cstr.as_ptr(), verbosity) }
            .result()?;
        let ty = match format {
            GraphDumpFormat::Png => "png",
            GraphDumpFormat::Svg => "svg",
            GraphDumpFormat::Dot => return Ok(()),
        };
        let command = Command::new("dot")
            .arg(format!("-T{ty}"))
            .arg(tmp)
            .output()
            .context("`candle_graph` requires the graphviz utility to be installed: https://graphviz.org/download/")?
            .stdout;
        std::fs::write(out, command)?;
        Ok(())
    }

    /// Retrieve the nodes for this graph. Node dependency information is not tracked.
    pub fn nodes(&self) -> anyhow::Result<Vec<Node<'_>>> {
        println!("Getting nodes");
        let mut num_nodes = unsafe {
            let mut num_nodes = MaybeUninit::uninit();
            driver::sys::lib()
                .cuGraphGetNodes(self.graph, ptr::null_mut(), num_nodes.as_mut_ptr())
                .result()?;
            num_nodes.assume_init()
        };
        let node_ptrs = unsafe {
            let mut nodes: Vec<CUgraphNode> = Vec::with_capacity(num_nodes);
            driver::sys::lib()
                .cuGraphGetNodes(self.graph, nodes.as_mut_ptr(), &mut num_nodes as *mut _)
                .result()?;
            nodes.set_len(num_nodes);
            nodes
        };

        let mut nodes = Vec::new();
        for node in &node_ptrs {
            let node_type = unsafe {
                let mut node_type = MaybeUninit::uninit();
                driver::sys::lib()
                    .cuGraphNodeGetType(*node, node_type.as_mut_ptr())
                    .result()?;
                node_type.assume_init()
            };
            #[allow(clippy::single_match)]
            let data = match node_type {
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => {
                    let node_params = unsafe {
                        let mut node_params = MaybeUninit::uninit();
                        driver::sys::lib()
                            .cuGraphKernelNodeGetParams_v2(*node, node_params.as_mut_ptr())
                            .result()?;
                        node_params.assume_init()
                    };
                    let vec = unsafe { std::slice::from_raw_parts(node_params.kernelParams, 1) };
                    for item in vec {
                        let arg_ptr = *item as *const u64;
                        if unsafe { *arg_ptr } == COPY2D_FINGERPRINT {
                            println!("found it ");
                        }
                    }
                    let params = KernelLaunchParams {
                        grid_dim_x: node_params.gridDimX,
                        grid_dim_y: node_params.gridDimY,
                        grid_dim_z: node_params.gridDimZ,
                        block_dim_x: node_params.blockDimX,
                        block_dim_y: node_params.blockDimY,
                        block_dim_z: node_params.blockDimZ,
                        shared_mem_bytes: node_params.sharedMemBytes,
                    };
                    NodeData::Kernel {
                        launch_params: params,
                    }
                }
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_BATCH_MEM_OP => {
                    NodeData::BatchMemOp
                }
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL => {
                    NodeData::Conditional
                }
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY => NodeData::Empty,
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EVENT_RECORD => {
                    NodeData::EventRecord
                }
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL => {
                    NodeData::ExtSemasSignal
                }
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT => {
                    NodeData::ExtSemasWait
                }
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_GRAPH => NodeData::Graph,
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_HOST => NodeData::Host,
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => NodeData::Memcpy,
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => NodeData::Memset,
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => NodeData::MemAlloc,
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => NodeData::MemFree,
                driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_WAIT_EVENT => NodeData::WaitEvent,
            };

            let node = Node {
                data,
                inner: *node,
                _marker: PhantomData,
            };
            nodes.push(node);
        }

        Ok(nodes)
    }
}

impl<T: GraphInput> Drop for Graph<T> {
    fn drop(&mut self) {
        if !self.ran_graph.get() {
            unsafe {
                driver::sys::lib()
                    .cuGraphLaunch(self.exec, self.stream)
                    .result()
                    .expect("Graph was not run, final run failed")
            }
            self.device
                .synchronize()
                .expect("Graph was not run, device sync failed")
        }
        unsafe { driver::sys::lib().cuGraphDestroy(self.graph) }
            .result()
            .expect("Graph destroy failed");
        unsafe { driver::sys::lib().cuGraphExecDestroy(self.exec) }
            .result()
            .expect("Graph destroy failed");
    }
}

#[cfg(test)]
mod test {
    use candle_core::{DType, Device, Tensor};

    use super::copy_inplace;

    /// The memcpy addresses *storages*, not tensors. A contiguous view -- what `Cache::append_at`
    /// is meant to be driven with, e.g. `positions.narrow(0, i, 1)` -- shares the storage of the
    /// buffer it was carved from, so ignoring the layouts reads that buffer's first element and
    /// writes its whole length, however small the destination's own allocation is.
    #[test]
    fn copy_inplace_honors_view_offsets() -> anyhow::Result<()> {
        let device = Device::new_cuda_with_stream(0)?;

        let positions = Tensor::new(&[10u32, 20, 30, 40], &device)?;
        let src = positions.narrow(0, 2, 1)?;

        // The destination is a view too: the write has to land on its element, not on the
        // start of the buffer behind it.
        let dst_buf = Tensor::zeros(4usize, DType::U32, &device)?;
        let dst = dst_buf.narrow(0, 1, 1)?;
        unsafe { copy_inplace(&src, &dst, &device)? };
        assert_eq!(dst_buf.to_vec1::<u32>()?, [0, 30, 0, 0]);

        // A destination whose entire storage is one element: copying the source's storage
        // wholesale would write four elements into it.
        let small = Tensor::zeros(1usize, DType::U32, &device)?;
        unsafe { copy_inplace(&src, &small, &device)? };
        assert_eq!(small.to_vec1::<u32>()?, [30]);

        Ok(())
    }
}
