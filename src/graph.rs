use candle_core::{
    backend::BackendStorage,
    cuda::cudarc::driver::{
        self,
        sys::{
            CUgraph, CUgraphDebugDot_flags, CUgraphExec, CUgraphInstantiate_flags, CUstream,
            CUstreamCaptureMode,
        },
        DevicePtr, DeviceSlice,
    },
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Storage, Tensor,
};
use candle_nn::Module;
use half::{bf16, f16};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    process::Command,
    ptr,
};

/// # Safety
/// It must be ensured that the storage of src can be cast to &mut. So no aliasing across threads.
unsafe fn copy_inplace(src: &Tensor, dst: &Tensor, device: &Device) -> anyhow::Result<()> {
    match (&*src.storage_and_layout().0, &*dst.storage_and_layout().0) {
        (Storage::Cuda(src), Storage::Cuda(tgt)) => {
            // What we are really doing:

            // unsafe fn cast_to_mut<T>(r: &T) -> &mut T {
            //     // Cast immutable reference to mutable reference
            //     #[allow(invalid_reference_casting)]
            //     &mut *(r as *const T as *mut T)
            // }
            // let dst = unsafe { cast_to_mut(tgt.as_cuda_slice::<bf16>()?) };
            // cu_device.dtod_copy(src, dst)?;

            anyhow::ensure!(src.dtype() == dst.dtype(), "DTypes must match!");

            match src.dtype() {
                DType::BF16 => {
                    let tgt = tgt.as_cuda_slice::<bf16>()?;
                    let src = src.as_cuda_slice::<bf16>()?;
                    driver::result::memcpy_dtod_sync(
                        *tgt.device_ptr(),
                        *src.device_ptr(),
                        src.len() * std::mem::size_of::<bf16>(),
                    )?;
                }
                DType::F16 => {
                    let tgt = tgt.as_cuda_slice::<f16>()?;
                    let src = src.as_cuda_slice::<f16>()?;
                    driver::result::memcpy_dtod_sync(
                        *tgt.device_ptr(),
                        *src.device_ptr(),
                        src.len() * std::mem::size_of::<f16>(),
                    )?;
                }
                DType::F32 => {
                    let tgt = tgt.as_cuda_slice::<f32>()?;
                    let src = src.as_cuda_slice::<f32>()?;
                    driver::result::memcpy_dtod_sync(
                        *tgt.device_ptr(),
                        *src.device_ptr(),
                        src.len() * std::mem::size_of::<f32>(),
                    )?;
                }
                DType::F64 => {
                    let tgt = tgt.as_cuda_slice::<f64>()?;
                    let src = src.as_cuda_slice::<f64>()?;
                    driver::result::memcpy_dtod_sync(
                        *tgt.device_ptr(),
                        *src.device_ptr(),
                        src.len() * std::mem::size_of::<f64>(),
                    )?;
                }
                DType::I64 => {
                    let tgt = tgt.as_cuda_slice::<i64>()?;
                    let src = src.as_cuda_slice::<i64>()?;
                    driver::result::memcpy_dtod_sync(
                        *tgt.device_ptr(),
                        *src.device_ptr(),
                        src.len() * std::mem::size_of::<i64>(),
                    )?;
                }
                DType::U32 => {
                    let tgt = tgt.as_cuda_slice::<u32>()?;
                    let src = src.as_cuda_slice::<u32>()?;
                    driver::result::memcpy_dtod_sync(
                        *tgt.device_ptr(),
                        *src.device_ptr(),
                        src.len() * std::mem::size_of::<u32>(),
                    )?;
                }
                DType::U8 => {
                    let tgt = tgt.as_cuda_slice::<u8>()?;
                    let src = src.as_cuda_slice::<u8>()?;
                    driver::result::memcpy_dtod_sync(
                        *tgt.device_ptr(),
                        *src.device_ptr(),
                        src.len() * std::mem::size_of::<u8>(),
                    )?;
                }
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

pub struct Graph {
    graph: CUgraph,
    exec: CUgraphExec,
    stream: CUstream,
    device: Device,
    input_tensors: HashMap<&'static str, Tensor>,
}

impl Graph {
    pub fn new(
        from_code: impl FnOnce() -> anyhow::Result<()>,
        device: &Device,
        input_tensors: HashMap<&'static str, Tensor>,
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

            device.synchronize()?;
        }

        let mut cu_graph: CUgraph = unsafe {
            let mut cu_graph = std::mem::MaybeUninit::uninit();
            driver::sys::lib()
                .cuGraphCreate(cu_graph.as_mut_ptr(), 0)
                .result()?;
            cu_graph.assume_init()
        };

        unsafe {
            driver::sys::lib()
                .cuStreamBeginCaptureToGraph(
                    *cu_stream,
                    cu_graph,
                    ptr::null(),
                    ptr::null(),
                    0,
                    CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                )
                .result()?
        }

        from_code()?;

        /////  END CAPTURE AND WRITE TO THE GRAPH
        unsafe {
            driver::sys::lib()
                .cuStreamEndCapture(*cu_stream, &mut cu_graph as *mut _)
                .result()?;
        }

        /////  CREATING THE GRAPH EXECUTOR
        let cu_graph_e: CUgraphExec = unsafe {
            let mut cu_graph_e = std::mem::MaybeUninit::uninit();
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

        Ok(Self {
            graph: cu_graph,
            exec: cu_graph_e,
            stream: *cu_stream,
            device: device.clone(),
            input_tensors,
        })
    }

    /// Run the graph
    pub fn replay(&self, input_tensors: HashMap<&'static str, &Tensor>) -> anyhow::Result<()> {
        let mut added = HashSet::new();
        for (name, input) in &input_tensors {
            if !added.insert(name) {
                panic!("Got duplicate inputs {name}");
            }
            if let Some(inp_ref) = self.input_tensors.get(name) {
                unsafe { copy_inplace(input, inp_ref, &self.device)? };
            } else {
                panic!("Graph has no input {name}");
            }
        }
        if added.len() != input_tensors.len() {
            panic!(
                "Some inputs were not provided: expected {:?}, got {added:?}",
                input_tensors.keys().collect::<Vec<_>>()
            );
        }
        unsafe {
            driver::sys::lib()
                .cuGraphLaunch(self.exec, self.stream)
                .result()?
        }
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
            out.as_ref().to_string_lossy().to_string()
        } else {
            format!("{}/candle-graph-dump.dot", std::env::temp_dir().display())
        };
        let verbosity = match verbosity {
            GraphDumpVerbosity::Verbose => {
                CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE as u32
            }
            GraphDumpVerbosity::Clean => 0,
        };
        unsafe {
            driver::sys::lib().cuGraphDebugDotPrint(
                self.graph,
                tmp.as_ptr() as *const i8,
                verbosity,
            )
        }
        .result()?;
        match format {
            GraphDumpFormat::Png | GraphDumpFormat::Svg => {
                let command = Command::new("dot").arg("-Tpng").arg(tmp).output()?.stdout;
                std::fs::write(out, command)?;
                Ok(())
            }
            GraphDumpFormat::Dot => Ok(()),
        }
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { driver::sys::lib().cuGraphDestroy(self.graph) }
            .result()
            .expect("Graph destroy failed");
        unsafe { driver::sys::lib().cuGraphExecDestroy(self.exec) }
            .result()
            .expect("Graph destroy failed");
    }
}
