use candle_core::{
    cuda::cudarc::driver::{
        self,
        sys::{
            CUgraph, CUgraphDebugDot_flags, CUgraphExec, CUgraphInstantiate_flags, CUstream,
            CUstreamCaptureMode,
        },
    },
    DType, Device, Tensor,
};
use std::{path::Path, process::Command, ptr};

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
}

impl Graph {
    pub fn new(
        from_code: impl FnOnce() -> anyhow::Result<()>,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let cu_device = match &device {
            Device::Cuda(dev) => dev,
            _ => anyhow::bail!("Must have CUDA device."),
        };

        let cu_stream = cu_device.cu_stream();

        // Initialize all ptx files
        {
            // load_ptx cannot be called while capturing the stream so we need this to happen
            // beforehand.
            let u = Tensor::zeros((4096, 4096), DType::F32, &device)?.to_dtype(DType::BF16)?;
            let x = Tensor::zeros((4096, 4096), DType::F32, &device)?.to_dtype(DType::BF16)?;
            let v = Tensor::zeros(4096, DType::F32, &device)?.to_dtype(DType::BF16)?;
            let _x = x.mul(&u)?.broadcast_add(&v)?;
            let _x = x.affine(1., 0.5)?;
            x.slice_set(&u, 0, 0)?;
            x.matmul(&x)?;
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
        })
    }

    /// Run the graph
    pub fn replay(&self) -> anyhow::Result<()> {
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
