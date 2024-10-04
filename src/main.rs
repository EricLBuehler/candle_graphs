use std::{process::Command, time::Instant};

use candle_core::{cuda::cudarc, DType, Device, Tensor};

const USE_CUDA_GRAPH: bool = true;

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cu_device = match &device {
        Device::Cuda(dev) => dev,
        _ => unreachable!(),
    };
    let cu_stream = cu_device.cu_stream();
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
    if USE_CUDA_GRAPH {
        unsafe {
            cudarc::driver::sys::lib()
            .cuStreamBeginCapture_v2(
                *cu_stream,
                cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .result()?
        }
    }
    let mut out_data = Tensor::zeros((32, 32), DType::BF16, &device)?;
    {
        let x = Tensor::ones((32, 32), DType::BF16, &device)?;
        let y = Tensor::ones((32, 32), DType::BF16, &device)?;
        out_data = (x.silu()? + &y.gelu()?)?;
    }
    if USE_CUDA_GRAPH {
        let cu_graph: cudarc::driver::sys::CUgraph = unsafe {
            let mut cu_graph = std::mem::MaybeUninit::uninit();
            cudarc::driver::sys::lib()
                .cuStreamEndCapture(*cu_stream, cu_graph.as_mut_ptr())
                .result()?;
            cu_graph.assume_init()
        };
        let cu_graph_e: cudarc::driver::sys::CUgraphExec = unsafe {
            let mut cu_graph_e = std::mem::MaybeUninit::uninit();
            cudarc::driver::sys::lib()
                .cuGraphInstantiateWithFlags(cu_graph_e.as_mut_ptr(), cu_graph, 0)
                .result()?;
            cu_graph_e.assume_init()
        };

        println!("graph captured!");
        let start = Instant::now();

        let out = String::from("out.dot");
        unsafe {
            cudarc::driver::sys::lib().cuGraphDebugDotPrint(cu_graph, out.as_ptr() as *const i8, 0)
        }
        .result()?;
        let command = Command::new("dot")
            .arg("-Tpng")
            .arg("out.dot")
            .output()?
            .stdout;
        std::fs::write("out.png", command)?;

        println!("graph exec");
        unsafe {
            cudarc::driver::sys::lib()
                .cuGraphLaunch(cu_graph_e, *cu_stream)
                .result()?
        }
        println!("sync");
        device.synchronize()?;
        println!("done syncing");

        println!("{out_data}");

        dbg!(&Instant::now().duration_since(start).as_secs_f64());
    } else {
        let start = Instant::now();
        let _u = Tensor::zeros((4096, 4096), DType::F32, &device)?.to_dtype(DType::BF16)?;
        let mut x = Tensor::zeros((4096, 4096), DType::F32, &device)?.to_dtype(DType::BF16)?;
        let _v = Tensor::zeros((4096, 1), DType::F32, &device)?.to_dtype(DType::BF16)?;
        for i in 0..100 * 100 {
            println!("exec {i}");
            // x.slice_set(&u, 0, 0)?;
            // x.broadcast_add(&v)?;
            x = x.matmul(&x)?;
            // x = (&u + &x)?;
        }
        dbg!(&Instant::now().duration_since(start).as_secs_f64());
        device.synchronize()?;
    }
    Ok(())
}
