use std::{f64::consts::E, process::Command, ptr};

use candle_core::{
    cuda::cudarc::{
        self,
        driver::{DevicePtr, DeviceSlice},
    },
    DType, Device, Storage, Tensor,
};
use half::bf16;

/// # Safety
/// It must be ensured that the storage of src can be cast to &mut. So no aliasing across threads.
unsafe fn copy_into(src: &Tensor, dst: &Tensor, device: &Device) -> anyhow::Result<()> {
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

            let tgt = tgt.as_cuda_slice::<bf16>()?;
            let src = src.as_cuda_slice::<bf16>()?;
            cudarc::driver::result::memcpy_dtod_sync(
                *tgt.device_ptr(),
                *src.device_ptr(),
                src.len() * std::mem::size_of::<bf16>(),
            )?;
            device.synchronize()?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

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

    println!("Creating graph");
    /////  CREATING THE GRAPH
    let mut cu_graph: cudarc::driver::sys::CUgraph = unsafe {
        let mut cu_graph = std::mem::MaybeUninit::uninit();
        cudarc::driver::sys::lib()
            .cuGraphCreate(cu_graph.as_mut_ptr(), 0)
            .result()?;
        cu_graph.assume_init()
    };

    println!("Created graph");

    let x = Tensor::ones((4, 4), DType::BF16, &device)?;
    let mut y: Option<Tensor> = None;

    /////  START CAPTURE INTO THE GRAPH
    unsafe {
        cudarc::driver::sys::lib()
            .cuStreamBeginCaptureToGraph(
                *cu_stream,
                cu_graph,
                ptr::null(),
                ptr::null(),
                0,
                cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED, //CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .result()?
    }

    println!("Begin capture");
    {
        let out_data = x.log()?;
        y = Some(out_data);
    };
    println!("Done with ops");

    /////  END CAPTURE AND WRITE TO THE GRAPH
    unsafe {
        cudarc::driver::sys::lib()
            .cuStreamEndCapture(*cu_stream, &mut cu_graph as *mut _)
            .result()?;
    }

    /////  CREATING THE GRAPH EXECUTOR
    let cu_graph_e: cudarc::driver::sys::CUgraphExec = unsafe {
        let mut cu_graph_e = std::mem::MaybeUninit::uninit();
        // https://github.com/pytorch/pytorch/blob/c7b0d4b148cf2e4e68f14193549945e1639bff40/aten/src/ATen/cuda/CUDAGraph.cpp#L166-L176
        cudarc::driver::sys::lib()
            .cuGraphInstantiateWithFlags(cu_graph_e.as_mut_ptr(), cu_graph, cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as u64)
            .result()?;
        cu_graph_e.assume_init()
    };

    println!("graph captured!");

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

    for i in 1..=10 {
        println!("{} Exec {i} {}", "=".repeat(10), "=".repeat(10));
        let new = Tensor::full(E.powi(i), (4, 4), &device)?.to_dtype(DType::BF16)?;

        unsafe { copy_into(&new, &x, &device)? };

        println!("x {x}");

        println!("graph exec");
        unsafe {
            cudarc::driver::sys::lib()
                .cuGraphLaunch(cu_graph_e, *cu_stream)
                .result()?
        }
        println!("sync");
        device.synchronize()?;
        println!("done syncing");

        if let Some(y) = &y {
            println!("out {y}");
        }
    }

    Ok(())
}
