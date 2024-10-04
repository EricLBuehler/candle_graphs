//! https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
//! https://github.com/pytorch/pytorch/blob/c7b0d4b148cf2e4e68f14193549945e1639bff40/aten/src/ATen/cuda/CUDAGraph.cpp

mod graph;
use graph::{Graph, GraphDumpFormat, GraphDumpVerbosity};

use std::f64::consts::E;

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

    let x = Tensor::ones((4, 4), DType::BF16, &device)?;
    let mut y: Option<Tensor> = None;

    let graph = Graph::new(
        || {
            let out_data = x.log()?.matmul(&x)?;
            y = Some(out_data);
            Ok(())
        },
        &device,
    )?;

    graph.output_dot("out.png", GraphDumpFormat::Png, GraphDumpVerbosity::Verbose)?;

    for i in 1..=10 {
        println!("{} Exec {i} {}", "=".repeat(10), "=".repeat(10));
        let new = Tensor::full(E.powi(i), (4, 4), &device)?.to_dtype(DType::BF16)?;

        unsafe { copy_into(&new, &x, &device)? };

        println!("x\n{x}");

        graph.replay()?;

        if let Some(y) = &y {
            println!("out\n{y}");
        }
    }

    Ok(())
}
