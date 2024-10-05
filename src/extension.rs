use candle_core::{
    cuda::{
        cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig},
        CudaError, WrapErr, S,
    },
    shape::Dim,
    CudaStorage, Error, Result, Storage, Tensor,
};

use crate::kernels;

// A random number.
pub const COPY2D_FINGERPRINT: u64 = 7472013941709904021;

trait CudaStorageExtension {
    #[allow(clippy::too_many_arguments)]
    fn copy2d_fingerprinted(
        &self,
        src: &CudaStorage,
        dst: &CudaStorage,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()>;
}

/// Provides extensions to Tensor.
pub trait CudaTensorExtension {
    /// `slice_set`.
    ///
    /// The `copy2d_*` function takes a u64 fingerprint as the first argument, specifically [`COPY2D_FINGERPRINT`].
    fn slice_set_fingerprinted<D: Dim>(&self, src: &Self, dim: D, offset: usize) -> Result<()>;
}

impl CudaTensorExtension for Tensor {
    /// Set the values on `self` using values from `src`. The copy starts at the specified
    /// `offset` for the target dimension `dim` on `self`.
    /// `self` and `src` must have the same shape except on dimension `dim` where the `self` size
    /// has to be greater than or equal to `offset` plus the `src` size.
    ///
    /// Note that this modifies `self` in place and as such is not compatibel with
    /// back-propagation.  
    fn slice_set_fingerprinted<D: Dim>(&self, src: &Self, dim: D, offset: usize) -> Result<()> {
        let dim = dim.to_index(self.shape(), "slice-set")?;
        if !self.is_contiguous() || !src.is_contiguous() {
            Err(Error::RequiresContiguous { op: "slice-set" }.bt())?
        }
        if self.dtype() != src.dtype() {
            Err(Error::DTypeMismatchBinaryOp {
                lhs: self.dtype(),
                rhs: src.dtype(),
                op: "slice-set",
            }
            .bt())?
        }
        if self.device().location() != src.device().location() {
            Err(Error::DeviceMismatchBinaryOp {
                lhs: self.device().location(),
                rhs: src.device().location(),
                op: "slice-set",
            }
            .bt())?
        }
        if self.rank() != src.rank() {
            Err(Error::UnexpectedNumberOfDims {
                expected: self.rank(),
                got: src.rank(),
                shape: self.shape().clone(),
            }
            .bt())?
        }
        for (dim_idx, (v1, v2)) in self.dims().iter().zip(src.dims().iter()).enumerate() {
            if dim_idx == dim && *v2 + offset > *v1 {
                candle_core::bail!("shape mismatch on target dim, dst: {v1}, src: {v2} + {offset}")
            }
            if dim_idx != dim && v1 != v2 {
                candle_core::bail!("shape mismatch on dim {dim_idx}, {v1} <> {v2}")
            }
        }
        let block_size: usize = src.dims().iter().skip(1 + dim).product();
        let d1: usize = src.dims().iter().take(dim).product();
        let d2 = block_size * src.dims()[dim];
        let dst_o = self.layout().start_offset() + offset * block_size;
        let src_o = src.layout().start_offset();
        match (&*self.storage_and_layout().0, &*src.storage_and_layout().0) {
            (Storage::Cuda(src), Storage::Cuda(dst)) => self.copy2d_fingerprinted(
                dst,
                src,
                d1,
                d2,
                d2,
                block_size * self.dims()[dim],
                src_o,
                dst_o,
            ),
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "copy2d CUDA",
            }
            .bt()),
        }
    }
}

impl CudaStorageExtension for Tensor {
    fn copy2d_fingerprinted(
        &self,
        src: &CudaStorage,
        dst: &CudaStorage,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        let dev = &src.device;
        let d1 = d1 as u32;
        let d2 = d2 as u32;
        // Nothing to copy so we exit early to avoid launching a kernel and some potential invalid
        // argument with a null pointer.
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        let dst_s = dst_s as u32;
        let src_s = src_s as u32;
        let (src, dst, kname) = match (&src.slice, &dst.slice) {
            (S::U8(s), S::U8(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_u8"),
            (S::U32(s), S::U32(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_u32"),
            (S::I64(s), S::I64(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_i64"),
            (S::BF16(s), S::BF16(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_bf16"),
            (S::F16(s), S::F16(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_f16"),
            (S::F32(s), S::F32(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_f32"),
            (S::F64(s), S::F64(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_f64"),
            _ => Err(CudaError::InternalError("dtype mismatch in copy2d"))?,
        };
        let func = dev.get_or_load_func(kname, kernels::UPDATE_KV)?;
        let cfg = LaunchConfig::for_num_elems(d1 * d2);
        let params = (
            COPY2D_FINGERPRINT,
            src,
            dst,
            d1,
            d2,
            src_o,
            dst_o,
            src_s,
            dst_s,
        );
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }.w()?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use candle_core::{DType, Device, Result, Tensor};

    use crate::extension::CudaTensorExtension;

    #[test]
    fn slice_set() -> Result<()> {
        let device = Device::new_cuda_with_stream(0)?;

        let (b, h, max_t, d) = (2, 4, 7, 3);
        let cache = Tensor::zeros((b, h, max_t, d), DType::F32, &device)?;
        let tensor = Tensor::randn(0f32, 1f32, (b, h, 4, d), &device)?;
        cache.slice_set_fingerprinted(&tensor, 2, 0)?;
        let cache_t = cache.narrow(2, 0, 4)?;
        let diff = (cache_t - &tensor)?.abs()?.sum_all()?.to_vec0::<f32>()?;
        assert_eq!(diff, 0.);
        cache.slice_set_fingerprinted(&tensor, 2, 1)?;
        let cache_t = cache.narrow(2, 1, 4)?;
        let diff = (cache_t - &tensor)?.abs()?.sum_all()?.to_vec0::<f32>()?;
        assert_eq!(diff, 0.);
        let ones = Tensor::ones((b, h, 1, d), DType::F32, &device)?;
        cache.slice_set_fingerprinted(&ones, 2, 6)?;
        let diff = cache.narrow(2, 5, 1)?.abs()?.sum_all()?.to_vec0::<f32>()?;
        assert_eq!(diff, 0.);
        let diff = (cache.narrow(2, 6, 1)? - 1.)?
            .abs()?
            .sum_all()?
            .to_vec0::<f32>()?;
        assert_eq!(diff, 0.);
        Ok(())
    }
}
