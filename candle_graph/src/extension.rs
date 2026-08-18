use candle_core::{
    builder_arg,
    cuda::{
        cudarc::driver::{LaunchConfig, PushKernelArg},
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
        let src_o = src_o as u32;
        let dst_o = dst_o as u32;
        let kname = match (&src.slice, &dst.slice) {
            (S::U8(_), S::U8(_)) => "copy2d_u8",
            (S::U32(_), S::U32(_)) => "copy2d_u32",
            (S::I16(_), S::I16(_)) => "copy2d_i16",
            (S::I32(_), S::I32(_)) => "copy2d_i32",
            (S::I64(_), S::I64(_)) => "copy2d_i64",
            (S::BF16(_), S::BF16(_)) => "copy2d_bf16",
            (S::F16(_), S::F16(_)) => "copy2d_f16",
            (S::F32(_), S::F32(_)) => "copy2d_f32",
            (S::F64(_), S::F64(_)) => "copy2d_f64",
            _ => Err(CudaError::InternalError("dtype mismatch in copy2d"))?,
        };
        let func =
            dev.get_or_load_custom_func(kname, "candle_graph_update_kv", kernels::UPDATE_KV)?;
        let cfg = LaunchConfig::for_num_elems(d1 * d2);
        let mut builder = func.builder();
        builder_arg!(builder, COPY2D_FINGERPRINT);
        // Pushes the (already dtype-matched) src/dst slices as the kernel's `src`/`dst`
        // pointer args. Each arm's `s`/`d` differ in element type, so this can't be
        // factored out of the match the way the kname lookup above was.
        match (&src.slice, &dst.slice) {
            (S::U8(s), S::U8(d)) => {
                builder.arg(s);
                builder.arg(d);
            }
            (S::U32(s), S::U32(d)) => {
                builder.arg(s);
                builder.arg(d);
            }
            (S::I16(s), S::I16(d)) => {
                builder.arg(s);
                builder.arg(d);
            }
            (S::I32(s), S::I32(d)) => {
                builder.arg(s);
                builder.arg(d);
            }
            (S::I64(s), S::I64(d)) => {
                builder.arg(s);
                builder.arg(d);
            }
            (S::BF16(s), S::BF16(d)) => {
                builder.arg(s);
                builder.arg(d);
            }
            (S::F16(s), S::F16(d)) => {
                builder.arg(s);
                builder.arg(d);
            }
            (S::F32(s), S::F32(d)) => {
                builder.arg(s);
                builder.arg(d);
            }
            (S::F64(s), S::F64(d)) => {
                builder.arg(s);
                builder.arg(d);
            }
            // Already validated as one of the arms above when `kname` was picked.
            _ => unreachable!("dtype mismatch in copy2d"),
        }
        builder_arg!(builder, d1);
        builder_arg!(builder, d2);
        builder_arg!(builder, src_o);
        builder_arg!(builder, dst_o);
        builder_arg!(builder, src_s);
        builder_arg!(builder, dst_s);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
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
