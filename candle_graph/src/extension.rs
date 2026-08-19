use candle_core::{
    cuda::{
        cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig},
        CudaError, WrapErr, S,
    },
    shape::Dim,
    CudaStorage, DType, Error, Result, Storage, Tensor,
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

    #[allow(clippy::too_many_arguments)]
    fn copy2d_fingerprinted_dynoffset(
        &self,
        src: &CudaStorage,
        dst: &CudaStorage,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o_base: usize,
        dst_o_stride: usize,
        dst_o_ptr: &CudaStorage,
        dst_o_ptr_offset: usize,
        dst_o_ptr_max: usize,
    ) -> Result<()>;
}

/// Provides extensions to Tensor.
pub trait CudaTensorExtension {
    /// `slice_set`.
    ///
    /// The `copy2d_*` function takes a u64 fingerprint as the first argument, specifically [`COPY2D_FINGERPRINT`].
    fn slice_set_fingerprinted<D: Dim>(&self, src: &Self, dim: D, offset: usize) -> Result<()>;

    /// Like [`slice_set_fingerprinted`](Self::slice_set_fingerprinted), except the offset on
    /// `dim` is read from `position` (a `U32` tensor holding exactly one element) at kernel
    /// launch time instead of being passed as a plain `usize`.
    ///
    /// A plain offset becomes a fixed kernel-launch argument once captured into a CUDA graph,
    /// so every replay would write to the same slot the capture did. Reading the offset from
    /// `position` at launch time instead means only the *pointer* is captured; updating
    /// `position`'s contents between replays (e.g. via [`copy_inplace`](crate::copy_inplace))
    /// changes where the next replay writes without re-capturing anything. See the
    /// `decode_replay` example.
    ///
    /// A `*position` that would run off the end of `self` on `dim` -- `offset + *position +
    /// src.dims()[dim] > self.dims()[dim]` -- makes the copy a no-op instead of writing out of
    /// bounds; the bound is enforced in the kernel, since `*position` is only readable there.
    /// The skip is silent: nothing is returned to report it, because on a replay there is no
    /// Rust code running to return it to. Callers that need to know a step was dropped have to
    /// track the position themselves (see [`Cache::append_at`](crate::Cache::append_at)).
    fn slice_set_fingerprinted_at<D: Dim>(
        &self,
        src: &Self,
        dim: D,
        offset: usize,
        position: &Self,
    ) -> Result<()>;
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

    fn slice_set_fingerprinted_at<D: Dim>(
        &self,
        src: &Self,
        dim: D,
        offset: usize,
        position: &Self,
    ) -> Result<()> {
        let dim = dim.to_index(self.shape(), "slice-set-at")?;
        if !self.is_contiguous() || !src.is_contiguous() {
            Err(Error::RequiresContiguous { op: "slice-set-at" }.bt())?
        }
        if self.dtype() != src.dtype() {
            Err(Error::DTypeMismatchBinaryOp {
                lhs: self.dtype(),
                rhs: src.dtype(),
                op: "slice-set-at",
            }
            .bt())?
        }
        if self.device().location() != src.device().location()
            || self.device().location() != position.device().location()
        {
            Err(Error::DeviceMismatchBinaryOp {
                lhs: self.device().location(),
                rhs: src.device().location(),
                op: "slice-set-at",
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
        if position.dtype() != DType::U32 {
            candle_core::bail!(
                "slice-set-at: position must be U32, got {:?}",
                position.dtype()
            )
        }
        if position.elem_count() != 1 {
            candle_core::bail!(
                "slice-set-at: position must hold exactly one element, got shape {:?}",
                position.shape()
            )
        }
        if !position.is_contiguous() {
            Err(Error::RequiresContiguous { op: "slice-set-at" }.bt())?
        }
        for (dim_idx, (v1, v2)) in self.dims().iter().zip(src.dims().iter()).enumerate() {
            // `dim` itself is handled below: unlike `slice_set_fingerprinted`, the write position
            // on `dim` is not fully known here, so the two halves of that bound are checked
            // separately.
            if dim_idx != dim && v1 != v2 {
                candle_core::bail!("shape mismatch on dim {dim_idx}, {v1} <> {v2}")
            }
        }
        // The write covers `offset + *position .. offset + *position + src_dim` on `dim`.
        // `offset` and `src_dim` are fixed at capture time, so the room they leave is known
        // here; only `*position` is not, since it lives on the device and is rewritten between
        // replays. So the host checks the static half and hands the kernel the largest position
        // that still fits, and the kernel drops the copy when `*position` exceeds it -- see
        // `dst_o_ptr_max` in `update_kv.cu`. Without that, an out-of-range position is scaled by
        // the destination stride and written past the allocation, which a safe API must not
        // allow whatever the caller passes in.
        let dst_dim = self.dims()[dim];
        let src_dim = src.dims()[dim];
        if offset + src_dim > dst_dim {
            candle_core::bail!(
                "slice-set-at: shape mismatch on target dim, dst: {dst_dim}, src: {src_dim} + {offset}"
            )
        }
        let position_max = dst_dim - offset - src_dim;
        let block_size: usize = src.dims().iter().skip(1 + dim).product();
        let d1: usize = src.dims().iter().take(dim).product();
        let d2 = block_size * src.dims()[dim];
        let dst_o_base = self.layout().start_offset() + offset * block_size;
        let src_o = src.layout().start_offset();
        // `position` may be a view into a larger buffer -- `positions.narrow(0, i, 1)` is
        // contiguous and holds one element, so it passes the checks above while its element
        // lives at `start_offset`, not at the start of the storage.
        let position_o = position.layout().start_offset();
        match (
            &*self.storage_and_layout().0,
            &*src.storage_and_layout().0,
            &*position.storage_and_layout().0,
        ) {
            (Storage::Cuda(dst), Storage::Cuda(src), Storage::Cuda(position)) => self
                .copy2d_fingerprinted_dynoffset(
                    src,
                    dst,
                    d1,
                    d2,
                    d2,
                    block_size * self.dims()[dim],
                    src_o,
                    dst_o_base,
                    block_size,
                    position,
                    position_o,
                    position_max,
                ),
            _ => Err(Error::DeviceMismatchBinaryOp {
                lhs: self.device().location(),
                rhs: src.device().location(),
                op: "copy2d CUDA",
            }
            .bt()),
        }
    }
}

/// Narrows a host-side `usize` to the `uint32_t` the `copy2d_*` kernels take, refusing to
/// truncate.
///
/// Every index the kernels use is 32-bit, while the checks that feed them are computed in
/// `usize`. A destination too big to address in 32 bits -- a bf16 KV cache of about 8 GiB is
/// already there -- would pass those checks and then have its offsets wrapped on the way to the
/// kernel, writing into the wrong region of the cache. Widening the kernels to 64-bit indices is
/// the other way out, and a larger change than this.
fn to_u32(v: usize, what: &str) -> Result<u32> {
    match u32::try_from(v) {
        Ok(v) => Ok(v),
        Err(_) => candle_core::bail!(
            "slice-set-at: {what} ({v}) does not fit in u32; the copy2d kernels index in 32 bits"
        ),
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

    #[allow(clippy::too_many_arguments)]
    fn copy2d_fingerprinted_dynoffset(
        &self,
        src: &CudaStorage,
        dst: &CudaStorage,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o_base: usize,
        dst_o_stride: usize,
        dst_o_ptr: &CudaStorage,
        dst_o_ptr_offset: usize,
        dst_o_ptr_max: usize,
    ) -> Result<()> {
        let dev = &src.device;
        let d1 = to_u32(d1, "d1")?;
        let d2 = to_u32(d2, "d2")?;
        // Nothing to copy so we exit early to avoid launching a kernel and some potential invalid
        // argument with a null pointer.
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        let dst_s = to_u32(dst_s, "dst stride")?;
        let src_s = to_u32(src_s, "src stride")?;
        let src_o = to_u32(src_o, "src offset")?;
        let dst_o_base = to_u32(dst_o_base, "dst base offset")?;
        let dst_o_stride = to_u32(dst_o_stride, "dst position stride")?;
        let dst_o_ptr_max = to_u32(dst_o_ptr_max, "max position")?;
        // `slice_set_fingerprinted_at` already checked `position.dtype() == DType::U32` before
        // calling here; this match is exhaustive-by-construction, not a runtime dtype guard.
        // The pointer is advanced to the view's own element: the storage pointer alone would
        // make the kernel read element 0 of the underlying buffer, which is the wrong slot for
        // any `position` that is a view (e.g. `positions.narrow(0, i, 1)`).
        let position_ptr = match &dst_o_ptr.slice {
            S::U32(p) => *p.device_ptr() + (dst_o_ptr_offset * std::mem::size_of::<u32>()) as u64,
            _ => Err(CudaError::InternalError(
                "slice-set-at: position storage must be U32",
            ))?,
        };
        let (src, dst, kname) = match (&src.slice, &dst.slice) {
            (S::U8(s), S::U8(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_dynoffset_u8"),
            (S::U32(s), S::U32(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_dynoffset_u32"),
            (S::I64(s), S::I64(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_dynoffset_i64"),
            (S::BF16(s), S::BF16(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_dynoffset_bf16"),
            (S::F16(s), S::F16(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_dynoffset_f16"),
            (S::F32(s), S::F32(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_dynoffset_f32"),
            (S::F64(s), S::F64(d)) => (*s.device_ptr(), *d.device_ptr(), "copy2d_dynoffset_f64"),
            _ => Err(CudaError::InternalError(
                "dtype mismatch in copy2d_dynoffset",
            ))?,
        };
        let func = dev.get_or_load_func(kname, kernels::UPDATE_KV)?;
        // The kernel compares its thread index against `d1 * d2` in 32 bits too, so this has to
        // hold there as well, not just for the launch config.
        let Some(num_elems) = d1.checked_mul(d2) else {
            candle_core::bail!(
                "slice-set-at: element count ({d1} * {d2}) does not fit in u32; \
                 the copy2d kernels index in 32 bits"
            )
        };
        let cfg = LaunchConfig::for_num_elems(num_elems);
        let params = (
            COPY2D_FINGERPRINT,
            src,
            dst,
            d1,
            d2,
            src_o,
            dst_o_base,
            dst_o_stride,
            src_s,
            dst_s,
            position_ptr,
            dst_o_ptr_max,
        );
        // SAFETY: ffi.
        unsafe { func.launch(cfg, params) }.w()?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Result, Tensor};

    use crate::{
        extension::CudaTensorExtension,
        graph::{copy_inplace, Graph, GraphInput},
    };

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

    #[test]
    fn slice_set_at_matches_slice_set() -> Result<()> {
        let device = Device::new_cuda_with_stream(0)?;

        let (b, h, max_t, d) = (2, 4, 7, 3);
        let cache = Tensor::zeros((b, h, max_t, d), DType::F32, &device)?;
        let tensor = Tensor::randn(0f32, 1f32, (b, h, 1, d), &device)?;
        let position = Tensor::new(&[3u32], &device)?;

        cache.slice_set_fingerprinted_at(&tensor, 2, 0, &position)?;
        let cache_t = cache.narrow(2, 3, 1)?;
        let diff = (cache_t - &tensor)?.abs()?.sum_all()?.to_vec0::<f32>()?;
        assert_eq!(diff, 0.);

        // Every other slot must be untouched.
        for other in [0usize, 1, 2, 4, 5, 6] {
            let diff = cache
                .narrow(2, other, 1)?
                .abs()?
                .sum_all()?
                .to_vec0::<f32>()?;
            assert_eq!(diff, 0., "slot {other} should still be zero");
        }
        Ok(())
    }

    /// `position` is allowed to be a view: a `narrow` of a positions buffer is contiguous and
    /// holds one element, so it passes every check, but its element sits at the view's
    /// `start_offset` rather than at the start of the storage. Reading the storage pointer
    /// alone would make the kernel use `positions[0]` -- here 5 -- and write to the wrong slot.
    #[test]
    fn slice_set_at_honors_position_view_offset() -> Result<()> {
        let device = Device::new_cuda_with_stream(0)?;

        let (b, h, max_t, d) = (1, 1, 7, 3);
        let cache = Tensor::zeros((b, h, max_t, d), DType::F32, &device)?;
        let tensor = Tensor::ones((b, h, 1, d), DType::F32, &device)?;

        // Element 0 is a decoy: if the offset is ignored, the write lands on slot 5.
        let positions = Tensor::new(&[5u32, 2], &device)?;
        let position = positions.narrow(0, 1, 1)?;
        assert_eq!(position.layout().start_offset(), 1);

        cache.slice_set_fingerprinted_at(&tensor, 2, 0, &position)?;

        for slot in 0usize..max_t {
            let sum = cache
                .narrow(2, slot, 1)?
                .abs()?
                .sum_all()?
                .to_vec0::<f32>()?;
            let expected = if slot == 2 { d as f32 } else { 0. };
            assert_eq!(sum, expected, "slot {slot}: write landed in the wrong slot");
        }
        Ok(())
    }

    /// `*position` lives on the device, so nothing on the host can stop a caller -- or a
    /// replay whose position tensor was advanced one step too far -- from naming a slot past
    /// the end of the cache. Unbounded, that offset is scaled by the destination stride and
    /// written straight past the allocation. The kernel drops the copy instead.
    #[test]
    fn slice_set_at_skips_out_of_range_position() -> Result<()> {
        let device = Device::new_cuda_with_stream(0)?;

        let (b, h, max_t, d) = (1, 1, 4, 3);
        let cache = Tensor::zeros((b, h, max_t, d), DType::F32, &device)?;
        let tensor = Tensor::ones((b, h, 1, d), DType::F32, &device)?;

        // One past the last writable slot, and a value that would overflow the u32 offset
        // arithmetic if it were left to wrap.
        for out_of_range in [max_t as u32, max_t as u32 + 7, u32::MAX] {
            let position = Tensor::new(&[out_of_range], &device)?;
            cache.slice_set_fingerprinted_at(&tensor, 2, 0, &position)?;
            device.synchronize()?;
            let touched = cache.abs()?.sum_all()?.to_vec0::<f32>()?;
            assert_eq!(touched, 0., "position {out_of_range} wrote into the cache");
        }

        // The last in-range slot still writes, so the bound is not off by one.
        let position = Tensor::new(&[max_t as u32 - 1], &device)?;
        cache.slice_set_fingerprinted_at(&tensor, 2, 0, &position)?;
        for slot in 0usize..max_t {
            let sum = cache
                .narrow(2, slot, 1)?
                .abs()?
                .sum_all()?
                .to_vec0::<f32>()?;
            let expected = if slot == max_t - 1 { d as f32 } else { 0. };
            assert_eq!(sum, expected, "slot {slot}");
        }
        Ok(())
    }

    /// The half of the bound that *is* known on the host -- `offset` plus the source's own
    /// extent -- is an error rather than a silent skip, since it is wrong for every position.
    #[test]
    fn slice_set_at_rejects_a_source_that_cannot_fit() -> Result<()> {
        let device = Device::new_cuda_with_stream(0)?;

        let (b, h, max_t, d) = (1, 1, 4, 3);
        let cache = Tensor::zeros((b, h, max_t, d), DType::F32, &device)?;
        let position = Tensor::new(&[0u32], &device)?;

        // Source longer than the cache: no position can make this fit.
        let too_long = Tensor::ones((b, h, max_t + 1, d), DType::F32, &device)?;
        assert!(cache
            .slice_set_fingerprinted_at(&too_long, 2, 0, &position)
            .is_err());

        // Fits on its own, but not at this `offset`.
        let tensor = Tensor::ones((b, h, 2, d), DType::F32, &device)?;
        assert!(cache
            .slice_set_fingerprinted_at(&tensor, 2, 3, &position)
            .is_err());
        assert!(cache
            .slice_set_fingerprinted_at(&tensor, 2, 2, &position)
            .is_ok());
        Ok(())
    }

    /// The bug this module exists to fix: without a device-resident position, a
    /// `slice_set_fingerprinted` call captured into a graph writes to the same
    /// slot on every replay, because the offset is baked into the graph as a
    /// plain kernel-launch argument at capture time. This test captures a graph
    /// containing a single `slice_set_fingerprinted_at` call once, replays it
    /// three times with `position` updated to 0, 1, 2 between replays (never
    /// re-capturing), and asserts each replay landed in its own slot -- i.e.
    /// that the write position tracks graph *replay* time, not capture time.
    #[test]
    fn slice_set_at_survives_graph_replay() -> anyhow::Result<()> {
        struct Inputs {
            x: Tensor,
            position: Tensor,
        }

        impl GraphInput for Inputs {
            fn to_inputs(&self) -> HashMap<String, Tensor> {
                [
                    ("x".to_string(), self.x.clone()),
                    ("position".to_string(), self.position.clone()),
                ]
                .into()
            }
            fn load_inputs_inplace(&self, input: Self, device: &Device) -> anyhow::Result<()> {
                unsafe {
                    copy_inplace(&input.x, &self.x, device)?;
                    copy_inplace(&input.position, &self.position, device)?;
                }
                Ok(())
            }
        }

        let device = Device::new_cuda_with_stream(0)?;
        let (b, h, max_t, d) = (1, 1, 4, 2);

        let cache = Tensor::zeros((b, h, max_t, d), DType::F32, &device)?;
        let x = Tensor::zeros((b, h, 1, d), DType::F32, &device)?;
        let position = Tensor::new(&[0u32], &device)?;

        let graph = Graph::new(
            |input: &Inputs| {
                cache.slice_set_fingerprinted_at(&input.x, 2, 0, &input.position)?;
                Ok(())
            },
            &device,
            Inputs { x, position },
        )?;

        for step in 0u32..3 {
            // `Tensor::full` broadcasts a 1-element storage, so it has to be materialized
            // before `copy_inplace` (which memcpy's the raw storage) can use it.
            let x = Tensor::full(step as f32 + 1., (b, h, 1, d), &device)?.contiguous()?;
            let position = Tensor::new(&[step], &device)?;
            graph.replay(Inputs { x, position })?;

            let written = cache
                .narrow(2, step as usize, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for &v in &written {
                assert_eq!(v, step as f32 + 1., "slot {step} after replay {step}");
            }
        }

        // The final replay must not have touched earlier slots -- if the offset
        // were still pinned at capture time (0), every replay would have
        // overwritten slot 0 instead of advancing.
        for slot in 0usize..2 {
            let expected = slot as f32 + 1.;
            let got = cache.narrow(2, slot, 1)?.flatten_all()?.to_vec1::<f32>()?;
            for &v in &got {
                assert_eq!(
                    v, expected,
                    "slot {slot} must retain its own replay's value"
                );
            }
        }

        Ok(())
    }
}
