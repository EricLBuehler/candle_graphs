#include <stdint.h>

template<typename T>
__device__ void copy2d(
  const uint64_t fingerprint,
  const T *src, T *dst,
  uint32_t d1, uint32_t d2,
  uint32_t src_o, uint32_t dst_o,
  uint32_t src_s, uint32_t dst_s
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= d1 * d2) {
    return;
  }
  uint32_t idx1 = idx / d2;
  uint32_t idx2 = idx - d2 * idx1;
  (dst + dst_o)[idx1 * dst_s + idx2] = (src + src_o)[idx1 * src_s + idx2];
}

// Same as `copy2d`, except the destination offset is read from `dst_o_ptr` at
// launch time instead of being passed as a plain scalar. A plain `dst_o` is a
// kernel-launch argument, so once a call is captured into a CUDA graph its
// value is frozen for every future replay: `Cache::append`'s current-seq-len
// offset would stay pinned at whatever it was during the one-time capture,
// and every replay would overwrite the same slot instead of advancing. Since
// `dst_o_ptr` is a device pointer, the *pointer* is what gets baked into the
// graph, not the value at that address -- writing a new value to `*dst_o_ptr`
// between replays (see `copy_inplace`) changes what the next replay writes to
// without re-capturing anything.
//
// That same indirection is why the bound has to be enforced here: `*dst_o_ptr`
// lives on the device and is rewritten between replays, so the host cannot
// check it the way it checks `copy2d`'s host-known `dst_o`. Left unchecked, a
// position past the end of the destination is scaled by `dst_o_stride` and
// written straight past the allocation. `dst_o_ptr_max` is the largest
// position that still leaves room for the whole source slab; the host computes
// it from the destination's extent on the copied dimension, and it is a plain
// scalar because it does not change between replays (the shapes are fixed at
// capture time -- only the position moves).
template<typename T>
__device__ void copy2d_dynoffset(
  const uint64_t fingerprint,
  const T *src, T *dst,
  uint32_t d1, uint32_t d2,
  uint32_t src_o, uint32_t dst_o_base, uint32_t dst_o_stride,
  uint32_t src_s, uint32_t dst_s,
  const uint32_t *dst_o_ptr, uint32_t dst_o_ptr_max
) {
  // Read once, then check and use that same value. Nothing stops `dst_o_ptr`
  // from aliasing a slot this very launch writes, and re-reading it for the
  // offset would let a neighbour's write land between the check and its use --
  // the bound would then be enforced against a position that is no longer the
  // one written at. One load per thread means each thread writes only where it
  // checked: an aliasing caller still gets garbage, but in bounds.
  const uint32_t position = *dst_o_ptr;
  // The whole copy is dropped rather than clipped -- a partially written slot
  // would leave the destination in a state the caller has no way to reason
  // about. Barring that aliasing case, every thread loads the same position, so
  // this is uniform and the launch drops as one.
  if (position > dst_o_ptr_max) {
    return;
  }
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= d1 * d2) {
    return;
  }
  uint32_t idx1 = idx / d2;
  uint32_t idx2 = idx - d2 * idx1;
  // Widened to 64 bits: the host refuses to launch unless each of `dst_o_base`,
  // `dst_o_stride` and `dst_o_ptr_max` fits in 32 bits, but their *sum* need not
  // -- a destination past 4G elements is reachable (a bf16 KV cache of ~8 GiB
  // already is), and a wrap would land the write at a wrong offset *inside* the
  // allocation, which no bounds check can catch because it is still in range.
  const uint64_t dst_index = (uint64_t)dst_o_base
                           + (uint64_t)position * dst_o_stride
                           + (uint64_t)idx1 * dst_s + idx2;
  const uint64_t src_index = (uint64_t)src_o + (uint64_t)idx1 * src_s + idx2;
  dst[dst_index] = src[src_index];
}

#define COPY2D_OP(TYPENAME, FNNAME) \
extern "C" __global__ \
void FNNAME( \
  const uint64_t fingerprint, \
  const TYPENAME *src, TYPENAME *dst, \
  uint32_t d1, uint32_t d2, \
  uint32_t src_o, uint32_t dst_o, \
  uint32_t src_s, uint32_t dst_s \
) { \
  copy2d(fingerprint, src, dst, d1, d2, src_o, dst_o, src_s, dst_s); \
} \

#define COPY2D_DYNOFFSET_OP(TYPENAME, FNNAME) \
extern "C" __global__ \
void FNNAME( \
  const uint64_t fingerprint, \
  const TYPENAME *src, TYPENAME *dst, \
  uint32_t d1, uint32_t d2, \
  uint32_t src_o, uint32_t dst_o_base, uint32_t dst_o_stride, \
  uint32_t src_s, uint32_t dst_s, \
  const uint32_t *dst_o_ptr, uint32_t dst_o_ptr_max \
) { \
  copy2d_dynoffset(fingerprint, src, dst, d1, d2, src_o, dst_o_base, dst_o_stride, src_s, dst_s, dst_o_ptr, dst_o_ptr_max); \
} \

COPY2D_OP(float, copy2d_f32)
COPY2D_OP(double, copy2d_f64)
COPY2D_OP(uint8_t, copy2d_u8)
COPY2D_OP(uint32_t, copy2d_u32)
COPY2D_OP(int16_t, copy2d_i16)
COPY2D_OP(int32_t, copy2d_i32)
COPY2D_OP(int64_t, copy2d_i64)

COPY2D_DYNOFFSET_OP(float, copy2d_dynoffset_f32)
COPY2D_DYNOFFSET_OP(double, copy2d_dynoffset_f64)
COPY2D_DYNOFFSET_OP(uint8_t, copy2d_dynoffset_u8)
COPY2D_DYNOFFSET_OP(uint32_t, copy2d_dynoffset_u32)
COPY2D_DYNOFFSET_OP(int16_t, copy2d_dynoffset_i16)
COPY2D_DYNOFFSET_OP(int32_t, copy2d_dynoffset_i32)
COPY2D_DYNOFFSET_OP(int64_t, copy2d_dynoffset_i64)

#if __CUDA_ARCH__ >= 530
#include "cuda_fp16.h"
COPY2D_OP(__half, copy2d_f16)
COPY2D_DYNOFFSET_OP(__half, copy2d_dynoffset_f16)
#endif

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
COPY2D_OP(__nv_bfloat16, copy2d_bf16)
COPY2D_DYNOFFSET_OP(__nv_bfloat16, copy2d_dynoffset_bf16)
#endif
