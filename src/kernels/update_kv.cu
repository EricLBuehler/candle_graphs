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

COPY2D_OP(float, copy2d_f32)
COPY2D_OP(double, copy2d_f64)
COPY2D_OP(uint8_t, copy2d_u8)
COPY2D_OP(uint32_t, copy2d_u32)
COPY2D_OP(int16_t, copy2d_i16)
COPY2D_OP(int32_t, copy2d_i32)
COPY2D_OP(int64_t, copy2d_i64)

#if __CUDA_ARCH__ >= 530
#include "cuda_fp16.h"
COPY2D_OP(__half, copy2d_f16)
#endif

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
COPY2D_OP(__nv_bfloat16, copy2d_bf16)
#endif
