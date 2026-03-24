#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void packed_f16(float* out, const _Float16* a, const _Float16* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i*2+1 < n) {
    _Float16 a0=a[i*2], a1=a[i*2+1], b0=b[i*2], b1=b[i*2+1];
    _Float16 r0 = a0 + b0;  // v_pk_add_f16
    _Float16 r1 = a1 * b1;  // v_pk_mul_f16
    out[i] = (float)r0 + (float)r1;
  }
}
