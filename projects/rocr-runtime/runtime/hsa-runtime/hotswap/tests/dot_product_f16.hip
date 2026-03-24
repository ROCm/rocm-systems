#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void dot_product_f16(float* out, const _Float16* a, const _Float16* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n/2) {
    float sum = 0;
    sum += (float)a[i*2] * (float)b[i*2];
    sum += (float)a[i*2+1] * (float)b[i*2+1];
    out[i] = sum;
  }
}
