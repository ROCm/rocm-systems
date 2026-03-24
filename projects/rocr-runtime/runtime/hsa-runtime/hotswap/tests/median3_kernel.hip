#include <hip/hip_runtime.h>
extern "C" __global__ void median3_kernel(float* out, const float* a, const float* b,
                                           const float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = a[i], y = b[i], z = c[i];
    out[i] = fmaxf(fminf(x, y), fminf(fmaxf(x, y), z));
  }
}
