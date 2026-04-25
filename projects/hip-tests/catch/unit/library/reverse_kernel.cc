#include <hip/hip_runtime.h>

extern "C" {
__global__ void reverse(int* d, int n) {
  __shared__ int shBuf[64];
  int t = threadIdx.x;
  int tr = n - t - 1;

  shBuf[t] = d[t];
  __syncthreads();
  d[t] = shBuf[tr];
}
__global__ void add_kernel(float* out, float* a, float* b) {
  size_t i = threadIdx.x;
  out[i] = a[i] + b[i];
}
}
