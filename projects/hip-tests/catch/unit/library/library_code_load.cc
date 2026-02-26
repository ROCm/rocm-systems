#include <hip/hip_runtime.h>

extern "C" {
__global__ void add_kernel(float* out, float* a, float* b) {
  size_t i = threadIdx.x;
  out[i] = a[i] + b[i];
}
__global__ void sub_kernel(float* out, float* a, float* b) {
  size_t i = threadIdx.x;
  out[i] = a[i] - b[i];
}
__global__ void mul_kernel(float* out, float* a, float* b) {
  size_t i = threadIdx.x;
  out[i] = a[i] * b[i];
}
__global__ void add_shared_kernel(float* out, float* a, float* b) {
  extern __shared__ float sBuf[];
  size_t i = threadIdx.x;
  sBuf[i] = a[i] + b[i];
  __syncthreads();
  out[i] = sBuf[i];
}
}
