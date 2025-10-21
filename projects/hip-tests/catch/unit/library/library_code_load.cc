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
}
