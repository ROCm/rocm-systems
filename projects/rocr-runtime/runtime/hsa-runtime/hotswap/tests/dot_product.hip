#include <hip/hip_runtime.h>
extern "C" __global__ void dot_product(float* out, const float* a, const float* b, int n) {
  extern __shared__ float sdata[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[threadIdx.x] = (i < n) ? a[i] * b[i] : 0.0f;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = sdata[0];
}
