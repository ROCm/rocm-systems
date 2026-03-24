#include <hip/hip_runtime.h>

// Sigmoid kernel: 1/(1+exp(-x)) — exercises device library expf chain.
// Uses many VGPRs for the polynomial approximation, making it a good
// test for the temp VGPR collision fix.
extern "C" __global__ void sigmoid_kernel(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 1.0f / (1.0f + expf(-in[i]));
}
