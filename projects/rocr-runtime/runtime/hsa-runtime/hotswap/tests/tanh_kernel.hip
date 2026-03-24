#include <hip/hip_runtime.h>

// Simple tanh kernel — exercises device library tanhf (uses many VGPRs
// including v5-v8 for polynomial evaluation).  This is a targeted test
// for the temp VGPR collision fix: the transpiler's instruction expansion
// previously clobbered v6 which tanhf uses as a live data register.
extern "C" __global__ void tanh_kernel(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tanhf(in[i]);
}
