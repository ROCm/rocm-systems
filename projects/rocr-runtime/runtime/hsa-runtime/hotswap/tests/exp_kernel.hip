#include <hip/hip_runtime.h>
// Element-wise exp: out[i] = expf(in[i])
// Tests transcendental sequence (v_exp_f32 + range reduction + VOPD patterns)
extern "C" __global__ void exp_kernel(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = expf(in[i]);
}
