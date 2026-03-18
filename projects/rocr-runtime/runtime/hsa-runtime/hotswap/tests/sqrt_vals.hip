#include <hip/hip_runtime.h>
// Element-wise square root — tests v_sqrt_f32 (transcendental instruction)
extern "C" __global__ void sqrt_vals(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = sqrtf(in[i]);
}
