#include <hip/hip_runtime.h>
// Reciprocal: out[i] = 1.0f / in[i]
// Tests v_rcp_f32 + Newton-Raphson FMA refinement (v_fmac_f32 / v_fma_f32)
extern "C" __global__ void recip_vals(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = 1.0f / in[i];
}
