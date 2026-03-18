#include <hip/hip_runtime.h>
// Fused multiply-add: out[i] = a[i]*b[i] + c[i]
// Tests v_fmac_f32 (GFX12 in-place FMA) → v_mac_f32 (GFX9)
extern "C" __global__ void fma_vals(float* out, const float* a, const float* b,
                                    const float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = __fmaf_rn(a[i], b[i], c[i]);
}
