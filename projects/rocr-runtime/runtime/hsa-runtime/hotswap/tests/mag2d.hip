#include <hip/hip_runtime.h>
// 2D vector magnitude: out[i] = sqrtf(a[i]*a[i] + b[i]*b[i])
// Tests FMA (a²+b² = fma(a,a,b*b)) + sqrt sequence; exercises VOPD patterns
extern "C" __global__ void mag2d(float* out, const float* a, const float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = sqrtf(__fmaf_rn(a[i], a[i], b[i] * b[i]));
}
