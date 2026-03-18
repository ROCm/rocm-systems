#include <hip/hip_runtime.h>
// 1D stencil: weighted average of neighbors [1,2,1]/4
// Tests neighbor loads with clamped index arithmetic (no scale_offset on loads)
extern "C" __global__ void stencil_1d(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int im1 = (i > 0) ? i - 1 : 0;
    int ip1 = (i < n - 1) ? i + 1 : n - 1;
    out[i] = (in[im1] + 2.0f * in[i] + in[ip1]) * 0.25f;
  }
}
