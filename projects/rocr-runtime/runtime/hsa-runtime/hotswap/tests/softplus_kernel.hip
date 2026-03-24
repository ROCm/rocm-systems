#include <hip/hip_runtime.h>
// Leaky ReLU: simple activation that exercises v_max, v_mul, v_cmp branches
// No device library dependency — pure VALU instructions.
extern "C" __global__ void leaky_relu_kernel(float* out, const float* in, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = in[i];
    out[i] = (x > 0.0f) ? x : alpha * x;
  }
}
