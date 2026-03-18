#include <hip/hip_runtime.h>
extern "C" __global__ void clamp_kernel(float* out, const float* in, float lo, float hi, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = in[i];
    out[i] = v < lo ? lo : (v > hi ? hi : v);
  }
}
