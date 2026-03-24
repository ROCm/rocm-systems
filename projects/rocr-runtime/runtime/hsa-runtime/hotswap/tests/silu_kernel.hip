#include <hip/hip_runtime.h>
extern "C" __global__ void silu_kernel(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = in[i];
    out[i] = x / (1.0f + expf(-x));  // SiLU = x * sigmoid(x)
  }
}
