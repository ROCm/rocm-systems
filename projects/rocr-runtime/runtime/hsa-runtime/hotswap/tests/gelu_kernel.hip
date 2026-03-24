#include <hip/hip_runtime.h>
extern "C" __global__ void gelu_kernel(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = in[i];
    // GELU = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    out[i] = 0.5f * x * (1.0f + tanhf(inner));
  }
}
