#include <hip/hip_runtime.h>
extern "C" __global__ void log_rcp_kernel(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = in[i];
    float lg = log2f(x);
    float rc = 1.0f / x;
    out[i] = lg * rc;
  }
}
