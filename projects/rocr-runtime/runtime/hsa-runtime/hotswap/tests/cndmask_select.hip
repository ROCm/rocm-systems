#include <hip/hip_runtime.h>
extern "C" __global__ void cndmask_select(float* out, const float* a, const float* b,
                                           const int* mask, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sel = mask[i] ? a[i] : b[i];
    float sel2 = (mask[i] > 0) ? a[i] * 2.0f : b[i] * 0.5f;
    out[i] = sel + sel2;
  }
}
