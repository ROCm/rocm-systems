#include <hip/hip_runtime.h>
// L2 distance: exercises v_sub, v_fma, v_sqrt (via sqrtf which uses rsqrt+NR)
// No device library dependency — uses hardware transcendentals only.
extern "C" __global__ void l2_distance_kernel(float* out, const float* a, const float* b, int D, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
      float diff = a[i*D+d] - b[i*D+d];
      sum += diff * diff;
    }
    out[i] = sqrtf(sum);
  }
}
