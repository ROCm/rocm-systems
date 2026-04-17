// Baseline kernel: pure VALU, no cross-lane ops, no wave-width
// dependencies.  Serves as the baseline probe in the correctness
// harness — a structural sanity check before anything more interesting.

#include <hip/hip_runtime.h>

extern "C" __global__ void vecadd(const float *a, const float *b, float *c,
                                  int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
