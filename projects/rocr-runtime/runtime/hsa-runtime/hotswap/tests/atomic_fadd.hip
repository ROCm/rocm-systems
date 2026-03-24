#include <hip/hip_runtime.h>
extern "C" __global__ void atomic_fadd(float* bins, const float* vals,
                                        const int* bin_ids, int n, int num_bins) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&bins[bin_ids[i]], vals[i]);
}
