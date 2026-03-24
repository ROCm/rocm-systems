#include <hip/hip_runtime.h>
extern "C" __global__ void atomic_histogram(int* global_hist, const int* data,
                                             int n, int num_bins) {
  extern __shared__ int shared_hist[];
  int tid = threadIdx.x;
  for (int b = tid; b < num_bins; b += blockDim.x) shared_hist[b] = 0;
  __syncthreads();
  int i = blockIdx.x * blockDim.x + tid;
  if (i < n) {
    int bin = data[i] % num_bins;
    if (bin < 0) bin += num_bins;
    atomicAdd(&shared_hist[bin], 1);
  }
  __syncthreads();
  for (int b = tid; b < num_bins; b += blockDim.x)
    atomicAdd(&global_hist[b], shared_hist[b]);
}
