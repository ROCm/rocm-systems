#include <hip/hip_runtime.h>

extern "C" __global__ void warp_shuffle(int* out, const int* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int val = in[i];
    // Broadcast lane 0's value to all lanes, then add own value
    int lane0_val = __shfl(val, 0);
    out[i] = val + lane0_val;
  }
}
