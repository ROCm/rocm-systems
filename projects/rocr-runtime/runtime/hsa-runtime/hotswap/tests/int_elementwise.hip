#include <hip/hip_runtime.h>
// Integer elementwise: out[i] = a[i]*2 + b[i] - tests integer VALU ops
extern "C" __global__ void int_elementwise(int* out, const int* a, const int* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * 2 + b[i];
}
