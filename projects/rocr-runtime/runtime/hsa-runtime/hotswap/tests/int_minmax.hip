#include <hip/hip_runtime.h>
extern "C" __global__ void int_minmax(int* out, const int* a, const int* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int x = a[i], y = b[i];
    int mn = min(x, y);
    int mx = max(x, y);
    int d = abs(x - y);
    out[i] = mn + mx - d;
  }
}
