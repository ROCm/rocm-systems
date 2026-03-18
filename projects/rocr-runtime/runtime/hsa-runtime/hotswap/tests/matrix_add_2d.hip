#include <hip/hip_runtime.h>
// 1D thread block, 2D grid: tests blockIdx.y extraction without
// the gfx1250 packed-thread-ID issue (v0 = (y<<10)|x).
extern "C" __global__ void matrix_add_2d(float* out, const float* a, const float* b, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (row < rows && col < cols) {
    int i = row * cols + col;
    out[i] = a[i] + b[i];
  }
}
