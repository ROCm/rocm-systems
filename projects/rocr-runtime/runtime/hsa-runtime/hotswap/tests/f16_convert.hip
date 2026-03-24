#include <hip/hip_runtime.h>
extern "C" __global__ void f16_convert(float* out, const unsigned short* in_f16, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    _Float16 h = *reinterpret_cast<const _Float16*>(&in_f16[i]);
    float f = (float)h;
    out[i] = f * 2.0f + 1.0f;
  }
}
