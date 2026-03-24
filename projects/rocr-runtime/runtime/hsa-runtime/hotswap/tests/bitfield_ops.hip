#include <hip/hip_runtime.h>
extern "C" __global__ void bitfield_ops(unsigned* out, const unsigned* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned v = in[i];
    unsigned byte1 = (v >> 8) & 0xFF;
    unsigned byte2 = (v >> 16) & 0xFF;
    unsigned result = (v & 0xFF0000FFu) | (byte1 << 16) | (byte2 << 8);
    out[i] = result;
  }
}
