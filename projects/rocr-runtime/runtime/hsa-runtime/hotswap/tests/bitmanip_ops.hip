#include <hip/hip_runtime.h>
extern "C" __global__ void bitmanip_ops(int* out, const unsigned* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned v = in[i];
    int clz = __clz(v);           // v_ffbh_u32
    int ffs_val = __ffs(v);       // v_ffbl_b32
    int popc = __popc(v);         // v_bcnt_u32_b32
    out[i] = clz + ffs_val * 100 + popc * 10000;
  }
}
