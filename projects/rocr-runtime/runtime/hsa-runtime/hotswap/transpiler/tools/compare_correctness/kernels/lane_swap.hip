// Per-thread lane swap: out[tid] = in[tid ^ 1].
//
// 1:1 output shape, computed via a single `__shfl_xor(x, 1)` cross-lane
// read followed by a store.  Crisp probe for whether a cross-lane op
// round-trips correctly — the expected behaviour is an elementwise
// pairwise swap of the input array; any deviation shows up at the
// specific index that was misrouted.
//
// The harness pre-fills the output buffer with a sentinel value before
// dispatch, so "wrong value" and "never written" are visually distinct
// in the diff.

#include <hip/hip_runtime.h>

extern "C" __global__ void lane_swap(const float *in, float *out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  float x = in[tid];
  float partner = __shfl_xor(x, 1);
  out[tid] = partner;
}
