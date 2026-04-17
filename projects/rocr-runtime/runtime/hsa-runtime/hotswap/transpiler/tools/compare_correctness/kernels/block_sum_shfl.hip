// Per-block reduction with a wave-size-AGNOSTIC output shape: one float
// per block, the sum of all block elements.
//
// The wave-size dependency is confined to the kernel's internals (the
// warp-level shuffle phase and the cross-warp shared-memory phase);
// the output buffer has `gridDim.x` floats regardless of what
// `warpSize` the kernel was compiled for.  That means element-by-
// element comparison of outputs across ISAs is meaningful — the two
// engines are computing the same quantity, not apples-to-oranges
// per-warp partial sums.
//
// This kernel probes two things at once: warp-level shuffle reductions
// (phase 1) and cross-warp reductions via shared memory (phase 2).

#include <hip/hip_runtime.h>

// Max warps per block we support.  256 threads / 32-lane warps = 8;
// 1024 / 32 = 32; pick a safe upper bound.
constexpr int MAX_WARPS = 64;

extern "C" __global__ void block_sum_shfl(const float *in, float *out, int n) {
  __shared__ float partial[MAX_WARPS];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float x = (tid < n) ? in[tid] : 0.0f;

  // Phase 1: per-warp reduction via shuffle.  warpSize is a compile-time
  // constant set by the offload-arch.
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    x += __shfl_xor(x, off);
  }

  int lane = threadIdx.x & (warpSize - 1);
  int warp = threadIdx.x / warpSize;
  int num_warps = blockDim.x / warpSize;
  if (lane == 0) partial[warp] = x;
  __syncthreads();

  // Phase 2: first warp reduces the per-warp partials.
  if (warp == 0) {
    float s = (lane < num_warps) ? partial[lane] : 0.0f;
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      s += __shfl_xor(s, off);
    }
    if (lane == 0) out[blockIdx.x] = s;
  }
}
