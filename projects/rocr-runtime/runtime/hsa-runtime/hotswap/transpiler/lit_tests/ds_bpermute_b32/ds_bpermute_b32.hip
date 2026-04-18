// ARCH: gfx942
//
// Fixture for the DS_BPERMUTE_B32 gather lit test
// (lit_tests/ds_bpermute_b32).
//
// The `handle_ds.cpp` handler for `DS_BPERMUTE_B32` was an identity
// copy of `src(1)` until the recent fix. That silently collapsed
// every `__shfl_xor` / `__shfl_down` / `__shfl` to "each lane keeps
// its own partial", which is how `lane_swap` and `block_sum_shfl`
// compare_correctness recipes failed with "every output equals the
// input". The fix lowers through `llvm.amdgcn.ds.bpermute`, which the
// backend maps to the real cross-lane gather on the target ISA.
//
// This fixture uses a native gfx942 kernel so the test isolates the
// handler's gather shape from any cross-wave projection concerns:
// source == target == wave64. The MODREP-tagged wave-size assumptions
// in `handle_ds.cpp` are covered in the handler comment; exercising
// them in a lit test would need a gfx1250 → gfx942 lift with
// __shfl_xor, which introduces its own lane-id translation noise
// (see the residual `lane_swap` / `block_sum_shfl` numerical
// mismatches in RESULTS.md).
//
// The kernel uses `__shfl_xor(x, 1)` — the smallest XOR selector — so
// hipcc reliably emits `ds_bpermute_b32` with a selector derived from
// `(lane ^ 1) * 4`. Taking `partner + x` for the output ensures the
// load is not DCE'd.

#include <hip/hip_runtime.h>

extern "C" __global__ void ds_bpermute_b32_kernel(const float *in, float *out,
                                                   int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  float x = in[tid];
  float partner = __shfl_xor(x, 1);
  out[tid] = x + partner;
}
