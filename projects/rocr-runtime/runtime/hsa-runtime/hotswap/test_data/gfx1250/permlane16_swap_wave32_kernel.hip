// Source for permlane16_swap_wave32_gfx1250.hsaco — wave32-SOURCE
// test kernel for the P4 rewrite's cross-widening path.
//
// Why this exists alongside `permlane16_swap_kernel.hip`
// =====================================================
//
// `permlane16_swap_kernel.hip` uses `__launch_bounds__(64)`, which
// forces wave64 on gfx1250 (even though gfx1250 defaults to wave32).
// That test verifies wave64 → wave64 SAME-WAVE lift of
// `v_permlane16_swap_b32` — it does NOT exercise the cross-widening
// direction where a wave32 source projects to a wave64 target.
//
// The deterministic `canary_tl_sort_fp32_deterministic` probe
// (compare_correctness/kernels/triton/) showed that Triton's
// `tl.sort(dim=1, descending=True)` at (BLOCK_M=32, BLOCK_N=32)
// produces two SORTED HALVES on salmon (cols 0..15 sorted desc,
// cols 16..31 sorted desc) — the final bitonic merge step that
// uses `v_permlane16_swap_b32` is compositionally broken under
// wave32 → wave64 cross-widening.  The `emitPermLaneSwapEmulation`
// helper in `handle_valu_cross_lane.cpp` emits
// `ds_bpermute(lane_id ^ 16, src)` which LOOKS correct in
// isolation, but empirically produces a no-op at the Triton
// composition layer.
//
// This fixture isolates the question: is the emulation ITSELF
// broken under wave32 → wave64 projection, or is the composition
// (compare-and-cndmask dance around the swap) what's collapsing?
//
// Build (regenerate when the kernel is changed):
//   hipcc --offload-arch=gfx1250 --genco \
//     test_data/gfx1250/permlane16_swap_wave32_kernel.hip \
//     -o test_data/gfx1250/permlane16_swap_wave32_gfx1250.hsaco
//   /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler \
//     --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
//     --input=test_data/gfx1250/permlane16_swap_wave32_gfx1250.hsaco \
//     --output=test_data/gfx1250/permlane16_swap_wave32_gfx1250.hsaco \
//     --unbundle
//
// The `__launch_bounds__(32)` annotation pins the source kernel
// to wave32 on gfx1250.  hipcc honours this by setting the kernel's
// max_flat_workgroup_size metadata to 32 and emitting wave32
// codegen (no wave64 expansion of cross-lane semantics).
//
// Per-lane setup:
//   vdst_in[L]  = L                       (for L in [0, 32))
//   src0_in[L]  = 1000 + L
//
// Expected after the swap:
//   new_vdst[L]      = src0_in[L XOR 16] = 1000 + (L XOR 16)
//   new_src0_out[L]  = vdst_in[L XOR 16] = (L XOR 16)
//
// Under wave32 → wave64 cross-widening (salmon target gfx942):
// the source wave32 has a single pair of 16-lane halves (lanes
// 0..15 ↔ lanes 16..31 within the wave).  Salmon's
// `emitPermLaneSwapEmulation` projects this onto a wave64 target
// where the XOR-16 partner of any lane stays within the same
// 32-lane half (bit 5 preservation).  With
// `WaveNativeProjection`, two source waves per target wave: the
// source's wave 0 data lives in target lanes 0..31, source's
// wave 1 data lives in target lanes 32..63.  Each source wave's
// swap is independent, so the expected per-lane output pattern
// is IDENTICAL across both halves (and across all 32 lanes of the
// source wave32 — since this kernel launches with blockDim.x=32,
// there's only ONE source wave worth of lanes, so target lanes
// 32..63 under WaveNative may carry synthesised data or be
// inactive; the test only inspects lanes 0..31 to avoid mixing
// up the projection's own replication vs the swap's correctness).

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(32) void permlane16_swap_wave32_kernel(
    int *out_vdst, int *out_src0_out) {
  int L = threadIdx.x;
  int v = L;
  int u = 1000 + L;
  asm volatile(
      "v_permlane16_swap_b32 %[v], %[u]\n\t"
      : [v] "+v"(v), [u] "+v"(u));
  out_vdst[L] = v;
  out_src0_out[L] = u;
}
