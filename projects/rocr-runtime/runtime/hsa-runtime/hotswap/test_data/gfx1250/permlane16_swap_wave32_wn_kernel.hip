// Source for permlane16_swap_wave32_wn_gfx1250.hsaco — wave32 source
// whose max_flat_workgroup_size forces WaveNativeProjection (NOT the
// phantom-lane ModRep fallback).
//
// Why this exists alongside the other two permlane16_swap fixtures
// ================================================================
//
// We now have three dimensions pinned for `v_permlane16_swap_b32`:
//
//   fixture                                source wave    projection
//   -------------------------------------  -------------  -------------
//   permlane16_swap_kernel                 wave64         SameWave
//   permlane16_swap_wave32_kernel          wave32         ModuloRep
//   permlane16_swap_wave32_wn_kernel       wave32         WaveNative   <-- this
//
// The middle one exercises the phantom-lane regime (block size 32
// < target wavefront 64), where `raiser.cpp` forces ModRep
// regardless of the `enableWaveNative` flag so phantom target
// lanes stay hardware-inactive.  That path passes, but it is NOT
// the path Triton `tl.sort` takes — `tl.sort` at BLOCK_M=32,
// BLOCK_N=32, num_warps=4 runs at max_flat_workgroup_size=128,
// which is at or above the target wavefront width (64) and
// therefore engages WaveNativeProjection per
// `wave-size-translation.md §2.2 / 5.6.1`.  That is the path
// `canary_tl_sort_fp32_deterministic` shows the cross-16 merge
// failing on.
//
// To exercise WaveNativeProjection on a wave32 source in
// isolation, we need max_flat_workgroup_size >= 64.  128 is the
// closest match to Triton's tl.sort dispatch shape (num_warps=4
// × 32 lanes/wave).  With 128 threads spread over 4 source
// wave32s, WaveNativeProjection lifts to 2 target wave64s, each
// holding two source waves' data (source wave 0 in target lanes
// 0..31, source wave 1 in target lanes 32..63; similarly for the
// second target wave64).
//
// Build (regenerate when the kernel is changed):
//   hipcc --offload-arch=gfx1250 --genco \
//     test_data/gfx1250/permlane16_swap_wave32_wn_kernel.hip \
//     -o test_data/gfx1250/permlane16_swap_wave32_wn_gfx1250.hsaco
//   /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler \
//     --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
//     --input=test_data/gfx1250/permlane16_swap_wave32_wn_gfx1250.hsaco \
//     --output=test_data/gfx1250/permlane16_swap_wave32_wn_gfx1250.hsaco \
//     --unbundle
//
// Per-lane setup (for every lane L in [0, 128)):
//   vdst_in[L]  = L
//   src0_in[L]  = 1000 + L
//
// The `v_permlane16_swap_b32` swaps WITHIN each source wave32's
// 32 lanes.  The "L XOR 16" partner stays within the same source
// wave.  Expected per-lane output on the lifted wave64 kernel:
//
//   wave-local partner = (L % 32) XOR 16         // 0..31 within wave
//   wave-base = L - (L % 32)                     // 0, 32, 64, 96
//   partner   = wave-base + wave_local_partner
//   new_vdst[L]      = 1000 + partner
//   new_src0_out[L]  = partner
//
// The test reads back all 128 lanes and validates this pattern.
// Under the hypothesis that the Triton `tl.sort` cross-16 merge
// is broken BECAUSE `emitPermLaneSwapEmulation` behaves wrong
// under WaveNativeProjection, this test will fire.  If the test
// passes, the emulation is correct under this projection too,
// and the root cause is in the compare-and-cndmask dance AROUND
// the swap (the `v_cmp → v_cndmask` whose VOP3 encoding
// `.long 0xd41b0002` is not yet pinned — next narrowing step).

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(128)
    void permlane16_swap_wave32_wn_kernel(int *out_vdst,
                                           int *out_src0_out) {
  int L = threadIdx.x;
  int v = L;
  int u = 1000 + L;
  asm volatile(
      "v_permlane16_swap_b32 %[v], %[u]\n\t"
      : [v] "+v"(v), [u] "+v"(u));
  out_vdst[L] = v;
  out_src0_out[L] = u;
}
