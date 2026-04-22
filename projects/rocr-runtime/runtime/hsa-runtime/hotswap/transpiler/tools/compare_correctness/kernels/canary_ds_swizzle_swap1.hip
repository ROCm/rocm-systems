// Canary for the `ds_swizzle_b32` P6 lift on its most GPT-OSS-relevant
// imm shape: SWAP-1 (BITMASK_PERM with and=0x1F, or=0, xor=1;
// offset imm = 0x041F), the exact pattern that GPT-OSS's
// `sum_bitmatrix_rows` kernel emits (`ds_swizzle_b32 ×8` per §5 of
// `hotswap/docs/gpt-oss-derisking.md`).  Complements the existing
// `Gfx1250Gpu.DsSwizzle` end-to-end GPU test (which intentionally uses
// SWAP-2 / imm=0x081F to catch XOR-mask imm-extraction bugs that
// happen to round-trip SWAP-1) by pinning the SWAP-1 imm specifically.
//
// Motivation (from `hotswap/docs/gpt-oss-derisking.md` §7.2 / §9.2
// item 4):
//
//   `sum_bitmatrix_rows` is the one GPT-OSS kernel that reaches
//   `ds_swizzle_b32`, and it does so exclusively in the SWAP-1
//   BITMASK_PERM shape.  The P6 handler in `handle_ds.cpp` lifts
//   every valid envelope (QUAD_PERM / BITMASK_PERM / FFT / ROTATE)
//   through `@llvm.amdgcn.ds.swizzle(src, immarg)` and relies on a
//   wave64-family-wide hardware contract that `ds_swizzle_b32`
//   preserves bit 5 of the lane id (i.e. each 32-lane half performs
//   the permutation independently).  The existing GPU test pins
//   SWAP-2; this canary adds the SWAP-1 coverage so a future imm-
//   decode bug that corrupts only the XOR=1 bit of the BITMASK_PERM
//   mask can't slip past both gates.
//
// Kernel shape:
//
//   * 64-thread workgroup, one workgroup.  On gfx1250 wave32 that is
//     2 source waves of 32 each; on gfx942 wave64 (native / legacy /
//     salmon targets) a single wave64 spans the whole workgroup, with
//     ds_swizzle_b32's bit-5 preservation giving per-half SWAP-1
//     semantics.
//   * Each lane seeds a VGPR with its 0..63 thread ID promoted to
//     float, stored via volatile load so hipcc can't fold the
//     inline-asm source across the swizzle boundary.
//   * `ds_swizzle_b32 %v, %v offset:0x041f; s_wait_dscnt 0` via
//     inline asm pins the exact MCInst the P6 handler decodes.
//   * The permuted value is written to `out[tid]`.
//
// Expected outputs and verdict contract (CpuReference gold):
//
//   Gold semantics = SWAP-1 within each 32-lane group, which is
//   identical across wave32 source and wave64 target thanks to the
//   bit-5-preservation hardware contract:
//
//     for tid in [0, N):
//       src_wave_group_base = tid & ~31    // 32-lane group origin
//       lane_in_group       = tid & 31     // lane index within group
//       swap_partner        = lane_in_group ^ 1
//       out[tid] = in[src_wave_group_base | swap_partner]
//
//   Expected verdicts (all three modes match):
//
//   * native  : match.  gfx942 compiles the inline-asm
//               `ds_swizzle_b32 offset:0x041f` directly; hardware
//               bit-5 preservation produces per-32-lane-half SWAP-1
//               which matches the gold.
//   * legacy  : match.  Text transpiler copies the MCInst byte-for-
//               byte; gfx942 hardware executes the identical swizzle.
//   * salmon  : match.  P6 handler in `handle_ds.cpp::DS_SWIZZLE_B32`
//               lifts via `@llvm.amdgcn.ds.swizzle(src, 0x041F)`; the
//               gfx942 backend re-lowers to the identical
//               `ds_swizzle_b32 offset:swizzle(SWAP,1)` instruction.
//
// Regression contract:
//
//   * salmon `match` -> `WRONG` on every shape  ==  P6 handler or
//     classifier regressed (either the imm extraction is now dropping
//     the XOR=1 bit, or the MODREP classifier started refusing this
//     envelope, or the intrinsic-to-backend path broke).  The
//     mismatch pattern is a stable per-32-lane SWAP-1 diff, so the
//     regression signal is clean.
//
//   * any mode `match` -> `EXIT=2`  ==  classifier refusal.  Intended
//     only if the SWAP-1 envelope is re-categorised as unsafe (e.g. a
//     future ISA adds a variant that breaks the bit-5 contract).
//     Otherwise the refusal is a regression.
//
//   * native / legacy drift  ==  hipcc or text-transpile regression;
//     investigate before relaxing the canary's match expectation.
//
// Why `.hip` / inline-asm rather than Triton: Triton's
// `tl.reduce` lowering on gfx1250 picks the ds_swizzle imm implicitly
// and can retarget between imm values across compiler versions.
// Inline asm binds the exact MCInst we need to probe, making the
// canary robust against Triton codegen drift.  See
// `kernels/triton/canary_bpermute_scan_fp32.py` for the contrast: the
// bpermute canary lets Triton pick the selector because its imm is
// uniform-expressible; ds_swizzle's 16-bit envelope encoding makes the
// Triton path fragile.

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(64) void
canary_ds_swizzle_swap1(const float *in, float *out) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Volatile load keeps the per-lane seed live across any whole-
  // program constant-folding across the inline-asm boundary.
  float v = *(const volatile float *)(in + tid);
  // ds_swizzle_b32 produces its value to a VGPR via the DS hardware
  // counter, and a subsequent consumer must wait for that counter
  // before reading the result.  hipcc's automatic scheduling across
  // an `asm volatile` boundary has been observed NOT to emit the
  // required wait on gfx942 (the store reads the pre-swizzle value
  // and the swap silently disappears from the output), so we bake
  // the wait into the inline asm itself.  The wait mnemonic is
  // ISA-specific: gfx9 uses the merged `s_waitcnt lgkmcnt(0)` and
  // gfx12+ uses the split `s_wait_dscnt 0`.  A universal spelling
  // does not exist; we dispatch on the target-ISA preprocessor
  // macros that the AMDGPU clang frontend defines per offload arch.
  //
  // Why this isn't an ISA-exclusive kernel (unlike
  // `test_data/gfx1250/ds_swizzle_kernel.hip`): the preprocessor
  // dispatch produces gfx-appropriate MCInst sequences for BOTH
  // compilation targets at build time, so the same source file
  // emits a valid gfx1250 and a valid gfx942 code object.  The
  // dispatch is an implementation detail of how to inject the
  // correct wait, not a semantic difference in what the kernel
  // computes.
#if defined(__gfx12__) || defined(__gfx1250__) || defined(__gfx1200__) || \
    defined(__gfx1201__)
  asm volatile(
      "ds_swizzle_b32 %[v], %[v] offset:0x041f\n\t"
      "s_wait_dscnt 0"
      : [v] "+v"(v));
#else
  asm volatile(
      "ds_swizzle_b32 %[v], %[v] offset:0x041f\n\t"
      "s_waitcnt lgkmcnt(0)"
      : [v] "+v"(v));
#endif
  out[tid] = v;
}
