// ARCH: gfx1250
//
// Regression fixture for the carry-chain SGPR-operand bug class
// (sibling to the VOPD `v_dual_cndmask_b32` SGPR-condition fix
// documented in hotswap/docs/modrep-predicate-chain.md §9.7).
//
// PRE-FIX BUG. The six V_{ADD,SUB,SUBREV}_CO_(CI_)U32 handlers in
// handle_valu.cpp hardcoded `ctx.regs.loadVCC` for carry-in and
// `ctx.regs.storeVCC` for carry-out, silently ignoring the explicit
// scalar operand the e64 / VOP3B encoding carries. On the e64
// forms the scalar can be EITHER `vcc_lo`/`vcc` OR an arbitrary
// `sN` — the compiler picks based on SGPR pressure. Hardcoding VCC
// collapsed the two cases, miscompiling any kernel that bound the
// carry to a non-VCC SGPR (address arithmetic silently reading
// stale VCC instead of the intended carry).
//
// POST-FIX. `handle_valu.cpp::readCarryInI1` / `writeCarryOutI1`
// mirror the SGPR-aware routing the non-VOPD V_CNDMASK_B32
// handler has always done (in `handle_valu_vop3p.cpp`):
//
//   * SGPR sN → prefer `ctx.lookupSgprWaveMaskI1(N)`'s fresh per-BB
//     V_CMP shadow `i1` (populated by V_CMP_*_e64 writers); fall
//     back to `projection.extractLaneBitFromWaveMask` on the raw
//     SGPR alloca for reads; ballot-and-write via
//     `writeRegExecWidth` + `recordSgprWaveMaskI1` for writes.
//   * VCC / absent → `loadVCC` / `storeVCC` (preserves the
//     pre-fix path for the all-VCC common case).
//
// No current corpus kernel (Triton on gfx1250 / gfx942, AITER
// TensileLite) exercises the non-VCC SGPR form of these
// instructions:
//
//   * Triton-gfx1250 uses `v_add_nc_u32` / `v_add_nc_u64`
//     (no-carry encoding; gfx1250 added native 64-bit add).
//   * Triton-gfx942 uses `v_add_u32` / `v_lshl_add_u64`
//     (no-carry on gfx9; fused add-shift for address math).
//   * AITER TensileLite uses `v_add_co_u32` / `v_add_co_ci_u32`
//     but always with `vcc_lo` for the carry (the compiler's
//     default when VCC is live-available).
//
// So the latent bug has zero corpus exposure today. We close it
// proactively because (a) it's the same bug class as the VOPD
// cndmask miscompile we already fixed, and (b) "never do silent
// fallbacks" is a non-negotiable project rule — the pre-fix
// handler literally did a silent fallback from "whatever scalar
// the encoding names" to "VCC always".
//
// The fixture forces the e64 form via inline asm with explicit
// `s[0:1]` for carry, so the assembler emits a VOP3B encoding
// whose sdst and ssrc2 fields point at s0 rather than vcc_lo.
// Two paired instructions exercise both endpoints:
//
//   1. `v_add_co_u32 v<dst>, s0, v<a>, v<b>` — e64 carry-OUT to
//      s0 (exercises `writeCarryOutI1` SGPR branch).
//   2. `v_add_co_ci_u32_e64 v<dst2>, s0, v<c>, v<d>, s0` — e64
//      carry-IN from s0 (produced above) AND carry-OUT back to
//      s0 (exercises `readCarryInI1` + `writeCarryOutI1` SGPR
//      branches in one shot). Models the canonical 64-bit
//      address-chain shape the compiler would emit if it
//      chose to use `s0` instead of `vcc_lo`.

#include <hip/hip_runtime.h>

extern "C" __global__ void v_add_co_u32_sgpr_carry_kernel(int *out,
                                                           const int *in) {
  int lane = (int)threadIdx.x;
  int a = in[lane];
  int b = in[(lane + 1) & 31];
  int c = in[(lane + 2) & 31];
  int d = in[(lane + 3) & 31];

  // The 64-bit address-arithmetic shape: add (a, b) producing a
  // 32-bit sum + 1-bit carry-out into s0, then add (c, d) + that
  // carry (reading s0) producing a second sum + carry.
  //
  // Both instructions use `s0` for their carry scalar rather than
  // the default `vcc_lo`. This is the exact shape the compiler
  // would emit under VCC pressure (VCC live for a simultaneous
  // predicate elsewhere). The inline asm clobber list declares
  // `s0` so hipcc knows we're trampling it; the `"=s"` output
  // bindings aren't right for an implicit carry-out, so we use a
  // dummy `"+v"` on the vdst to keep the compiler from DCE'ing
  // the sequence.
  int sum1 = 0;
  int sum2 = 0;
  asm volatile(
      // Carry-OUT to s0 (first add).
      "v_add_co_u32 %[d1], s0, %[a], %[b]\n\t"
      // Carry-IN from s0, carry-OUT back to s0 (second add).
      "v_add_co_ci_u32_e64 %[d2], s0, %[c], %[d], s0\n\t"
      : [d1] "=v"(sum1), [d2] "=v"(sum2)
      : [a] "v"(a), [b] "v"(b), [c] "v"(c), [d] "v"(d)
      : "s0");

  out[lane * 2 + 0] = sum1;
  out[lane * 2 + 1] = sum2;
}
