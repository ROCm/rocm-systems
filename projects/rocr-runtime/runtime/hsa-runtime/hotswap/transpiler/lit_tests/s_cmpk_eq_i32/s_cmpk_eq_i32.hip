// ARCH: gfx950
//
// Lit fixture for `s_cmpk_eq_i32` (SOPK_SCC class). Pins the
// operand-index contract for the family of SOPK compare-and-SCC
// opcodes (`S_CMPK_{EQ,LG,GT,GE,LT,LE}_I32/U32`) after the fix
// in `handle_sopk.cpp`.
//
// ISA note: `s_cmpk_*_i32/u32` was dropped on gfx12; it exists
// only up through gfx11 (per the SOPK_Real defm in
// SOPInstructions.td). So this fixture compiles for gfx950, not
// gfx1250 like the sibling `s_addk_i32` / `s_mulk_i32` fixtures.
// The raise direction is the canonical AITER-corpus source
// (gfx950 -> gfx942).
//
// Operand layout (SOPK_SCC):
//     (outs)                       ; empty — no defs
//     (ins SReg_32:$sdst,          ; operand 0: $sdst as SOURCE
//          s16imm:$simm16)         ; operand 1: the immediate
// With `getNumDefs() == 0`, the decoder's `buildSrcMap` keeps
// both in srcMap — `op.src(0)` = sdst, `op.src(1)` = simm16.
// Pre-fix handler read `op.src(0)` thinking it was the
// immediate, producing `icmp eq %sdst, %sdst` = always true.
// Post-fix: `icmp eq %sdst, <literal>`.
//
// The inline-asm sequence forces the exact MC opcode and a
// distinguishable `s_cselect_b32` consumer so the SCC write
// observably feeds the output — if the fix regresses to the
// pre-fix "SCC always set" shape, `result` below collapses to
// the "true" arm of the cselect across the whole wave.

#include <hip/hip_runtime.h>

extern "C" __global__ void s_cmpk_eq_i32_kernel(int *out) {
  int tid = (int)threadIdx.x;
  int x = (int)blockIdx.x;
  int result = 0;
  asm volatile(
      // s_cmpk_eq_i32 sdst, simm16: SCC = (sdst == sext(simm16)).
      // Immediate 0x400 (= 1024).
      "s_cmpk_eq_i32 %[x], 0x400\n\t"
      // Observable SCC consumer: cselect materialises 1 if SCC
      // else 0. Under the pre-fix bug SCC was always 1; under
      // the fix SCC is 1 only when %[x] actually equals 1024.
      "s_cselect_b32 %[r], 1, 0\n\t"
      : [r] "=s"(result)
      : [x] "s"(x));
  out[tid] = result;
}
