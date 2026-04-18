// ARCH: gfx1250
//
// Lit fixture for SOP1 s_set_pc_i64 — Pattern A (statically
// resolvable intra-kernel direct branch). See SemOp::S_SET_PC_I64
// in semop.hpp and the symbolic-PC tracker in setpc_analysis.cpp.
// The handler in handle_sop1.cpp must lower this site to a plain
//   br label %BB_target
// because the static analysis can prove the source SGPR pair came
// from a complete, intra-block s_get_pc_i64 + s_add_co_u32 +
// s_add_co_ci_u32 chain whose folded value is the offset of an
// in-kernel basic block leader.
//
// The fixture's chain math:
//   * `s_get_pc_i64 s[10:11]` writes (PC_of_get_pc + 4) into the
//     pair — i.e. the offset of the very next instruction.
//   * `s_add_co_u32 s10, s10, 16` adds 16 bytes (four 32-bit
//     scalar instruction slots) to the low half. Inline immediates
//     in [0, 64] don't require a literal slot, so this stays a
//     4-byte SOP2 encoding (which keeps the offset arithmetic below
//     stable across LLVM AMDGPU asm versions).
//   * `s_add_co_ci_u32 s11, s11, 0` propagates carry into the
//     high half (the 5-bit inline 0 keeps this 4-byte too) and
//     finalises the chain at this instruction's offset (the
//     analysis records this as the chain terminator).
//
// Resolved chain value, in offsets from the kernel's first
// instruction (call it K):
//   K +  0:  s_get_pc_i64 s[10:11]   (4 bytes)
//   K +  4:  s_add_co_u32  s10, s10, 16
//   K +  8:  s_add_co_ci_u32 s11, s11, 0
//   K + 12:  s_set_pc_i64  s[10:11]
//   K + 16:  v_mov_b32     v0, 0xDEAD0001  (8 bytes — 32-bit literal)
//   K + 24:  v_mov_b32     v0, 0xDEAD0002  (8 bytes — 32-bit literal)
//   K + 32:  ret-target instructions ...
//
// Wait — the `s_get_pc_i64` writes (its own offset + 4) = K + 4
// into s[10:11], so after `s_add_co_u32 s10, s10, 16` we get
// s[10:11] = K + 4 + 16 = K + 20. But K + 20 is mid-instruction
// (inside the first v_mov_b32's literal slot). To land on a
// real instruction boundary we must use an immediate that lands
// the chain on K + N where N is a 4-byte multiple AND coincides
// with an instruction start.
//
// Simpler: pick the immediate so the chain lands on the
// instruction right after the s_set_pc_i64 chain itself. The
// s_set_pc_i64 is at K+12 and is 4 bytes, so its fall-through is
// K+16. We want s[10:11] = K+16. Setting the low-half addend to
// (16 - 4) = 12 gives s[10:11] = K + 4 + 12 = K + 16 — the
// fall-through of the s_set_pc_i64. This is an "indirect
// fall-through" which is morally a direct branch; the analysis
// classifies it as Pattern A.
//
// We also force a 32-bit-literal v_mov_b32 at K+16 (8 bytes)
// followed by another v_mov_b32 at K+24 (8 bytes). The chain
// targets K+16, the first v_mov_b32, which is a real 4-byte
// instruction boundary. We mark that v_mov as the "after-branch"
// landing pad and assert FileCheck sees a `br label` to the BB
// containing it.
//
// MAINTENANCE: if the AMDGPU assembler grows a literal-encoding
// for this inline asm sequence (causing the s_get_pc_i64 to
// occupy more than 4 bytes), the K + 4 + IMM math above breaks.
// Inspect the disassembly with `llvm-objdump -d --mcpu=gfx1250
// <co>`; the chain-terminator comment in handle_sop1.cpp's
// S_SET_PC_I64 case explains how to re-derive the immediate.

#include <hip/hip_runtime.h>

extern "C" __global__ void
setpc_pattern_a_kernel(int *out) {
  int tid = (int)threadIdx.x;
  int result = 0;
  asm volatile(
      // PC chain: s[10:11] = (s_get_pc_i64 PC + 4) + 12
      //                    = address of the instruction
      //                      immediately after s_set_pc_i64.
      // The static analysis classifies this as Pattern A and
      // the handler emits `br label %BB_K+16`.
      "s_get_pc_i64 s[10:11]\n\t"
      "s_add_co_u32 s10, s10, 12\n\t"
      "s_add_co_ci_u32 s11, s11, 0\n\t"
      // Pattern A direct branch — must lower to `br label %BB`.
      "s_set_pc_i64 s[10:11]\n\t"
      // Branch target lives here. The v_mov_b32 with a 32-bit
      // literal pins this as a real, observable instruction in
      // the lifted IR (so a regression that drops the branch and
      // falls through would still execute *something* but the
      // FileCheck below would fail because the IR shape changes).
      "v_mov_b32 %[r], 0xDEAD0001\n\t"
      : [r] "=v"(result)
      :
      : "s10", "s11");
  out[tid] = result;
}
