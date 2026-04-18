// ARCH: gfx1250
//
// Lift fixture for SOP1 s_cmov_b32 (scalar conditional move on
// SCC). See the S_CMOV_B32 SemOp in transpiler/semop.hpp and the
// matching handler block in transpiler/handle_sop1.cpp; this
// fixture pins the IR shape both reference (handler ↔ test
// back-reference is non-negotiable per the transpiler test
// policy).
//
// The hardware semantics are:
//   if (SCC) sdst = src; else sdst stays unchanged
// SCC is read but not written. LLVM's SOP1_32 pseudo for
// S_CMOV_B32 declares `(outs sdst), (ins src0)` *without* a tied
// sdst_in input — the dst-on-SCC=0 read-modify is implicit in
// the hardware encoding rather than modeled at the MachineInstr
// level — so the handler reads the prior dst value via
// `regs.readReg32(op.dst())` rather than off srcMap.
//
// Construction: produce SCC via `s_cmp_lg_u32 src1, 0` (set SCC
// to src1 != 0), then `s_cmov_b32 sdst, src0`. The result is
// `select(src1 != 0, src0, prior_sdst)`. We pre-load `sdst` with
// a known "prior" sentinel (`s_mov_b32 sdst, sentinel`) so the
// SCC=0 path is observable in the lifted IR — a bare cmov whose
// prior dst is undef would let the lift collapse the select into
// just `src0` and the test would not catch a regression that
// dropped the SCC condition entirely.

#include <hip/hip_runtime.h>

extern "C" __global__ void s_cmov_b32_kernel(int *out, int cond, int src) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Force scalar register allocation for the asm operands. The
  // kernel-argument loads land in VGPRs by default once the kernarg
  // dwords are broadcast; `__builtin_amdgcn_readfirstlane` lifts
  // them back into SGPRs so the inline asm's "s" constraints can be
  // honoured. Without this, hipcc emits e.g. `s_cmov_b32 s0, v2`
  // and the assembler rejects it ("invalid operand for instruction").
  int sCond = __builtin_amdgcn_readfirstlane(cond);
  int sSrc = __builtin_amdgcn_readfirstlane(src);
  int result;
  asm volatile(
      // Sentinel pre-load gives the cmov a known prior dst value
      // so the lift's select cannot fold to just src on the false
      // branch.
      "s_mov_b32 %[r], 0xDEADBEEF\n\t"
      "s_cmp_lg_u32 %[c], 0\n\t"
      "s_cmov_b32 %[r], %[s]\n\t"
      : [r] "=&s"(result)
      : [c] "s"(sCond), [s] "s"(sSrc));
  out[tid] = result;
}
