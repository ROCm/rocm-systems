// ARCH: gfx1250
//
// Lift fixture for SOP1 s_bitset0_b32 (clear a single bit in sdst).
// See SemOp::S_BITSET0_B32 in transpiler/semop.hpp and the matching
// handler block in transpiler/handle_sop1.cpp; this fixture pins
// the IR shape both reference (handler ↔ test back-reference is
// non-negotiable per the transpiler test policy).
//
// Hardware semantics:
//   sdst = sdst & ~(1 << (src0 & 0x1F))    // bit-clear
// SCC is not updated.  TableGen declares the pseudo via `SOP1_32`
// with `tied_in=1`:
//   `(outs SReg_32:$sdst), (ins SSrc_b32:$src0, SReg_32:$sdst_in)`
// with `Constraints = "$sdst = $sdst_in"`.
//
// The MC layer collapses the tied `$sdst_in` slot — the AMDGPU
// disassembler emits a 2-operand MCInst (`sdst`, `src0`), and the
// tie is reconstituted only at MachineInstr lowering time. So the
// handler reads the prior dst value via
// `ctx.regs.readReg32(op.dst())` rather than off srcMap (mirroring
// the S_CMOV_B32 pattern in handle_sop1.cpp); a previous version
// of the handler asserted `op.nSrcs() >= 2` expecting `sdst_in`
// in srcMap and aborted SIGABRT on every real s_bitset corpus
// site (every kernel that issued an actual s_bitset instruction,
// including triton f16 GEMM TDM-pipelined kernels and the
// compare_correctness s_bitset0_b{32,64} fixtures themselves).
// This fixture exists to regression-pin that "prior-dst is read
// from the destination register, not from srcMap" behaviour.
//
// Construction: pre-load the destination scalar with a recognisable
// sentinel (`s_mov_b32 sdst, 0xDEADBEEF`) via inline asm, then
// issue `s_bitset0_b32 sdst, bit_index` with the bit index supplied
// as an SGPR via a `+s` constraint.  The lifted IR must therefore
// AND the sentinel with `~(1 << (bit_index & 0x1F))`.  Without the
// sentinel pre-load, the lift's AND would have an `undef` operand
// and the test could not distinguish "prior dst correctly read"
// from "prior dst silently dropped".

#include <hip/hip_runtime.h>

extern "C" __global__ void s_bitset0_b32_kernel(unsigned int *out,
                                                 unsigned int bit_pos) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Force scalar register allocation for the asm operands. The
  // kernarg-loaded `bit_pos` lands in a VGPR by default;
  // `__builtin_amdgcn_readfirstlane` lifts it to an SGPR so the
  // inline-asm `s` constraint can be honoured.
  unsigned int sBitPos = __builtin_amdgcn_readfirstlane(bit_pos);
  unsigned int result;
  asm volatile(
      // Sentinel pre-load: gives the bitset's RMW a known prior dst
      // value.  Without this, the lifted AND would have `undef` on
      // one side and the test could not pin "prior dst correctly
      // read from sdst" vs. "prior dst silently dropped".
      "s_mov_b32 %[r], 0xDEADBEEF\n\t"
      "s_bitset0_b32 %[r], %[b]\n\t"
      : [r] "=&s"(result)
      : [b] "s"(sBitPos));
  out[tid] = result;
}
