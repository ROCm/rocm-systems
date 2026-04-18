// ARCH: gfx1250
//
// Lit fixture for SOP1 s_swap_pc_i64 — Pattern A (statically
// resolvable direct call). See SemOp::S_SWAP_PC_I64 in semop.hpp,
// the swap branch in transpiler/setpc_analysis.cpp, and the
// `if (sop == SemOp::S_SWAP_PC_I64)` block in
// transpiler/handle_sop1.cpp. The handler must lower this site to:
//
//   %swap_ret_blockaddr = ptrtoint
//       ptr blockaddress(@kernel, %BB_returnAddr) to i64
//   store i64 %swap_ret_blockaddr, ptr %sdst_alloca   ; for sdst
//   br label %BB_callee
//
// because the static analysis can prove the source SGPR pair (call
// target ssrc) was produced by a complete intra-block
// `s_get_pc_i64 + s_add_co_u32 + s_add_co_ci_u32` chain whose folded
// value is the offset of an in-kernel basic block leader. The
// blockaddress materialisation into sdst replaces the binary return
// PC the hardware would deposit there with a real LLVM constant a
// downstream Pattern B `s_set_pc_i64` could consume via indirectbr.
//
// Layout (kernel-relative offsets after the kernarg-load prologue
// hipcc auto-emits — call the inline-asm start K, with K = 0x08 in
// practice because the prologue is one 8-byte s_load_b64; the
// FileCheck offsets below assume K = 0x08):
//
//   K + 0x00:  s_get_pc_i64 s[10:11]            ; s[10:11] = K+0x04
//   K + 0x04:  s_add_co_u32 s10, s10, 20        ; s[10:11] = K+0x18
//   K + 0x08:  s_add_co_ci_u32 s11, s11, 0      ; chain done; pair = K+0x18
//   K + 0x0C:  s_swap_pc_i64 s[20:21], s[10:11] ; ssrc = K+0x18 (callee),
//                                               ; sdst stores K+0x10 (return)
//   K + 0x10:  v_mov_b32 vR, 0xCAFE0001          ; .Lret (8 bytes — return BB)
//   K + 0x18:  v_mov_b32 vR, 0xDEAD0001          ; .Lcallee (8 bytes — callee body)
//   K + 0x20:  s_endpgm                          ; terminate kernel
//
// Chain math: s_get_pc_i64 records (its own offset + 4) = K+0x04;
// s_add_co_u32 with IMM=20 (decimal) folds to K+0x04+0x14 = K+0x18.
// IMM=20 fits in a 7-bit inline immediate so the SOP2 stays 4 bytes,
// which keeps the offset arithmetic above stable.
//
// The swap is at K+0x0C (4 bytes) → return offset = K+0x10. The
// analysis adds K+0x10 (return) and K+0x18 (callee) to
// extraBlockStarts so both are basic-block leaders in the lifted IR.
// The handler emits `br label %bb_<K+0x18>` (callee) and writes
// `blockaddress(@kernel, %bb_<K+0x10>)` (return) into sdst.
//
// MAINTENANCE: the offset arithmetic above depends on the AMDGPU
// assembler picking 4-byte SOP1/SOP2 encodings for every scalar
// instruction in the asm block (inline immediates in [0, 64] keep
// s_add_co_u32 4 bytes; s_get_pc_i64 / s_swap_pc_i64 / s_endpgm are
// SOP1/SOPP forms with no literal slot). The two `v_mov_b32` use
// 32-bit literal payloads so they're 8 bytes each. If the encoder
// ever changes any of these, re-derive the immediate from
// `llvm-objdump -d --mcpu=gfx1250 <co>`.

#include <hip/hip_runtime.h>

extern "C" __global__ void
setpc_swap_pattern_a_kernel(int *out) {
  int tid = (int)threadIdx.x;
  int result = 0;
  asm volatile(
      // Pattern A call-target chain: s[10:11] = address of .Lcallee
      // at K+0x18.
      "s_get_pc_i64 s[10:11]\n\t"
      "s_add_co_u32 s10, s10, 20\n\t"
      "s_add_co_ci_u32 s11, s11, 0\n\t"
      // Branch-and-link. sdst = s[20:21] receives the return PC
      // (K+0x10); ssrc = s[10:11] is the call target (K+0x18).
      "s_swap_pc_i64 s[20:21], s[10:11]\n\t"
      // .Lret: the BB the swap's blockaddress points at. Pinned
      // here as a real, observable instruction so the lifted IR
      // includes a `bb_<K+0x10>` label that survives DCE.
      "v_mov_b32 %[r], 0xCAFE0001\n\t"
      // .Lcallee: the swap's `br label` lands here.
      "v_mov_b32 %[r], 0xDEAD0001\n\t"
      // Terminate the wave so the post-asm hipcc-generated store
      // tail is unreachable in practice (still lifts cleanly into
      // its own BBs — the handler only cares about the swap site).
      "s_endpgm\n\t"
      : [r] "=v"(result)
      :
      : "s10", "s11", "s20", "s21");
  out[tid] = result;
}
