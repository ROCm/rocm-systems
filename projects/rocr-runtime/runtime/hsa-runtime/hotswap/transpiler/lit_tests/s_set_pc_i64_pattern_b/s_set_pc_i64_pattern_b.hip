// ARCH: gfx1250
//
// Lit fixture for SOP1 s_set_pc_i64 — Pattern B (subroutine
// return via an SGPR pair stashed at the call site). See
// SemOp::S_SET_PC_I64 in semop.hpp and the symbolic-PC tracker
// in setpc_analysis.cpp. The handler in handle_sop1.cpp must
// lower this site to
//   indirectbr ptr %ret_pc, [ list of resolved return targets ]
// because the source SGPR pair was constructed at a call site
// (its getpc+add chain finalised in a *different* basic block —
// the analysis enforces a per-basic-block reset of its symbolic
// PC table — and the chain's value is the return address). The
// raiser also post-processes the chain-terminator s_add_co_ci_u32
// to overwrite the ret-pair with `blockaddress(@kernel, %ret_BB)`
// instead of the binary PC the chain would otherwise yield.
//
// Layout (kernel-relative offsets after the kernarg-load prologue
// hipcc auto-emits — call the inline-asm start K, with K = 0x08
// because the prologue is one 8-byte s_load_b64):
//
//   K+0x00:  s_get_pc_i64 s[8:9]            ; s[8:9] = K+0x04
//   K+0x04:  s_add_co_u32 s8, s8, 28        ; s[8:9] = K+0x04 + 28
//                                           ;        = K+0x20 (.Lret)
//   K+0x08:  s_add_co_ci_u32 s9, s9, 0      ; chain terminator
//                                           ; chainTerminators[K+0x08] =
//                                           ;   (resolvedReturnAddr=K+0x20,
//                                           ;    retPair=8)
//   K+0x0C:  s_get_pc_i64 s[10:11]          ; s[10:11] = K+0x10
//   K+0x10:  s_add_co_u32 s10, s10, 24      ; s[10:11] = K+0x10 + 24
//                                           ;         = K+0x28 (.Lcallee)
//   K+0x14:  s_add_co_ci_u32 s11, s11, 0    ; chain terminator (Pattern A)
//   K+0x18:  s_set_pc_i64 s[10:11]          ; Pattern A jump → .Lcallee
//                                           ; extraBlockStarts: K+0x1C,
//                                           ; K+0x28
//   K+0x1C:  s_branch 1                     ; unreachable in execution
//                                           ; (the previous s_set_pc_i64
//                                           ; is a terminator), present
//                                           ; ONLY to seed .Lcallee in
//                                           ; the decoder's blockStarts.
//                                           ; target = K+0x1C + 4 + 1*4
//                                           ;        = K+0x24 — but we
//                                           ; want K+0x28; see below.
//   K+0x20:  v_mov_b32 vR, 0xCAFE0001       ; .Lret (8 bytes)
//   K+0x28:  v_mov_b32 vR, 0xDEAD0001       ; .Lcallee (8 bytes)
//   K+0x30:  s_set_pc_i64 s[8:9]            ; Pattern B return
//   K+0x34:  s_endpgm
//
// To make the s_branch target K+0x28 (.Lcallee), the immediate must
// satisfy:  K+0x1C + 4 + IMM*4 = K+0x28  ⇒  IMM = 2.  We use
// `s_branch 2`. The branch is dynamically dead (the preceding
// s_set_pc_i64 already transferred control), so its purpose is
// purely to seed K+0x28 as a basic-block leader in the decoder's
// blockStarts set so the symbolic-PC analysis resets state at
// .Lcallee — without that reset, s[8:9]'s tracked chain would
// survive into the callee and the second s_set_pc_i64 would be
// misclassified as Pattern A.
//
// MAINTENANCE: this fixture's offset arithmetic depends on the
// AMDGPU asm encoder picking a 4-byte SOP2 encoding for every
// scalar instruction in the asm block. Inline immediates in
// [0, 64] keep s_add_co_u32 4 bytes; s_get_pc_i64 / s_set_pc_i64 /
// s_branch / s_endpgm are all SOP1/SOPP forms with no literal slot.
// The two `v_mov_b32 vR, 0x...0001` use 32-bit literal payloads so
// they're 8 bytes each; the chain math depends on that. If the
// encoder ever changes any of these, re-derive the immediates from
// `llvm-objdump -d --mcpu=gfx1250 <co>`.

#include <hip/hip_runtime.h>

extern "C" __global__ void
setpc_pattern_b_kernel(int *out) {
  int tid = (int)threadIdx.x;
  int result = 0;
  asm volatile(
      // Call-site chain construction: s[8:9] = return target =
      // address of the v_mov_b32 at .Lret (kernel offset K+0x20).
      "s_get_pc_i64 s[8:9]\n\t"
      "s_add_co_u32 s8, s8, 28\n\t"
      "s_add_co_ci_u32 s9, s9, 0\n\t"
      // Pattern A chain construction + jump to .Lcallee at K+0x28.
      "s_get_pc_i64 s[10:11]\n\t"
      "s_add_co_u32 s10, s10, 24\n\t"
      "s_add_co_ci_u32 s11, s11, 0\n\t"
      "s_set_pc_i64 s[10:11]\n\t"
      // Decoder-visible immediate-target s_branch. Dynamically
      // dead (the preceding s_set_pc_i64 already terminated the
      // wave's path), present only to put K+0x28 (.Lcallee) into
      // the decoder's blockStarts so the analysis pass resets its
      // symbolic-PC state at .Lcallee.
      "s_branch 2\n\t"
      // .Lret: indirectbr destination for Pattern B's return.
      "v_mov_b32 %[r], 0xCAFE0001\n\t"
      // .Lcallee: target of the Pattern A branch.
      "v_mov_b32 %[r], 0xDEAD0001\n\t"
      // Pattern B indirect return — must lower to `indirectbr`
      // with destination list [bb_0x20] (the .Lret label).
      "s_set_pc_i64 s[8:9]\n\t"
      : [r] "=v"(result)
      :
      : "s8", "s9", "s10", "s11");
  out[tid] = result;
}
