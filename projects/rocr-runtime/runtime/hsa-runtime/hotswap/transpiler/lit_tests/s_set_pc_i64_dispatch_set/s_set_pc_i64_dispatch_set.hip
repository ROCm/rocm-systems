// ARCH: gfx1250
//
// Lit fixture for SOP1 s_set_pc_i64 — DispatchSet (multi-target
// indirect branch resolved by inter-block PC-chain dataflow). See
// SemOp::S_SET_PC_I64 in semop.hpp, the five-phase dataflow in
// transpiler/setpc_analysis.cpp, and the
// `case SetPcSiteInfo::Kind::DispatchSet:` branch in
// transpiler/handle_sop1.cpp.
//
// The DispatchSet shape is the dominant tensilelite "activation
// function dispatcher" pattern. Two (or more) CFG paths each compute
// a different statically-known absolute kernel offset into the same
// SGPR pair via independent `s_get_pc_i64 + s_add_co_u32 +
// s_add_co_ci_u32` chains, then converge at a join block whose first
// terminator is `s_set_pc_i64 sX:X+1`. The intra-block analysis sees
// the source pair pristine through the join block; the Phase 3 inter-
// block dataflow joins the per-path SET facts and Phase 4 lifts the
// site to a `cmp eq + br` cascade enumerating the statically-known
// callees, terminating in an `unreachable` trap BB (via the
// file-local `emitEnumeratedDispatch` helper in handle_sop1.cpp). See
// that helper's rationale block for why this is a cascade and not
// `indirectbr` (FixIrreducible compatibility under the irreducible
// CFGs this shape produces).
//
// The S_ADDC_U32 post-handler hook in raiser.cpp also rewrites each
// path's chain terminator to materialise
// `blockaddress(@kernel, %BB_target_i)` into the ret-pair instead of
// the binary PC the chain would yield (Phase 5 retains both
// terminators because their `resolvedReturnAddr` matches a target in
// the dispatched set).
//
// Layout (kernel-relative offsets after the kernarg-load prologue
// hipcc auto-emits — call the inline-asm start K, with K = 0x08
// because the prologue is one 8-byte s_load_b64; the FileCheck
// offsets in the .ll assume K = 0x08 so the absolute kernel offsets
// quoted in the bb_<hex> labels are K + the relative offset below):
//
//   K + 0x00:  s_cmp_eq_u32 s2, s3              ; SOPC, sets SCC
//   K + 0x04:  s_cbranch_scc1 4                 ; SCC=1 → .Lpath2
//                                                ; (target K + 0x18,
//                                                ;  fallthrough K + 0x08)
//   ; .Lpath1 (fallthrough block):
//   K + 0x08:  s_get_pc_i64 s[10:11]            ; s[10:11] = K + 0x0C
//   K + 0x0C:  s_add_co_u32 s10, s10, 32        ; s[10:11] = K + 0x2C
//   K + 0x10:  s_add_co_ci_u32 s11, s11, 0      ; chain done; pair=K+0x2C
//                                                ; chainTerminators[K+0x10]
//                                                ;   = (K+0x2C, retPair=10)
//   K + 0x14:  s_branch 4                       ; → .Lconverge (K+0x28)
//   ; .Lpath2 (cbranch target block):
//   K + 0x18:  s_get_pc_i64 s[10:11]            ; s[10:11] = K + 0x1C
//   K + 0x1C:  s_add_co_u32 s10, s10, 24        ; s[10:11] = K + 0x34
//   K + 0x20:  s_add_co_ci_u32 s11, s11, 0      ; chain done; pair=K+0x34
//                                                ; chainTerminators[K+0x20]
//                                                ;   = (K+0x34, retPair=10)
//   K + 0x24:  s_branch 0                       ; → .Lconverge (K+0x28)
//                                                ; (IMM=0 ⇒ jump to PC+4
//                                                ;  = next inst, equivalent
//                                                ;  to fallthrough but
//                                                ;  ensures K+0x28 is in
//                                                ;  the decoder's
//                                                ;  blockStarts via the
//                                                ;  decoded immediate)
//   ; .Lconverge (join block):
//   K + 0x28:  s_set_pc_i64 s[10:11]            ; DispatchSet ⇒ cmp+br
//                                                ; cascade over
//                                                ;   [bb_<K+0x2C>,
//                                                ;    bb_<K+0x34>]
//                                                ; + unreachable trap BB
//   K + 0x2C:  v_mov_b32 vR, 0xCAFE0001         ; .Ltarget1 (8 bytes)
//   K + 0x34:  v_mov_b32 vR, 0xDEAD0001         ; .Ltarget2 (8 bytes)
//   K + 0x3C:  s_endpgm                         ; terminate kernel
//
// Chain math (Phase 2 finishHighAdd):
//   path1: getpc records value = K + 0x0C; add IMM=32 ⇒ K + 0x2C.
//          IMM=32 (= 0x20) fits the 7-bit inline imm so SOP2 stays
//          4 bytes.
//   path2: getpc records value = K + 0x1C; add IMM=24 ⇒ K + 0x34.
//          IMM=24 (= 0x18) fits the 7-bit inline imm so SOP2 stays
//          4 bytes.
//
// Branch math:
//   s_cbranch_scc1 IMM at K+0x04 (size=4): target = K+0x04 + 4 + IMM*4.
//     IMM=4 ⇒ target = K + 0x18 = .Lpath2.
//   s_branch IMM at K+0x14 (size=4): target = K+0x14 + 4 + IMM*4.
//     IMM=4 ⇒ target = K + 0x28 = .Lconverge.
//   s_branch IMM at K+0x24 (size=4): target = K+0x24 + 4 + IMM*4.
//     IMM=0 ⇒ target = K + 0x28 = .Lconverge (next inst).
//
// Phase 3 dataflow (worklist forward, JOIN over predecessors):
//   entry exit: PASS for s10 (no chain ops).
//   path1 exit: SET(K+0x2C) for s10 (incomplete=false, single-value).
//   path2 exit: SET(K+0x34) for s10 (incomplete=false, single-value).
//   converge entry: JOIN({K+0x2C}, {K+0x34}) = {K+0x2C, K+0x34}
//                   (incomplete=false; both predecessors mention s10).
//
// Phase 4: deferred site at K+0x28 with srcPair=10 sees converged
//          facts {K+0x2C, K+0x34}. Two values within the cap ⇒
//          DispatchSet with indirectTargets = [K+0x2C, K+0x34],
//          indirectRetPairLowReg = 10. Both targets get added to
//          extraBlockStarts so the BB-layout phase materialises the
//          named bb_0x<absolute-offset> labels.
//
// Phase 5: chainTerminators[K+0x10] and chainTerminators[K+0x20]
//          retained because their resolvedReturnAddr ∈ DispatchSet
//          targets. Each terminator's S_ADDC_U32 post-handler hook
//          rewrites the ret-pair write to materialise
//          `blockaddress(@kernel, %bb_<target>)` into s[10:11].
//
// MAINTENANCE: this fixture's offset arithmetic depends on the AMDGPU
// asm encoder picking 4-byte SOP1/SOP2/SOPC/SOPP encodings for every
// scalar instruction in the asm block. Inline immediates in [0, 64]
// keep s_add_co_u32 / s_branch / s_cbranch_scc1 4 bytes; s_get_pc_i64
// / s_set_pc_i64 / s_cmp_eq_u32 / s_endpgm are SOP1/SOPC/SOPP forms
// with no literal slot. The two `v_mov_b32 vR, 0x...0001` use 32-bit
// literal payloads so they're 8 bytes each. If the encoder ever
// changes any of these, re-derive the immediates from
// `llvm-objdump -d --mcpu=gfx1250 <co>`.
//
// The kernel must NEVER be dispatched — the asm intentionally
// performs an indirect set-PC to a synthetic chain target that is not
// a real callee body. Only the raiser pre-translation classification
// is exercised (via raise_cli --emit-ir + FileCheck in the .ll).

#include <hip/hip_runtime.h>

extern "C" __global__ void
setpc_set_dispatch_set_kernel(int *out) {
  int tid = (int)threadIdx.x;
  int result = 0;
  asm volatile(
      // Set SCC from a runtime scalar comparison; the actual outcome
      // doesn't matter for static analysis (the dataflow follows BOTH
      // sides of the cbranch). The clobbers cover s2/s3 so the
      // compiler does not assume any prior contents survive.
      "s_cmp_eq_u32 s2, s3\n\t"
      "s_cbranch_scc1 4\n\t"
      // .Lpath1 (fallthrough): chain target = .Ltarget1 (K+0x2C).
      "s_get_pc_i64 s[10:11]\n\t"
      "s_add_co_u32 s10, s10, 32\n\t"
      "s_add_co_ci_u32 s11, s11, 0\n\t"
      "s_branch 4\n\t"
      // .Lpath2 (cbranch target): chain target = .Ltarget2 (K+0x34).
      "s_get_pc_i64 s[10:11]\n\t"
      "s_add_co_u32 s10, s10, 24\n\t"
      "s_add_co_ci_u32 s11, s11, 0\n\t"
      "s_branch 0\n\t"
      // .Lconverge: dataflow has converged, src pair s[10:11] holds
      // one of {K+0x2C, K+0x34}. Lifts to a cmp+br cascade.
      "s_set_pc_i64 s[10:11]\n\t"
      // .Ltarget1 / .Ltarget2: pinned-by-construction observable
      // instructions so the lifted IR includes named bb_0x<...>
      // labels matching the dispatch set.
      "v_mov_b32 %[r], 0xCAFE0001\n\t"
      "v_mov_b32 %[r], 0xDEAD0001\n\t"
      "s_endpgm\n\t"
      : [r] "=v"(result)
      :
      : "s2", "s3", "s10", "s11");
  out[tid] = result;
}
