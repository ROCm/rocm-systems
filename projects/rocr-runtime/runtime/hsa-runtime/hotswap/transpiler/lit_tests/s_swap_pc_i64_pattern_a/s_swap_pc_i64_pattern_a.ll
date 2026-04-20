; RUN: %raise_cli %s_swap_pc_i64_pattern_a_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=setpc_swap_pattern_a_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for s_swap_pc_i64 — Pattern A (statically resolvable
; direct call). Pins that the SOP1 branch-and-link lowers to:
;   1. an i64 return-address marker (the plain source-MC byte
;      offset of the BB immediately following the swap) written to
;      sdst (the ret SGPR pair), and
;   2. an unconditional `br label %bb_<callee-offset>`
; whenever the static analysis in transpiler/setpc_analysis.cpp can
; fold the call-target SGPR pair's provenance through the canonical
; chain
;   s_get_pc_i64 → s_add_co_u32 → s_add_co_ci_u32
; into a single in-kernel offset. The handler lives in
; transpiler/handle_sop1.cpp under
; `if (sop == SemOp::S_SWAP_PC_I64) { ... DirectA path }`. The SemOp
; doc in transpiler/semop.hpp pins the lowering contract.
;
; Why we write an integer marker and not
; `ptrtoint(blockaddress(@kernel, %bb_<ret>))` (as earlier revisions
; of this fixture pinned): AMDGPU's instruction selector has no
; pattern to materialise a `BlockAddress` constant as an i64
; register value, so any `BlockAddress` SDNode that survives the
; middle-end pipeline aborts llc with
;   `Cannot select: t1: i64 = BlockAddress<@kernel, %bb_N>`.
; Using a plain integer offset sidesteps ISel entirely — see the
; rationale block on `emitEnumeratedDispatch` in handle_sop1.cpp.
;
; Why these CHECKs:
;   * `bb_0x18` is the fixture's resolved return target. Derived in
;     the .hip header doc:
;        K = 0x08 (kernarg-load prologue)
;        swap site at K+0x0C ⇒ return offset = K+0x10 = 0x18
;     The analysis adds 0x18 to extraBlockStarts (Phase 1 promotes
;     swap.offset+size to a leader) so `bb_0x18` is a real, named
;     BB in the lifted IR.
;   * `bb_0x20` is the chain-resolved callee. Derived as:
;        K+0x18 = 0x20 (the s_add_co_u32 IMM=20 lands the chain at
;        K+0x18; absolute kernel offset is 0x08+0x18 = 0x20)
;     The handler emits `br label %bb_0x20`.
;   * In this fixture sdst is never consumed (the downstream code is
;     straight-line and the return-path BB is an immediate
;     fall-through), so mem2reg + DCE promote-and-drop the sdst
;     alloca. We therefore do NOT pin the sdst write in the IR text;
;     the invariant that matters is pinned negatively below
;     (`CHECK-NOT: blockaddress(`) so a regression that re-introduced
;     the old `ptrtoint(blockaddress(...))` sdst materialisation
;     would fail the negative check.
;   * `CHECK-NOT: indirectbr` pins that we did NOT degenerate into
;     an IndirectB-shaped lowering; the DirectA arm must emit a
;     single unconditional `br`.
;   * `CHECK-NOT: blockaddress(` pins the ISel-safety fix: the
;     swap handler's sdst write must use a plain integer marker,
;     not `ptrtoint(blockaddress(...))`.

; CHECK-LABEL: define amdgpu_kernel void @setpc_swap_pattern_a_kernel(

; The swap site emits the direct branch to the chain-resolved
; callee. The return-address marker is written into sdst by the
; handler (as a plain `ConstantInt::get(i64Ty, 0x18)`) but is dead
; in this fixture and does not appear in the emitted IR.
; CHECK: br label %bb_0x20

; Both the return-target BB and the callee BB are real, named blocks
; in the lifted IR (bb_0x18 from extraBlockStarts via swap.offset+
; size; bb_0x20 from extraBlockStarts via the resolved chain target).
; CHECK: bb_0x18:
; CHECK: bb_0x20:

; No `indirectbr` may leak back into the lifted IR.
; CHECK-NOT: indirectbr
; No `blockaddress` constant may leak back into the lifted IR —
; the sdst write must use a plain integer marker, not
; `ptrtoint(blockaddress(...))`.
; CHECK-NOT: blockaddress(
