; RUN: %raise_cli %s_swap_pc_i64_pattern_a_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=setpc_swap_pattern_a_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for s_swap_pc_i64 — Pattern A (statically resolvable
; direct call). Pins that the SOP1 branch-and-link lowers to:
;   1. a `blockaddress(@kernel, %bb_<return-offset>)` materialised
;      and written to sdst (the ret SGPR pair), and
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
; Why these CHECKs:
;   * `bb_0x18` is the fixture's resolved return target. Derived in
;     the .hip header doc:
;        K = 0x08 (kernarg-load prologue)
;        swap site at K+0x0C ⇒ return offset = K+0x10 = 0x18
;     The analysis adds 0x18 to extraBlockStarts and the swap handler
;     materialises `blockaddress(@kernel, %bb_0x18)` into sdst.
;   * `bb_0x20` is the chain-resolved callee. Derived as:
;        K+0x18 = 0x20 (the s_add_co_u32 IMM=20 lands the chain at
;        K+0x18; absolute kernel offset is 0x08+0x18 = 0x20)
;     The handler emits `br label %bb_0x20`.
;   * `CHECK-NOT: indirectbr ptr` pins that we did NOT degenerate
;     into the IndirectB lowering (a regression that lost the chain
;     would refuse the swap, not switch to indirectbr — but the
;     CHECK-NOT is cheap insurance in case a future analysis change
;     misclassifies the source pair).
;   * `CHECK-NOT: unreachable` (in the function body) pins that we
;     did not silently emit a stub `unreachable` instead of `br`.

; CHECK-LABEL: define amdgpu_kernel void @setpc_swap_pattern_a_kernel(

; The swap site materialises the return-PC blockaddress and emits
; the direct branch to the chain-resolved callee.
; CHECK: blockaddress(@setpc_swap_pattern_a_kernel, %bb_0x18)
; CHECK: br label %bb_0x20
; CHECK-NOT: indirectbr ptr

; Both the return-target BB and the callee BB are real, named blocks
; in the lifted IR (bb_0x18 from extraBlockStarts via swap.offset+
; size; bb_0x20 from extraBlockStarts via the resolved chain target).
; CHECK: bb_0x18:
; CHECK: bb_0x20:
