; RUN: %raise_cli %s_set_pc_i64_pattern_a_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=setpc_pattern_a_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for s_set_pc_i64 — Pattern A (statically resolvable
; intra-kernel direct branch). Pins that the SOP1 indirect set-PC
; lowers to a plain `br label %BB_target` when the static analysis
; in transpiler/setpc_analysis.cpp can fold the source SGPR pair's
; provenance through the canonical chain
;   s_get_pc_i64 → s_add_co_u32 → s_add_co_ci_u32
; into a single in-kernel offset.
;
; The handler lives in transpiler/handle_sop1.cpp under
; `if (sop == SemOp::S_SET_PC_I64) { ... case DirectA: }`. The
; SemOp doc in transpiler/semop.hpp pins the lowering contract.
;
; Why these CHECKs:
;   * `bb_0x18` is the fixture's chain-resolved target. It is
;     derived in the .hip header doc:
;        s_get_pc_i64 records s[10:11] = (di.offset + di.size)
;                                      = 0x08 + 4 = 0x0C (kernel-relative)
;        s_add_co_u32 with imm=12 ⇒ s[10:11] = 0x0C + 12 = 0x18
;     The analysis adds 0x18 to extraBlockStarts; the raiser names
;     basic blocks `bb_0x<hex offset>`.
;   * The `br label %bb_0x18` is the Direct-A lowering. A Pattern B
;     regression would emit `indirectbr ptr ..., [`; an Unresolvable
;     regression would refuse and never produce IR; a silent stub
;     regression would emit `unreachable` instead of `br`.
;   * `bb_0x18:` confirms the target block actually exists in the
;     emitted IR (catches a regression where the chain target gets
;     promoted to a leader but no block is materialised).

; CHECK-LABEL: define amdgpu_kernel void @setpc_pattern_a_kernel(

; The s_set_pc_i64 site lowers to a direct br to the chain-resolved
; offset. CHECK-NOT pins that we did NOT degenerate to indirectbr or
; unreachable in the same function (a Pattern B / Unresolvable
; regression would surface as one of those).
; CHECK: br label %bb_0x18
; CHECK-NOT: indirectbr ptr
; CHECK-NOT: unreachable

; The branch target is a real, named block.
; CHECK: bb_0x18:
