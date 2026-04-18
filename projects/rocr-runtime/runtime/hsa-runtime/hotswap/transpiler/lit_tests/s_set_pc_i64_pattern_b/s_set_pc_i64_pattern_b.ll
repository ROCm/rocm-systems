; RUN: %raise_cli %s_set_pc_i64_pattern_b_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=setpc_pattern_b_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for s_set_pc_i64 — Pattern B (subroutine return via an
; SGPR pair stashed at the call site). Pins that the SOP1
; indirect set-PC lowers to an LLVM `indirectbr` enumerating the
; resolved return targets when the static analysis in
; transpiler/setpc_analysis.cpp can match the source SGPR pair to
; a chainTerminator recorded at a call site (the canonical
; getpc + add chain whose value is the return address). The
; raiser also rewrites the chain-terminator s_add_co_ci_u32 to
; materialise `blockaddress(@kernel, %ret_BB)` into the ret-pair
; — the binary PC the chain would otherwise yield is unusable as
; an LLVM `indirectbr` operand.
;
; The handler lives in transpiler/handle_sop1.cpp under
; `if (sop == SemOp::S_SET_PC_I64) { ... case IndirectB: }`. The
; SemOp doc in transpiler/semop.hpp pins the lowering contract;
; the call-site rewrite hook lives in raiser.cpp's main loop on
; `S_ADDC_U32` (see the post-handler block keyed on
; `setpcAnalysis.chainTerminators`).
;
; Why these CHECKs:
;   * The first s_set_pc_i64 in the .hip fixture is Pattern A and
;     lowers to `br label %bb_0x30` (the .Lcallee target). That
;     half of the fixture is also pinned here so a Pattern B
;     regression that accidentally mis-classified the Pattern A
;     site as IndirectB would surface as an `indirectbr` in
;     bb_0x0 instead of a `br`.
;   * The second s_set_pc_i64 is Pattern B and must lower to an
;     `indirectbr`. The destination list must contain `bb_0x28`
;     — the .Lret label — derived in the .hip header from the
;     chain math:
;        s_get_pc_i64 records s[8:9] = (di.offset + di.size)
;                                    = 0x08 + 4 = 0x0C
;        s_add_co_u32 with imm=28 ⇒ s[8:9] = 0x0C + 28 = 0x28
;     The analysis records chainTerminators[0x10] = (0x28, 8) and
;     pendingB at 0x38 with retPair=8; Pass 2 builds
;     targetsByPair[8] = [0x28], so the IndirectB site emits
;     `indirectbr ... [label %bb_0x28]`.
;   * `bb_0x28:` confirms the indirectbr target block actually
;     exists in the emitted IR.
;
; The indirectbr handler reads the ret-pair via
; `regs.loadSGPR64(B, retPairLowReg)` and then `inttoptr`s the
; i64 to a `ptr` — we pin the `ret_pc_ptr` SSA name and the
; pointer-to-i64 round-trip to lock the lowering shape.

; CHECK-LABEL: define amdgpu_kernel void @setpc_pattern_b_kernel(

; Pattern A site (first s_set_pc_i64) lowers to a direct br to
; the chain-resolved callee at offset 0x30.
; CHECK: bb_0x0:
; CHECK: br label %bb_0x30

; The indirectbr's destination block must exist in the emitted
; IR — pin it before the indirectbr line because LLVM emits BB
; label definitions in CFG order, and bb_0x28 (the .Lret target)
; appears in the function body before bb_0x30 / spe_skip11
; where the indirectbr is materialised.
; CHECK: bb_0x28:

; Pattern B site (second s_set_pc_i64) lowers to indirectbr.
; The ret-pair is read as i64, cast to ptr (`inttoptr`), and the
; destination list enumerates each resolved return target. With
; one call site recorded by the analysis, the destination list
; contains exactly one label: %bb_0x28 (the .Lret target).
; The handler names the `inttoptr` result `ret_pc_ptr`; this
; pins both the lowering shape and the conversion direction.
; CHECK: %ret_pc_ptr = inttoptr i64 %{{[^ ]+}} to ptr
; CHECK-NEXT: indirectbr ptr %ret_pc_ptr, [label %bb_0x28]
