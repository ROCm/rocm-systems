; RUN: %raise_cli %c2_dpp_quad_perm_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c2_dpp_quad_perm_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; The P5 rewrite (DPP modifier intrinsic lift) has landed — see
; the DPP row of hotswap/docs/wave-size-translation.md §5.3. This
; test was originally a refuse-loud fixture asserting
; `cross-wave-shuffle-rewrite-pending`. Per its MAINTENANCE block we
; flipped it to a positive test that asserts:
;
;   1. The raise now succeeds (no `%not`; the classifier accepts
;      `DppCrossLane` sites as outcome (b) rewrite-implemented, see
;      `wave_size_obstruction.cpp`'s DPP case).
;   2. The emitted IR contains a call to `llvm.amdgcn.update.dpp.i32`
;      with the DPP16 operand set from the source instruction.
;   3. The intrinsic is overloaded on i32 (the corpus-wide 32-bit
;      DPP assumption documented in `raise_context.cpp:emitUpdateDpp`
;      — 64-bit DPP `report_fatal_error`s pending a future lift).
;
; The DPP modifier values in the fixture are
; `quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1` —
; which encode to `dpp_ctrl = 0xB1 = 177`, `row_mask = 0xF = 15`,
; `bank_mask = 0xF = 15`, `bound_ctrl = true`.

; CHECK-LABEL: define amdgpu_kernel void @c2_dpp_quad_perm_kernel(

; The update.dpp call: 6 args — %old, %src, ctrl, row_mask, bank_mask,
; bound_ctrl. %old and %src reference the same SSA value here because
; the inline-asm fixture uses an in/out `+v` constraint on the only
; operand.
; CHECK: call i32 @llvm.amdgcn.update.dpp.i32(i32 %{{[^,]+}}, i32 %{{[^,]+}}, i32 177, i32 15, i32 15, i1 true)

; Declaration of the intrinsic with the correct overload signature.
; CHECK: declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg)
