; RUN: %raise_cli %v_cmp_class_f32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_cmp_class_f32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_cmp_class_f32_e64 (VOPC, floating-point IEEE class
; predicate). Pins the dual contract that completes the parser-level
; gap closure in `parseVCmpPseudoName`:
;
;   1. The MC pseudo `V_CMP_CLASS_F32_e64` parses as the class
;      special-case (not an FCmp predicate compare); the handler in
;      `handle_valu_vcmp.cpp` takes the `if (m->isClass)` branch
;      and emits `llvm.amdgcn.class.f32`, NOT a CreateFCmp.
;   2. The wave-mask write-back to the SGPR-pair destination shares
;      the same `ballot` + `trunc` discipline as the V_CMP/V_CMPX
;      predicate compares (the reused `cmp` -> `vcmp_ballot` tail
;      in handle_valu_vcmp.cpp). The cross-wave wave32 -> wave64
;      direction here pins the `i64 -> i32` truncation step.
;
; The mask is the literal `i32 512` (= 0x200, bit 9 = +inf), threaded
; through to the intrinsic call without any rewriting; if a future
; change re-encodes class masks we want a loud failure here, not a
; silent one. The companion fixture `v_cmpx_ballot` covers the
; predicate-compare side of the same wave-mask plumbing.

; CHECK-LABEL: define amdgpu_kernel void @v_cmp_class_f32_kernel(

; The class call: per-lane i1 result, FP source bitcast to f32, mask
; threaded through as the literal `i32 512`. The `vclass` name is
; pinned by the handler (search for `"vclass"` in handle_valu_vcmp.cpp).
; CHECK: %vclass{{[0-9]*}} = call i1 @llvm.amdgcn.class.f32(float %{{[^,]+}}, i32 512)

; The wave-mask write-back, shared with the predicate-compare path:
; the i1 feeds amdgcn.ballot.i64 (target wave64), the i64 result is
; truncated back to the source's wave32 execTy (i32), and the
; truncated mask is then stored to the SGPR alloca. `vcmp_ballot` /
; `vcmp_ballot_trunc` are the names pinned by the V_CMP -> SGPR
; branch in handle_valu_vcmp.cpp (same identifiers asserted by the
; v_cmpx_ballot fixture).
; CHECK: %vcmp_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %vclass{{[0-9]*}})
; CHECK-NEXT: %vcmp_ballot_trunc = trunc i64 %vcmp_ballot to i32

; Negative assertions: the lift MUST NOT take the FCmp path (which
; would compare the float operand to the i32 mask reinterpreted as a
; float — silently miscompiling the entire kernel). It also MUST NOT
; sext the i1 directly into the SGPR — that's the pre-fix shape from
; the v_cmpx_ballot regression and would re-introduce the divergent
; SSA value into a wave-mask consumer.
; CHECK-NOT: fcmp {{.*}} float %{{[^,]+}}, {{.*}}i32
; CHECK-NOT: sext i1 %vclass{{[0-9]*}} to i32
