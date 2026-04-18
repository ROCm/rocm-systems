; RUN: %raise_cli %v_minmax_num_f32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_minmax_num_f32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_minmax_num_f32. Pins:
;   * Inner `@llvm.maxnum.f32` (named `vminmax_inner`).
;   * Outer `@llvm.minnum.f32` (named `vminmax_num`).
; Same layered-intrinsic pattern as V_MAX3_F32 / V_MIN3_F32 /
; V_MED3_F32 in the same handler file. The handler lives in
; transpiler/handle_valu.cpp under
; `if (sop == SemOp::V_MINMAX_NUM_F32) { ... }`; the SemOp
; lives in transpiler/semop.hpp under VOP3.

; CHECK-LABEL: define amdgpu_kernel void @v_minmax_num_f32_kernel(

; The inner maxnum and the outer minnum, both named.
; CHECK: %vminmax_inner{{[0-9]*}} = call float @llvm.maxnum.f32(float %{{[^,]+}}, float %{{[^)]+}})
; CHECK: %vminmax_num{{[0-9]*}} = call float @llvm.minnum.f32(float %vminmax_inner{{[0-9]*}}, float %{{[^)]+}})

; Negative checks: must NOT lift via the IEEE-2019
; NaN-propagating @llvm.maximum / @llvm.minimum (those are
; reserved for V_MINIMUMMAXIMUM_F32 / V_MAXIMUMMINIMUM_F32 at
; opcodes 0x26c / 0x26d) — confusing the .NUM and .non-.NUM
; families would silently flip NaN-handling semantics.
; CHECK-NOT: call {{.*}}@llvm.maximum
; CHECK-NOT: call {{.*}}@llvm.minimum
