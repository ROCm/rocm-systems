; RUN: %raise_cli %v_med3_i32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_med3_i32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_med3_i32 (VOP3, signed-integer median-of-three).
; Pins the four-call cascade the handler emits:
;   * `vmed3_lo` = `llvm.smin.i32(s0, s1)`    (the lower of the pair)
;   * `vmed3_hi` = `llvm.smax.i32(s0, s1)`    (the upper of the pair)
;   * `vmed3_clamp` = `llvm.smin.i32(vmed3_hi, s2)` (clamp pair-max to s2)
;   * `vmed3` = `llvm.smax.i32(vmed3_lo, vmed3_clamp)`
;     -> the median.
;
; The shape mirrors the standard `smax(smin(a, b), smin(smax(a, b), c))`
; identity for med3 over signed ints. The AMDGPU backend's
; AMDGPUsmed3 SDAG pattern (AMDGPUInstructions.td) matches this exact
; nesting back to V_MED3_I32, so the round-trip is structure-preserving.
;
; The handler lives in transpiler/handle_valu.cpp under
; `if (sop == SemOp::V_MED3_I32) { ... }`; the SemOp lives in
; transpiler/semop.hpp under VOP3.

; CHECK-LABEL: define amdgpu_kernel void @v_med3_i32_kernel(

; The four-call cascade, in emission order. Each call is named
; (`vmed3_lo`, `vmed3_hi`, `vmed3_clamp`, `vmed3`) so a future
; reorder or rename pattern-fails this fixture immediately.
; CHECK: %vmed3_lo{{[0-9]*}} = call i32 @llvm.smin.i32(i32 %{{[^,]+}}, i32 %{{[^)]+}})
; CHECK: %vmed3_hi{{[0-9]*}} = call i32 @llvm.smax.i32(i32 %{{[^,]+}}, i32 %{{[^)]+}})
; CHECK: %vmed3_clamp{{[0-9]*}} = call i32 @llvm.smin.i32(i32 %vmed3_hi{{[0-9]*}}, i32 %{{[^)]+}})
; CHECK: %vmed3{{[0-9]*}} = call i32 @llvm.smax.i32(i32 %vmed3_lo{{[0-9]*}}, i32 %vmed3_clamp{{[0-9]*}})

; Negative checks: must NOT lower via the unsigned forms (umin/umax)
; — that would silently flip the sign-handling on negative i32
; operands. Must also NOT lower via the dedicated
; `llvm.amdgcn.smed3` intrinsic — we deliberately use the open
; smin/smax form so peephole IR optimisations can compose with it,
; per the comment on the V_MED3_I32 SemOp in semop.hpp.
; CHECK-NOT: call {{.*}}@llvm.umin
; CHECK-NOT: call {{.*}}@llvm.umax
; CHECK-NOT: call {{.*}}@llvm.amdgcn.smed3
