; RUN: %raise_cli %v_max3_u32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_max3_u32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_max3_u32. Pins that the VOP3 ternary unsigned
; max lowers to the canonical 2-step ICmpUGT+Select chain
; (mirrors the V_MAX_U32 binary handler one block above in the
; same file, intentionally so a future refactor that switches
; V_MAX_U32 to llvm.umax can propagate to V_MAX3_U32 in one go).
; The handler lives in transpiler/handle_valu.cpp under
; `if (sop == SemOp::V_MAX3_U32) { ... }`; the SemOp lives in
; transpiler/semop.hpp under VOP3.
;
; The shape difference vs V_MAX3_F32 (FP fmax-of-fmax via
; @llvm.maxnum.f32) is the use of integer ICmp + Select — the
; .td pattern is `AMDGPUumax3` which is `(umax (umax a, b), c)`.
; A regression that swaps `ICmpUGT` → `ICmpSGT` would silently
; turn the unsigned max into a signed one (same shape, wrong
; semantics on the high half of the i32 range).

; CHECK-LABEL: define amdgpu_kernel void @v_max3_u32_kernel(

; The handler emits two icmp/select pairs with names `vmax3_lo`
; (intermediate `umax(a, b)`) and `vmax3` (final `umax(_, c)`).
; CHECK: icmp ugt i32 %{{[^,]+}}, %{{[^ ]+}}
; CHECK: select i1 %{{[^,]+}}, i32 %{{[^,]+}}, i32 %{{[^ ]+}}
; CHECK: icmp ugt i32 %{{[^,]+}}, %{{[^ ]+}}
; CHECK: select i1 %{{[^,]+}}, i32 %{{[^,]+}}, i32 %{{[^ ]+}}

; Negative checks: must NOT lift via signed compare or via the
; FP @llvm.maxnum intrinsic (that would imply the f32 family
; handler accidentally absorbed this op).
; CHECK-NOT: icmp sgt
; CHECK-NOT: call {{.*}}@llvm.maxnum
