; RUN: %raise_cli %v_mul_u64_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_mul_u64_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_mul_u64. Pins that the gfx1250 VOP2 64-bit
; unsigned multiply lowers to a single i64 `mul`. The handler
; lives in transpiler/handle_valu.cpp under
; `if (sop == SemOp::V_MUL_U64) { ... }`; the SemOp lives in
; transpiler/semop.hpp under "64-bit vector ops".
;
; Per AMDGPU/VOP2Instructions.td:942 V_MUL_U64 has SDPattern
;   DivergentBinFrag<mul>
; on i64 — there's no high-half output and no carry/SCC side-effect,
; so the lowered IR must be a bare `mul i64`, not a wider/narrower
; form, and never a *.with.overflow intrinsic.

; CHECK-LABEL: define amdgpu_kernel void @v_mul_u64_kernel(

; The handler's `vmul64` value-name is the canonical breadcrumb
; (mirrors `vadd64` from the V_ADD_NC_U64 sibling in the same file).
; CHECK: mul {{.*}}i64 %{{[^,]+}}, %{{[^,]+}}

; Negative checks:
;   - no overflow-tracked variant
;   - no high-half multiply intrinsic (e.g. v_mul_hi_u32 fallback)
;   - no narrowing to a 32-bit mul for *this* kernel body
; CHECK-NOT: @llvm.umul.with.overflow
; CHECK-NOT: @llvm.amdgcn.mul.hi
; CHECK-NOT: mul {{.*}}i32
