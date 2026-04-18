; RUN: %raise_cli %s_mul_hi_i32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=s_mul_hi_i32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for s_mul_hi_i32. Pins that the SOP2 signed mul-high
; lowers to a 64-bit `mul` of two `sext`-widened i32 inputs,
; followed by `lshr ... 32` and `trunc to i32`. The handler lives
; in transpiler/handle_sop2.cpp under
; `if (sop == SemOp::S_MUL_HI_I32) { ... }`; the SemOp lives in
; transpiler/semop.hpp under SOP2.
;
; The shape difference vs S_MUL_HI_U32 (which uses `zext`) is the
; only operational distinction — that's exactly what this test
; pins. A regression that swaps `sext` → `zext` would silently
; turn a signed mul-high into an unsigned one and produce the
; wrong sign on the upper bits when either operand is negative.

; CHECK-LABEL: define amdgpu_kernel void @s_mul_hi_i32_kernel(

; The two inputs must be sign-extended (not zero-extended) before
; the wide multiply. Pin both sext sites; if either degenerates to
; zext the test fires.
; CHECK-DAG: sext i32 %{{[^ ]+}} to i64
; CHECK-DAG: sext i32 %{{[^ ]+}} to i64

; Wide multiply, then take the upper 32 bits (lshr 32) and trunc.
; Names use the `_i_wide` / `mulhi_i` suffixes the handler emits so
; that a future refactor that drops the I-suffix (e.g. unifies the
; two handlers into a single template) is caught.
; CHECK: mul {{.*}}i64 %{{.*}}, %{{.*}}
; CHECK: lshr {{.*}}i64 %{{.*}}, 32
; CHECK: trunc {{.*}}i64 %{{.*}} to i32
