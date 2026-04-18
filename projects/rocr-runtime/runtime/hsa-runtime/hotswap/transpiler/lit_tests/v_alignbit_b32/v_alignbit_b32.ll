; RUN: %raise_cli %v_alignbit_b32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_alignbit_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_alignbit_b32. Pins the .td `fshr` semantics
; (VOP3Instructions.td:222) lifting to a direct
; `@llvm.fshr.i32` call with the shift amount masked to 5 bits
; (matching the hardware's src2[4:0] field). The handler lives
; in transpiler/handle_valu.cpp under
; `if (sop == SemOp::V_ALIGNBIT_B32) { ... }`; the SemOp lives
; in transpiler/semop.hpp under VOP3.

; CHECK-LABEL: define amdgpu_kernel void @v_alignbit_b32_kernel(

; Shift-amount mask (5 bits) is the named `valign_shamt`.
; CHECK: %valign_shamt{{[0-9]*}} = and i32 %{{[^,]+}}, 31

; Funnel-shift call, named `valignbit`.
; CHECK: %valignbit{{[0-9]*}} = call i32 @llvm.fshr.i32(i32 %{{[^,]+}}, i32 %{{[^,]+}}, i32 %valign_shamt{{[0-9]*}})

; Negative check: must NOT lift via fshl (would imply confused
; operand order — fshl would left-shift instead of right). A
; broader CHECK-NOT for `shl i64` is too noisy because the
; kernarg / pointer-arithmetic boilerplate emits an unrelated
; `shl i64 %_, 32` to assemble a 64-bit address from two i32
; halves.
; CHECK-NOT: call {{.*}}@llvm.fshl
