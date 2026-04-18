; RUN: %raise_cli %v_xor3_b32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_xor3_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_xor3_b32. Pins that the VOP3 ternary xor lowers
; to two nested `xor i32` instructions (matches the .td iselect
; pattern at VOP3Instructions.td:1350 — `(xor (xor a, b), c)`).
; The handler lives in transpiler/handle_valu.cpp under
; `if (sop == SemOp::V_XOR3_B32) { ... }`; the SemOp lives in
; transpiler/semop.hpp under VOP3.
;
; Direct mirror of V_OR3_B32 in the same file (one block above) —
; if a future refactor changes one, both should change in
; lockstep.

; CHECK-LABEL: define amdgpu_kernel void @v_xor3_b32_kernel(

; The handler emits the inner xor unnamed (default IRBuilder
; behaviour) and the outer xor as `vxor3` — pin both shapes.
; CHECK: xor i32 %{{[^,]+}}, %{{[^ ]+}}
; CHECK: xor i32 %{{[^,]+}}, %{{[^ ]+}}

; Negative checks: bitwise ternary must NOT lift via the bitop3
; LUT-expansion path (V_BITOP3_B32 is for the explicit imm8 LUT
; instruction — confusing the two would balloon the IR with 8
; minterm AND/OR chains).
; CHECK-NOT: call {{.*}}@llvm.amdgcn.perm
; CHECK-NOT: bitop3
