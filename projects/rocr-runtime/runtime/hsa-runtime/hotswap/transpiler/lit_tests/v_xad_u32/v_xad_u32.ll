; RUN: %raise_cli %v_xad_u32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_xad_u32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_xad_u32. Pins the .td iselect pattern at
; VOP3Instructions.td:831 — `add(xor(a, b), c)` — as `xor i32`
; followed by `add i32 %{xor}, c`. The handler lives in
; transpiler/handle_valu.cpp under
; `if (sop == SemOp::V_XAD_U32) { ... }` (next to V_OR3_B32 /
; V_AND_OR_B32 / V_LSHL_OR_B32 — same skeleton, different inner
; and outer ops). The SemOp lives in transpiler/semop.hpp under
; VOP3.

; CHECK-LABEL: define amdgpu_kernel void @v_xad_u32_kernel(

; The handler emits the inner xor unnamed and the outer add as
; `vxad`. Pin both the xor and the add-on-xor.
; CHECK: %{{[^ ]+}} = xor i32 %{{[^,]+}}, %{{[^ ]+}}
; CHECK: %vxad{{[0-9]*}} = add i32 %{{[^,]+}}, %{{[^ ]+}}

; Negative checks: must NOT lift via V_XOR3 (3-way xor) or as a
; plain add of three operands (would imply the iselect pattern
; was misread as `add(a, b)` with `c` discarded).
; CHECK-NOT: vxor3
; CHECK-NOT: vor3
