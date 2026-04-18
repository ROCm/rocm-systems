; RUN: %raise_cli %s_sub_f32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=s_sub_f32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for s_sub_f32. Pins that the SOP2 scalar FP subtract
; lowers to a plain `fsub float` between the two i32 operands
; bit-cast to f32, with no source modifiers (SOP2 has none). The
; handler lives in transpiler/handle_sop2.cpp under
; `if (sop == SemOp::S_SUB_F32) { ... }`; the SemOp lives in
; transpiler/semop.hpp under SOP2.
;
; The shape difference vs S_ADD_F32 (one block above in the same
; handler file) is the FSub vs FAdd choice — that's the only
; operational distinction this test pins. A regression that swaps
; `fsub` → `fadd` (e.g. a copy-paste from the add handler that
; forgets to flip the opcode) would silently invert the operation.

; CHECK-LABEL: define amdgpu_kernel void @s_sub_f32_kernel(

; The handler's named value is `s_fsub` (mirrors `s_fadd` /
; `s_fmul` siblings). Pin both the bitcast-in and the fsub.
; CHECK-DAG: bitcast i32 %{{[^ ]+}} to float
; CHECK-DAG: bitcast i32 %{{[^ ]+}} to float
; CHECK: fsub {{.*}}float %{{[^,]+}}, %{{[^,]+}}
; CHECK: bitcast float %{{[^ ]+}} to i32

; Negative checks: SOP2 has no source-modifier slots, so the lift
; must NOT introduce fneg/fabs around the inputs. If a future
; refactor unifies the f32 SOP2 family with a VOP3-style modifier
; pipeline, this fires.
; CHECK-NOT: fneg
; CHECK-NOT: call {{.*}}@llvm.fabs
