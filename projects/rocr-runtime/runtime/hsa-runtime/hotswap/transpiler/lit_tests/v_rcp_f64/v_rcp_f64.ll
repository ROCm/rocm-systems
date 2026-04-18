; RUN: %raise_cli %v_rcp_f64_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_rcp_f64_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_rcp_f64. Pins that the VOP1 64-bit reciprocal lowers
; to `llvm.amdgcn.rcp.f64`. The handler lives in
; transpiler/handle_valu.cpp under `if (sop == SemOp::V_RCP_F64) { ... }`;
; the SemOp lives in transpiler/semop.hpp under the FP64 group.
;
; The lift goes through the AMDGPU intrinsic rather than a generic
; `fdiv 1.0, x` because:
;   * gfx942 isels `int_amdgcn_rcp.f64` straight back to v_rcp_f64
;     (the source op's exact ~26-bit-accurate semantics).
;   * `fdiv 1.0, x` on f64 would lower to a software divide sequence
;     on the target (mul + Newton-Raphson refinement) unless `arcp` /
;     fast-math flags are set — that's a silent semantics change.

; CHECK-LABEL: define amdgpu_kernel void @v_rcp_f64_kernel(

; The lifted IR must contain a call to the amdgcn rcp intrinsic at
; f64 precision. The exact SSA register names are unimportant; the
; presence of the intrinsic call is what pins the lift shape.
; CHECK: call {{.*}}double @llvm.amdgcn.rcp.f64(double {{.*}})

; The intrinsic declaration must be present (proves the call wasn't
; created against the wrong overload).
; CHECK: declare {{.*}}double @llvm.amdgcn.rcp.f64(double)
