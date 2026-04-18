; RUN: %raise_cli %v_add_nc_u16_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_add_nc_u16_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_add_nc_u16 (default op_sel:[0,0,0]). Pins:
;   * src0/src1 are truncated from i32 to i16 (no LShr because
;     op_sel sources are both 0 = lo half).
;   * `add i16` is the named `vadd_nc_u16` value.
;   * Result is zero-extended back to i32 and merged into dst via
;     mask-OR with `0xFFFF0000` against the prior dst value
;     (`vadd_u16_merge_lo`). The merge is what makes the dst-half
;     preservation semantics observable in the IR shape — without
;     it the handler would silently miscompile op_sel:[*, *, 1]
;     and sibling 16-bit ops with non-default dst.
;
; The handler lives in transpiler/handle_valu.cpp under
; `if (sop == SemOp::V_ADD_NC_U16) { ... }`; the SemOp lives in
; transpiler/semop.hpp under VOP3.

; CHECK-LABEL: define amdgpu_kernel void @v_add_nc_u16_kernel(

; Source half-extraction (default op_sel:[0,0]: trunc only, no
; lshr). Names are unnamed (default IRBuilder behaviour).
; CHECK: trunc i32 %{{[^ ]+}} to i16
; CHECK: trunc i32 %{{[^ ]+}} to i16

; The 16-bit add itself, named to match the handler value name.
; CHECK: %vadd_nc_u16{{[0-9]*}} = add i16 %{{[^,]+}}, %{{[^ ]+}}

; Dst-half merge: zext, AND with high-half mask, OR back in.
; CHECK-DAG: zext i16 %vadd_nc_u16{{[0-9]*}} to i32
; CHECK-DAG: and i32 %{{[^,]+}}, -65536
; CHECK: %vadd_u16_merge_lo{{[0-9]*}} = or {{(disjoint )?}}i32 %{{[^,]+}}, %{{[^ ]+}}

; Negative checks: must NOT produce a 32-bit add (would imply
; integer-promotion-style lift) or call any 16-bit add intrinsic
; (LLVM has no such thing; would imply a wrong intrinsic was
; introduced).
; CHECK-NOT: add i32
; CHECK-NOT: call {{.*}}@llvm.amdgcn.add{{.*}}u16
