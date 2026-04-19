; RUN: %raise_cli %tensor_load_to_lds_co --isa=gfx1250 \
; RUN:     --target-isa=gfx1250 --emit-ir=tensor_load_to_lds_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Lift fixture for VIMAGE TENSOR `tensor_load_to_lds_d2` — same-target
; (gfx1250 -> gfx1250) intrinsic-emit path. Pins the principled
; lift in transpiler/handle_vimage.cpp under
; SemOp::TENSOR_LOAD_TO_LDS when `ctx.targetIsa.hasTensorOps` is
; true. Companion fixture to `tensor_load_to_lds.ll`, which pins
; the cross-target (gfx942) loud refusal.
;
; The MIMGInstructions.td:2049-2113 `VIMAGE_TENSOR_Pseudo` operand
; layout for the `_d2` form is `vaddr0:SReg_128, vaddr1:SReg_256,
; r128:imm, cpol:imm`. The matching LLVM intrinsic
; (IntrinsicsAMDGPU.td:4197-4214) is
;
;   void llvm.amdgcn.tensor.load.to.lds(<4 x i32> grp0,
;                                        <8 x i32> grp1,
;                                        <4 x i32> grp2,
;                                        <4 x i32> grp3,
;                                        <8 x i32> grp4_reserved,
;                                        i32 immarg cachepolicy)
;
; The handler marshals each SGPR range into the matching
; `<n x i32>` via `loadSGPR32` + `insertelement`, zero-fills the
; unused `_d2` groups (2 and 3) and the always-reserved group 4,
; and threads the `cpol` immediate through as `i32 0` for the
; corpus encoding (the hand-encoded `D0710001 7C000000 7C7C0428`
; payload sets cpol=0).
;
; We pin two things:
;   1. The eight insertelement chain that materialises group 0 from
;      s40..s43 and group 1 from s4..s11. The fixture's inline-asm
;      clobber list never assigns those SGPRs, so the loads fold to
;      `undef` — the structural shape of the chain is what matters.
;   2. The intrinsic call's argument vector: groups 0/1 are the
;      marshalled SGPR vectors; groups 2/3 are <4 x i32>
;      zeroinitializer; group 4 is <8 x i32> zeroinitializer; cpol
;      is the constant `i32 0`.
;
; Drift indicators:
;   * If a future LLVM rename swaps the intrinsic name (e.g. drops
;     `tensor.` prefix) the IR check fails immediately and pinpoints
;     the rename rather than letting a silently mis-named intrinsic
;     reach the backend.
;   * If the operand-marshalling order changes (group ordering,
;     vector widths, or the reserved-group convention) the
;     insertelement / call-argument shape diverges and FileCheck
;     reports the exact line.

; Group 0: <4 x i32> built from four sequential SGPR reads (s40..s43).
; The first lane seeds the chain off `poison`; the last lane
; (`i64 3`) closes it. LLVM's instnamer suffixes the SSA values
; (`%td_grp0`, `%td_grp02`, ...) so we use a regex on the trailing
; numeric.
; IR: %td_grp0{{[0-9]*}} = insertelement <4 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp0{{[0-9]*}} = insertelement <4 x i32> %td_grp0{{[0-9]*}}, i32 {{.*}}, i64 3

; Group 1: <8 x i32> built from eight sequential SGPR reads (s4..s11).
; IR: %td_grp1{{[0-9]*}} = insertelement <8 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp1{{[0-9]*}} = insertelement <8 x i32> %td_grp1{{[0-9]*}}, i32 {{.*}}, i64 7

; The intrinsic call: groups 2/3 are <4 x i32> zeroinitializer
; (unused for the `_d2` form), group 4 is <8 x i32> zeroinitializer
; (always reserved), and the cpol immediate is `i32 0` for the
; corpus payload.
; IR: call void @llvm.amdgcn.tensor.load.to.lds(
; IR-SAME: <4 x i32> %td_grp0
; IR-SAME: <8 x i32> %td_grp1
; IR-SAME: <4 x i32> zeroinitializer
; IR-SAME: <4 x i32> zeroinitializer
; IR-SAME: <8 x i32> zeroinitializer
; IR-SAME: i32 0
