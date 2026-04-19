; RUN: %raise_cli %v_pk_add_f32_lit_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_pk_add_f32_lit_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for VOP3P `v_pk_add_f32` with an inline 32-bit literal
; source.  Before the supporting handler change in
; transpiler/handle_valu_vop3p.cpp, this shape was rejected with
; `non-register source 1 (immediate in VOP3P not supported)`,
; bouncing the swiglu tensilelite kernel out to
; `unsupportedOpcode [VALU]` despite the SemOp + register/register
; lowering being present.
;
; Invariants pinned below:
;
;   1. A `<2 x float> fadd` (the FAdd of two packed lanes) appears,
;      against a `splat (float 1.000000e+00)` operand.  The literal
;      `1.0` (encoded as 0x3F800000 in the inline asm) is broadcast
;      to both lanes by the VOP3P literal path; LLVM's IRBuilder
;      then constant-folds the two `insertelement` ops into a single
;      `splat` constant — that's the surface form we pin here.
;   2. NO `non-register source` diagnostic and NO `unsupportedOpcode`
;      refusal: the test exits 0.  (FileCheck would report empty
;      stdin if either fired.)

; CHECK-LABEL: define amdgpu_kernel void @v_pk_add_f32_lit_kernel(

; The packed add of the register source against the broadcast 1.0
; literal.  The handler emits `B.CreateFAdd(s0, s1, "pk_add")`.
; CHECK: %pk_add = fadd {{(reassoc |nnan |ninf |nsz |arcp |contract |afn |fast )*}}<2 x float> %{{[0-9a-zA-Z_.]+}}, splat (float 1.000000e+00)

; Negative pin: the previous refusal path must not appear.
; CHECK-NOT: unsupportedOpcode
