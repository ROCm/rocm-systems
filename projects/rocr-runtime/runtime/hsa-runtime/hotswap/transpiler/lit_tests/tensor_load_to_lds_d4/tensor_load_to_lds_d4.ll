; RUN: %raise_cli %tensor_load_to_lds_d4_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=tensor_load_to_lds_d4_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Lift fixture for VIMAGE TENSOR `tensor_load_to_lds_d4` — cross-target
; TDM-emulation path for the 4/5D Tensor Descriptor form. Companion to
; `tensor_load_to_lds.ll` (the _d2 form, groups 2/3 zeroinitialized).
;
; The _d4 form's InOperandList is `vaddr0:SReg_128, vaddr1:SReg_256,
; vaddr2:SReg_128, vaddr3:SReg_128, r128:imm, cpol:imm`
; (MIMGInstructions.td:2073). `marshalTDMArgs` recovers the form from
; `op.nSrcs()` (6 rather than 4) and marshals groups 2 AND 3 from the
; matching SGPR ranges via `loadSGPR32` + `insertelement`, instead of
; zero-filling them as in the _d2 form. The six-argument tuple that
; reaches the helper call is:
;
;   call salmon_tdm_load_to_lds(<4 x i32> grp0,  ; from s40..s43
;                                <8 x i32> grp1,  ; from s4..s11
;                                <4 x i32> grp2,  ; from s12..s15  <-- _d4
;                                <4 x i32> grp3,  ; from s16..s19  <-- _d4
;                                <8 x i32> zeroinitializer,
;                                i32 0)
;
; The IR shape that distinguishes _d4 from _d2 is the presence of the
; group 2 and group 3 insertelement chains — _d2 passes both as
; `<4 x i32> zeroinitializer` directly. A regression that demotes _d4
; to _d2 (zeroing groups 2/3) silently miscompiles any kernel using
; the higher dims of its Tensor Descriptor.

; Group 0: <4 x i32> from s40..s43.
; IR: %td_grp0{{[0-9]*}} = insertelement <4 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp0{{[0-9]*}} = insertelement <4 x i32> %td_grp0{{[0-9]*}}, i32 {{.*}}, i64 3

; Group 1: <8 x i32> from s4..s11.
; IR: %td_grp1{{[0-9]*}} = insertelement <8 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp1{{[0-9]*}} = insertelement <8 x i32> %td_grp1{{[0-9]*}}, i32 {{.*}}, i64 7

; Group 2: <4 x i32> from s12..s15 — NOT zeroinitializer.
; IR: %td_grp2{{[0-9]*}} = insertelement <4 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp2{{[0-9]*}} = insertelement <4 x i32> %td_grp2{{[0-9]*}}, i32 {{.*}}, i64 3

; Group 3: <4 x i32> from s16..s19 — NOT zeroinitializer.
; IR: %td_grp3{{[0-9]*}} = insertelement <4 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp3{{[0-9]*}} = insertelement <4 x i32> %td_grp3{{[0-9]*}}, i32 {{.*}}, i64 3

; The helper call receives all four marshalled groups. The helper's
; signature does NOT carry the intrinsic's trailing `<8 x i32> grp4`
; or `i32 cpol` — see `tdm_runtime.hpp` for the rationale. An
; accidental revert to the six-arg helper signature would fail this
; CHECK at the closing `)`.
; IR: call void @salmon_tdm_load_to_lds(
; IR-SAME: <4 x i32> %td_grp0
; IR-SAME: <8 x i32> %td_grp1
; IR-SAME: <4 x i32> %td_grp2
; IR-SAME: <4 x i32> %td_grp3
; IR-SAME: )
