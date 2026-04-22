; RUN: %raise_cli %tensor_load_to_lds_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=tensor_load_to_lds_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Lift fixture for VIMAGE TENSOR `tensor_load_to_lds_d2` — cross-target
; (gfx1250 -> gfx942) TDM emulation path. Pins the principled lift in
; transpiler/handle_vimage.cpp under SemOp::TENSOR_LOAD_TO_LDS when the
; compilation target does not have the gfx1250 TENSORcnt unit and the
; embedded TDM runtime bitcode is available (hipcc was on PATH at CMake
; configure time — otherwise the pre-TDM refusal path still fires; see
; `tdmRuntimeAvailable()` in `tdm_runtime.{hpp,cpp}`).
;
; The _d2 form's `InOperandList` is `vaddr0:SReg_128, vaddr1:SReg_256,
; r128:imm, cpol:imm` (MIMGInstructions.td:2073). The NULL sentinel
; 0x7C in vaddr2/vaddr3 in the wire encoding (`7C7C0428` low word of
; dword 2) indicates the _d2 form. `marshalTDMArgs` marshals groups
; 0/1 from the two SGPR ranges via `loadSGPR32` + `insertelement`,
; zero-fills the unused groups 2 and 3, and threads cpol through as
; `i32 0` for this encoding. The same six-argument tuple feeds the
; same-target intrinsic emit (see `tensor_load_to_lds_same_target.ll`)
; and the cross-target helper call pinned here.
;
; We assert two things:
;
;   1. Group 0 <4 x i32> materialises from four sequential SGPR reads
;      (s40..s43), group 1 <8 x i32> materialises from eight sequential
;      SGPR reads (s4..s11). LLVM's instnamer suffixes the SSA values
;      (`%td_grp0`, `%td_grp02`, ...) so we regex the trailing numeric.
;
;   2. The call to the link-merged helper `salmon_tdm_load_to_lds`
;      receives groups 2/3 as `<4 x i32> zeroinitializer` (the `_d2`
;      form contract), group 4 as `<8 x i32> zeroinitializer` (always
;      reserved in the intrinsic signature, carried through by the
;      handler for shape parity), and cpol as the constant `i32 0`.
;
; Drift indicators:
;   * If the helper symbol is renamed (e.g. `salmon_tdm_load` drops the
;     `_to_lds` suffix) the call-line check fails and pinpoints the
;     rename.
;   * If the handler forgets to zero groups 2/3 for the `_d2` form the
;     zeroinitializer check fails, which would be a functional bug
;     because the walker then iterates against garbage SGPRs.

; Group 0: <4 x i32> from four consecutive SGPR reads (s40..s43).
; IR: %td_grp0{{[0-9]*}} = insertelement <4 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp0{{[0-9]*}} = insertelement <4 x i32> %td_grp0{{[0-9]*}}, i32 {{.*}}, i64 3

; Group 1: <8 x i32> from eight consecutive SGPR reads (s4..s11).
; IR: %td_grp1{{[0-9]*}} = insertelement <8 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp1{{[0-9]*}} = insertelement <8 x i32> %td_grp1{{[0-9]*}}, i32 {{.*}}, i64 7

; The helper call: groups 2/3 are <4 x i32> zeroinitializer (unused
; in the `_d2` form). The helper's signature carries only the four
; D# groups (see `tdm_runtime.hpp`) — the intrinsic's trailing
; `<8 x i32> grp4` (reserved) and `i32 cpol` are NOT forwarded to
; the helper because the emulation walk does not consume them; the
; same-target intrinsic call still carries all six.
; IR: call void @salmon_tdm_load_to_lds(
; IR-SAME: <4 x i32> %td_grp0
; IR-SAME: <8 x i32> %td_grp1
; IR-SAME: <4 x i32> zeroinitializer
; IR-SAME: <4 x i32> zeroinitializer
; IR-SAME: )
