; RUN: %raise_cli %tensor_store_from_lds_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=tensor_store_from_lds_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Lift fixture for VIMAGE TENSOR `tensor_store_from_lds_d2` —
; cross-target TDM-emulation path. Companion to
; `tensor_load_to_lds.ll`; this one pins the store direction.
;
; The store handler's output differs from the load handler's output
; in exactly one place: the emitted call target is
; `salmon_tdm_store_from_lds` instead of `salmon_tdm_load_to_lds`.
; Everything else — the group 0/1 insertelement chains, the
; zero-fill of groups 2/3 for the _d2 form, the reserved group 4,
; the cpol immediate — is structurally identical (same
; `marshalTDMArgs` code path; see `handle_vimage.cpp`). Pinning both
; directions catches an accidental wire-up swap (emitting the load
; helper for the store SemOp, which would silently turn every
; hand-authored store into a load).

; Group 0: <4 x i32> from s40..s43.
; IR: %td_grp0{{[0-9]*}} = insertelement <4 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp0{{[0-9]*}} = insertelement <4 x i32> %td_grp0{{[0-9]*}}, i32 {{.*}}, i64 3

; Group 1: <8 x i32> from s4..s11.
; IR: %td_grp1{{[0-9]*}} = insertelement <8 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp1{{[0-9]*}} = insertelement <8 x i32> %td_grp1{{[0-9]*}}, i32 {{.*}}, i64 7

; The store-direction helper call: _d2 form, so groups 2/3 are zero.
; The helper's signature carries only the four D# groups — the
; intrinsic's trailing `<8 x i32> grp4` and `i32 cpol` are NOT
; forwarded (see `tdm_runtime.hpp`).
; IR: call void @salmon_tdm_store_from_lds(
; IR-SAME: <4 x i32> %td_grp0
; IR-SAME: <8 x i32> %td_grp1
; IR-SAME: <4 x i32> zeroinitializer
; IR-SAME: <4 x i32> zeroinitializer
; IR-SAME: )

; The load helper symbol must NOT appear in a store lift — catches an
; accidental wire-up that calls load instead of store. `@salmon_tdm_load_to_lds`
; appears in `llvm.compiler.used` (the bitcode merge hoists both
; helpers with `__attribute__((used))`), so we restrict the check to
; "not called as a function".
; IR-NOT: call void @salmon_tdm_load_to_lds
