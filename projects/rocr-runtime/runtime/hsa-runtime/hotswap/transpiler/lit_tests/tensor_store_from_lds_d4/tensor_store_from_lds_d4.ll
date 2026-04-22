; RUN: %raise_cli %tensor_store_from_lds_d4_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=tensor_store_from_lds_d4_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Lift fixture for VIMAGE TENSOR `tensor_store_from_lds_d4` —
; cross-target TDM-emulation path for the 4/5D Tensor Descriptor
; form, store direction. Completes the 2x2 encoding matrix started
; by `tensor_load_to_lds.ll` (_d2 load),
; `tensor_load_to_lds_d4.ll` (_d4 load), and
; `tensor_store_from_lds.ll` (_d2 store).
;
; The _d4 form requires that groups 2 and 3 be marshalled from real
; SGPR ranges (see `tensor_load_to_lds_d4.ll`). A regression that
; zeroed them would change the 3D/4D/5D walk semantics in both
; directions; pinning both directions guards against a one-sided
; fix that only addresses load.

; Group 0: <4 x i32> from s40..s43.
; IR: %td_grp0{{[0-9]*}} = insertelement <4 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp0{{[0-9]*}} = insertelement <4 x i32> %td_grp0{{[0-9]*}}, i32 {{.*}}, i64 3

; Group 1: <8 x i32> from s4..s11.
; IR: %td_grp1{{[0-9]*}} = insertelement <8 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp1{{[0-9]*}} = insertelement <8 x i32> %td_grp1{{[0-9]*}}, i32 {{.*}}, i64 7

; Group 2: <4 x i32> from s12..s15.
; IR: %td_grp2{{[0-9]*}} = insertelement <4 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp2{{[0-9]*}} = insertelement <4 x i32> %td_grp2{{[0-9]*}}, i32 {{.*}}, i64 3

; Group 3: <4 x i32> from s16..s19.
; IR: %td_grp3{{[0-9]*}} = insertelement <4 x i32> poison, i32 {{.*}}, i64 0
; IR: %td_grp3{{[0-9]*}} = insertelement <4 x i32> %td_grp3{{[0-9]*}}, i32 {{.*}}, i64 3

; The store-direction helper call: all four groups marshalled from
; SGPRs (_d4 form). The helper's signature does NOT carry the
; intrinsic's trailing `<8 x i32> grp4` or `i32 cpol` (see
; `tdm_runtime.hpp`).
; IR: call void @salmon_tdm_store_from_lds(
; IR-SAME: <4 x i32> %td_grp0
; IR-SAME: <8 x i32> %td_grp1
; IR-SAME: <4 x i32> %td_grp2
; IR-SAME: <4 x i32> %td_grp3
; IR-SAME: )

; IR-NOT: call void @salmon_tdm_load_to_lds
