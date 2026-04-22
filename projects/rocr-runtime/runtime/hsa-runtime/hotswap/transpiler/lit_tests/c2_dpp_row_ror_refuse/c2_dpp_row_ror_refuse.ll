; Negative fixture: the DPP cross-widen rewrite must refuse
; loudly on any `dpp_ctrl` outside the supported family
; (quad_perm / row_shl / row_shr).  This fixture pins the
; refusal diagnostic for `row_ror:1` (ctrl = 0x121); the
; companion positive fixture is `c2_dpp_quad_perm.ll`.
;
; Contract: `raise_cli` under cross-widening (gfx1250 -> gfx942)
; must exit non-zero AND the stderr must name the specific
; unsupported ctrl.  This closes the pair with the positive
; fixture: together they pin BOTH sides of the rewrite's
; all-or-nothing symmetry invariant.

; RUN: %not %raise_cli %c2_dpp_row_ror_refuse_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=c2_dpp_row_ror_refuse_kernel \
; RUN:   2>&1 \
; RUN:   | %FileCheck %s

; The refusal diagnostic MUST name:
;
;   1. The failing kernel, so `grep function '` pinpoints it in a
;      batch-raise run.
;   2. The specific unsupported ctrl, so the extension path is
;      obvious (add the case to `buildDppLaneMap` + widen
;      `isDppCtrlRewritable`).
;   3. The reference to wave-size-translation.md §5.3, so the next
;      session can read the rewrite invariant without digging
;      through the rewrite pass's source.
;
; CHECK-DAG: function 'c2_dpp_row_ror_refuse_kernel'
; CHECK-DAG: unsupported row_ror:1
; CHECK-DAG: wave-size-translation.md

; And the supported-family list MUST appear so a reviewer seeing a
; new refusal knows the current rewrite scope without cross-
; referencing source.
; CHECK-DAG: quad_perm, row_shl:N and row_shr:N
