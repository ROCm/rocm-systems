; RUN: %raise_cli %c5_predicate_chain_masked_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c5_predicate_chain_masked_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Non-refusal sibling for the Class-5 "wave-size-sensitive predicate
; chain" narrow-O1 classifier
; (hotswap/docs/modrep-predicate-chain.md §5, transpiler/
;  c5_predicate_chain_classifier.{hpp,cpp}).
;
; Paired with `lit_tests/c5_predicate_chain_tid/` which pins the
; REFUSAL path on the SAME compile-time K=15 predicate. The
; distinguishing feature of this fixture is an `and %tid, 31` on the
; chain BEFORE the icmp — the classifier's
; `isSourceWaveMaskAnd` recognises the mask as collapsing replica-1
; lanes onto `[0, W_s)` and stops walking, so the icmp-against-K=15
; never triggers a refusal.
;
; Together, the two fixtures are the principled regression fence for
; the classifier's soundness-not-completeness contract (see
; `hotswap/docs/wave-size-translation.md` §7):
;   - the tid-only fixture (refuses) pins false-positives stay
;     refused — a future iteration must not sneak an unmasked scan-
;     stage predicate past the gate;
;   - the masked fixture (this file, OK) pins the masked case does
;     not over-refuse — a future iteration must not strip the
;     `isSourceWaveMaskAnd` recognition and start refusing the SPE
;     prelude's own `lane_id mod execBits`, the §5.6.2 `wave_id`
;     lift's mask, or any future `tid AND (W_s - 1)` rewrite that
;     §5 O2 may eventually land.
;
; We assert:
;   1) Raise succeeds (no `%not`; the RUN line's exit-zero expectation
;      is enough to fail the test if the classifier regresses into
;      refusing here).
;   2) The kernel body is present (IR-LABEL anchors on it).
;   3) The SOURCE-WAVE MASK `and i32 <...>, 31` appears on the chain
;      that the classifier used to decide the icmp was safe. This
;      assertion is stable because the fixture's inline-asm
;      `v_and_b32_e64 v, t, 31` lifts directly to `and i32 X, 31`;
;      any future IR-printer change that renames the SSA but
;      preserves the bitwise shape still matches.

; IR-LABEL: define amdgpu_kernel void @c5_predicate_chain_masked_kernel(
; IR: call i32 @llvm.amdgcn.workitem.id.x()
; IR: and i32 {{.*}}, 31
