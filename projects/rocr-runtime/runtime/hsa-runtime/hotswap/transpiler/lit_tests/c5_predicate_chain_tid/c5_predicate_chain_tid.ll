; RUN: %not %raise_cli %c5_predicate_chain_tid_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --disable-wave-native \
; RUN:     --emit-ir=c5_predicate_chain_tid_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; `--disable-wave-native` forces `ModuloReplicationProjection` — the
; narrow-O1 classifier's refusal rationale is MODREP-scoped (the
; replica-1-vs-source-wave-0 divergence it catches is a MODREP
; artefact), so the classifier's `waveNative` gate short-circuits
; refusal under the post-graduation WaveNative default. This
; fixture pins the refusal on MODREP specifically. See
; c5_predicate_chain_classifier.hpp's `waveNative` parameter
; docstring and hotswap/docs/modrep-predicate-chain.md §6
; "Picked: WaveNative as default" for the graduation rationale.
;
; Regression fence for the Class-5 "wave-size-sensitive predicate chain"
; narrow-O1 classifier
; (hotswap/docs/modrep-predicate-chain.md §5, Phase-2 narrowing per
;  §9.6 of that doc;
;  transpiler/c5_predicate_chain_classifier.{hpp,cpp}).
;
; Under cross-widening (gfx1250 → gfx942 / gfx950), a wave32 source
; kernel whose `@llvm.amdgcn.workitem.id.x()` flows into an `icmp`
; against a compile-time constant `K` with `0 < K <= W_s - 1 = 31`,
; and whose chain to that icmp does NOT pass through an `and v, K'`
; with `K' <= W_s - 1`, partitions lanes by position within a single
; source wave. Modulo-replication's target-replica-1 lanes have
; architectural `tid >= W_s` so the predicate evaluates differently
; for them than for source wave 0's lane L — a miscompile the
; classifier has no safe rewrite for today (§5 O2 is deferred per
; §9.6). The only correct outcome is the (c) refusal branch of
; wave-size-translation.md §7's 3-outcome decision procedure.
;
; Paired with lit_tests/c5_predicate_chain_masked/ which pins the
; non-refusal path (chain AND-masked by 31 before reaching the icmp).
; The pair is the principled soundness-not-completeness regression
; fence for the classifier: REFUSE on unmasked + compile-time-K,
; OK on masked + same constant.
;
; Non-refusal contract for passing Triton baselines
; (`vecadd_f16`, `rope_fp32`, `canary_dpp_compound_add_fp32`) is
; enforced by `compare_correctness`'s end-to-end MATCH row rather
; than lit; see the §7 Test-surface table in
; `hotswap/docs/modrep-predicate-chain.md`. The narrowing rule ensures
; the classifier does not fire on those recipes — they use dynamic
; kernargs (`tid < %arg_N`), not compile-time constants, as the
; icmp's other operand.

; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-predicate-chain
; STDERR-SAME: workitem.id.x-predicate-chain-classifier

; Per-site trace names the ObstructionKind in human-readable form
; and includes the Class-5 cross-reference. The narrow-O1 classifier's
; refusal detail also carries the offending `icmp` predicate name and
; the failing constant so a reader of the diagnostic knows the
; triggering instruction's operand, not just the class.
; STDERR: icmp ult
; STDERR-SAME: compile-time constant 16
; STDERR-SAME: W_s-1=31
; STDERR: outcome: (c) refuse
; STDERR-SAME: WorkitemIdPredicateChain
; STDERR-SAME: Class 5

; raise_cli's outer-tool failure line names the kernel in
; kerneldex-style so coverage tooling can bucket on it.
; STDERR: raise_cli: kernel 'c5_predicate_chain_tid_kernel' failed to raise:
; STDERR-SAME: cross-wave-predicate-chain
