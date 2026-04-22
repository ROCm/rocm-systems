; RUN: %not %raise_cli %c5_predicate_chain_phantom_lane_co \
; RUN:     --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c5_predicate_chain_phantom_lane_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR_WN
;
; Paired MODREP invocation: `--disable-wave-native` forces the
; MODREP refusal arm, which fires on ANY C5 site regardless of the
; WG size. This pins that the phantom-lane bit is a WaveNative-
; specific annotation — under MODREP the diagnostic does NOT carry
; the "phantom-lane regime" prefix because the refusal reason is
; the unconditional MODREP replica-1 EXEC-share trap, not the
; WaveNative phantom-lane collapse.
; RUN: %not %raise_cli %c5_predicate_chain_phantom_lane_co \
; RUN:     --isa=gfx1250 --target-isa=gfx942 --disable-wave-native \
; RUN:     --emit-ir=c5_predicate_chain_phantom_lane_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR_MR
;
; Regression fence for the phantom-lane sub-case of the Class-5
; classifier (`c5_predicate_chain_classifier.hpp` file-header
; docstring + the `phantomLaneGuaranteed` predicate in
; `classifyPredicateChain`).
;
; Under the post-graduation WaveNative default, the classifier
; suppresses its MODREP-specific refusal on the premise that
; `init_whole_wave` + per-source-lane-modeled EXEC give each target
; lane a 1:1 source-lane mapping — so `tid < K` evaluates the same
; way the source HW would have. That premise HOLDS for the common
; case `max_flat_workgroup_size >= targetWaveSize` (where every
; target wavefront lane corresponds to a source-kernel thread), but
; it COLLAPSES when the kernel's launch bounds force a WG smaller
; than the target wavefront width: hardware activates the spare
; lanes via `init_whole_wave`, their architectural `tid` is their
; hardware lane index (32..63 on gfx942 wave64), and the source
; kernel never modelled computation at those lane positions. The
; compiler's `@llvm.amdgcn.workitem.id.x()` lift returns those
; out-of-range `tid` values verbatim, the compile-time-K predicate
; partitions lanes into "modelled" vs "phantom", and any convergent
; cross-lane op (`ds_bpermute`, `ds_swizzle`, `permlane*`) reading
; from a phantom lane observes unmodelled state — the silent
; miscompile `canary_bitmatrix_composite` empirically produces
; (see the canary's docstring for the runtime evidence).
;
; The classifier's phantom-lane arm closes this gap by making the
; refusal fire under WaveNative iff the HSACO's
; `max_flat_workgroup_size` is statically below the target
; wavefront width. This fixture forces that state via
; `__launch_bounds__(32)` on a gfx1250 (wave32) source compiled
; against a gfx942 (wave64) target.
;
; Paired with `lit_tests/c5_predicate_chain_tid/` (same icmp shape,
; no launch_bounds → max_flat_workgroup_size = 1024 > 64, no
; phantom lanes, WaveNative still suppresses). The pair pins BOTH
; WaveNative arms end-to-end.

; STDERR_WN: transpiler: pre-translation abort:
; STDERR_WN-SAME: cross-wave-predicate-chain
; STDERR_WN-SAME: workitem.id.x-predicate-chain-classifier

; Phantom-lane prefix names the distinguishing evidence — the HSACO
; WG bound and the target wavefront width.
; STDERR_WN: phantom-lane regime:
; STDERR_WN-SAME: max_flat_workgroup_size
; STDERR_WN-SAME: 32
; STDERR_WN-SAME: target wavefront width
; STDERR_WN-SAME: 64

; The base C5-shape reason is still present (the phantom-lane
; detail is a prefix on the underlying refusal reason, not a
; replacement) so operators can follow the chain from the
; WaveNative-specific cause back to the C5 class.
; STDERR_WN: icmp ult
; STDERR_WN-SAME: compile-time constant 16
; STDERR_WN-SAME: W_s-1=31

; Outcome line carries the phantom-lane sub-case annotation so
; triage tools can bucket separately from the baseline MODREP
; refusal.
; STDERR_WN: outcome: (c) refuse
; STDERR_WN-SAME: WorkitemIdPredicateChain
; STDERR_WN-SAME: Class 5 phantom-lane sub-case

; Outer-tool failure line.
; STDERR_WN: raise_cli: kernel 'c5_predicate_chain_phantom_lane_kernel' failed to raise:
; STDERR_WN-SAME: cross-wave-predicate-chain

; MODREP assertions: same refusal CLASS fires, but WITHOUT the
; phantom-lane prefix — MODREP refuses every C5 site unconditionally
; (the `waveNative=false` arm doesn't consult `maxFlatWorkgroupSize`).
; Pins that the phantom-lane annotation is WaveNative-only.
; STDERR_MR: transpiler: pre-translation abort:
; STDERR_MR-SAME: cross-wave-predicate-chain
; STDERR_MR: icmp ult
; STDERR_MR-SAME: compile-time constant 16
; STDERR_MR-SAME: W_s-1=31
; STDERR_MR: outcome: (c) refuse
; STDERR_MR-SAME: WorkitemIdPredicateChain
; STDERR_MR-SAME: Class 5

; MODREP outcome line MUST NOT carry the phantom-lane sub-case
; annotation — if this CHECK-NOT fires it means the MODREP arm
; started consulting `maxFlatWorkgroupSize` (which would blur the
; two refusal reasons and break the WaveNative-specific
; attribution contract).
; STDERR_MR-NOT: phantom-lane sub-case
; STDERR_MR-NOT: phantom-lane regime
