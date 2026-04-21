; RUN: %not %raise_cli %c1_wave_id_lift_scalarized_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c1_wave_id_lift_scalarized_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; hotswap/docs/wave-size-translation.md §6 Class 1, refuse row for
; the "canonical wave_id BFE lift scalarised through a cross-lane
; primitive under WMMA" pattern. Companion to
; lit_tests/c1_ttmp_wave_id_lift (rescue row) — together they pin the
; two terminal outcomes for every downstream consumer of the
; `s_bfe_u32 sDST, ttmp8, 0x50019` wave_id read under cross-widening
; (gfx1250 → gfx942 / gfx950).
;
; Detection logic lives in wave_size_obstruction.cpp's
; buildObstructionReport, which tracks the three conditions
; (`canonicalWaveIdBfeSites`, `crossLaneScalarSites`, `haveWMMA`) in
; separate vectors and emits one ObstructionKind::WaveIdLiftScalarized
; site per v_writelane / v_readlane once all three are non-empty. The
; mapping in selectFailureFromReport routes the site to
; RaiseFailure::crossWaveLaneIdLeak.
;
; We assert the three stable anchors the classifier + raise_cli
; surface at the refusal boundary:
;
;   (a) the pre-translation abort line carries the cross-wave-lane-id
;       leak diagnostic, names the offending mnemonic
;       (v_writelane_b32), and references the wave-size-translation.md
;       §7 unrewritable table;
;   (b) at least one per-site trace names the ObstructionKind in
;       human-readable form AND includes the Class 1 cross-reference
;       AND the WMMA co-occurrence rationale ("v_writelane/v_readlane
;       + WMMA"); and
;   (c) the raise_cli wrapper's kerneldex-style failure line pins the
;       kernel name and mnemonic so coverage tooling can bucket on it.
;
; Matching is substring-based throughout so the test stays resilient
; to future wording tightenings in the detail text.

; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-lane-id-leak
; STDERR-SAME: v_writelane_b32
; STDERR-SAME: wave-size-translation.md

; STDERR: WaveIdLiftScalarized
; STDERR-SAME: Class 1
; STDERR-SAME: v_writelane/v_readlane
; STDERR-SAME: WMMA

; The outcome tag in the post-site summary keys on the (c) refuse
; branch of §7's 3-outcome decision procedure, same contract as the
; sibling c1_lane_id_leak fixture asserts for its MbcntHiLaneIdLeak
; path.
; STDERR: outcome: (c) refuse

; raise_cli's outer-tool failure line names the kernel and mnemonic
; in kerneldex-style so coverage tooling can bucket on it.
; STDERR: raise_cli: kernel 'c1_wave_id_lift_scalarized_kernel' failed to raise:
; STDERR-SAME: v_writelane_b32
