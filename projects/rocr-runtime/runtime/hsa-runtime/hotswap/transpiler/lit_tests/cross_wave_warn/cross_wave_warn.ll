; RUN: %raise_cli %cross_wave_warn_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=cross_wave_writer_kernel 2>&1 >/dev/null \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; Cross-wave translation (wave32 source → wave64 target) of a kernel
; that manipulates EXEC must emit the modulo-replication warning from
; raiser.cpp's Phase-1.4 gate. Today the policy is warn-only, so the
; raise still succeeds; downstream CI is free to escalate by grepping
; stderr.
;
; We assert the diagnostic contains the stable substrings that make
; the warning actionable:
;
;   * `cross-wave translation` — identifies the class of issue.
;   * `modulo-replication`     — names the policy the warning is about.
;   * the concrete wave sizes on each side, so the operator can see
;     the cross-family delta at a glance.
;   * the offending instruction mnemonic (v_cmpx_lt_u32_e64) and the
;     offset — both useful for triage.
;
; The warning is NOT expected to abort; a separate companion lit
; test (abort_gate.ll) guards the different — allow-list — abort
; path. A cross-wave lit test that asserts an ABORT would be the
; right thing to add if we ever tighten the policy in the future.

; STDERR: transpiler: WARNING: cross-wave translation
; STDERR-SAME: EXEC-manipulating kernel
; STDERR-SAME: modulo-replication

; Wave sizes reported explicitly so the operator sees the cross-family
; delta. Source is gfx1250 (wave32), target is gfx942 (wave64).
; STDERR:      source ISA wave size: 32 (gfx1250)
; STDERR-NEXT: target ISA wave size: 64 (gfx942)

; First EXEC-writer in the fixture is our `v_cmpx_lt_u32_e64`.
; STDERR: first EXEC-writer: v_cmpx_lt_u32_e64
