; RUN: %not %raise_cli %c1_lane_id_leak_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c1_lane_id_leak_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; SPE_DESIGN.md §3 Class 1: v_mbcnt_hi_u32_b32 on a wave32 source
; binary leaks the absolute target-hardware lane position into an
; observable value whenever the raised IR runs on a wider target
; wave. No rewrite in §4's rewrite table recovers the original
; wave32 semantics (the source never named "lane_id mod W_s" as a
; distinct quantity), so the only correct outcome is the (c) refusal
; branch of the 3-outcome decision procedure.
;
; The classifier must flag the v_mbcnt_hi site at raise time and
; abort with the stable diagnostic substrings asserted below.
; Matching on substrings rather than the full sentence keeps the
; test resilient to future rewordings.
;
; We also assert `%not` inverts the exit code, so the test fails
; loudly if the abort silently degrades into a warning.

; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-lane-id-leak
; STDERR-SAME: v_mbcnt_hi

; The per-site trace emitted after the abort line names the
; ObstructionKind in human-readable form and includes the
; SPE_DESIGN.md §3 cross-reference parenthetically. We key on
; stable substrings only.
; STDERR: MbcntHiLaneIdLeak
; STDERR-SAME: Class 1
; STDERR: outcome: (c) refuse

; The raise_cli wrapper reports the failure once more in its
; kerneldex-style format so coverage tooling bucket on it.
; STDERR: raise_cli: kernel 'c1_lane_id_leak_kernel' failed to raise:
; STDERR-SAME: v_mbcnt_hi
