; RUN: %not %raise_cli %c4_lane_dep_cmpx_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c4_lane_dep_cmpx_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; SPE_DESIGN.md §3 Class 4 "lane-position-dependent EXEC writes".
; The v_cmpx's LHS flows from v_mbcnt_lo, which makes the compare
; semantically "enable lanes below an absolute-lane-id threshold".
; This pattern has no rewrite in §4's table — modulo-replication
; would enable target lanes {0..15, 32..47} under the source's
; wave32 semantics, but the raised IR would enable only target
; lanes 0..15 (because v_mbcnt_lo runs on target hardware and
; produces values in the range determined by target EXEC). Neither
; outcome matches the source's wave32 intent; the correct behaviour
; is to refuse.
;
; This is the structural counter-example to lit_tests/cross_wave_warn,
; whose v_cmpx uses a lane-position-INDEPENDENT bounds expression
; (an 8-vs-lane-value compare with a constant bound, where the
; lane_id is not routed through a mbcnt-derived path). That warn-
; only fixture remains valid under the new gate because its
; compare is provably lane-position-independent.
;
; DATAFLOW FOLLOW-UP. The current syntactic classifier flags this
; kernel by matching on v_cmpx co-occurring with v_mbcnt_lo /
; v_mbcnt_hi in the same kernel. The principled check — proving
; the v_cmpx's operand chain is rooted in an absolute-lane-id
; expression — requires LLVM Uniformity Analysis on the raised IR
; and is tracked as wave_size_obstruction.cpp's
; TODO(dataflow-upgrade). Until then, false positives (syntactically
; co-located v_cmpx and mbcnt that don't actually flow into each
; other) fail closed, which is the correct direction.

; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-lane-predicated-exec
; STDERR-SAME: v_cmpx

; STDERR: CmpxFromLaneId
; STDERR-SAME: Class 4
; STDERR: outcome: (c) refuse

; STDERR: raise_cli: kernel 'c4_lane_dep_cmpx_kernel' failed to raise:
; STDERR-SAME: v_cmpx
