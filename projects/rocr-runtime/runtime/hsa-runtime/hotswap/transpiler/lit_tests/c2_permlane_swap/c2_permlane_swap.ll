; RUN: %not %raise_cli %c2_permlane_swap_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c2_permlane_swap_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; SPE_DESIGN.md §3 Class 2 "wave-size-baked cross-lane ops". The
; v_permlane16_swap_b32 instruction has a clean rewrite on paper
; (CROSS_LANE_SURVEY.md P2/P3/P4 — lift to llvm.amdgcn.permlane16),
; but until that handler lands the classifier must refuse the kernel
; rather than emit the current same-lane-move fallback that
; CROSS_LANE_SURVEY.md documents as "❌ broken".
;
; Once the P2/P3/P4 handler lands this test should be updated to
; assert the kernel raises successfully and the permlane16
; intrinsic is present in the emitted IR (mirroring
; lit_tests/ds_bpermute_b32/ for P1's landed handler).
;
; MAINTENANCE CONTRACT. When you implement P2/P3/P4 in handle_valu:
;   1. Flip this test's RUN line from `%not %raise_cli` to
;      `%raise_cli`.
;   2. Replace the STDERR CHECK block with a CHECK block asserting
;      `call {{.*}}@llvm.amdgcn.permlane16` in the raised IR.
;   3. Update the obstruction table in wave_size_obstruction.cpp so
;      the V_PERMLANE16_SWAP_B32 SemOp is marked `rewriteImplemented
;      = true`.

; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-shuffle-rewrite-pending
; STDERR-SAME: v_permlane16_swap

; STDERR: LaneGroupShuffle
; STDERR-SAME: Class 2
; STDERR: rewrite: P4
; STDERR-SAME: pending
; STDERR: outcome: (c) refuse

; STDERR: raise_cli: kernel 'c2_permlane_swap_kernel' failed to raise:
; STDERR-SAME: v_permlane16_swap
