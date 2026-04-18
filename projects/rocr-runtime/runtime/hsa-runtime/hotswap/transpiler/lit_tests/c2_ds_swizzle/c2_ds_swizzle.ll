; RUN: %not %raise_cli %c2_ds_swizzle_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c2_ds_swizzle_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; SPE_DESIGN.md §3 Class 2 (ds_swizzle sub-category). The classifier
; must refuse any kernel with ds_swizzle_b32 until
; CROSS_LANE_SURVEY.md P6 lands the intrinsic lift.
;
; MAINTENANCE. Flip protocol identical to c2_permlane_swap.ll once
; P6 lands. The test should then assert
; `call {{.*}}@llvm.amdgcn.ds.swizzle` in the raised IR.

; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-shuffle-rewrite-pending
; STDERR-SAME: ds_swizzle

; STDERR: DsSwizzle
; STDERR-SAME: Class 2
; STDERR: rewrite: P6
; STDERR-SAME: pending
; STDERR: outcome: (c) refuse

; STDERR: raise_cli: kernel 'c2_ds_swizzle_kernel' failed to raise:
; STDERR-SAME: ds_swizzle
