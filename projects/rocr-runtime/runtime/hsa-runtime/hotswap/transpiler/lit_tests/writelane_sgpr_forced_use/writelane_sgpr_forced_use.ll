; RUN: %not %raise_cli %writelane_sgpr_forced_use_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --enable-writelane-rewrite \
; RUN:     --emit-ir=writelane_sgpr_forced_use_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE
;
; RUN: %raise_cli %writelane_sgpr_forced_use_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --disable-writelane-rewrite \
; RUN:     --emit-ir=writelane_sgpr_forced_use_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=UNCHANGED
;
; Pins the forward use-chain classifier's refusal path in
; `rewrite_cross_lane_divergent.cpp::classifyForwardUseChain`. The
; fixture kernel (writelane_sgpr_forced_use.hip) chains a divergent
; `v_writelane_b32` into a `v_readfirstlane_b32`; the classifier must
; detect the readfirstlane as an SGPR-forced sink and refuse the
; whole function rather than emit a `ds_bpermute`-shaped rewrite
; that the backend would immediately re-collapse via readfirstlane.
;
; REFUSE path (--enable-writelane-rewrite on):
;   Asserts three stable anchors the raiser surfaces at the refusal
;   boundary:
;     (a) the post-raise abort line carries the
;         cross-wave-lane-id-leak diagnostic (the
;         `RaiseFailure::crossWaveRewriteOracleDisagreement` bucket);
;     (b) the classifier's detail names the offending consumer
;         (`readfirstlane`) so the next investigation knows where to
;         look — the exact string is the intrinsic name as rendered
;         by `Function::getName`, stable across LLVM versions;
;     (c) the raise_cli wrapper's kerneldex-style failure line pins
;         the kernel name and the refusal bucket, letting coverage
;         tooling bucket regressions without parsing the detail.
;
; UNCHANGED path (flag off):
;   The kernel has no WMMA, so the Phase 1.4.5
;   `WaveIdLiftScalarized` classifier does NOT refuse the lift (the
;   three-way co-occurrence requires WMMA). The raiser emits
;   `@llvm.amdgcn.writelane` + `@llvm.amdgcn.readfirstlane` verbatim
;   and exits successfully. This arm is the regression guard for the
;   "no behaviour change when the flag is off" contract — ensures
;   the classifier does not leak refusal into the flag-off path.

; REFUSE: transpiler: post-raise abort:
; REFUSE-SAME: cross-wave-lane-id-leak

; REFUSE: SGPR-forced consumer
; REFUSE-SAME: readfirstlane

; REFUSE: raise_cli: kernel 'writelane_sgpr_forced_use_kernel' failed to raise:
; REFUSE-SAME: cross-wave-lane-id-leak

; UNCHANGED-LABEL: define amdgpu_kernel void @writelane_sgpr_forced_use_kernel(
; UNCHANGED: call i32 @llvm.amdgcn.writelane
; UNCHANGED: call i32 @llvm.amdgcn.readfirstlane
; UNCHANGED-NOT: cwd_lane_id_lo
; UNCHANGED-NOT: cwd_writelane_rewritten
