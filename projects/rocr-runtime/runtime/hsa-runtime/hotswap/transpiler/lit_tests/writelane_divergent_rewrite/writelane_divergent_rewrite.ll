; RUN: %raise_cli %writelane_divergent_rewrite_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --enable-writelane-rewrite \
; RUN:     --emit-ir=writelane_divergent_rewrite_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=REWRITE
;
; RUN: %raise_cli %writelane_divergent_rewrite_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=writelane_divergent_rewrite_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=UNCHANGED
;
; Divergence-triggered `v_writelane_b32` rewrite contract.
;
; REWRITE path (--enable-writelane-rewrite on):
;   * `rewriteCrossLaneDivergent` in
;     `rewrite_cross_lane_divergent.cpp` replaces the
;     `@llvm.amdgcn.writelane` call with
;       `select ((lane_id & (W_s-1)) == lane_idx), val_div, old`
;     preceded by the canonical `mbcnt_lo(-1, 0); mbcnt_hi(-1, prev)`
;     target-wave lane_id construction in the function's entry block.
;   * Asserts the four stable signatures:
;       (a) the `cwd_lane_id_lo` + `cwd_lane_id` two-step mbcnt pair
;           (rewriter-specific variable names — distinct from the
;           raiser's `lane_lo` / `lane_id` EXEC-mask-predication pair
;           around cross-lane primitive sites),
;       (b) the `cwd_wl_mask` lane-equality predicate,
;       (c) the `cwd_writelane_rewritten` select,
;       (d) the absence of any `cwd_readlane_rewritten` sibling
;           (read half must NOT fire on this fixture).
;
; UNCHANGED path (flag off -> commit 1 default-off invariant):
;   * `@llvm.amdgcn.writelane` survives verbatim; the rewrite-emitted
;     `cwd_*` values are absent (the raiser's own SPE-wrapper lane_id
;     construction for EXEC-mask predication still appears — that
;     machinery is ORTHOGONAL to the rewrite pass and predates it, so
;     we key the negative assertion on the rewriter-specific prefix
;     `cwd_` rather than on `mbcnt`). Regression guard for commit 1's
;     "no behaviour change when the flag is off" contract; once
;     commit 2 flips the classifier to accept rewrite-implemented paths
;     under the flag, this arm continues to prove the default-off
;     plumbing works.

; REWRITE-LABEL: define amdgpu_kernel void @writelane_divergent_rewrite_kernel(
; REWRITE: %cwd_lane_id_lo = call i32 @llvm.amdgcn.mbcnt.lo
; REWRITE: %cwd_lane_id = call i32 @llvm.amdgcn.mbcnt.hi
; REWRITE: %cwd_wl_mask = icmp eq
; REWRITE: %cwd_writelane_rewritten = select i1
; REWRITE-NOT: cwd_readlane_rewritten

; UNCHANGED-LABEL: define amdgpu_kernel void @writelane_divergent_rewrite_kernel(
; UNCHANGED: call i32 @llvm.amdgcn.writelane
; UNCHANGED-NOT: cwd_lane_id_lo
; UNCHANGED-NOT: cwd_writelane_rewritten
