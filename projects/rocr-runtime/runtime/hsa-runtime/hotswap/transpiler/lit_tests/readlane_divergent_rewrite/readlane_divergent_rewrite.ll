; RUN: %raise_cli %readlane_divergent_rewrite_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --enable-writelane-rewrite \
; RUN:     --emit-ir=readlane_divergent_rewrite_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=REWRITE
;
; RUN: %raise_cli %readlane_divergent_rewrite_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=readlane_divergent_rewrite_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=UNCHANGED
;
; Divergence-triggered `v_readlane_b32` rewrite contract.
;
; REWRITE path (--enable-writelane-rewrite on):
;   * `rewriteCrossLaneDivergent` replaces the
;     `@llvm.amdgcn.readlane` call with a `ds_bpermute` whose selector
;     is `((lane_id & ~(W_s-1)) | lane_idx) << 2`. The lane_id helper
;     is the canonical `mbcnt_lo(-1, 0); mbcnt_hi(-1, prev)` pair built
;     once at the entry block.
;   * Asserts the four stable signatures, all keyed on
;     rewriter-specific `cwd_*` variable names (disambiguates from the
;     raiser's SPE-wrapper lane_id construction around writelane /
;     readlane primitive sites):
;       (a) the `cwd_lane_id_lo` + `cwd_lane_id` two-step mbcnt pair,
;       (b) the `cwd_rl_selector` byte-offset (shl by 2),
;       (c) the `cwd_readlane_rewritten` ds_bpermute call,
;       (d) the absence of any `cwd_writelane_rewritten` sibling
;           (write half must NOT fire on this fixture).
;
; UNCHANGED path (flag off): `@llvm.amdgcn.readlane` survives; the
; rewriter-emitted `cwd_*` values are absent. Commit 1's default-off
; invariant. The raiser's own `ds_bpermute` lowerings for other
; opcodes (e.g. the dedicated DS_BPERMUTE handler exercised by
; `ds_bpermute_b32`) remain unrelated to this contract; we key the
; negative assertion on the rewriter's `cwd_readlane_rewritten` name
; rather than on `ds_bpermute` to keep the fixture robust against
; corpus shifts.

; REWRITE-LABEL: define amdgpu_kernel void @readlane_divergent_rewrite_kernel(
; REWRITE: %cwd_lane_id_lo = call i32 @llvm.amdgcn.mbcnt.lo
; REWRITE: %cwd_lane_id = call i32 @llvm.amdgcn.mbcnt.hi
; REWRITE: %cwd_rl_selector = shl
; REWRITE: %cwd_readlane_rewritten = call i32 @llvm.amdgcn.ds.bpermute
; REWRITE-NOT: cwd_writelane_rewritten

; UNCHANGED-LABEL: define amdgpu_kernel void @readlane_divergent_rewrite_kernel(
; UNCHANGED: call i32 @llvm.amdgcn.readlane
; UNCHANGED-NOT: cwd_lane_id_lo
; UNCHANGED-NOT: cwd_readlane_rewritten
