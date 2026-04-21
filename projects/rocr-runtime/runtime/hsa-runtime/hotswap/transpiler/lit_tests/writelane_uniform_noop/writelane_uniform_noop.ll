; RUN: %raise_cli %writelane_uniform_noop_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --enable-writelane-rewrite \
; RUN:     --emit-ir=writelane_uniform_noop_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=NOOP
;
; Uniform-operand `v_writelane_b32` no-op contract.
;
; With `--enable-writelane-rewrite` on AND both operands uniform
; (val = `workgroup.id.x`, old = constant 0 via `+v` tied constraint):
;   * `rewriteCrossLaneDivergent` must NOT convert the call into a
;     `select`-based per-lane form (would regress codegen without a
;     correctness benefit).
;   * The `amdgcn.writelane` intrinsic survives verbatim.
;   * The rewrite pass must ALSO not materialise its `cwd_lane_id*`
;     mbcnt helper, because the helper is lazy (see
;     `rewriteCrossLaneDivergent`'s `laneIdCached` closure: no
;     divergent site -> no helper).
;
; Negative assertions key on the rewriter-specific `cwd_` prefix
; rather than on generic `mbcnt`, because the raiser's own
; SPE-wrapper predication around every writelane site already emits
; a `lane_lo` / `lane_id` mbcnt pair for EXEC-mask gating. That
; machinery predates this rewrite pass and is orthogonal to its
; contract; the only stable "rewrite did not fire" signal is the
; absence of the rewriter's `cwd_*` named values.

; NOOP-LABEL: define amdgpu_kernel void @writelane_uniform_noop_kernel(
; NOOP: call i32 @llvm.amdgcn.writelane
; NOOP-NOT: cwd_lane_id_lo
; NOOP-NOT: cwd_writelane_rewritten
; NOOP-NOT: cwd_readlane_rewritten
