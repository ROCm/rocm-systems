; RUN: %raise_cli %writelane_uniform_noop_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --enable-writelane-rewrite \
; RUN:     --emit-ir=writelane_uniform_noop_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=REWRITE
;
; RUN: %raise_cli %writelane_uniform_noop_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=writelane_uniform_noop_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=UNCHANGED
;
; Uniform-operand `v_writelane_b32` symmetry contract.
;
; The original pass kept uniform writelane sites as the native
; intrinsic on codegen-quality grounds ("convert only when divergent").
; That rule turned out to be silently UNSOUND when paired with a
; rewritten readlane on the same VGPR: the native `v_writelane_b32`
; writes hardware lane N only, while a `ds_bpermute`-rewritten
; sibling readlane reads BOTH lane N and lane N + W_s, so the upper
; source-wave replica observes undef and downstream address math
; faults (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION; see the
; Gfx1250Gpu.Matmul128x128* triage in tests/xfail.cmake and
; hotswap/docs/wave-size-translation.md §5.6.3 "symmetry contract").
;
; Under the symmetry fix, every writelane site is rewritten under
; cross-widening regardless of operand divergence. Uniform operands
; still produce the same value at lanes N and N+W_s (the `select` is
; a no-op on uniform `val`), so the rewrite is correctness-preserving.
; This fixture pins the post-fix invariant.
;
; REWRITE path (--enable-writelane-rewrite on):
;   * The uniform `v_writelane_b32` is rewritten to the
;     `select ((lane_id & (W_s-1)) == lane_idx), val, old` shape
;     just like the divergent-operand fixture
;     (lit_tests/writelane_divergent_rewrite).
;   * The rewriter's `cwd_lane_id` mbcnt helper is materialised
;     exactly once at function entry because at least one site now
;     needs it.
;   * The original `@llvm.amdgcn.writelane` call is gone.
;
; UNCHANGED path (flag off): the rewrite pass never runs, the
; intrinsic survives verbatim, and the rewriter's `cwd_*` values are
; absent. This is the commit-1 "no behaviour change when flag is off"
; regression guard — unchanged from the pre-symmetry-fix contract.

; REWRITE-LABEL: define amdgpu_kernel void @writelane_uniform_noop_kernel(
; REWRITE: %cwd_lane_id_lo = call i32 @llvm.amdgcn.mbcnt.lo
; REWRITE: %cwd_lane_id = call i32 @llvm.amdgcn.mbcnt.hi
; REWRITE: %cwd_wl_mask = icmp eq
; REWRITE: %cwd_writelane_rewritten = select i1
; REWRITE-NOT: call i32 @llvm.amdgcn.writelane

; UNCHANGED-LABEL: define amdgpu_kernel void @writelane_uniform_noop_kernel(
; UNCHANGED: call i32 @llvm.amdgcn.writelane
; UNCHANGED-NOT: cwd_lane_id_lo
; UNCHANGED-NOT: cwd_writelane_rewritten
