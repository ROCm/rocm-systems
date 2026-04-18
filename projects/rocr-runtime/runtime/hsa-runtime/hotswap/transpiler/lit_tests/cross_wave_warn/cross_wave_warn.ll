; RUN: %raise_cli %cross_wave_warn_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=cross_wave_writer_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Cross-wave translation (wave32 source → wave64 target) of a kernel
; whose only EXEC writer is lane-position-INDEPENDENT — here, a
; `v_cmpx_lt_u32_e64 threadIdx.x, 8` bounds check where the LHS is
; workitem_id_x (uniform-across-replicas, not derived from mbcnt).
; The Phase 1.4.5 classifier in `wave_size_obstruction.cpp` should
; classify this as outcome (a) oblivious: the v_cmpx is an EXEC
; writer, but with no v_mbcnt_* in the kernel the syntactic
; co-occurrence heuristic does not flag it, and no refusal fires.
;
; This is the regression-fence counterpart to the c4_lane_dep_cmpx
; lit test, which uses exactly the same v_cmpx shape but adds an
; `v_mbcnt_lo_u32_b32` to the body so the co-occurrence heuristic
; catches it. The pair of fixtures pins both directions of the
; classifier's decision.
;
; Historical note. Before Phase 1.4.5 landed, the raiser's warn-
; only Phase 1.4 gate printed a stderr banner (`transpiler:
; WARNING: cross-wave translation of an EXEC-manipulating kernel
; relies on modulo-replication, which is not provably correct in
; general...`). That banner is now routed through LLVM_DEBUG under
; DEBUG_TYPE="wave-projection" — it surfaces only with
; `-debug-only=wave-projection`. The classifier's per-site trace,
; emitted via the same DEBUG_TYPE, subsumes the banner's content.
; See `wave_projection.cpp:emitCrossWaveWarning` for the shim and
; `wave_size_obstruction.cpp:renderObstructionTrace` for the new
; format.
;
; We assert structurally: the raise succeeds and the emitted IR
; contains the kernel body — not a particular SSA name, which is
; not stable across LLVM versions.

; CHECK-LABEL: define amdgpu_kernel void @cross_wave_writer_kernel(
