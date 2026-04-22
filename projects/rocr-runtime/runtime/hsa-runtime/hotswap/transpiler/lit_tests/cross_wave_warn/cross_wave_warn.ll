; RUN: %raise_cli %cross_wave_warn_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=cross_wave_writer_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Cross-wave translation (wave32 source → wave64 target) of a kernel
; whose only EXEC writer is lane-position-INDEPENDENT — here, a
; `v_cmpx_lt_u32_e64 threadIdx.x, 64` bounds check. The constant 64
; is structurally ≥ W_s so the Class-5 predicate-chain classifier
; in `c5_predicate_chain_classifier.cpp` accepts it as a bounds
; check rather than a lane-position gate. The Phase 1.4.5 MC-level
; classifier in `wave_size_obstruction.cpp` similarly classifies it
; as outcome (a) oblivious: the v_cmpx is an EXEC writer, but with
; no v_mbcnt_* in the kernel the syntactic co-occurrence heuristic
; does not flag it, and no refusal fires.
;
; Pre-C5-classifier-landing this fixture used K=8, exercising a
; genuine lane-position-scoped predicate under cross-wave — the C5
; classifier now refuses that exact shape (lit_tests/c5_predicate_chain_tid
; is the principled positive fixture for the refusal). The fixture
; was updated to K=64 rather than xfail'd so the cross-wave warning
; path still has an IR-generating producer.
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
