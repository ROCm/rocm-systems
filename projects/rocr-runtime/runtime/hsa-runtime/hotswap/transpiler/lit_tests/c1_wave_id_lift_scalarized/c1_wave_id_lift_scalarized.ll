; RUN: %not %raise_cli %c1_wave_id_lift_scalarized_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --disable-writelane-rewrite \
; RUN:     --emit-ir=c1_wave_id_lift_scalarized_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE
;
; RUN: %raise_cli %c1_wave_id_lift_scalarized_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --enable-writelane-rewrite \
; RUN:     --emit-ir=c1_wave_id_lift_scalarized_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=REWRITTEN
;
; hotswap/docs/wave-size-translation.md §6 Class 1 table — the two
; terminal outcomes for the "canonical wave_id BFE lift scalarised
; through a cross-lane primitive under WMMA" pattern. Companion to
; lit_tests/c1_ttmp_wave_id_lift (rescue row) — together with this
; fixture's REFUSE / REWRITTEN siblings they pin all three outcomes
; for every downstream consumer of the `s_bfe_u32 sDST, ttmp8,
; 0x50019` wave_id read under cross-widening (gfx1250 → gfx942 /
; gfx950):
;
;   (a) rescue  — no v_writelane / v_readlane under WMMA: pinned by
;                 the c1_ttmp_wave_id_lift fixture;
;   (b) refuse  — all three co-occur AND the --enable-writelane-
;                 rewrite flag is OFF: the REFUSE RUN line pins the
;                 pre-Phase-6.5 refusal contract verbatim so the
;                 flag-off path stays honest even after the rewrite
;                 pass lands;
;   (c) rewrite — all three co-occur AND the --enable-writelane-
;                 rewrite flag is ON: the REWRITTEN RUN line pins the
;                 Phase 6.5 rewrite-to-select IR shape that
;                 `rewriteCrossLaneDivergent` in
;                 `rewrite_cross_lane_divergent.cpp` emits once the
;                 classifier in
;                 `wave_size_obstruction.cpp::buildObstructionReport`
;                 tags the WaveIdLiftScalarized site as
;                 RewriteId::PostRaiseCrossLaneRewrite (an
;                 *implemented* rewrite) instead of refusing outright.
;
; Detection logic lives in wave_size_obstruction.cpp's
; buildObstructionReport, which tracks the three conditions
; (`canonicalWaveIdBfeSites`, `crossLaneScalarSites`, `haveWMMA`) in
; separate vectors and emits one ObstructionKind::WaveIdLiftScalarized
; site per v_writelane / v_readlane once all three are non-empty.
;
; REFUSE path (--disable-writelane-rewrite — default is now ON, so
; the REFUSE sibling explicitly opts out of the rewrite):
;   We assert the three stable anchors the classifier + raise_cli
;   surface at the refusal boundary:
;
;     (a) the pre-translation abort line carries the cross-wave-lane-
;         id leak diagnostic, names the offending mnemonic
;         (v_writelane_b32), and references the wave-size-
;         translation.md §7 unrewritable table;
;     (b) at least one per-site trace names the ObstructionKind in
;         human-readable form AND includes the Class 1 cross-
;         reference AND the WMMA co-occurrence rationale
;         ("v_writelane/v_readlane + WMMA"); and
;     (c) the raise_cli wrapper's kerneldex-style failure line pins
;         the kernel name and mnemonic so coverage tooling can
;         bucket on it.
;
;   Matching is substring-based throughout so the test stays
;   resilient to future wording tightenings in the detail text.

; REFUSE: transpiler: pre-translation abort:
; REFUSE-SAME: cross-wave-lane-id-leak
; REFUSE-SAME: v_writelane_b32
; REFUSE-SAME: wave-size-translation.md

; REFUSE: WaveIdLiftScalarized
; REFUSE-SAME: Class 1
; REFUSE-SAME: v_writelane/v_readlane
; REFUSE-SAME: WMMA

; The outcome tag in the post-site summary keys on the (c) refuse
; branch of §7's 3-outcome decision procedure, same contract as the
; sibling c1_lane_id_leak fixture asserts for its MbcntHiLaneIdLeak
; path.
; REFUSE: outcome: (c) refuse

; raise_cli's outer-tool failure line names the kernel and mnemonic
; in kerneldex-style so coverage tooling can bucket on it.
; REFUSE: raise_cli: kernel 'c1_wave_id_lift_scalarized_kernel' failed to raise:
; REFUSE-SAME: v_writelane_b32

; REWRITTEN path (--enable-writelane-rewrite ON):
;   The classifier in buildObstructionReport re-tags the site as
;   RewriteId::PostRaiseCrossLaneRewrite and `rewriteImplemented =
;   true`, which lets it pass the Phase 1.4.5 pre-translation refusal
;   gate. In Phase 6.5 (post-mem2reg) the rewrite pass then replaces
;   the scalar-source v_writelane with a per-lane `select` keyed on
;   `(lane_id & (W_s-1)) == lane_idx`, materialising the canonical
;   `mbcnt_lo(-1, 0); mbcnt_hi(-1, prev)` target-wave lane_id in the
;   function's entry block with the rewriter-specific `cwd_` prefix.
;
;   We assert the four stable signatures unique to the rewrite
;   (matching the writelane_divergent_rewrite fixture's IR-shape
;   contract exactly — same mbcnt pair, same predicate shape, same
;   select name — so any future refactor that generalises the three
;   fixtures shares one ground truth):
;     (a) raise succeeds (no `transpiler: pre-translation abort`
;         line reaches stdout — it would leak through to stdout only
;         on a pipeline-internal failure, so its absence together
;         with the kernel body below is sufficient);
;     (b) the `cwd_lane_id_lo` + `cwd_lane_id` two-step mbcnt pair;
;     (c) the `cwd_wl_mask` lane-equality predicate;
;     (d) the `cwd_writelane_rewritten` select replaces the original
;         `@llvm.amdgcn.writelane` call.

; REWRITTEN-LABEL: define amdgpu_kernel void @c1_wave_id_lift_scalarized_kernel(
; REWRITTEN: %cwd_lane_id_lo = call i32 @llvm.amdgcn.mbcnt.lo
; REWRITTEN: %cwd_lane_id = call i32 @llvm.amdgcn.mbcnt.hi
; REWRITTEN: %cwd_wl_mask = icmp eq
; REWRITTEN: %cwd_writelane_rewritten = select i1
; REWRITTEN-NOT: call i32 @llvm.amdgcn.writelane
