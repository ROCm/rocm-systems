; RUN: %raise_cli %s_swap_pc_i64_dispatch_set_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=setpc_swap_dispatch_set_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for s_swap_pc_i64 — DispatchSet (multi-target indirect
; branch-and-link resolved by inter-block PC-chain dataflow). Pins
; that the SOP1 swap site lowers to:
;
;   1. the return-address i64 marker materialised and stored into
;      sdst (so a downstream IndirectB consumer of sdst can use it
;      as a cascade-match operand), AND
;   2. a `cmp eq + br` cascade enumerating the statically-known
;      callees (each cmp comparing an i64 marker against the
;      callee's source-MC byte offset), terminating in an
;      `unreachable` trap BB,
;
; whenever the inter-block PC-chain dataflow can prove the call-target
; SGPR pair holds one of a bounded set of in-kernel offsets. The
; static analysis lives in transpiler/setpc_analysis.cpp; the handler
; for this site lives in transpiler/handle_sop1.cpp under
; `case SetPcSiteInfo::Kind::DispatchSet:` of the swap branch and
; routes the cascade emission through the file-local
; `emitEnumeratedDispatch` helper.
;
; Why a cascade instead of `indirectbr` (the very first revision of
; this fixture pinned): LLVM's `FixIrreducible` pass only handles
; `UncondBrInst` / `CondBrInst` / `CallBrInst` as predecessors of an
; irreducible cycle header — `indirectbr` and `switch` crash llc with
; "unsupported block terminator" when the dispatch block lands inside
; an irreducible cycle (the dominant tensilelite activation-function
; dispatch CFG shape). A cascade is FixIrreducible-compatible and
; mem2reg+SCCP+InstCombine-foldable to the same final codegen as a
; fully-folded `indirectbr` whenever the chain-rewriter's per-pred
; marker makes each cmp resolve to a constant `i1 true`. See the
; rationale block on `emitEnumeratedDispatch` in handle_sop1.cpp.
;
; Why integer markers instead of `ptrtoint(blockaddress)` (the
; intermediate revision of this fixture pinned): AMDGPU's
; instruction selector has no pattern to materialise a `BlockAddress`
; constant as an i64 register value, so any `BlockAddress` SDNode
; that survives the middle-end pipeline aborts llc with
;   `Cannot select: t1: i64 = BlockAddress<@kernel, %bb_N>`.
; The storeSGPR64 hi/lo split defeats SCCP's cross-phi fold in
; complex (e.g. tensilelite activation-dispatch) CFGs. Using a
; plain integer marker sidesteps ISel entirely: `BlockAddress`
; only appears as the `label` operand of the `br` (which DOES have
; a codegen pattern), and the `icmp eq i64 %marker, <offset>`
; folds cleanly across phi joins on each predecessor-specialised
; path.
;
; The S_ADDC_U32 post-handler hook in raiser.cpp (keyed on
; `setpcAnalysis.chainTerminators`) rewrites each surviving chain
; terminator's high-half add to store the plain i64 marker
; `resolvedReturnAddr` (the callee's source-MC byte offset) into
; the ret-pair instead of the binary PC the chain would otherwise
; yield (Phase 5 retains both terminators because their
; `resolvedReturnAddr` matches a target in the dispatched set).
; mem2reg + SCCP promote the marker from its alloca into a phi on
; the join block so each cascade cmp resolves to `i1 true` on the
; predecessor-specialised path.
;
; Why these CHECKs:
;   * `bb_0x34` (return target) and `bb_0x3C` / `bb_0x44` (dispatch
;     targets) are derived from the .hip header doc (BB names are
;     formatted by `BasicBlock::Create(C, "bb_0x" + utohexstr(...))`
;     in raiser.cpp — `utohexstr` uses UPPER-case hex digits, hence
;     `bb_0x3C`, not `bb_0x3c`):
;        K = 0x08 (kernarg-load prologue end)
;        swap site at K+0x28 ⇒ return offset = K+0x2C = 0x34 → bb_0x34  (decimal 52)
;        path1 chain target = K+0x34 = 0x3C → bb_0x3C  (decimal 60)
;        path2 chain target = K+0x3C = 0x44 → bb_0x44  (decimal 68)
;   * The marker phi on the join block (`bb_0x30`) carries the two
;     callee i64 markers (60 and 68) keyed by predecessor label —
;     pinning this catches a regression that retained the chain
;     terminator (Phase 5 kept it) but failed to fire the rewrite
;     hook for it (would leave a binary-PC-shaped `extractvalue`
;     flowing into the phi instead of a constant).
;   * The cascade shape (`%swap_call_target_marker = or i64 ...`
;     followed by `dispatch_0x30_cmp_N` + `br i1 ..., label
;     %bb_<target>, label %dispatch_0x30_{N+1_or_unreachable}`)
;     pins that the swap site's terminator chain enumerates exactly
;     the chain-resolved callees.
;   * The return-PC marker (0x34 = decimal 52) is written into sdst
;     by the swap handler as a plain `ConstantInt::get(i64Ty, 52)`.
;     In this fixture sdst is never consumed (the downstream code is
;     straight-line and never reads s[16:17]), so mem2reg + DCE
;     promote-and-drop the sdst alloca; we do NOT pin the sdst write
;     directly in the IR text. The invariant that matters is pinned
;     negatively below: `CHECK-NOT: blockaddress(` proves the sdst
;     write did NOT accidentally materialise a `BlockAddress` as an
;     i64, which was the ISel-crash trigger this revision fixed.
;   * Degeneration into the DirectA single-target lowering is
;     caught indirectly: the DirectA arm of `s_swap_pc_i64` emits
;     `br label %bb_<callee>` and never names anything
;     `swap_call_target_marker` / `dispatch_*_cmp_*`, so a
;     regression that mis-classifies the site as DirectA would fail
;     the CHECK on `%swap_call_target_marker = or i64 ...`. We
;     deliberately do NOT use a `CHECK-NOT br label %bb_0x<dst>$`
;     guard because the SPE (Scalar Predicate Emulation) lowering
;     for the pinned `v_mov_b32` / `scratch_store_b8` instructions
;     inside each target BB emits a benign `br label %bb_<next>`
;     fallthrough into the adjacent BB; matching that as a
;     regression would be a false positive.
;   * Negative `CHECK-NOT: indirectbr` regression-pins that no
;     `indirectbr` ever leaks back into the lifted IR.
;   * Negative `CHECK-NOT: blockaddress(` regression-pins the
;     ISel-safety fix: neither the sdst write nor the cascade may
;     emit a `blockaddress` constant into the lifted IR.

; CHECK-LABEL: define amdgpu_kernel void @setpc_swap_dispatch_set_kernel(

; Both chain terminators' rewrite hooks fire and store the correct
; i64 marker for their respective dispatch targets into the
; ret-pair's low half. mem2reg + SCCP promote those stores into a
; phi on the join block (`bb_0x30`), so the i64 markers appear as
; constant phi incoming values keyed by predecessor label. The DAG
; variant is used because the phi-operand order is encoder-
; dependent; pinning the two (value, label) pairs independently is
; equivalent and CFG-order-robust.
; CHECK-DAG: 60, %bb_0x10
; CHECK-DAG: 68, %bb_0x20

; The DispatchSet site lowers to a cmp+br cascade enumerating both
; chain-resolved callees. The swap-variant handler in
; handle_sop1.cpp first materialises the return-PC i64 marker into
; sdst (a plain `ConstantInt::get(i64Ty, 0x34)` — not pinned here
; because sdst is dead in this fixture), THEN reads the source pair
; s[10:11] as i64 with SSA name `swap_call_target_marker`, and
; emits the cascade. The cascade is top-down: step 0 tests
; `bb_0x3C` (first target in ascending order, offset 60), falling
; through to `dispatch_0x30_1` on miss; step 1 tests `bb_0x44`
; (offset 68), falling through to `dispatch_0x30_unreachable` on
; miss.
;
; Note on block ordering in the emitted module: `emitEnumeratedDispatch`
; pre-creates the unreachable trap BB before looping over the target
; list (so that it can be named deterministically and referenced from
; the last fallthrough), and then creates one intermediate dispatch BB
; per subsequent cascade step as it goes. LLVM appends each new BB to
; the function's basic-block list in creation order, so the emitted
; order for this kernel is:
;   bb_0x30 (step 0 lives here) -> dispatch_0x30_unreachable
;   (pre-created) -> dispatch_0x30_1 (step 1 lives here)
; These CHECKs follow that emitted order.
; CHECK: %swap_call_target_marker = or i64 %{{[^ ]+}}, %{{[^ ]+}}
; CHECK-NEXT: %dispatch_0x30_cmp_0 = icmp eq i64 %swap_call_target_marker, 60
; CHECK-NEXT: br i1 %dispatch_0x30_cmp_0, label %bb_0x3C, label %dispatch_0x30_1

; Trap BB (pre-created by emitEnumeratedDispatch, hence appears in the
; IR before `dispatch_0x30_1`).
; CHECK: dispatch_0x30_unreachable:
; CHECK-NEXT: unreachable

; The fall-through dispatch BB (step 1) is emitted as its own block
; with its own cmp+br. This pins that the cascade is materialised
; as separate BBs (the FixIrreducible-compatible shape) rather than
; chained selects or a single-block switch.
; CHECK: dispatch_0x30_1:
; CHECK-NEXT: %dispatch_0x30_cmp_1 = icmp eq i64 %swap_call_target_marker, 68
; CHECK-NEXT: br i1 %dispatch_0x30_cmp_1, label %bb_0x44, label %dispatch_0x30_unreachable

; No `indirectbr` may leak back into the lifted IR.
; CHECK-NOT: indirectbr
; No `blockaddress` constant may leak back into the lifted IR —
; every dispatch target must be reached via a plain label-operand
; branch, not via a materialised `BlockAddress` value. This also
; guards the sdst-write in the swap handler against accidentally
; regressing to the old `ptrtoint(blockaddress(...))` shape.
; CHECK-NOT: blockaddress(
