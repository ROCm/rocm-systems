; RUN: %raise_cli %s_set_pc_i64_dispatch_set_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=setpc_set_dispatch_set_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for s_set_pc_i64 — DispatchSet (multi-target indirect
; branch resolved by inter-block PC-chain dataflow). Pins that the
; SOP1 indirect set-PC lowers to a `cmp eq + br` cascade enumerating
; the statically-known callees when:
;
;   1. Two CFG paths each compute a different complete getpc+add
;      chain into the same SGPR pair.
;   2. The paths converge into a join block that begins with
;      `s_set_pc_i64 sX:X+1`.
;
; The static analysis (transpiler/setpc_analysis.cpp) Phase 3 inter-
; block dataflow joins the per-path SET facts; Phase 4 enumerates
; both targets and emits a DispatchSet site. The handler in
; transpiler/handle_sop1.cpp under
; `case SetPcSiteInfo::Kind::DispatchSet:` reads the source SGPR pair
; as i64 and routes through the file-local `emitEnumeratedDispatch`
; helper which emits one cmp+br step per resolved callee (comparing
; an i64 marker against the target's source-MC byte offset),
; terminating in an `unreachable` trap BB.
;
; Why a cascade instead of `indirectbr` (the very first revision of
; this fixture pinned): LLVM's `FixIrreducible` pass only handles
; `UncondBrInst` / `CondBrInst` / `CallBrInst` as predecessors of an
; irreducible cycle header — `indirectbr` and `switch` crash llc with
; "unsupported block terminator" when the dispatch block lands inside
; an irreducible cycle. A cascade is FixIrreducible-compatible and
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
; path so SimplifyCFG collapses the cascade to a direct branch.
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
;   * `bb_0x34` and `bb_0x3C` are the two dispatch targets (BB names
;     are formatted by `BasicBlock::Create(C, "bb_0x" + utohexstr(...))`
;     in raiser.cpp — `utohexstr` uses UPPER-case hex digits, hence
;     `bb_0x3C`, not `bb_0x3c`):
;       K = 0x08 (kernarg-load prologue end, see .hip header)
;       path1 chain target = K + 0x2C = 0x34 → bb_0x34  (decimal 52)
;       path2 chain target = K + 0x34 = 0x3C → bb_0x3C  (decimal 60)
;     Both must appear as named blocks in the lifted IR (added to
;     extraBlockStarts in Phase 4) AND both must appear as on-hit
;     destinations in the cascade.
;   * The marker phi on the join block (`bb_0x30`) carries 52 on
;     the bb_0x10 edge and 60 on the bb_0x20 edge — this pins that
;     each predecessor's chain terminator fired its rewrite hook and
;     materialised the correct i64 marker into the ret-pair's low
;     half (the low-half is all that survives because the high-half
;     of every valid in-kernel offset is 0; the hi alloca's promoted
;     phi folds to a constant 0 and is shifted / or'd into the final
;     i64 marker). A regression that retained the chain terminator
;     (Phase 5 kept it) but failed to fire the rewrite hook would
;     leave a binary-PC-shaped `extractvalue` flowing into the phi
;     instead of a constant, and the cascade cmp would not fold.
;   * The `or i64` / `icmp eq i64` / `br i1` sequence and the
;     `dispatch_<off>_cmp_N` / `dispatch_<off>_N` /
;     `dispatch_<off>_unreachable` names pin the cascade shape
;     shared with Pattern B (IndirectB); the SSA name
;     `ret_pc_marker` is set by the shared `case IndirectB / case
;     DispatchSet` branch in handle_sop1.cpp, which lets the same
;     handler code path serve both shapes. The site-offset infix
;     (`0x30`, the offset of the dispatching s_set_pc_i64) is
;     deterministic and unique across dispatch sites in the same
;     kernel.
;   * Degeneration into the DirectA single-target lowering is
;     caught indirectly: the DirectA arm of `s_set_pc_i64` emits
;     `br label %bb_<dst>` and never names anything
;     `ret_pc_marker` / `dispatch_*_cmp_*`, so a regression that
;     mis-classifies the site as DirectA would fail the CHECK on
;     `%ret_pc_marker = ...`. We deliberately do NOT use a
;     `CHECK-NOT br label %bb_0x<dst>$` guard because the SPE
;     (Scalar Predicate Emulation) lowering for the pinned
;     `v_mov_b32` instructions inside each target BB emits a
;     benign `br label %bb_<next>` fallthrough into the adjacent
;     BB; matching that as a regression would be a false positive.
;   * Negative `CHECK-NOT: indirectbr` regression-pins that no
;     `indirectbr` ever leaks back into the lifted IR.
;   * Negative `CHECK-NOT: blockaddress(` regression-pins the
;     ISel-safety fix: the enumerated-dispatch lowering must NEVER
;     emit a `blockaddress` constant into the lifted IR.

; CHECK-LABEL: define amdgpu_kernel void @setpc_set_dispatch_set_kernel(

; Both chain terminators' rewrite hooks fire and store the correct
; i64 marker for their respective dispatch targets into the
; ret-pair's low half. mem2reg + SCCP promote those stores into a
; phi on the join block (`bb_0x30`), so the i64 markers appear as
; constant phi incoming values keyed by predecessor label. The DAG
; variant is used because the phi-operand order is encoder-
; dependent; pinning the two (value, label) pairs independently is
; equivalent and CFG-order-robust.
; CHECK-DAG: 52, %bb_0x10
; CHECK-DAG: 60, %bb_0x20

; The DispatchSet site lowers to a cmp+br cascade enumerating both
; chain-resolved callees. The handler in handle_sop1.cpp's
; `case SetPcSiteInfo::Kind` arm DispatchSet (shared with IndirectB)
; reads s[10:11] as i64 with SSA name `ret_pc_marker` — pinning the
; lowering shape, the marker-comparison direction, and the per-target
; cascade steps. The cascade is top-down: step 0 tests `bb_0x34` (the
; first enumerated target in ascending offset order, offset 52),
; falling through to `dispatch_0x30_1` on miss; step 1 tests
; `bb_0x3C` (offset 60), falling through to
; `dispatch_0x30_unreachable` on miss.
;
; Note on block ordering in the emitted module: `emitEnumeratedDispatch`
; pre-creates the unreachable trap BB before looping over the target
; list (so that it can be named deterministically and referenced from
; the last fallthrough), and then creates one intermediate dispatch BB
; per subsequent cascade step as it goes. LLVM appends each new BB to
; the function's basic-block list in creation order, so the emitted
; order is:
;   bb_0x30 (step 0 lives here) -> dispatch_0x30_unreachable
;   (pre-created)                -> dispatch_0x30_1 (step 1 lives here)
; These CHECKs follow that emitted order.
; CHECK: %ret_pc_marker = or i64 %{{[^ ]+}}, %{{[^ ]+}}
; CHECK-NEXT: %dispatch_0x30_cmp_0 = icmp eq i64 %ret_pc_marker, 52
; CHECK-NEXT: br i1 %dispatch_0x30_cmp_0, label %bb_0x34, label %dispatch_0x30_1

; Trap BB (pre-created by emitEnumeratedDispatch, hence appears in the
; IR before `dispatch_0x30_1`).
; CHECK: dispatch_0x30_unreachable:
; CHECK-NEXT: unreachable

; The fall-through dispatch BB (step 1) is emitted as its own block
; with its own cmp+br. This pins that the cascade is materialised as
; separate BBs (the FixIrreducible-compatible shape) rather than
; chained selects or a single-block switch.
; CHECK: dispatch_0x30_1:
; CHECK-NEXT: %dispatch_0x30_cmp_1 = icmp eq i64 %ret_pc_marker, 60
; CHECK-NEXT: br i1 %dispatch_0x30_cmp_1, label %bb_0x3C, label %dispatch_0x30_unreachable

; No `indirectbr` may leak back into the lifted IR.
; CHECK-NOT: indirectbr
; No `blockaddress` constant may leak back into the lifted IR —
; every dispatch target must be reached via a plain label-operand
; branch, not via a materialised `BlockAddress` value.
; CHECK-NOT: blockaddress(
