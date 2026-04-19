; RUN: %raise_cli %s_swap_pc_i64_dispatch_set_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=setpc_swap_dispatch_set_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for s_swap_pc_i64 — DispatchSet (multi-target indirect
; branch-and-link resolved by inter-block PC-chain dataflow). Pins
; that the SOP1 swap site lowers to:
;
;   1. a `blockaddress(@kernel, %bb_<return-offset>)` materialised
;      and stored into sdst (so a downstream IndirectB consumer of
;      sdst can use it as an indirectbr operand), AND
;   2. an `indirectbr ptr %target, [list]` enumerating the
;      statically-known callees,
;
; whenever the inter-block PC-chain dataflow can prove the call-target
; SGPR pair holds one of a bounded set of in-kernel offsets. The
; static analysis lives in transpiler/setpc_analysis.cpp; the handler
; for this site lives in transpiler/handle_sop1.cpp under
; `case SetPcSiteInfo::Kind::DispatchSet:` of the swap branch.
;
; The S_ADDC_U32 post-handler hook in raiser.cpp (keyed on
; `setpcAnalysis.chainTerminators`) rewrites each surviving chain
; terminator's high-half add to materialise
; `blockaddress(@kernel, %bb_<target>)` into the ret-pair instead of
; the binary PC the chain would otherwise yield (Phase 5 retains both
; terminators because their `resolvedReturnAddr` matches a target in
; the dispatched set).
;
; Why these CHECKs:
;   * `bb_0x34` (return target) and `bb_0x3C` / `bb_0x44` (dispatch
;     targets) are derived from the .hip header doc (BB names are
;     formatted by `BasicBlock::Create(C, "bb_0x" + utohexstr(...))`
;     in raiser.cpp — `utohexstr` uses UPPER-case hex digits, hence
;     `bb_0x3C`, not `bb_0x3c`):
;        K = 0x08 (kernarg-load prologue end)
;        swap site at K+0x28 ⇒ return offset = K+0x2C = 0x34 → bb_0x34
;        path1 chain target = K+0x34 = 0x3C → bb_0x3C
;        path2 chain target = K+0x3C = 0x44 → bb_0x44
;   * The ret-PC blockaddress (bb_0x34) materialised in the swap
;     handler must appear in the IR — pinning the contract that the
;     swap variant of DispatchSet writes the return PC to sdst (the
;     SET-PC variant does not).
;   * Each path's chain terminator must also materialise the
;     corresponding callee blockaddress into the ret-pair — pinning
;     this catches a regression that retained the chain terminator
;     (Phase 5 kept it) but failed to fire the rewrite hook for it
;     (would store the binary PC instead).
;   * `indirectbr ptr %... , [label %bb_0x3C, label %bb_0x44]` pins
;     that the swap site's terminator is an indirectbr enumerating
;     exactly the chain-resolved callees.
;   * Degeneration into the DirectA single-target lowering is
;     caught indirectly by the CHECK-NEXT indirectbr directive
;     below: the DirectA arm of `s_swap_pc_i64` emits
;     `br label %bb_<callee>` and never names anything
;     `swap_call_target_ptr`, so a regression that mis-classifies
;     the site as DirectA would fail the CHECK on
;     `%swap_call_target_ptr = inttoptr i64 ...`. We deliberately
;     do NOT use a `CHECK-NOT br label %bb_0x<dst>$` guard because
;     the SPE (Scalar Predicate Emulation) lowering for the pinned
;     `v_mov_b32` instructions inside each target BB emits a benign
;     `br label %bb_<next>` fallthrough into the adjacent BB;
;     matching that as a regression would be a false positive.

; CHECK-LABEL: define amdgpu_kernel void @setpc_swap_dispatch_set_kernel(

; The swap handler materialises the return-PC blockaddress (sdst
; write) and both chain terminators' rewrite hooks materialise the
; correct callee blockaddresses for the dispatch set. The
; `ptrtoint (ptr blockaddress(...) to i64)` constant-expression form
; is what raiser.cpp's S_ADDC_U32 post-handler hook emits for the
; chain rewrites and what the swap handler's
; `CreatePtrToInt(ba, i64Ty, "swap_ret_blockaddr")` emits for sdst;
; matching the `blockaddress(@kernel, %bb_<target>)` substring is
; robust against the surrounding ptrtoint/lshr/trunc rearranging.
; We use the DAG variant because the IR emits the path1/path2 chain
; rewrites in two predecessor blocks AND the swap-handler sdst
; materialisation in the converge block, all before the indirectbr
; — the relative ordering of these three is encoder/CFG-order
; dependent.
; CHECK-DAG: blockaddress(@setpc_swap_dispatch_set_kernel, %bb_0x34)
; CHECK-DAG: blockaddress(@setpc_swap_dispatch_set_kernel, %bb_0x3C)
; CHECK-DAG: blockaddress(@setpc_swap_dispatch_set_kernel, %bb_0x44)

; The DispatchSet site lowers to an indirectbr enumerating both
; chain-resolved callees. The swap-variant handler in
; handle_sop1.cpp first materialises the return-PC blockaddress into
; sdst (named `swap_ret_blockaddr` in the handler — pinned via the
; three blockaddress DAG checks above), THEN reads the source pair
; s[10:11] as i64, casts it to ptr with SSA name
; `swap_call_target_ptr`, and emits the indirectbr. The strict
; CHECK directive below (after the DAG block above) forces all
; three blockaddresses to appear before the inttoptr in the IR —
; matching the actual emission order (chain rewrites + swap-handler
; sdst write all precede the converge block's indirectbr).
; CHECK: %swap_call_target_ptr = inttoptr i64 %{{[^ ]+}} to ptr
; CHECK-NEXT: indirectbr ptr %swap_call_target_ptr, [label %bb_0x3C, label %bb_0x44]

; The return-target BB and both dispatch-target BBs must be real,
; named blocks in the lifted IR (bb_0x34 from extraBlockStarts via
; swap.offset+size in Phase 1; bb_0x3C / bb_0x44 from
; extraBlockStarts via the resolved DispatchSet targets in Phase 4).
; They are placed AFTER the indirectbr CHECK-NEXT because LLVM
; emits BB label definitions in CFG order — these blocks appear in
; the function body AFTER the converge block where the indirectbr
; lives.
; CHECK-DAG: bb_0x34{{:}}
; CHECK-DAG: bb_0x3C{{:}}
; CHECK-DAG: bb_0x44{{:}}

