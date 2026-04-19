; RUN: %raise_cli %s_set_pc_i64_dispatch_set_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=setpc_set_dispatch_set_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for s_set_pc_i64 — DispatchSet (multi-target indirect
; branch resolved by inter-block PC-chain dataflow). Pins that the
; SOP1 indirect set-PC lowers to an LLVM `indirectbr` enumerating the
; statically-known callees when:
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
; as i64, casts to ptr, and emits `indirectbr ptr %target, [list]`
; with one destination per resolved callee.
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
;   * `bb_0x34` and `bb_0x3C` are the two dispatch targets (BB names
;     are formatted by `BasicBlock::Create(C, "bb_0x" + utohexstr(...))`
;     in raiser.cpp — `utohexstr` uses UPPER-case hex digits, hence
;     `bb_0x3C`, not `bb_0x3c`):
;       K = 0x08 (kernarg-load prologue end, see .hip header)
;       path1 chain target = K + 0x2C = 0x34 → bb_0x34
;       path2 chain target = K + 0x34 = 0x3C → bb_0x3C
;     Both must appear as named blocks in the lifted IR (added to
;     extraBlockStarts in Phase 4) AND both must appear in the
;     indirectbr's destination list.
;   * Each path's chain terminator must materialise the corresponding
;     blockaddress into the ret-pair — pinning this catches a
;     regression that retained the chain terminator (Phase 5 kept it)
;     but failed to fire the rewrite hook for it (would store the
;     binary PC instead).
;   * `inttoptr` and `indirectbr ptr` shape is shared with Pattern B
;     (IndirectB); the SSA name `ret_pc_ptr` is set by the shared
;     `case IndirectB / case DispatchSet` branch in handle_sop1.cpp,
;     which lets the same handler code path serve both shapes.
;   * Degeneration into the DirectA single-target lowering is
;     caught indirectly by the CHECK-NEXT indirectbr directive
;     below: the DirectA arm of `s_set_pc_i64` emits
;     `br label %bb_<dst>` and never names anything `ret_pc_ptr`,
;     so a regression that mis-classifies the site as DirectA
;     would fail the CHECK on `%ret_pc_ptr = inttoptr i64 ...`.
;     We deliberately do NOT use a `CHECK-NOT br label %bb_0x<dst>$`
;     guard because the SPE (Scalar Predicate Emulation) lowering
;     for the pinned `v_mov_b32` instructions inside each target BB
;     emits a benign `br label %bb_<next>` fallthrough into the
;     adjacent BB; matching that as a regression would be a false
;     positive.

; CHECK-LABEL: define amdgpu_kernel void @setpc_set_dispatch_set_kernel(

; Both chain terminators' rewrite hooks fire and materialise the
; correct blockaddress for their respective dispatch targets. The
; `ptrtoint (ptr blockaddress(...) to i64)` constant-expression form
; is what raiser.cpp's S_ADDC_U32 post-handler hook emits before
; lshr/trunc/store-into-ret-pair; matching the substring
; `blockaddress(@kernel, %bb_<target>)` is robust against the
; surrounding ptrtoint/lshr/trunc rearranging. We use the DAG
; variant because the IR emits the path1/path2 chain rewrites in
; two predecessor blocks whose CFG order is encoder-dependent.
; CHECK-DAG: blockaddress(@setpc_set_dispatch_set_kernel, %bb_0x34)
; CHECK-DAG: blockaddress(@setpc_set_dispatch_set_kernel, %bb_0x3C)

; The DispatchSet site lowers to an indirectbr enumerating both
; chain-resolved callees. The handler in handle_sop1.cpp's
; `case SetPcSiteInfo::Kind` arm DispatchSet (shared with IndirectB)
; reads s[10:11] as i64 and `inttoptr`s to ptr with SSA name
; `ret_pc_ptr` — pinning both the lowering shape, the conversion
; direction, and the destination list. The strict CHECK directive
; below (after the DAG block above) forces the blockaddresses to be
; found before the inttoptr in the IR — matching the actual emission
; order (chain rewrites in predecessor BBs, then the indirectbr in
; the converge BB).
; CHECK: %ret_pc_ptr = inttoptr i64 %{{[^ ]+}} to ptr
; CHECK-NEXT: indirectbr ptr %ret_pc_ptr, [label %bb_0x34, label %bb_0x3C]

; Both target blocks must be real, named labels in the lifted IR.
; They are placed AFTER the indirectbr CHECK-NEXT because LLVM
; emits BB label definitions in CFG order — bb_0x34 / bb_0x3C
; appear in the function body AFTER the converge block where the
; indirectbr lives.
; CHECK-DAG: bb_0x34{{:}}
; CHECK-DAG: bb_0x3C{{:}}

