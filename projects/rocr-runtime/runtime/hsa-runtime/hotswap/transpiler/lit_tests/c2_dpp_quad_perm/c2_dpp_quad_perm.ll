; RUN: %raise_cli %c2_dpp_quad_perm_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c2_dpp_quad_perm_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; The P5 rewrite (DPP modifier intrinsic lift) has landed — see
; the DPP row of hotswap/docs/wave-size-translation.md §5.3.  This
; test was originally a refuse-loud fixture asserting
; `cross-wave-shuffle-rewrite-pending`, then flipped to a positive
; faithful-lift fixture asserting `llvm.amdgcn.update.dpp.i32` in
; the emitted IR.
;
; A subsequent pass (`rewrite_cross_lane_divergent.cpp`) brought
; DPP under the same cross-widening rewrite invariant as
; writelane / readlane / permlane16 / permlanex16: under cross-
; widening (wave32 -> wave64) every cross-lane primitive is
; rewritten to a `ds_bpermute + select` shape whose correctness
; depends only on ds_bpermute's explicit per-lane read semantics
; (stable across gfx9+) rather than on target ISA and source ISA
; sharing the same bank_mask / row_mask interpretation.  This
; test's source kernel targets gfx1250 (wave32) and is raised
; against gfx942 (wave64), so the rewrite fires.
;
; The DPP modifier values in the fixture are
; `quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1` —
; which encode to `dpp_ctrl = 0xB1 = 177`, `row_mask = 0xF = 15`,
; `bank_mask = 0xF = 15`, `bound_ctrl = true`.  The
; `quad_perm:[1,0,3,2]` pattern is a within-4-lane swap pair: lane
; 0 reads lane 1, lane 1 reads lane 0, lane 2 reads lane 3, lane 3
; reads lane 2.  In the rewrite, `withinRow = lane & 0xF`,
; `quadBase = withinRow & ~3`, `quadWithin = withinRow & 3`,
; `selector = (0xB1 >> (quadWithin * 2)) & 3`; lane 0 decodes
; `selector = 0xB1 & 3 = 1`, lane 1 decodes `(0xB1 >> 2) & 3 = 0`,
; etc.  srcLaneAbs = rowBase | (quadBase | selector).
;
; Since both row_mask and bank_mask are 0xF (every target lane
; active), the per-lane mask-gating select collapses to the
; dpp-value select (see the `rowMaskImm == 0xF && bankMaskImm ==
; 0xF` shortcut in rewrite_cross_lane_divergent.cpp).  And since
; quad_perm is always in-range (every 2-bit selector resolves to
; a valid lane within the quad), the `inRange` predicate folds to
; `i1 true` and the bperm result flows directly through.  What
; survives in the IR is: lane-topology mbcnt / and / or / shl
; scaffolding plus one `ds_bpermute` call.

; CHECK-LABEL: define amdgpu_kernel void @c2_dpp_quad_perm_kernel(

; The faithful-lift `update.dpp` call MUST NOT appear in the
; emitted IR under cross-widening — the rewrite replaces every
; i32 update.dpp with a ds_bpermute + select chain.  Only the
; intrinsic declaration remains (inserted by other passes); the
; rewrite's `CreateCall` of `@llvm.amdgcn.ds.bpermute` does not
; re-insert `update.dpp`.
; CHECK-NOT: call i32 @llvm.amdgcn.update.dpp.i32(

; The rewritten shape is built from lane-topology primitives:
; within-row, quad-base, quad-within, 2-bit-selector, row-base.
; These name the per-lane mapping for quad_perm:[1,0,3,2].
; CHECK-DAG: %cwd_dpp_within_row = and i32 %{{.+}}, 15
; CHECK-DAG: %cwd_dpp_row_base = and i32 %{{.+}}, -16
; CHECK-DAG: %cwd_dpp_quad_base = and i32 %cwd_dpp_within_row, -4
; CHECK-DAG: %cwd_dpp_quad_within = and i32 %cwd_dpp_within_row, 3
; CHECK-DAG: %cwd_dpp_quad_shift = shl i32 %cwd_dpp_quad_within, 1
; CHECK-DAG: lshr i32 177, %cwd_dpp_quad_shift
; CHECK-DAG: %cwd_dpp_quad_sel = and i32 %{{.+}}, 3
; CHECK-DAG: %cwd_dpp_quad_src = or i32 %cwd_dpp_quad_base, %cwd_dpp_quad_sel
; CHECK-DAG: %cwd_dpp_src_abs = or i32 %cwd_dpp_row_base, %{{.+}}
; CHECK-DAG: %cwd_dpp_selector = shl i32 %cwd_dpp_src_abs, 2

; The ds_bpermute call carries the per-lane selector and the
; source value unchanged.  The declaration below is the only
; `amdgcn.update.dpp.i32` symbol that can remain — it is inserted
; by passes unrelated to the rewrite and is safely unused.
; CHECK: call i32 @llvm.amdgcn.ds.bpermute(i32 %cwd_dpp_selector, i32 %{{[^,]+}})

; Declaration of the bpermute intrinsic (emitted by the rewrite
; when it inserts the call).  Overloaded on i32 by fixed-shape.
; CHECK: declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)
