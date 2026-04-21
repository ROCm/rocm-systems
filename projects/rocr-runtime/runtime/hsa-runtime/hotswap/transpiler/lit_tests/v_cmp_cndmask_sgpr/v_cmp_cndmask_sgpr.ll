; RUN: %raise_cli %v_cmp_cndmask_sgpr_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_cmp_cndmask_sgpr_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift fixture for the V_CMP_*_e64 -> SGPR -> V_CNDMASK_B32_e64
; per-lane-predication idiom. Pins the post-fix IR shape in
; `handle_valu_vop3p.cpp`'s V_CNDMASK_B32 arm: when the mask source
; is an SGPR (e64 form, not VCC), the handler routes through the
; WaveProjection's `extractLaneBitFromWaveMask` — matching the
; consumer symmetry of VCC's `readVCCAsWaveMask`.
;
; Pre-fix, the SGPR arm did `icmp ne <sgpr-as-integer>, 0`, folding
; the entire wave mask into one uniform `i1`. Every lane of the
; wave then took the same side of the subsequent `select`, silently
; miscompiling every data-dependent predication idiom: libdevice
; math branches (asin / acos / atan / log / exp), copysign, Triton
; epilogues that survive an e64 compare across a basic-block
; boundary. See the regression probe `v_cmp_cndmask_sgpr` under
; `tools/compare_correctness/` for the runtime evidence.
;
; Three contract pieces covered:
;
;   1. The compare writes a single wave32 SGPR: the handler in
;      `handle_valu_vcmp.cpp` runs a full-width ballot on the
;      target (wave64 -> i64) and truncates to source width (i32)
;      before the alloca store. Shared identifiers
;      `vcmp_ballot` + `vcmp_ballot_trunc` with the sibling
;      `v_cmpx_ballot` / `v_cmp_class_f32` fixtures.
;   2. The cndmask consumes the SGPR as a per-lane mask: zext /
;      trunc to wave-mask width, then an `mbcnt`-derived lane id
;      indexes the mask via `lshr + and 1 + icmp ne 0`. This is
;      the shared tail of `WaveProjection::extractLaneBitFromWaveMask`
;      (both ModRep and WaveNative variants reach the same shape;
;      WaveNative additionally replicates the narrow source-width
;      mask into the wider target wave mask before the lshr).
;   3. The final select carries an `i1` (per-lane), not a wide
;      integer (which would be the pre-fix uniform shape).
;
; CHECK-LABEL: define amdgpu_kernel void @v_cmp_cndmask_sgpr_kernel(

; (1) Full-width target ballot + truncate to the single-SGPR
; source-width write-back. `vcmp_ballot` / `vcmp_ballot_trunc` are
; the pinned identifiers from the V_CMP -> SGPR branch in
; handle_valu_vcmp.cpp (same shape as the v_cmpx_ballot and
; v_cmp_class_f32 fixtures).
; CHECK: %vcmp_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %{{[^)]+}})
; CHECK-NEXT: %vcmp_ballot_trunc = trunc i64 %vcmp_ballot to i32

; (2) Per-lane extract: lane id from `mbcnt.lo` + `mbcnt.hi`, then
; `lshr <mask>, lane_id` -> `and 1` -> `icmp ne 0`. The projection
; names the intermediates; `{{.*}}` absorbs the `wn_` prefix on the
; wave-native variant vs. the bare names on modulo-replication.
; CHECK: call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK: call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %{{[^)]+}})
; CHECK: lshr i{{32|64}} %{{[^,]+}}, %{{.*}}lane_idx{{.*}}
; CHECK-NEXT: and i{{32|64}} %{{[^,]+}}, 1
; CHECK-NEXT: icmp ne i{{32|64}} %{{[^,]+}}, 0

; (3) The cndmask itself: a per-lane `select i1` with the extracted
; bit, not a wide-integer `select` driven by a wave-uniform bool.
; 1065353216 = 0x3f800000 = +1.0f; -1082130432 = 0xbf800000 = -1.0f.
; CHECK: %cndmask = select i1 %{{[^,]+}}, i32 1065353216, i32 -1082130432

; NEGATIVE: the pre-fix lowering folded the entire SGPR mask to a
; single uniform `i1` via `icmp ne <sgpr-load>, 0`, feeding every
; lane the same bit. The positive CHECK chain above already catches
; that regression structurally — a pre-fix lowering has no mbcnt
; calls or per-lane lshr/and/icmp between the ballot-trunc and the
; cndmask select, so those CHECK lines fail to match. The one
; additional shape-level assertion that adds value is the cndmask
; `select` MUST carry an `i1` condition (per-lane), not an `i32`
; (which would be the wide-integer pre-fix shape threaded directly
; from the SGPR load).
; CHECK-NOT: select i32 %{{.*}}, i32 {{.*}}, i32 {{.*}}
