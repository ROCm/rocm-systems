; RUN: %raise_cli %v_cmp_cndmask_sgpr_class_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 \
; RUN:     --emit-ir=v_cmp_cndmask_sgpr_class_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Class-compare companion to the predicate-compare shadow fixture
; in `v_cmp_cndmask_sgpr/`. Pins that
; `v_cmp_class_f32_e64 -> SGPR -> v_cndmask_b32_e64` takes the
; same direct-i1 shadow path that predicate compares take — i.e.
; the class intrinsic's `i1` result feeds the cndmask's `select
; i1` directly, bypassing the lossy
; `extractLaneBitFromWaveMask` round-trip.
;
; Regression this fixture guards. A previous version of the fix
; gated the shadow recording on `!m->isClass`, leaving class-
; compare kernels miscompiling under cross-widening for no reason.
; The gate was removed in the same change that added this
; fixture. If a future rewrite re-introduces the gate (or
; otherwise breaks class-compare shadow recording), the `select
; i1 %vclass, ...` CHECK line fails and the fallback extract
; chain `%mask_lane_idx` / `%mask_at_lane` / `%mask_lane_bit` /
; `%mask_lane_i1` reappears — either mode is caught by the
; positive CHECK below and the CHECK-NOTs on the extract chain.
;
; Mask 0x200 = bit 9 = +inf singleton class, same as
; `v_cmp_class_f32/`. The fixture otherwise shares the predicate-
; compare sibling's kernel shape (single wave32 SGPR destination,
; distinctive -1.0 / 1.0 cndmask immediates).
;
; CHECK-LABEL: define amdgpu_kernel void @v_cmp_cndmask_sgpr_class_kernel(

; Producer: class intrinsic emits the per-lane `i1` that the
; handler's class arm (`if (m->isClass)` in handle_valu_vcmp.cpp)
; stores as `%vclass`. The `i32 512` mask immediate is the IEEE-
; +inf singleton; any future rewrite that re-encodes the mask in
; flight would break this CHECK line.
; CHECK: [[CMP:%vclass[0-9]*]] = call i1 @llvm.amdgcn.class.f32(float %{{[^,]+}}, i32 512)

; Writer's wave-mask store preserved for scalar consumers of s4:
; full-width ballot + trunc to source-width SGPR. Shared shape with
; the predicate-compare / v_cmpx_ballot / v_cmp_class_f32 fixtures.
; CHECK: %vcmp_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 [[CMP]])
; CHECK-NEXT: %vcmp_ballot_trunc = trunc i64 %vcmp_ballot to i32

; Consumer reads the cached class-compare `i1` directly. SAME SSA
; value [[CMP]] threads through the cndmask's `select i1`
; condition — no `mask_lane_*` extract chain between producer and
; consumer.
; CHECK: %cndmask = select i1 [[CMP]], i32 1065353216, i32 -1082130432

; NEGATIVE: the extract-fallback chain must NOT fire for this
; fused-path fixture. These are the `mask_lane_*` identifiers
; emitted by `ModuloReplicationProjection::extractLaneBitFromWaveMask`
; in wave_projection.cpp; their absence here is the evidence that
; the class-compare shadow path is active.
; CHECK-NOT: %mask_lane_idx{{[0-9]*}} = zext i32 %{{[^ ]+}} to i64
; CHECK-NOT: %mask_at_lane{{[0-9]*}} = lshr i64 %{{[^,]+}},
; CHECK-NOT: %mask_lane_i1{{[0-9]*}} = icmp ne i64

; NEGATIVE: the pre-shadow-fix wide-integer uniform select shape
; (select i32 on an sgpr-load non-zero test) is forbidden across
; all fixtures in this series.
; CHECK-NOT: select i32 %{{.*}}, i32 {{.*}}, i32 {{.*}}
