; RUN: %raise_cli %v_cmp_cndmask_sgpr_scalar_clobber_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 \
; RUN:     --emit-ir=v_cmp_cndmask_sgpr_scalar_clobber_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Fallback-path fixture for the V_CMP -> SGPR -> V_CNDMASK_B32_e64
; idiom. A scalar `s_mov_b32 s4, imm` sits between the V_CMP
; producer and the V_CNDMASK consumer (see companion .hip). That
; scalar write fires `AllocaRegFile::onSgprWritten`, which
; invalidates the per-lane i1 shadow entry in
; `RaiseContext::lastSgprWaveMaskI1`. The consumer then takes the
; fallback extract chain — exactly the "any interference defeats
; the cache" invariant from `sgpr-wave-mask-translation.md`
; section 3.1 / I3.
;
; Why this fixture exists. Without it, a future change that forgets
; the invalidation hook (wires `onSgprWritten` but doesn't call
; `invalidateSgprWaveMaskI1`, drops the hook from `storeSGPR32`,
; etc.) would silently read a stale per-lane `i1` that no longer
; corresponds to the scalar `s4` at the consumer site. The
; fused-path fixture in the sibling `v_cmp_cndmask_sgpr/` directory
; would still pass in that case. Only this fixture catches that
; specific regression mode.
;
; CHECK-LABEL: define amdgpu_kernel void @v_cmp_cndmask_sgpr_scalar_clobber_kernel(

; The producer's compare still emits the wave-mask store (ballot
; + trunc). Shared with every V_CMP -> SGPR fixture; the trunc is
; the information-loss boundary documented in handle_valu_vcmp.cpp
; and in the design doc's section 2.
; CHECK: fcmp oge float %{{[^,]+}}, 5.000000e-01
; CHECK: %vcmp_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %{{[^)]+}})
; CHECK-NEXT: %vcmp_ballot_trunc = trunc i64 %vcmp_ballot to i32

; Invalidation is not directly observable in the raised IR (it is a
; raise-time C++ DenseMap::erase call that leaves no IR artefact).
; What IS observable is its consequence: the cndmask's mask source
; goes through the full `extractLaneBitFromWaveMask` chain because
; the shadow lookup returned null. Pin the extract chain
; identifiers (`mask_lane_idx` / `mask_at_lane` / `mask_lane_bit` /
; `mask_lane_i1`) — these are the names emitted by
; `ModuloReplicationProjection::extractLaneBitFromWaveMask` in
; wave_projection.cpp.
; CHECK: %mask_lane_idx{{[0-9]*}} = zext i32 %{{[^ ]+}} to i64
; CHECK: %mask_at_lane{{[0-9]*}} = lshr i64 %{{[^,]+}}, %mask_lane_idx{{[0-9]*}}
; CHECK: %mask_lane_bit{{[0-9]*}} = and i64 %mask_at_lane{{[0-9]*}}, 1
; CHECK: %mask_lane_i1{{[0-9]*}} = icmp ne i64 %mask_lane_bit{{[0-9]*}}, 0

; The cndmask's condition MUST be the extract chain's `i1`, NOT the
; compare's `%vcmpf`. Under the shadow-path shortcut, `%vcmpf`
; would flow directly into the select; the invalidation correctly
; prevents that here because by the time of the cndmask, `s4` no
; longer carries the original compare's wave mask.
; CHECK: %cndmask = select i1 %mask_lane_i1{{[0-9]*}}, i32 1065353216, i32 -1082130432

; NEGATIVE: the fused / direct-i1 shape must NOT be taken here. The
; absence of `select i1 %vcmpf, ...` in the clobber kernel is the
; core invariant this fixture exists to enforce.
; CHECK-NOT: %cndmask = select i1 %vcmpf{{[0-9]*}},
