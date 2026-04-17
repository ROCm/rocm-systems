; RUN: %raise_cli %divergent_vgpr_ir_co --emit-ir=divergent_vgpr_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; Per-lane VGPR divergence — asserts the SPE diamond structure in the
; raised IR matches what SPE's `emitUnderExec` + mem2reg are supposed
; to produce.
;
; The fixture (divergent_vgpr_ir.hip) has two v_cmpx-gated regions that
; write different per-lane constants (0xAA then 0xBB) into the same
; VGPR, with 0xCC as the pre-initialised default. A single unmasked
; store at the end observes each lane's final value. The post-mem2reg
; IR must carry the lane divergence via phi nodes that join the
; "active" branch (spe_do) with the "skip" branch (spe_skip).
;
; If the test breaks, the most likely regressions are:
;
;   * `emitUnderExec` no longer wraps v_mov_b32 stores to VGPRs in an
;     SPE diamond — the phi disappears and every lane sees the same
;     value.
;   * The EXEC SSA value is not updated after v_cmpx / s_and_b64 — the
;     `lshr i64 <exec>, %spe_lane_mod` after the second diamond still
;     uses `-1` instead of `%and64`.
;   * The lane-id / mask computation changes shape (e.g. a different
;     `and i64 ..., 63` mask for the wavefront size).

; CHECK-LABEL: define amdgpu_kernel void @divergent_vgpr_kernel(

; Lane id and initial (full-EXEC) active-bit computation.
;
; The `and i64 %spe_lane_idx, 63` mask is the wave64 (gfx942) lane
; modulus (waveSize - 1). The fixture is compiled `--offload-arch=
; gfx942` and raised for gfx942, so wave64 is correct. A wave32
; sibling of this test (e.g. for gfx1250) would match `31` instead;
; do not loosen the pattern to `{{(63|31)}}` because the concrete
; mask is part of the invariant — the test deliberately pins down
; that the raiser's lane-id math matches the target's wave size.
; CHECK:       %[[LANE_LO:[^ ]+]] = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK-NEXT:  %[[LANE_ID:[^ ]+]] = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %[[LANE_LO]])
; CHECK-NEXT:  %[[LANE_IDX:[^ ]+]] = zext i32 %[[LANE_ID]] to i64
; CHECK-NEXT:  %[[LANE_MOD:[^ ]+]] = and i64 %[[LANE_IDX]], 63
; CHECK-NEXT:  %[[AT_LANE:[^ ]+]] = lshr i64 -1, %[[LANE_MOD]]
; CHECK-NEXT:  %[[EXEC_BIT:[^ ]+]] = and i64 %[[AT_LANE]], 1
; CHECK-NEXT:  %[[ACTIVE:[^ ]+]] = icmp ne i64 %[[EXEC_BIT]], 0
; CHECK-NEXT:  br i1 %[[ACTIVE]], label %[[DO:[^ ,]+]], label %[[SKIP:[^ ,]+]]

; First v_cmpx narrows EXEC; `%cmpx_exec` names the new wave mask.
; CHECK:       %{{[^ ]+}} = icmp ult i32 %tid, 16
; CHECK:       %cmpx_exec = and i64 -1, %{{[^ ]+}}

; Second SPE diamond — keyed on %cmpx_exec, NOT on -1.
; CHECK:       %[[AT_LANE2:[^ ]+]] = lshr i64 %cmpx_exec, %{{[^ ]+}}
; CHECK-NEXT:  %[[EXEC_BIT2:[^ ]+]] = and i64 %[[AT_LANE2]], 1
; CHECK-NEXT:  %[[ACTIVE2:[^ ]+]] = icmp ne i64 %[[EXEC_BIT2]], 0
; CHECK-NEXT:  br i1 %[[ACTIVE2]], label %[[DO2:[^ ,]+]], label %[[SKIP2:[^ ,]+]]

; Phi joining the 0xAA active path with the 0xCC default path. The
; per-lane divergence invariant lives here: EVERY lane's final value
; is selected by its own %spe_lane_active predicate.
; 0xAA = 170, 0xCC = 204.
; CHECK:       {{%[^ ]+}} = phi i32 [ 170, %[[DO2]] ], [ 204, %[[SKIP]] ]

; Third SPE diamond — keyed on %and64 (EXEC after s_and_b64).
; CHECK:       %and64 = and i64 %{{[^ ]+}}, %{{[^ ]+}}
; CHECK:       %[[AT_LANE3:[^ ]+]] = lshr i64 %and64, %{{[^ ]+}}
; CHECK:       br i1 %{{[^ ]+}}, label %[[DO3:[^ ,]+]], label %[[SKIP3:[^ ,]+]]

; Phi joining the 0xBB active path with the previous phi's result.
; 0xBB = 187.
; CHECK:       %[[VAL_PHI:[^ ]+]] = phi i32 [ 187, %[[DO3]] ], [ %{{[^ ]+}}, %[[SKIP2]] ]

; Skip past the third (vlshl-address) diamond by anchoring on the
; %vgpr0.0 phi — it uniquely follows that diamond's merge block —
; then match the FINAL SPE diamond wrapping the address-space(1)
; store of the divergent VGPR value (%[[VAL_PHI]]).
; CHECK:       phi i32 [ %vlshl, %{{[^ ,]+}} ], [ %tid, %{{[^ ,]+}} ]
; CHECK:       %[[FINAL_AT_LANE:[^ ]+]] = lshr i64 -1, %{{[^ ]+}}
; CHECK-NEXT:  %[[FINAL_BIT:[^ ]+]] = and i64 %[[FINAL_AT_LANE]], 1
; CHECK-NEXT:  %[[FINAL_ACTIVE:[^ ]+]] = icmp ne i64 %[[FINAL_BIT]], 0
; CHECK-NEXT:  br i1 %[[FINAL_ACTIVE]], label %[[FINAL_DO:[^ ,]+]], label %{{[^ ,]+}}
; CHECK:       [[FINAL_DO]]:
; CHECK-NEXT:    store i32 %[[VAL_PHI]], ptr addrspace(1) %{{[^ ]+}}, align 4
