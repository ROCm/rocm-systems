; RUN: %raise_cli %c2_permlane_swap_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c2_permlane_swap_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; CROSS_LANE_SURVEY.md item P4 (v_permlane16_swap_b32 lift) has
; landed. The classifier's LaneGroupShuffle site accepts
; V_PERMLANE16_SWAP_B32 as outcome (b) because
; `handle_valu_cross_lane.cpp` emulates the two-VGPR exchange
; through paired `llvm.amdgcn.ds.bpermute` calls (the same target-
; independent path the P2 permlane16/permlanex16 emulation uses,
; for the same reason: gfx942 lacks native isel for
; `llvm.amdgcn.permlane16.swap`, per upstream LLVM's
; `test/CodeGen/AMDGPU/llvm.amdgcn.permlane16.swap.ll` ERR-SDAG
; assertion).
;
; This test asserts:
;   1. The raise succeeds (the classifier marks
;      V_PERMLANE16_SWAP_B32 [implemented]).
;   2. The emitted IR contains TWO calls to `llvm.amdgcn.ds.bpermute`
;      — one per output VGPR (vdst and src0_out). The P4 handler
;      reuses the partner-lane / byte-address chain across both
;      calls, so the bpermutes share their first operand under CSE
;      but each consumes a different second operand (vdst_in vs
;      src0_in).
;   3. The signature property is the partner-lane XOR with 0x10
;      (= 16): each lane's source-lane index is `lane_id XOR 16`.
;      Matching `xor i32 %{{.*}}, 16` in the byte-address chain
;      pins the swap-partner semantics without asserting on SSA
;      names.
;   4. The intrinsic declaration is present.

; CHECK-LABEL: define amdgpu_kernel void @c2_permlane_swap_kernel(

; The XOR-16 partner computation must precede the bpermutes.
; CHECK:      xor i32 %{{[^,]+}}, 16

; Two ds_bpermute calls, one per output VGPR.
; CHECK:      call i32 @llvm.amdgcn.ds.bpermute(
; CHECK:      call i32 @llvm.amdgcn.ds.bpermute(

; The intrinsic declaration must be present.
; CHECK:      declare {{.*}}i32 @llvm.amdgcn.ds.bpermute(i32, i32)
