; RUN: %raise_cli %c2_permlane16_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c2_permlane16_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; CROSS_LANE_SURVEY.md item P2 (permlane16 / permlanex16 lift) has
; landed. The classifier's LaneGroupShuffle site accepts
; V_PERMLANE16_B32 and V_PERMLANEX16_B32 as outcome (b) because
; `handle_valu_cross_lane.cpp` emulates both through
; `llvm.amdgcn.ds.bpermute`.
;
; Why the emulation rather than `llvm.amdgcn.permlane16` directly?
; `v_permlane16_b32` is an RDNA/gfx10+ instruction and is absent on
; CDNA (gfx942). Emitting the native intrinsic fails LLVM isel with
; "Cannot select: intrinsic %llvm.amdgcn.permlanex16" when the
; target is gfx942. `ds_bpermute_b32` is available on gfx8+, so
; emulating via bpermute keeps the lowering target-independent.
;
; This test asserts:
;   1. The raise succeeds (the classifier's LaneGroupShuffle site is
;      `[implemented]` for the non-swap variants).
;   2. The emitted IR contains a call to `llvm.amdgcn.ds.bpermute`
;      — one per `v_permlane16` / `v_permlanex16` input, though
;      CSE may merge identical calls if the fixture happens to
;      produce the same byte address.
;   3. The intrinsic is declared.
;
; We do NOT check the byte-address computation structurally (the
; per-lane lane-id + selector-nibble arithmetic surrounding the
; call) because that's codified in the handler's source-level
; comment block, and structural IR checks would be brittle across
; LLVM SSA-renumbering changes.

; CHECK-LABEL: define amdgpu_kernel void @c2_permlane16_kernel(

; Both permlane16 and permlanex16 emit a ds_bpermute call. The
; handler's SSA names are `permlane16_emu` / `permlanex16_emu`, but
; FileCheck against the intrinsic name alone stays stable across
; rename refactors.
; CHECK:      call i32 @llvm.amdgcn.ds.bpermute(
; CHECK:      call i32 @llvm.amdgcn.ds.bpermute(

; The intrinsic declaration must be present.
; CHECK:      declare {{.*}}i32 @llvm.amdgcn.ds.bpermute(i32, i32)
