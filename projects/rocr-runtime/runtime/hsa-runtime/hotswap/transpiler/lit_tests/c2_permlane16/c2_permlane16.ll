; RUN: %raise_cli %c2_permlane16_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c2_permlane16_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; The P2 rewrite (permlane16 / permlanex16 lift) has landed — see
; the permlane16/permlanex16 row of hotswap/docs/wave-size-
; translation.md §5.3. The classifier's LaneGroupShuffle site
; accepts
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

; The permlanex16 emulation's signature property is the group-swap
; XOR by 0x10 (= 16) on the computed group-base before the byte-
; address shift feeds into ds_bpermute. Matching the constant 16 in
; a `xor` instruction that then feeds (directly or indirectly) into
; the bpermute's index operand is the minimal check that pins the
; cross-16-lane-group semantics without asserting on SSA names.
;
; permlane16 (non-swap) must NOT have a `xor %..., 16` anywhere in
; its byte-address chain — FileCheck-NOT would be brittle here
; since the later permlanex16 DOES carry the xor, so we instead rely
; on the `ds_bpermute` call order: the first bpermute (permlane16)
; precedes the xor (which belongs to permlanex16's chain).

; First bpermute: the permlane16 (non-swap) call. Its byte-address
; chain has no `xor …, 16`.
; CHECK:      %{{permlane16_emu[0-9]*}} = call i32 @llvm.amdgcn.ds.bpermute(

; After the first permlane16 emits, the permlanex16 group-swap
; xor with 0x10 (= 16) must appear somewhere before the second
; ds_bpermute call consumes its byte-address operand.
; CHECK:      xor i32 %{{[^,]+}}, 16

; Second bpermute: the permlanex16 call.
; CHECK:      %{{permlanex16_emu[0-9]*}} = call i32 @llvm.amdgcn.ds.bpermute(

; The intrinsic declaration must be present.
; CHECK:      declare {{.*}}i32 @llvm.amdgcn.ds.bpermute(i32, i32)
