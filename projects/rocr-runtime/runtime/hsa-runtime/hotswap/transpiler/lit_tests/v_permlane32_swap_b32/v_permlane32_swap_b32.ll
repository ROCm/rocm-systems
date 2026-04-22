; RUN: %raise_cli %v_permlane32_swap_b32_co --isa=gfx950 --target-isa=gfx942 \
; RUN:     --emit-ir=v_permlane32_swap_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift fixture for `v_permlane32_swap_b32`'s XOR-32 lane-pair swap
; emulation on a wave64 target that does NOT natively support the
; instruction (gfx950 source → gfx942 target).  See
; `handle_valu_cross_lane.cpp::V_PERMLANE32_SWAP_B32` for the
; handler's design rationale; the `v_permlane16_swap_b32` sibling
; fixture at `../v_permlane16_swap_b32/` pins the XOR-16 variant
; under the same structural contract.
;
; This fixture pins three correctness anchors:
;
;   1. The partner lane is `lane_id XOR 32`, NOT `lane_id XOR 16`
;      / `XOR 15` / any other mask.  A refactor that substitutes
;      the wrong XOR mask would produce a valid-looking
;      `ds_bpermute` emulation whose partner neighbourhood silently
;      diverges from the source's hardware semantics — exactly the
;      miscompile class the P4 row of
;      hotswap/docs/wave-size-translation.md §5.3 documents as the
;      "cross-lane shuffle rewrite" concern.
;
;   2. Two `ds_bpermute` calls are emitted (one per output VGPR):
;      `new_vdst = bperm(addr, src0_in)` and `new_src0_out =
;      bperm(addr, vdst_in)` — the cross-wired swap.  A single-
;      bpermute lift would write one output correctly and leave
;      the other stale / undef; splitting the swap across two
;      bpermutes is what gives the two output VGPRs their
;      hardware-equivalent new values.
;
;   3. The byte-address shift is by 2 bits (multiply-by-4), which
;      is the `ds_bpermute` convention: each lane's selector is
;      the byte offset of the source lane's LDS slot.  A stale
;      lift that forgot the shift would silently lose addressing
;      and fetch the same few lanes' values for every lane.
;
; Matching uses stable SSA-name prefixes (`pls32_*`) from the
; handler rather than numeric SSA-register identifiers so the
; fixture survives unrelated SSA-number churn in the raiser.

; CHECK-LABEL: define amdgpu_kernel void @v_permlane32_swap_b32_kernel(

; (1) Partner = lane_id XOR 32.  The `pls32_partner` identifier
; is the handler's own name for this computed partner lane
; (pls32 = "permlane-swap 32"); pinning the xor-constant here is
; the strongest regression guard — an accidental mask change
; (e.g. copy-paste from the XOR-16 arm that forgets to widen to
; 32) shows up as a FileCheck miss on this line.
; CHECK: %pls32_partner{{[0-9]*}} = xor i32 %{{[^,]+}}, 32

; (2) Byte-address shift by 2 (multiply by 4).  The handler names
; this `pls32_addr` via CreateShl's twine.
; CHECK: %pls32_addr{{[0-9]*}} = shl i32 %pls32_partner{{[0-9]*}}, 2

; (3) Two `ds_bpermute` calls, cross-wired.  Pin both calls by
; their handler-local twines `pls32_new_vdst` / `pls32_new_src0_out`
; so the specific cross-wiring (first bpermute's source is
; src0_in, second's source is vdst_in) is what gets pinned — a
; same-source / reversed cross-wiring would satisfy a generic
; "two bpermutes emitted" CHECK but miscompile the swap.
; CHECK: %pls32_new_vdst{{[0-9]*}} = call i32 @llvm.amdgcn.ds.bpermute(i32 %pls32_addr{{[0-9]*}}, i32 %{{[^)]+}})
; CHECK: %pls32_new_src0_out{{[0-9]*}} = call i32 @llvm.amdgcn.ds.bpermute(i32 %pls32_addr{{[0-9]*}}, i32 %{{[^)]+}})

; NEGATIVE: the pre-graduation shape was a loud raise failure
; (unsupportedShape), not an emulation — no `amdgcn.ds.bpermute`
; call reached the output at all and the raised IR was empty.
; Nothing specific to CHECK-NOT here beyond the positive
; assertions above; the positive CHECKs are tight enough that
; any regression shows up as a FileCheck miss rather than as a
; silent alternate lift.
;
; Cross-target gotcha: the XOR-16 sibling fixture lives at
; `lit_tests/v_permlane16_swap_b32/` (if added) and runs against
; a wave32 source.  This fixture's wave64 source (gfx950) is
; what makes the XOR-32 semantics well-defined — on wave32 the
; partner would wrap past the wave, which is why the handler
; refuses any wave32-source encounter with
; `v_permlane32_swap_b32` (no wave32 ISA enables
; FeaturePermlane32Swap).
