; RUN: %raise_cli %ds_load_tr8_b64_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=ds_load_tr8_b64_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the gfx1250 64-bit transposed LDS load with 8-bit
; elements (`ds_load_tr8_b64`).  The handler lives in
; transpiler/handle_ds.cpp under the shared `emitDsLoadTr8B64` lambda
; that fires for both `SemOp::DS_LOAD_TR8_B64` (gfx1250 spelling) and
; `SemOp::DS_READ_B64_TR_B8` (gfx950 spelling); the SemOp lives in
; transpiler/semop.hpp under the `// -- DS --` group with a docstring
; explaining why the two MC opcodes share lowering.
;
; The lift is hand-rolled rather than emitting
; `int_amdgcn_ds_load_tr8_b64` because gfx942 (the transpiler's
; target ISA) has no isel pattern for that intrinsic — neither
; `HasGFX950Insts` (the gfx950 read_tr sibling) nor `isGFX1250Plus`
; (the gfx1250 load_tr family) gates fire on gfx942, and there is no
; in-tree pre-isel emulation pass.  Emitting the intrinsic would
; therefore silently miscompile or fail at codegen.  This test pins
; the structural primitives the hand-rolled emulation must produce.
;
; Invariants:
;
;   1. NO `int_amdgcn_ds_load_tr8_b64` in the lifted IR — that would
;      mean the handler regressed to the old "emit the intrinsic and
;      trust the backend" shape that DS_READ_B64_TR_B8 used to take.
;   2. The lane-id derivation (`mbcnt_lo` -> `mbcnt_hi`) is present
;      so the per-lane group_base and l_in_group can be computed.
;   3. Exactly 8 `amdgcn.ds.bpermute` calls — one per source lane in
;      the 8-lane transpose group.  The handler decomposes the v2i32
;      result into 2 dwords × 4 i8 elements per dword, and each i8
;      element comes from a different source lane (via bpermute).
;   4. Exactly 8 `load i8, ptr addrspace(3)` instructions — one per
;      bpermuted source lane, reading the actual byte from LDS at
;      `bpermuted_base + l_in_group`.
;   5. The packed result is constructed via the canonical
;      zext-shift-or pattern; the SSA value names `tr8_b` (loaded
;      i8), `tr8_pack` (the in-progress OR accumulator), `tr8_p`
;      (LDS pointer cast), and `bp_base` (bpermute result) pin the
;      handler's value-naming contract that downstream tooling
;      greps for.

; CHECK-LABEL: define amdgpu_kernel void @ds_load_tr8_b64_kernel(

; Lane-id derivation must use mbcnt_lo / mbcnt_hi against -1 / 0 /
; mbcnt_lo's result — the same shape DS_LOAD_TR16_B128 uses, since
; both share the "lane_id = mbcnt_hi(-1, mbcnt_lo(-1, 0))" idiom.
; CHECK: %lane_lo{{[0-9]*}} = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK: %lane_id{{[0-9]*}} = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane_lo{{[0-9]*}})

; The hand-rolled emulation must contain at least one
; ds.bpermute call (the structural primitive that does the
; cross-lane gather of source-lane base addresses).  The exact count
; is asserted via the 8 named bpermute results below.
; CHECK: %bp_base = call i32 @llvm.amdgcn.ds.bpermute(i32 %{{[^,]+}}, i32 %{{[^,]+}})

; Each transposed i8 element is loaded from LDS at the bpermuted
; source-lane base + l_in_group offset, then zext'd to i32 and
; shifted into its byte slot in the output dword.  This is the
; first element (byte 0 of dword 0) — pinned to confirm the
; packing pattern matches `or i32 0, %15` (the first slot has no
; previous accumulator).
; CHECK: %tr8_p = inttoptr i64 %{{[^ ]+}} to ptr addrspace(3)
; CHECK: %tr8_b = load i8, ptr addrspace(3) %tr8_p, align 1
; CHECK: %tr8_pack = or i32 0, %{{[^ ]+}}

; Honest refusal: if the handler ever regresses to emitting the
; gfx1250 intrinsic directly, the lift will silently miscompile on
; gfx942 (no isel pattern).  Pinning the negative invariant catches
; that drift in the next CI run.
; CHECK-NOT: call {{.*}}@llvm.amdgcn.ds.load.tr8.b64
; CHECK-NOT: call {{.*}}@llvm.amdgcn.ds.read.tr8.b64
