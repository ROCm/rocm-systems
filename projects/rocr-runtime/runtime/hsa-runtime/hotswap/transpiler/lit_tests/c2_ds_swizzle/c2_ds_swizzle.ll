; RUN: %raise_cli %c2_ds_swizzle_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c2_ds_swizzle_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; The P6 rewrite (ds_swizzle intrinsic lift) has landed — see the
; ds_swizzle_b32 row of hotswap/docs/wave-size-translation.md §5.3.
; The classifier's DsSwizzle site accepts the QUAD_PERM and
; BITMASK_PERM swizzle sub-modes as outcome (b) because
; `handle_ds.cpp` emits `llvm.amdgcn.ds.swizzle(value, offset_imm)`
; with the offset extracted via `AMDGPU::OpName::offset`.
;
; The fixture's swizzle imm `0x041F` decodes to BITMASK_PERM with
; xor_mask=1 (the SWAP-pairs pattern that the GPT-OSS
; `sum_bitmatrix_rows` kernel emits). BITMASK_PERM is structurally
; wave-size-oblivious because its 5-bit AND/OR/XOR masks address only
; bits 0..4 of the lane index — never reaching bit 5 (the bit that
; distinguishes the lower vs upper 32-lane half of a wave64). Each
; 32-lane half of the gfx942 wave64 target therefore reproduces the
; source's wave32 swizzle independently.
;
; FFT_MODE / ROTATE_MODE / unknown-sub-mode imms are NOT modulo-
; replication-safe and would still refuse via
; `cross-wave-shuffle-rewrite-pending`. That gate is exercised by the
; classifier's `dsSwizzleSafeForModRep` helper; a follow-up fixture
; could pin the negative case if we add a kernel that uses one of
; those modes.
;
; This test asserts:
;   1. The raise succeeds (the classifier passes the kernel through).
;   2. The emitted IR contains a call to `llvm.amdgcn.ds.swizzle`.
;   3. The intrinsic's offset arg is the literal 0x041F (= 1055
;      decimal) — pinning that the 16-bit MC immediate flows through
;      to the IR's ImmArg without truncation or sign-extension.
;   4. The intrinsic is declared.

; CHECK-LABEL: define amdgpu_kernel void @c2_ds_swizzle_kernel(

; The lift's signature property is the constant 0x041F = 1055
; appearing as the second arg of the intrinsic call. Matching the
; literal pins the imm-extraction path against silent truncation
; (e.g. an i8 cast that would clip 0x041F down to 0x1F).
; CHECK:      call i32 @llvm.amdgcn.ds.swizzle(i32 %{{.*}}, i32 1055)

; The intrinsic declaration must be present.
; CHECK:      declare {{.*}}i32 @llvm.amdgcn.ds.swizzle(i32, i32 immarg{{.*}})
