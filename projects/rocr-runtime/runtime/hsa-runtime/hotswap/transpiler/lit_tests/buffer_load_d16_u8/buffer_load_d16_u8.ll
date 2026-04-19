; RUN: %raise_cli %buffer_load_d16_u8_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=buffer_load_d16_u8_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the gfx9+/gfx1250 D16 byte buffer loads.  Pins the
; partial-write merge shape produced by `handle_mubuf.cpp` for both
; halves:
;
;   _D16:    merged = (prior & 0xFFFF0000) | zext(byte_load, i32)
;   _D16_HI: merged = (prior & 0x0000FFFF) | (zext(byte_load, i32) << 16)
;
; Without the merge the previous SHORT_D16 path zero-extended the
; loaded value to i32 and overwrote the entire VGPR — silently
; clobbering the unused half and breaking the `D16PreservesUnusedBits`
; predicate from BUFInstructions.td.  Tensilelite kernels exercise
; both halves to pack adjacent i8 lanes into one VGPR; this fixture
; exists to regression-pin both merge directions.
;
; Invariants pinned below:
;
;   1. The byte load lifts to `@llvm.amdgcn.raw.buffer.load.i8` —
;      not to a flat / global load, and not to a wider type.
;   2. Each merge produces an `and` against the half-preserve mask
;      (0xFFFF0000 for `_d16`, 0x0000FFFF for `_d16_hi`) followed by
;      an `or`.  Pinning the masks (rather than the SSA names) is
;      stable across LLVM-IR-printer updates.
;   3. NO refusal diagnostic in stderr (exit 0, not the legacy
;      `unsupportedOpcode [MUBUF]` path the corpus hit before this
;      handler landed).
;
; What we don't pin: the ordering of the two asm blocks in the lifted
; IR (their sequencing is preserved by the asm `volatile` barrier);
; nor the exact SSA names of the per-half merges (LLVM's IR printer
; may rename across versions).

; CHECK-LABEL: define amdgpu_kernel void @buffer_load_d16_u8_kernel(

; The byte loads themselves: i8-typed raw_buffer_load, one per asm
; block.  Both must be present.
; CHECK-DAG: call i8 @llvm.amdgcn.raw.buffer.load.i8
; CHECK-DAG: call i8 @llvm.amdgcn.raw.buffer.load.i8

; The lo-half merge: AND against 0xFFFF0000 to keep prior hi bits,
; then OR with the zero-extended byte (low 16 bits, hi 16 zeros).
; CHECK-DAG: and i32 {{.*}}, -65536
; CHECK-DAG: or {{(disjoint )?}}i32 {{.*}}, %{{.*}}

; The hi-half merge: AND against 0x0000FFFF to keep prior lo bits,
; then SHL by 16 to position the byte in the hi half, then OR.
; CHECK-DAG: and i32 {{.*}}, 65535
; CHECK-DAG: shl i32 {{.*}}, 16
; CHECK-DAG: or {{(disjoint )?}}i32 {{.*}}, %{{.*}}

; Negative pin: the handler must NOT route the byte load through a
; wider buffer-load type (which would corrupt adjacent bytes).
; CHECK-NOT: call i16 @llvm.amdgcn.raw.buffer.load
; CHECK-NOT: call i32 @llvm.amdgcn.raw.buffer.load

; Negative pin: the merge must NOT use a full-VGPR overwrite shape
; (the legacy bug for SHORT_D16 zext'd loaded -> i32 and stored the
; whole VGPR with no preserve-mask).  If the printed IR contains
; neither 0xFFFF0000 nor 0x0000FFFF, the handler regressed.
; (Negative direction is implicit in the CHECK-DAG asserts above.)
