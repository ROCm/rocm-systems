; RUN: %not %raise_cli %c2_ds_swizzle_fft_reserved_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c2_ds_swizzle_fft_reserved_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; Negative case for the P6 lift after the audit follow-up that
; tightened the FFT_MODE / ROTATE_MODE envelope checks to require
; reserved bits clear (matches the AsmParser's encoding contract
; and the `Swizzle::EncBits` table in SIDefines.h).
;
; Companion to c2_ds_swizzle_unsafe.ll, which pins refusal of the
; RESERVED top-nibble envelope (top nibble in {0x9, 0xA, 0xB, 0xD,
; 0xF}). This fixture pins refusal of imms that fall *within* a
; valid envelope (FFT_MODE here) but have reserved bits set —
; another category whose hardware semantics are undefined.
;
; The imm `offset:0xe020` selects FFT_MODE (top nibble 0xE) with FFT
; swizzle selector 0 (bits 0..4 = 0) BUT reserved bit 5 = 1. Without
; the reserved-bit validation, the broader-envelope check would
; have accepted this imm; with the validation, the classifier
; refuses with a "not a valid swizzle encoding" detail line.

; Same cross-wave-shuffle-rewrite-pending failure as before.
; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-shuffle-rewrite-pending
; STDERR-SAME: ds_swizzle

; The classifier trace must surface the imm-specific rejection.
; The detail line wording covers both refusal categories (RESERVED
; top-nibble OR FFT/ROTATE reserved bits set) — we assert the parts
; that are stable across either category so the same .ll wording
; could pin both fixtures' refusals.
; STDERR: DsSwizzle
; STDERR-SAME: Class 2
; STDERR: rewrite: P6
; STDERR-SAME: pending
; STDERR: detail: ds_swizzle_b32 imm 0xe020
; STDERR-SAME: not a valid swizzle encoding
; STDERR-SAME: FFT/ROTATE reserved bits set
; STDERR-SAME: AMDGPU hardware semantics undefined
; STDERR: outcome: (c) refuse

; Final raise_cli refusal line.
; STDERR: raise_cli: kernel 'c2_ds_swizzle_fft_reserved_kernel' failed to raise:
; STDERR-SAME: ds_swizzle
