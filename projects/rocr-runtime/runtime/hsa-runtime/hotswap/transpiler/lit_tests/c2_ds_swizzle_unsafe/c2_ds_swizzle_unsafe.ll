; RUN: %not %raise_cli %c2_ds_swizzle_unsafe_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c2_ds_swizzle_unsafe_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; Negative case for the P6 lift after the audit follow-up that
; widened the safe set to all four valid swizzle-mode envelopes
; (QUAD_PERM / BITMASK_PERM / FFT_MODE / ROTATE_MODE). What remains
; refused is the RESERVED top-nibble envelope (top nibble in
; {0x9, 0xA, 0xB, 0xD, 0xF}) where AMDGPU SIDefines.h `Swizzle::
; EncBits` assigns no semantics — hardware behavior is undefined and
; a silent lift would map the source's imm to whatever the wave64
; backend happens to do.
;
; This complements c2_ds_swizzle.ll (positive case for the
; BITMASK_PERM/SWAP-1 corpus pattern) by pinning the safe/unsafe
; boundary exactly at the RESERVED envelope.

; The pre-translation abort uses the same
; cross-wave-shuffle-rewrite-pending failure as the original P6
; refusal — only the cause changed: previously every imm that wasn't
; QUAD_PERM/BITMASK_PERM was refused; now the gate fires only for the
; RESERVED top-nibble envelope.
; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-shuffle-rewrite-pending
; STDERR-SAME: ds_swizzle

; Classifier trace must surface the DsSwizzle site with `[pending]`
; and a detail line citing the RESERVED top-nibble (one of the
; broader "not a valid swizzle encoding" refusal categories). The
; exact imm value (0x9000) is asserted to pin the imm-extraction
; path against silent truncation.
; STDERR: DsSwizzle
; STDERR-SAME: Class 2
; STDERR: rewrite: P6
; STDERR-SAME: pending
; STDERR: detail: ds_swizzle_b32 imm 0x9000
; STDERR-SAME: not a valid swizzle encoding
; STDERR-SAME: RESERVED top-nibble
; STDERR-SAME: AMDGPU hardware semantics undefined
; STDERR: outcome: (c) refuse

; Final raise_cli refusal line — the `failed to raise` message must
; cite ds_swizzle by mnemonic so corpus-sweep tooling can bucket the
; failure correctly.
; STDERR: raise_cli: kernel 'c2_ds_swizzle_unsafe_kernel' failed to raise:
; STDERR-SAME: ds_swizzle
