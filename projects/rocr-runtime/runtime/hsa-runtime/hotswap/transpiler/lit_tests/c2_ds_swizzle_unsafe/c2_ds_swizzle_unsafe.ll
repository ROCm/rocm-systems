; RUN: %not %raise_cli %c2_ds_swizzle_unsafe_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c2_ds_swizzle_unsafe_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; Negative case for the P6 lift. The classifier must refuse
; FFT_MODE / ROTATE_MODE / unknown-sub-mode ds_swizzle imms because
; those span the full source-wave width and have no half-wave-
; independent semantics — a wave64 lift of an FFT_MODE wave32 imm
; would silently change the swizzle pattern.
;
; This complements c2_ds_swizzle.ll (positive case for the safe
; BITMASK_PERM/QUAD_PERM sub-modes) by pinning the safe/unsafe
; boundary, so a future change to `dsSwizzleSafeForModRep` that
; widens the safe set without intent (e.g. accidentally accepting
; FFT_MODE) is caught here.

; The pre-translation abort is keyed on the same
; cross-wave-shuffle-rewrite-pending failure as before P6 — only the
; *cause* changed: previously every ds_swizzle imm was unconditionally
; pending; now the gate fires only for unsafe sub-modes.
; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-shuffle-rewrite-pending
; STDERR-SAME: ds_swizzle

; Classifier trace must surface the DsSwizzle site with
; `[pending]` and a detail line naming the offending imm and the
; FFT_MODE/ROTATE_MODE sub-mode envelope. The exact imm value
; (0xe03f) is asserted to pin the imm-extraction path against silent
; truncation (e.g. an i8 cast that would clip 0xe03f down to 0x3f
; and accidentally classify it as BITMASK_PERM-safe).
; STDERR: DsSwizzle
; STDERR-SAME: Class 2
; STDERR: rewrite: P6
; STDERR-SAME: pending
; STDERR: detail: ds_swizzle_b32 imm 0xe03f
; STDERR-SAME: FFT_MODE/ROTATE_MODE/unknown sub-mode
; STDERR-SAME: not modulo-replication-safe
; STDERR: outcome: (c) refuse

; Final raise_cli refusal line — the `failed to raise` message must
; cite ds_swizzle by mnemonic so corpus-sweep tooling can bucket the
; failure correctly.
; STDERR: raise_cli: kernel 'c2_ds_swizzle_unsafe_kernel' failed to raise:
; STDERR-SAME: ds_swizzle
