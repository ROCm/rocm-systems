; RUN: %raise_cli %flat_store_short_d16_hi_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 \
; RUN:     --emit-ir=flat_store_short_d16_hi_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; FLAT-addressing companion to
; `global_store_short_d16_hi/global_store_short_d16_hi.ll`.  Same
; "upper 16 bits of source VGPR" lift shape; different addrspace
; literal on the store (addrspace(0) for FLAT, addrspace(1) for
; GLOBAL).  Both route through the shared
; `emitD16HiHalfTruncI16` helper in handle_flat.cpp so the
; `d16hi_shift` / `d16hi_trunc` breadcrumbs are identical across
; the two fixtures.
;
; Refusing to let a future refactor break ONLY the FLAT branch:
; this fixture is what surfaces such a regression.  Without it,
; the GLOBAL fixture alone would pass-while-FLAT-silently-broke.

; CHECK-LABEL: define amdgpu_kernel void @flat_store_short_d16_hi_kernel(

; Same shared-helper IR shape as the GLOBAL fixture.
; CHECK-DAG: %d16hi_shift = lshr i32 %{{.+}}, 16
; CHECK-DAG: %d16hi_trunc = trunc i32 %d16hi_shift to i16

; i16-wide store.  FLAT address space is addrspace(0), which the
; backend lowers to either `flat_store_short` (flat aperture) or
; `global_store_short` (when the pointer is provably global) — we
; don't care which at the IR level; we care that the VALUE stored
; is the upper-half-bf16 and NOT the low-16 of the source.
; CHECK: store i16 %d16hi_trunc, ptr %{{[^,]+}}

; Width regressions.
; CHECK-NOT: store i32 %d16hi_trunc, ptr %
; CHECK-NOT: store i8 %d16hi_trunc, ptr %
