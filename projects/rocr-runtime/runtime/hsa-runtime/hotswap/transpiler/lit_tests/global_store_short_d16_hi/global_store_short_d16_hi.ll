; RUN: %raise_cli %global_store_short_d16_hi_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 \
; RUN:     --emit-ir=global_store_short_d16_hi_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the `global_store_short_d16_hi` /
; `global_store_d16_hi_b16` half-register store.  Companion to
; `ds_store_b16_d16_hi/ds_store_b16_d16_hi.ll` — same "upper 16
; bits of the source VGPR" semantics, different address space
; (addrspace(1) here, addrspace(3) for the DS sibling).
;
; INVARIANTS PINNED:
;
;   1. The lift surfaces the UPPER 16 bits of the source VGPR
;      (bits [31:16]) via `lshr i32 %src, 16` followed by
;      `trunc i32 ... to i16`.  The handler's value-name
;      `d16hi_shift` is the canonical breadcrumb on the lshr.
;
;   2. The 16-bit value is stored to global memory
;      (addrspace(1)).
;
;   3. The store is `i16`-wide (NOT i32).  A regression that
;      collapsed the lift to a full-dword store would still
;      compile but would clobber 16 bits of unrelated global
;      state beyond the target half-word.
;
; NEGATIVE PINS:
;
;   * NO `trunc i32 %{{.*}} to i16` against the source VGPR
;     without a preceding `lshr 16` — the pre-fix regression
;     shape.  The positive pin below requires `lshr 16` to
;     come first, and the negative pin below explicitly forbids
;     the no-shift form against the raw source.
;
;   * NO `store i32` or `store i8` to addrspace(1) — width
;     regressions.

; CHECK-LABEL: define amdgpu_kernel void @global_store_short_d16_hi_kernel(

; The defining lift pattern: lshr-16 then trunc-to-i16 with the
; canonical breadcrumb value-name `d16hi_shift` on the lshr.
; CHECK-DAG: %d16hi_shift = lshr i32 %{{.+}}, 16
; CHECK-DAG: trunc i32 %d16hi_shift to i16

; The store is i16-wide and lands in addrspace(1) (global).
; CHECK: store i16 %{{.+}}, ptr addrspace(1) %{{[^,]+}}

; No full-dword or byte store to global for this instruction.
; CHECK-NOT: store i32 %{{.+}}, ptr addrspace(1) %
; CHECK-NOT: store i8 %{{.+}}, ptr addrspace(1) %
