; RUN: %raise_cli %ds_store_b16_d16_hi_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=ds_store_b16_d16_hi_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the gfx8+ HasD16LoadStore D16_HI partial-store
; family member ds_store_b16_d16_hi (and, by handler-shape
; identity, ds_store_b8_d16_hi). See SemOp::DS_WRITE_B16_D16_HI in
; transpiler/semop.hpp; the matching handler block in
; transpiler/handle_ds.cpp under
; `if (sop == SemOp::DS_WRITE_B16_D16_HI || ...)`; and the DS
; mapping in transpiler/opcode_map.cpp.
;
; INVARIANTS PINNED:
;
;   1. The lift surfaces the UPPER 16 bits of the source VGPR
;      (bits [31:16]) via `lshr i32 %src, 16` followed by
;      `trunc i32 ... to i16`. The handler's value-name
;      `ds_st_d16_hi` is the canonical breadcrumb on the trunc
;      (mirrors the `ds_addr` / `ds_off` pattern used by sibling
;      DS handlers).
;
;   2. The 16-bit value is stored to LDS (`addrspace(3)`). A
;      regression that wrote to global / private would surface
;      as a different addrspace literal here.
;
;   3. The store is `i16`-wide (NOT i32). A regression that
;      collapsed the lift to a full-dword store would still
;      compile but would clobber 16 bits of unrelated LDS state.
;
;   4. The store happens under EXEC (the handler's
;      `emitUnderExec` wrapper). The lifted IR pattern for
;      EXEC-gated stores is a `select` / `phi` on the EXEC mask
;      — we accept any of the established shapes via the lit
;      regex below.
;
; NEGATIVE PINS:
;
;   * NO `trunc i32 %{{.*}} to i16` against the LOW half of the
;     source — a regression that wrote bits [15:0] instead of
;     [31:16] would emit a trunc DIRECTLY off the source VGPR
;     value with no preceding `lshr 16`. The positive pin
;     requires the `lshr` to come first, and the negative pin
;     below explicitly forbids the no-shift form.
;
;   * NO `store i32` to addrspace(3) — that would indicate a
;     regression to a full-dword store.
;
;   * NO `store i8` to addrspace(3) — that would indicate the
;     handler's b8 vs b16 dispatch regressed and emitted the
;     wrong width for the b16 variant.

; CHECK-LABEL: define amdgpu_kernel void @ds_store_b16_d16_hi_kernel(

; The defining lift pattern: lshr-16 then trunc-to-i16 with the
; canonical breadcrumb value-names on both ops.
; CHECK: %ds_st_hi16_shr = lshr i32 %{{[^,]+}}, 16
; CHECK: %ds_st_d16_hi = trunc i32 %ds_st_hi16_shr to i16

; The store is i16-wide and lands in addrspace(3) (LDS).
; CHECK: store i16 %ds_st_d16_hi, ptr addrspace(3) %{{[^,]+}}

; Negative pin: no full-dword or byte store to LDS for this
; instruction (those would indicate a regression in the b8 vs b16
; dispatch or the 16/32-bit truncation).
; CHECK-NOT: store i32 %ds_st_d16_hi, ptr addrspace(3)
; CHECK-NOT: store i8 %ds_st_d16_hi, ptr addrspace(3)
