; RUN: %raise_cli %buffer_atomic_cmpswap_b32_nortn_co --isa=gfx1250 --target-isa=gfx1250 \
; RUN:     --emit-ir=buffer_atomic_cmpswap_b32_nortn_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Non-RTN companion of `lit_tests/buffer_atomic_cmpswap_b32/`.  Pins
; that the MUBUF-atomic handler in `handle_mubuf.cpp` emits the
; `cmpxchg` without the `extractvalue ..., 0` write-back when the
; source instruction is the non-RTN form.  See the companion
; `.hip` block comment for the full rationale.
;
; The cmp/new value-pair read uses `op.dst(0)` + `baseIdx + 1`
; synthesis in the handler — same path as the RTN variant, because
; operand 0 (vdata) is present in both forms, just differing in
; whether it's tied to a destination.  What differs:
;
;   1. RTN: `cmpxchg` result is `extractvalue ..., 0`'d and written
;      back to `op.dst()` via writeReg32 — the lit fixture for the
;      RTN variant pins both the `cmpxchg` and the `extractvalue 0`.
;   2. Non-RTN (this fixture): `cmpxchg` still emits (the compare-
;      and-exchange atomic side-effect is preserved), but the
;      `di.numDefs > 0` guard skips the write-back.  No
;      `extractvalue ..., 0` should appear in the lifted IR for
;      this kernel.

; CHECK-LABEL: define amdgpu_kernel void @buffer_atomic_cmpswap_b32_nortn_kernel(

; The cmpxchg itself — same shape as the RTN variant.
; CHECK: cmpxchg ptr {{.*}} monotonic monotonic

; Negative pin for the RTN write-back.  If the handler accidentally
; routes non-RTN cmpswap through the RTN write-back path, an
; `extractvalue ..., 0` would appear after the cmpxchg.  For the
; non-RTN form the cmpxchg's result must be TRIVIALLY UNUSED in
; SSA — no extract, no writeReg.
; CHECK-NOT: extractvalue {{.*}}, 0

; Negative pin: not routed through the atomicrmw / buffer-
; intrinsic path.  Same conventions as the RTN fixture.
; CHECK-NOT: atomicrmw xchg
; CHECK-NOT: call {{.*}}@llvm.amdgcn.raw.buffer.atomic.cmpswap
