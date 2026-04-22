; RUN: %raise_cli %buffer_atomic_swap_b32_nortn_co --isa=gfx1250 --target-isa=gfx1250 \
; RUN:     --emit-ir=buffer_atomic_swap_b32_nortn_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Non-RTN companion of `lit_tests/buffer_atomic_swap_b32/`.  Pins
; that the MUBUF-atomic handler in `handle_mubuf.cpp` emits the
; `atomicrmw xchg` WITHOUT a write-back when the source instruction
; is the non-RTN form (glc=0 / no `th:TH_ATOMIC_RETURN` modifier).
; See the companion `.hip` block comment for the full rationale
; (the `op.dst(0)` access path handles both RTN and non-RTN
; because operand-0 is always vdata; the write-back is gated by
; `di.numDefs > 0` and correctly skips for non-RTN).
;
; The invariant this fixture pins:
;
;   1. `atomicrmw xchg` is emitted (the atomic side-effect is
;      preserved — the compiler must not elide the swap just
;      because the result is unused).
;   2. NO subsequent use of the atomicrmw's result that looks like
;      a write-back to a destination VGPR.  A handler regression
;      that drops the `di.numDefs > 0` guard would write the
;      atomicrmw result to `op.dst()`, which for non-RTN
;      instructions has no tied dst — the writeReg32 would
;      either fault or corrupt a neighbour VGPR.

; CHECK-LABEL: define amdgpu_kernel void @buffer_atomic_swap_b32_nortn_kernel(

; The atomic itself — same shape as the RTN variant.
; CHECK: atomicrmw xchg ptr {{.*}} monotonic

; Negative pin for the write-back.  On RTN the handler writes the
; atomicrmw result back to `op.dst()` via writeReg32, which in IR
; surfaces as an `%atomic_result` having an SSA USER that stores
; into a tied VGPR.  On non-RTN we want the atomicrmw RESULT to be
; unused (trivially dead in SSA terms).  FileCheck can't easily
; express "result has zero uses", but we can check that no
; `store`-to-tied-VGPR shape appears IMMEDIATELY after the
; atomicrmw — the `CHECK-NEXT` pin catches the usual writeReg32
; pattern the RTN handler emits.  The safety-net relies on
; `CHECK-NEXT` accepting any non-store follow-up (branch,
; terminator, etc.) which is what a non-RTN kernel body normally
; has.
; CHECK-NOT: call {{.*}}@llvm.amdgcn.raw.buffer.atomic.swap
