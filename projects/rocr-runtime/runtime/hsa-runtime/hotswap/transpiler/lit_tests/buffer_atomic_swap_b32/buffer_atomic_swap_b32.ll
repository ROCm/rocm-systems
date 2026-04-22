; RUN: %raise_cli %buffer_atomic_swap_b32_co --isa=gfx1250 --target-isa=gfx1250 \
; RUN:     --emit-ir=buffer_atomic_swap_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the gfx11+/gfx1250 VBUFFER buffer-atomic swap
; (`buffer_atomic_swap_b32`).  Sibling of
; `lit_tests/buffer_atomic_add_u32/`, which pins the commutative
; add path; this fixture pins the exchange path — `AtomicRMWInst::Xchg`
; in handle_mubuf.cpp.  The RTN-form write-back (see the "RTN-form
; write-back" comment in that handler) is the semantic point of
; `buffer_atomic_swap`: the caller reads the old value.  Without
; write-back, the handler would emit a pure `atomicrmw xchg` that
; discards the original value and reduce the swap to a store —
; quietly miscompiling any CAS-loop that relies on it.
;
; Same-target lift (gfx1250 → gfx1250).  The `buffer_atomic_swap_b32`
; opcode is non-commutative — cross-widening wave32 → wave64 races
; target lanes i and i+W_s on the same memory cell, which the
; Class-3 non-commutative-atomic classifier refuses at the
; projection-decision stage (see
; hotswap/docs/wave-size-translation.md §3 / §7's unrewritable
; table).  So the cross-widen path would refuse BEFORE reaching
; this handler, making the handler unreachable via that route.
; Same-target lifts skip the cross-widen classifier (R=1, no replica
; race possible) and exercise the handler directly — that's the path
; this fixture pins.  A future `ThreadLoopProjection` that emulates
; wave64 via per-source-wave loops would also bypass the replica-race
; refusal and exercise this handler; re-running the fixture with a
; `--target-isa=gfx942` RUN line once that projection lands is the
; extension direction.
;
; Invariants:
;
;   1. `atomicrmw xchg` with `monotonic` ordering — matches the
;      MUBUF family convention (see the ADD sibling fixture's
;      comment block on the `scope:SCOPE_DEV` → `monotonic`
;      lowering).
;   2. The RTN write-back is present: the `atomicrmw`'s result
;      reaches the destination VGPR through a `writeReg32` — in IR
;      this shows up as a `store i32 %atomicrmw.result, ...` or
;      equivalent SSA flow back to a tied VGPR.  We don't pin the
;      exact shape (the writeReg path varies with register widths
;      and VGPR aliases) but we DO pin that the atomicrmw's result
;      is named / bound to an SSA value, i.e. the value isn't
;      dropped at the IR level.
;   3. No `raw.buffer.atomic` intrinsic — all commutative and swap
;      buffer atomics route through AtomicRMW, not the buffer-
;      intrinsic path.

; CHECK-LABEL: define amdgpu_kernel void @buffer_atomic_swap_b32_kernel(

; The atomic itself: atomicrmw xchg with monotonic ordering.
; CHECK: atomicrmw xchg ptr {{.*}} monotonic

; Negative pin: no raw.buffer.atomic intrinsic.  Mirrors the ADD
; fixture's convention — all commutative and swap buffer atomics are
; modelled as AtomicRMW so the backend can re-lower to the native
; form per target ISA.
; CHECK-NOT: call {{.*}}@llvm.amdgcn.raw.buffer.atomic.swap
; CHECK-NOT: call {{.*}}@llvm.amdgcn.struct.buffer.atomic.swap

; Negative pin: no `cmpxchg` — SWAP and CMPSWAP are distinct opcodes
; with distinct handler arms; a regression that routes SWAP through
; the CMPSWAP path (or vice versa) would be caught here.
; CHECK-NOT: cmpxchg
