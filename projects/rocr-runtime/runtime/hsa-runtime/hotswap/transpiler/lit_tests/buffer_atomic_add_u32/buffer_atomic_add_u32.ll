; RUN: %raise_cli %buffer_atomic_add_u32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=buffer_atomic_add_u32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the gfx11+/gfx1250 VBUFFER buffer-atomic add
; (`buffer_atomic_add_u32`).  The kernel is dispatched via the
; existing AtomicRMW lowering branch in transpiler/handle_mubuf.cpp
; (`if (sop >= BUFFER_ATOMIC_ADD && sop <= BUFFER_ATOMIC_PK_ADD_F16)
; { ... AtomicRMWInst::Add ... }`); the new bit is the
; `VBUF4(BUFFER_ATOMIC_ADD, BUFFER_ATOMIC_ADD)` entry in
; transpiler/opcode_map.cpp that maps the gfx12 `_VBUFFER_*`
; addressing-mode opcodes to the same SemOp the legacy `_OFFSET /
; _OFFEN / _IDXEN / _BOTHEN` MUBUF forms already mapped to.
;
; This regression-pins the failure mode that motivated the change:
; scope_discovery___sum_bitmatrix_rows refused on
; `buffer_atomic_add_u32 v0, v1, s[4:7], null offen` with
; `unsupportedOpcode [MUBUF]` because no opcode_map entry existed
; for `BUFFER_ATOMIC_ADD_VBUFFER_OFFEN`, despite the SemOp +
; handler being present.
;
; Invariants:
;
;   1. The buffer atomic lifts to an `atomicrmw add` IR instruction
;      (not to an `@llvm.amdgcn.raw.buffer.atomic.add` intrinsic
;      call) — the existing handler models all the commutative
;      buffer atomics as AtomicRMW so the gfx942 backend can
;      re-lower to whichever native form fits the target ISA.
;   2. Ordering is `monotonic` — the `scope:SCOPE_DEV` modifier
;      on gfx12 maps to monotonic in the existing handler (no
;      acquire/release/seq_cst implied by SCOPE_DEV alone).
;   3. NO refusal diagnostic in stderr — exit 0, not the legacy
;      `unsupportedOpcode [MUBUF]` path.
;
; What we don't pin: the exact pointer-construction shape upstream
; of the atomic (the SRD-to-pointer translation goes through the
; raw_buffer descriptor decoder which has its own dedicated
; coverage); nor the address space of the resulting pointer (the
; backend chooses based on the SRD).

; CHECK-LABEL: define amdgpu_kernel void @buffer_atomic_add_u32_kernel(

; The atomic itself: `atomicrmw add` with monotonic ordering.
; Using a loose match on the pointer + value operands because their
; SSA names depend on the kernarg lowering pipeline upstream.
; CHECK: atomicrmw add ptr {{.*}} monotonic

; Negative pin: no `raw.buffer.atomic` intrinsic should appear —
; the handler models all commutative buffer atomics as AtomicRMW
; rather than as a buffer-intrinsic call.  A regression that
; routed BUFFER_ATOMIC_ADD through the intrinsic path would be a
; semantic change worth catching here.
; CHECK-NOT: call {{.*}}@llvm.amdgcn.raw.buffer.atomic.add
; CHECK-NOT: call {{.*}}@llvm.amdgcn.struct.buffer.atomic.add
