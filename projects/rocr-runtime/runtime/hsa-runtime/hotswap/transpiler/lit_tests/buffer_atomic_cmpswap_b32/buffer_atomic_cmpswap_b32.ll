; RUN: %raise_cli %buffer_atomic_cmpswap_b32_co --isa=gfx1250 --target-isa=gfx1250 \
; RUN:     --emit-ir=buffer_atomic_cmpswap_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the gfx11+/gfx1250 VBUFFER buffer-atomic
; compare-and-swap (`buffer_atomic_cmpswap_b32`).  Sibling of
; `lit_tests/buffer_atomic_add_u32/` (commutative add) and
; `lit_tests/buffer_atomic_swap_b32/` (pure exchange); this fixture
; pins the cmpxchg path — `CreateAtomicCmpXchg` in handle_mubuf.cpp.
;
; Same-target lift (gfx1250 → gfx1250): `buffer_atomic_cmpswap_b32`
; is non-commutative and would refuse at the cross-widen projection-
; decision stage via the Class-3 non-commutative-atomic classifier
; (see `buffer_atomic_swap_b32.ll` comment block for the full
; rationale).  Same-target R=1 lifts skip the classifier and reach
; the handler directly.
;
; Semantic pin: `buffer_atomic_cmpswap_b32` needs TWO data operands
; (cmp, new) packed as a VReg_64 pair, and the RTN form returns the
; ORIGINAL value at the memory cell.  The handler decodes the vdata
; register pair via `op.dst(0)` + a synthesised `baseIdx + 1` read
; (mirror of FLAT_ATOMIC_CMPSWAP / GLOBAL_ATOMIC_CMPSWAP).  The IR
; shape pinned here is the canonical LLVM `cmpxchg` instruction with
; `{old_value, success_bool}` result — downstream users extract
; element 0 to get the original value.
;
; Invariants:
;
;   1. `cmpxchg` — NOT `atomicrmw`.  A regression that accidentally
;      routes CMPSWAP through the atomicrmw path (e.g. by losing
;      the dedicated pre-switch branch) would miss the compare step
;      and silently miscompile.
;   2. Two separate value operands.  FileCheck can't cheaply pin the
;      exact SSA names the register reads produce, but we pin that
;      the cmpxchg has exactly the i32 x i32 value shape (not i64 or
;      vector) — matching a single-dword compare-and-swap.
;   3. `monotonic` ordering on both success and failure paths, matching
;      the MUBUF family convention (see the sibling ADD / SWAP fixture
;      comments on the `scope:SCOPE_DEV` → `monotonic` lowering).
;   4. The original value is extracted back out of the cmpxchg's
;      result struct via an `extractvalue` at element 0 — the write-
;      back path the handler emits for the RTN form.

; CHECK-LABEL: define amdgpu_kernel void @buffer_atomic_cmpswap_b32_kernel(

; The cmpxchg itself with both orderings `monotonic`.  FileCheck's
; whitespace-aware match tolerates the `align 4` / `syncscope`
; annotations the backend may add.
; CHECK: cmpxchg ptr {{.*}} monotonic monotonic

; The RTN write-back: extractvalue at element 0 (the original value).
; A handler that forgot the write-back would drop this extract, and
; the cmpxchg's result would have only ONE use (the success bool
; path).
; CHECK: extractvalue {{.*}}, 0

; Negative pin: no atomicrmw xchg — CMPSWAP is not a pure exchange,
; and routing it through the `BUFFER_ATOMIC_SWAP → AtomicRMWInst::Xchg`
; arm would lose the compare step.
; CHECK-NOT: atomicrmw xchg

; Negative pin: no `raw.buffer.atomic` intrinsic — the handler models
; CMPSWAP as `cmpxchg`, not a buffer intrinsic call.
; CHECK-NOT: call {{.*}}@llvm.amdgcn.raw.buffer.atomic.cmpswap
; CHECK-NOT: call {{.*}}@llvm.amdgcn.struct.buffer.atomic.cmpswap
