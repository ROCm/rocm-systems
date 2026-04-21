; RUN: %raise_cli %s_atomic_dec_co --isa=gfx950 --target-isa=gfx942 \
; RUN:     --emit-ir=s_atomic_dec_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the gfx8+ scalar-cache wrap-decrement
; (`s_atomic_dec sDST, sBASE, sOFFSET`). The kernel is dispatched via
; the shared scalar-atomic block at the bottom of
; transpiler/handle_smem.cpp, parameterised on
; `AtomicRMWInst::BinOp` — S_ATOMIC_DEC picks `UDecWrap`, matching
; AMDGPU's wrap-at-zero HW semantics exactly
;   new = (old == 0 || old > src) ? src : old - 1
; which landed as a first-class binop in LLVM 19.
;
; Source → target shape: gfx950 (AITER corpus source ISA) → gfx942
; (the hotswap race box).  Mirrors what
; BatchRaise.AiterGfx950 now threads through to `raiseToIR` via the
; newly-added `compilationTargetIsa` parameter, so that a change
; affecting either ISA's capability branches fails here rather than
; silently at the batch-raise boundary.
;
; This fixture regression-pins the failure mode that motivated the
; change: the 12 AITER `bf16gemm_*_splitk_clean.co` kernels failed to
; raise with `unsupportedOpcode [SMEM] (s_atomic_dec)` because
; semop.hpp + opcode_map.cpp had no entry for the opcode and
; handle_smem.cpp had no dispatch arm, despite the gfx942 ISA itself
; supporting the instruction natively (so this is purely a lift-side
; gap, not a cross-target capability mismatch).
;
; Invariants pinned:
;
;   1. The atomic lifts to `atomicrmw udec_wrap` — NOT to
;      `atomicrmw sub`, `atomicrmw add`, or any
;      `@llvm.amdgcn.s.atomic.*` intrinsic. The sub/add shapes would
;      silently miscompile the wrap-at-zero semantics into a regular
;      decrement (the AITER barrier value wraps from 1 → 0 and back
;      to the source threshold; a plain `sub` would underflow to
;      0xFFFFFFFF on the next decrement and break the `old == 1`
;      "am I last?" check).
;   2. AtomicOrdering is `monotonic` — the scalar-cache atomics carry
;      no implicit acquire/release, matching the existing
;      S_ATOMIC_SWAP handler.
;   3. Address space is `1` (global / default SMEM base-pointer
;      address space) — the same `ptr addrspace(1)` the dword-granular
;      S_LOAD_B* path produces, so downstream passes see a uniform
;      SMEM pointer type.
;   4. `align 4` — dword width, matching the hardware element size.
;
; What we don't pin: the exact SSA name of the pointer or the
; threshold value (both depend on kernarg lowering upstream); the
; `s_waitcnt lgkmcnt(0)` token after the atomic (codegen-side, lift
; drops it).

; CHECK-LABEL: define amdgpu_kernel void @s_atomic_dec_kernel(

; The atomic itself: udec_wrap, monotonic, align 4.  Wrap threshold
; and pointer SSA names float so the test is robust to kernarg-decode
; refactors.
; CHECK: atomicrmw udec_wrap ptr addrspace(1) {{.*}} monotonic, align 4

; Negative pins: no plain decrement/subtraction shape, and no
; s_atomic.* intrinsic dispatch (handler is required to go through
; the generic atomicrmw family so the gfx942 backend picks the
; right native encoding for the target).
; CHECK-NOT: atomicrmw sub
; CHECK-NOT: atomicrmw add
; CHECK-NOT: call {{.*}}@llvm.amdgcn.s.atomic
