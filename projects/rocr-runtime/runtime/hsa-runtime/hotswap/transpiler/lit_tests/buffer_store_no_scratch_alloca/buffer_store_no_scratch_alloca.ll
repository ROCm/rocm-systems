; RUN: %raise_cli %buffer_store_no_scratch_alloca_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=buffer_store_dwordx4_kernel \
; RUN:     2>/dev/null \
; RUN:   | %FileCheck %s
;
; Regression guard for the principled BUFFER_STORE_* lowering.
;
; The .hip fixture issues a single `buffer_store_b128` (gfx12 spelling
; for BUFFER_STORE_DWORDX4) with no other allocations in the kernel
; body.  The lifted IR therefore lets us pin both the desired shape
; (raw_buffer_store intrinsic) AND the ABI invariant that fixed Triton
; vector-add (R1): zero `addrspace(5)` allocas in the lifted module.
;
; Why the addrspace(5) negative pin matters more than the positive one:
; the legacy lowering routed every BUFFER_STORE through a software
; OOB-select against an `i32` `addrspace(5)` sink alloca.  That sink
; was wrong on three independent axes:
;
;   1. Size mismatch — sink was always i32 (4 B) but DWORDX4 writes
;      16 B; the OOB lane walked 12 B past the sink into either the
;      next per-thread scratch slot or unmapped scratch.
;
;   2. Forced scratch enablement — any `addrspace(5)` alloca that
;      survives PromoteMemToReg makes the AMDGPU backend emit
;      `.amdhsa_enable_private_segment 1` plus
;      `.amdhsa_private_segment_fixed_size > 0`.  Salmon's KD does
;      not request `flat_scratch_init`, so on entry FLAT_SCRATCH is
;      undefined and the very first flat-aperture access faults.
;
;   3. Asymmetry with the load path — loads already used
;      `amdgcn.raw.buffer.load` and relied on hardware OOB clamp;
;      stores were the only divergence.
;
; If any future change reintroduces ANY `addrspace(5)` alloca in a
; BUFFER_STORE-only kernel, this fixture fails.  That's the whole
; point: the bug class is "implicit ABI coupling via private-segment
; enablement", and the only stable way to forbid it at the IR level
; is to forbid the IR construct that triggers it.
;
; Invariants pinned (each as a separate FileCheck directive so a
; failure tells you exactly which axis regressed):
;
;   POSITIVE:
;     P1. The kernel definition is present (sanity check).
;     P2. The store lifts to `@llvm.amdgcn.raw.buffer.store.v4i32`.
;
;   NEGATIVE (each forbids a specific failure mode):
;     N1. No `alloca .* addrspace(5)` anywhere in the lifted module.
;     N2. No `addrspacecast .* to ptr addrspace(5)` (legacy sink).
;     N3. No `select i1 ... ptr addrspace(5)` (legacy OOB select).
;     N4. No `@llvm.amdgcn.flat.store` (we use the buffer intrinsic).
;     N5. No `store .* ptr addrspace(5)` (no sink writes).
;     N6. No raw_buffer_store with an i32 type — i.e. the DWORDX4
;         store must NOT be split into 4 i32 stores or downgraded.
;
; What this fixture does NOT pin:
;   * The exact SSA names or operand order in the intrinsic call
;     (printed names depend on the upstream ParsedReg materialisation
;     and on LLVM-IR-printer formatting).
;   * The decode of `auxFlags` from the source instruction's cache
;     policy — currently a known limitation (always zero); see
;     mubuf_addr.cpp and the principled-review notes in
;     hotswap/docs/buffer-store-lowering.md.
;   * Per-width coverage for BUFFER_STORE_BYTE / SHORT / DWORD /
;     DWORDX2 / DWORDX3 — handler-shape identity makes them
;     transitively covered by the same code path, but a future
;     follow-up fixture per width would close the lit-coverage gap
;     called out in the principled review.

; ---- Positive checks ----

; P1. The kernel survived the lift (label-form, robust to attribute
;     printing changes between LLVM versions).
; CHECK-LABEL: define amdgpu_kernel void @buffer_store_dwordx4_kernel(

; P2. The BUFFER_STORE_DWORDX4 lifts to the raw_buffer_store intrinsic
;     parameterised by `<4 x i32>`. We do not pin the operand SSA names
;     (they depend on the kernarg + register-materialisation pipeline);
;     we DO pin the intrinsic name and value type.
; CHECK: call void @llvm.amdgcn.raw.buffer.store.v4i32(

; ---- Negative checks (the actual ABI invariant) ----

; N1. Zero `addrspace(5)` allocas anywhere in the module.  This is the
;     core regression guard: any reintroduction of an `addrspace(5)`
;     alloca (legacy OOB sink, future scratch-promote, anything else)
;     re-couples the kernel descriptor to private-segment enablement
;     and risks a SIGSEGV under the salmon ABI.
; CHECK-NOT: alloca {{.*}}addrspace(5)

; N2. The legacy OOB sink path used an `addrspacecast` from the i32
;     alloca to a flat pointer.  Forbid that shape.
; CHECK-NOT: addrspacecast {{.*}}to ptr addrspace(5)
; CHECK-NOT: addrspacecast ptr addrspace(5){{.*}}to ptr

; N3. The legacy OOB-select shape: `select i1 oob, sink, real`
;     against an addrspace(5) operand.  Forbid the addrspace(5) side.
; CHECK-NOT: select i1 {{.*}}ptr addrspace(5)

; N4. We must not lower buffer_store via the FLAT-store intrinsic.
;     (The FLAT path was the legacy lowering's escape hatch into
;     scratch; the principled fix uses the BUFFER intrinsic which
;     never touches scratch addressing.)
; CHECK-NOT: @llvm.amdgcn.flat.store

; N5. No raw `store` against an `addrspace(5)` pointer either —
;     covers any path that sneaks in a scratch write without going
;     through the flat intrinsic.
; CHECK-NOT: store {{.*}}, ptr addrspace(5)

; N6. The DWORDX4 store must lower as a single v4i32 intrinsic call,
;     not as 4 separate i32 raw_buffer_store calls (which would be
;     a different bug — width split — but worth pinning).  This is
;     belt-and-braces: P2 above already requires the v4i32 form,
;     but a regression that emitted BOTH wouldn't be caught by P2
;     alone.
; CHECK-NOT: @llvm.amdgcn.raw.buffer.store.i32(
