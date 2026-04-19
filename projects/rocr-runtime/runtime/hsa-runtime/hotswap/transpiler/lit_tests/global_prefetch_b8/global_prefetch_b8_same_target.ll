; RUN: %raise_cli %global_prefetch_b8_co --isa=gfx1250 \
; RUN:     --target-isa=gfx1250 --emit-ir=global_prefetch_b8_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Lift fixture for FLAT `global_prefetch_b8` — the same-target
; (gfx1250 → gfx1250) intrinsic-emit path.  Pins the principled
; lift in transpiler/handle_flat.cpp under
; `SemOp::GLOBAL_PREFETCH_B8` when `ctx.targetIsa.hasTensorOps` is
; true (the proxy for `FeatureVmemPrefInsts` / `+vmem-pref-insts`,
; both of which are co-resident with the gfx1250-only
; `+gfx1250-insts` feature on the only sub-target that owns the
; encoding).  Companion fixture to `global_prefetch_b8.ll`, which
; pins the cross-target (gfx942) loud refusal.
;
; The FLATInstructions.td `FLAT_Prefetch_Pseudo` multiclass yields
; two operand-shape variants — plain (vaddr:VGPR_64, off:imm,
; cpol:imm) and SADDR (saddr:SGPR_64, vaddr:VGPR_32, off:imm,
; cpol:imm), both with `has_vdst = 0` (the prefetch returns no
; data, only injects a cache-line warm-up hint).  The fixture's
; HIP source uses the clang builtin
; `__builtin_amdgcn_global_prefetch` which lowers to the VFLAT
; 0x05D real (`global_prefetch_b8`); for kernarg-derived pointers
; the AMDGPU disassembler prints the SADDR variant
; `global_prefetch_b8 v0, s[0:1] [offset:N]`).
;
; The matching LLVM intrinsic (IntrinsicsAMDGPU.td:3211, added in
; PR #150466) is
;
;   void llvm.amdgcn.global.prefetch(
;       ptr addrspace(1)        %gaddr,    // captures(none)
;       i32 immarg              %cpol)
;
; The handler:
;   * decodes the global address — for SADDR it issues
;     `add i64 saddr, zext i32 vaddr → i64 (named `saddr_vaddr`)`,
;     for plain it reuses the FLAT decode helper that produces a
;     `i64` from `VGPR_64`;
;   * casts the resulting `i64` to `ptr addrspace(1)` via
;     `inttoptr i64 ... to ptr addrspace(1)`;
;   * folds the FLAT `flat_offset` immediate (when non-zero) onto
;     the pointer via a non-inbounds `getelementptr i8, ptr
;     addrspace(1) %p, i64 <off>` (named `prefetch_addr` in the
;     emitted IR).  This matches the AMDGPU backend's expectation
;     that `flat_offset` re-folds back into the VFLAT real's
;     `offset:` field rather than burning a separate ALU
;     instruction;
;   * threads the FLAT `cpol` immediate through as the trailing
;     `i32 immarg`;
;   * does NOT wrap the call in `ctx.emitUnderExec`: the intrinsic
;     carries the EXEC mask implicitly through
;     `IntrInaccessibleMemOrArgMemOnly` — it is semantically a
;     hint with no observable side effect on inactive lanes, so an
;     extra `if-spe-active` guard would gratuitously inflate IR
;     for what hardware executes as a single broadcast hint.
;
; We pin two concrete sub-shapes from the fixture's pair of
; prefetch calls:
;
;   1. The plain (no-offset) call:
;      * `add i64 saddr, zext_voff   → i64 %saddr_vaddr*`
;      * `inttoptr i64 ...           → ptr addrspace(1)`
;      * `call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) ..., i32 8)`
;
;   2. The byte-offset (flat_offset=256) call:
;      * the same SADDR sum
;      * `getelementptr i8, ptr addrspace(1) ..., i64 256` named
;        `prefetch_addr`
;      * `call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %prefetch_addr, i32 8)`
;
; Drift indicators:
;   * If a future LLVM rename swaps the intrinsic name (e.g.
;     `amdgcn.global.prefetch.b8` to expose the width), the IR
;     check fails immediately and pinpoints the rename rather than
;     letting a silently mis-named intrinsic reach the backend.
;   * If the FLAT-offset folding regresses (e.g. handler emits
;     `add i64` instead of `getelementptr i8`), the AMDGPU backend
;     loses the ability to re-fold `offset:` and the regression
;     surfaces as a stray ALU instruction in the disasm.

; First call: SADDR sum with flat_offset == 0 — no GEP, just the
; intrinsic on the inttoptr'd sum.
; IR: %saddr_vaddr{{[0-9]*}} = add i64
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %{{[0-9]+}}, i32 8)

; Second call: SADDR sum with flat_offset == 256 — non-inbounds
; `getelementptr i8` named `prefetch_addr`.
; IR: %saddr_vaddr{{[0-9]*}} = add i64
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %prefetch_addr{{[0-9]*}} = getelementptr i8, ptr addrspace(1) %{{[0-9]+}}, i64 256
; IR: call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %prefetch_addr{{[0-9]*}}, i32 8)

; The intrinsic declaration must match the upstream signature: the
; `captures(none)` attribute on the pointer argument is what
; permits the AMDGPU backend to safely fold the prefetch through
; passes that move pointers.  Drift here would silently weaken the
; aliasing guarantee.
; IR: declare void @llvm.amdgcn.global.prefetch(ptr addrspace(1) captures(none), i32 immarg)
