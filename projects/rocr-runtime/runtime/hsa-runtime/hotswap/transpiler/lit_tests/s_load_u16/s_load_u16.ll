; RUN: %raise_cli %s_load_u16_co --isa=gfx1250 --target-isa=gfx1250 \
; RUN:     --emit-ir=s_load_u16_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Principled lift of the gfx12+ scalar narrow-load family
; (s_load_{u8,i8,u16,i16}), exercising the u16 positive path through
; a runtime-materialised SGPR-pair base pointer. This is the only
; shape the kerneldex corpus exercises today (1 site in Triton's
; `_attn_fwd`); the u8 / i8 / i16 siblings share the handler body
; and differ only in the element type, Align immediate, and
; zext-vs-sext finale.
;
; The handler in `handle_smem.cpp` emits, for an
; `s_load_u16 s*, s[2:3], 0x0` MC pseudo:
;
;   1. `loadSGPR64` of the base SGPR pair, composed from the two
;      i32 dwords already live in the reg-file (these in turn come
;      from the earlier `s_load_b128 s[0:3], s[0:1], 0x0` kernarg
;      preload, which routes through `extractKernargDword`).
;   2. `inttoptr` of the i64 base into `ptr addrspace(1)` — global
;      address space, matching the existing `S_LOAD_B*` convention
;      (no `addrspace(4)` opt-in because we cannot prove the memory
;      is immutable).
;   3. `CreateAlignedLoad` with an EXPLICIT `align 2` (the hardware
;      requirement for s_load_u16). The alignment MUST be pinned in
;      IR — an implicit / under-constrained alignment would regress
;      if LLVM changes the default for i16 loads. This test catches
;      that class of regression.
;   4. `zext` (NOT `sext`) to i32 — U16 is the zero-extending
;      variant; the I16 sibling would pick `sext` through the same
;      handler branch.
;   5. `storeSGPR32` of the extended value back into the dest SGPR.
;
; The IR shape pinned here survives through the AMDGPU backend
; round-trip: on gfx1250, `load i16` over a uniform SGPR-derived
; pointer re-codegens back to `s_load_u16` (identity-preserving
; same-target lift). On gfx942 the backend lowers to VMEM
; (`global_load_ushort`) — semantically correct (broadcast), but
; uniformity-lossy; the handler permits this demotion rather than
; refusing, because `load i16` IR is a semantically well-defined
; lift and the backend's VMEM choice is the ISA-correct lowering
; on a target without scalar narrow loads. See the "Position-α
; permissive lift" design notes in `handle_smem.cpp`.

; CHECK-LABEL: define amdgpu_kernel void @s_load_u16_kernel(

; The narrow load. The key assertions are, in order:
;   * `i16` element type — pins the U16 half-word width (vs the
;     U8 / I8 byte paths that emit i8).
;   * `ptr addrspace(1)` — pins the global address-space choice
;     (consistent with `S_LOAD_B*`; `addrspace(4)` would over-
;     promise immutability).
;   * `align 2` — pins the hardware-mandated alignment. The
;     explicit annotation is the whole point of using
;     `CreateAlignedLoad` over the default-alignment `CreateLoad`.
; CHECK:      %smem_load_h = load i16, ptr addrspace(1) %{{[^ ,]+}}, align 2

; Zero-extension to i32 (NOT sign-extension) — this is the U16
; (unsigned) variant; the I16 sibling routes through `sext` via the
; same handler branch. The name binding (`smem_load_zext`) is the
; handler's convention and is pinned here so a rename breaks the
; fixture loudly.
; CHECK-NEXT: %smem_load_zext = zext i16 %smem_load_h to i32

; Negative assertions: the pre-handler failure surfaced here was a
; generic "Unsupported instruction" (SemOp::Unknown) dispatch miss.
; A regression to that path would produce NEITHER the `load i16`
; NOR the `zext` — `CHECK-NOT` those shapes so a future bug that
; re-routes the narrow-SMEM family to a generic handler gets caught
; at fixture time rather than at corpus sweep time.
;
; Specifically guard against the two adjacent mis-lifts:
;   * A dword-widen-and-mask lift (reads 4 bytes then `and 0xFFFF`)
;     — design-rejected because it risks OOB reads past the end of
;     a 2-byte allocation. A `mul` or `and` against the loaded i16
;     value here would indicate that regression.
;   * A sign-extension for the unsigned variant — would corrupt
;     every high-bit-set u16 value silently. The `zext` pin above
;     is the positive side of this guard.
; CHECK-NOT:  load i32, ptr addrspace(1) %{{.*}}, align 4
; CHECK-NOT:  sext i16 %smem_load_h to i32
