; RUN: %raise_cli %s_load_b32_scale_offset_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=s_load_b32_scale_offset_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the `scale_offset` (CPol::SCAL) modifier on an
; SGPR-offset `s_load_b32`.  Pins that the handler honours
; `di.hasScaleOffset` by scaling the zero-extended SGPR offset by the
; load's data-type size (4B for b32) BEFORE adding it to the base.
;
; The handler lives in transpiler/handle_smem.cpp under the
; dword-granular `S_LOAD_B*` block; the scaling decision is keyed on
; `di.hasScaleOffset` (populated by `decodeScaleOffset` in
; transpiler/decode.cpp from the CPol::SCAL bit).  The SemOp lives
; in transpiler/semop.hpp under SMEM.
;
; Why this fixture matters: before the fix, the SGPR-offset arm
; emitted a raw-byte `getelementptr i8, ptr, i64 <zext offset>` that
; was numerically correct only for `blockIdx.x == 0` / offset == 0.
; Every multi-block launch that indexed `mask[blockIdx.x]` through
; the `scale_offset` shape produced an off-by-(N-1)*3 byte offset
; and loaded the wrong dword.  The regression surfaced on the
; `s_and_saveexec_b32` compare_correctness probe as a multi-block
; WRONG pattern (single-block-matches, multi-block-mismatches) that
; no single-shape IR inspection would have caught.  The fix
; multiplies the offset by `loadBytes` (= 4 for b32) when
; `hasScaleOffset` is set, mirroring the FLAT/GLOBAL sibling paths
; in `flat_addr.cpp`.  This fixture pins that scaling.
;
; The FLAT/GLOBAL sibling fixtures
; (`global_load_ushort_saddr/` + the implicit coverage via corpus
; sweeps) were already pinning their own `hasScaleOffset` shape;
; the SMEM path had no equivalent fixture, which is exactly how the
; regression slipped in.  This fixture closes that gap.

; CHECK-LABEL: define amdgpu_kernel void @s_load_b32_scale_offset_kernel(

; The SGPR offset is zero-extended to i64 before any arithmetic,
; named by the handler as `smem_roff`.  Name pinning is the back-
; reference from fixture to handler.
; CHECK: %smem_roff = zext i32 %{{[^ ,]+}} to i64

; The `scale_offset` scaling: the zero-extended offset is multiplied
; by the element size (4 bytes for s_load_b32).  The product carries
; the `smem_roff_scaled` name from the handler.  A regression that
; dropped this multiplication would produce a raw-byte GEP and load
; the wrong dword on any non-zero offset.
; CHECK: %smem_roff_scaled = mul i64 %smem_roff, 4

; The address arithmetic: the scaled offset goes into a byte-unit
; inbounds GEP off the kernarg-derived pointer.  We pin the use of
; the scaled value (not the raw `smem_roff`) in the GEP index slot
; because the GEP's element type is `i8` — passing the unscaled
; value here would be the exact regression we guard against.
; CHECK: getelementptr inbounds i8, ptr addrspace(1) %{{[^ ,]+}}, i64 %smem_roff_scaled

; The final scalar load lands at the scaled address.  Name binding
; (`smem_load`) is the handler's convention.
; CHECK: %smem_load = load i32, ptr addrspace(1) %{{[^ ,]+}}, align 4

; Negative pin: the raw-byte shape must NOT appear.  Specifically,
; the GEP must not consume `smem_roff` directly (unscaled) — that
; would be the pre-fix miscompile.  The unscaled value is emitted
; (as a named i64) but only as an input to the mul; it must not
; reach a GEP.  Pinning on the literal GEP-with-unscaled-offset
; would be over-constraining since GEPs over i8 are perfectly
; legitimate in the lift; instead we pin on the absence of a GEP
; whose index is a register named `smem_roff` (the unscaled value).
; CHECK-NOT: getelementptr inbounds i8, ptr addrspace(1) %{{[^ ,]+}}, i64 %smem_roff{{$}}
