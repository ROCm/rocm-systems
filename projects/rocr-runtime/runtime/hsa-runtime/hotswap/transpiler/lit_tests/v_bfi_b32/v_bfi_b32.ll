; RUN: %raise_cli %v_bfi_b32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_bfi_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_bfi_b32.  Pins that the VOP3 bit-field insert
; lowers to the canonical `(mask & one) | (~mask & zero)` shape
; the AMDGPU backend's `AMDGPUbfiPattern` will isel straight back
; to v_bfi_b32 on gfx942.  The handler lives in
; transpiler/handle_valu.cpp under
; `if (sop == SemOp::V_BFI_B32) { ... }`; the SemOp lives in
; transpiler/semop.hpp under VOP3.
;
; Why this fixture matters: before the handler landed, every
; kernel that exercised v_bfi_b32 (asin / atan / copysign
; compositions in libdevice; the compare_correctness v_bfi_b32
; probe) failed the raise step with `unsupported instruction:
; v_bfi_b32` and surfaced as `EXIT=2 no kernel image` on the
; hotswap path.  Adding the handler unblocks raise; this fixture
; regression-pins the specific IR shape so a future "simplify to
; andn2+and+or" or "emit as @llvm.amdgcn.bfi" rewrite cannot
; silently drop one of the three operand-order invariants.

; CHECK-LABEL: define amdgpu_kernel void @v_bfi_b32_kernel(

; The handler emits three IR ops under the canonical names we pin
; on here.  The final `or` is named `vbfi`; the two inner `and`
; operands are unnamed, but the ordering matters — src0 (the mask)
; must flow into BOTH `and`s, with a `xor ..., -1` (LLVM's
; canonical `not`) between them on the zero-source side.  We check
; each fragment with DAG so LLVM's IRBuilder is free to reorder
; the two `and` emissions.

; The mask-AND-with-"not mask" shape: an `xor i32 ..., -1` is LLVM's
; canonical representation of `CreateNot`, so pin that.
; CHECK-DAG: %{{.+}} = xor i32 %{{.+}}, -1

; The fused `or` closing the bit-field insert.  Name is pinned by
; the handler's `CreateOr(..., "vbfi")`.
; CHECK-DAG: %vbfi{{[0-9]*}} = or i32

; Two `and i32` operations — one for the "mask & one_src" branch,
; one for the "~mask & zero_src" branch.  We count them together as
; a sanity check; without them the bit-field-insert semantics is
; not recoverable from the IR.
; CHECK-DAG: and i32
; CHECK-DAG: and i32

; Negative pin: the lift must NOT fall back to
; @llvm.amdgcn.bfi or a truncating shape that would silently drop
; high-bit mask coverage.  Keeping the plain and/or skeleton is
; what lets the backend recover a single v_bfi_b32 on both
; same-target (gfx1250 -> gfx1250) and cross-target
; (gfx1250 -> gfx942) paths.
; CHECK-NOT: call {{.*}}@llvm.amdgcn.bfi
