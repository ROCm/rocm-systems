; RUN: %raise_cli %s_sub_nc_u64_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=s_sub_nc_u64_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for s_sub_nc_u64. Pins that the SOP2 64-bit no-carry
; subtract lowers to a single `sub i64`. The handler lives in
; transpiler/handle_sop2.cpp under
; `if (sop == SemOp::S_SUB_NC_U64) { ... }`; the SemOp lives in
; transpiler/semop.hpp under SOP2.
;
; The "nc" suffix matters: the gfx12 form intentionally does NOT
; update SCC (see SOPInstructions.td around line 661 — `S_SUB_U64`
; is defined outside the surrounding `let Defs = [SCC]` block). A
; regression that emits a `usub.with.overflow` intrinsic (or any
; SCC-writing variant) would defeat that, so we negative-CHECK
; both shapes.

; CHECK-LABEL: define amdgpu_kernel void @s_sub_nc_u64_kernel(

; The lifted body must contain a single i64 sub — the handler's
; `ssub64` value-name is the canonical breadcrumb (mirrors the
; `sadd64` / `smul64` siblings in the same handler file).
; CHECK: sub {{.*}}i64 %{{[^,]+}}, %{{[^,]+}}

; Negative checks: no SCC-defining intrinsic and no narrower-width
; sub for this kernel. If the handler ever shrinks to 32-bit halves
; or grows an overflow-tracked variant, this fires.
; CHECK-NOT: @llvm.usub.with.overflow
; CHECK-NOT: sub {{.*}}i32
