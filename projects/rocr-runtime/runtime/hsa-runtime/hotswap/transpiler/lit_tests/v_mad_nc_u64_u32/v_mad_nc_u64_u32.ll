; RUN: %raise_cli %v_mad_nc_u64_u32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_mad_nc_u64_u32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_mad_nc_u64_u32. Unsigned sibling of
; `v_mad_nc_i64_i32.ll` — differs only in the widening direction
; (zext i32 → i64 rather than sext).  See that fixture for the full
; handler / decoder cross-reference; the handler body lives in
; transpiler/handle_valu.cpp under
;   `if (sop == SemOp::V_MAD_NC_U64_U32) { ... }`.

; CHECK-LABEL: define amdgpu_kernel void @v_mad_nc_u64_u32_kernel(

; Two zero-widenings of the 32-bit factors (NOT sext — that would
; mean we accidentally routed through the signed handler arm).
; CHECK: zext i32 %{{[^ ]+}} to i64
; CHECK: zext i32 %{{[^ ]+}} to i64

; Widening multiply.
; CHECK: mul {{.*}}i64 %{{[^,]+}}, %{{[^,]+}}

; Canonical accumulator add with the unsigned sibling's breadcrumb
; (`vmad_nc_u64`, mirroring `vmad_co64` from V_MAD_CO_U64_U32).
; CHECK: %vmad_nc_u64 = add {{.*}}i64

; Negative: no accidental routing through the signed arm.
; CHECK-NOT: %vmad_nc_i64 = add {{.*}}i64

; Negative: no with-overflow intrinsic CALL-site (same rationale as
; the signed fixture — "nc" = no carry, so deliberately plain add).
; The `call` keyword pins this to actual use-sites rather than
; module-level `declare` lines the LLVM textual writer emits even
; for unused intrinsics pulled in transitively by other passes.
; CHECK-NOT: call {{.*}}@llvm.umul.with.overflow
; CHECK-NOT: call {{.*}}@llvm.uadd.with.overflow

; Negative: no saturating-add — this fixture's inline `asm volatile`
; encodes `clamp = 0`, so the handler must take the plain `add i64`
; fast-path.  The `clamp = 1` encoding is a raise-time refusal
; today (see `handle_valu.cpp`'s V_MAD_NC_* block comment for the
; `llvm.uadd.sat.i64` upgrade path when a corpus producer
; surfaces); any appearance of the saturating intrinsic here
; would mean the handler silently promoted without us noticing.
; CHECK-NOT: call {{.*}}@llvm.uadd.sat.i64
