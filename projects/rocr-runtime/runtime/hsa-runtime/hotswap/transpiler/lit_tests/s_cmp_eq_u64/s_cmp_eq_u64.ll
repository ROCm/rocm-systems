; RUN: %raise_cli %s_cmp_eq_u64_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=s_cmp_eq_u64_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the gfx8+ SOPC s_cmp_eq_u64 (and its sibling
; s_cmp_lg_u64 — both are SOPC_CMP_64 in SOPInstructions.td and
; share the same handler-block shape). See SemOp::S_CMP_EQ_U64 in
; transpiler/semop.hpp; the matching handler block in
; transpiler/handle_sopc.cpp under `if (sop == SemOp::S_CMP_EQ_U64)`;
; and the SOPC mapping in transpiler/opcode_map.cpp.
;
; INVARIANTS PINNED:
;
;   1. The kernel signature carries the i64 source-operand arguments
;      directly (`i64 %arg1, i64 %arg2`); no narrowing to i32. This
;      pins the typical SOPC_CMP_64 corpus shape.
;
;   2. The lift emits a single `icmp eq i64`. The handler reads
;      both operands via `op.src64(...)` (i64 SGPR-pair reads,
;      since SOPC_CMP_64 takes two 64-bit operands) and emits
;      `CreateICmpEQ`. A regression that narrowed the operands to
;      i32 — a common shortcut against 64-bit SGPR pairs — would
;      break the corpus's per-thread-mask compares used in
;      tensilelite gemm dispatch. The `scmp64` value-name is the
;      canonical breadcrumb (mirrors `scmp` for the 32-bit
;      compares).
;
;   3. The compare result drives `s_cselect_b32` via SCC: the lift
;      stores the i1 to SCC (the storeSCC path), and the cselect
;      reads it back as the predicate of `select i1 %scmp64, i32 1,
;      i32 0`. This pin verifies the SCC-writeback path end-to-end.
;
;   4. NO 32-bit compare on the source operands — a regression to
;      a pair-of-i32 lift would emit `icmp eq i32` against the
;      i64-wide source.

; CHECK-LABEL: define amdgpu_kernel void @s_cmp_eq_u64_kernel(
; CHECK-SAME: ptr addrspace(1) %arg0
; CHECK-SAME: i64 %arg1
; CHECK-SAME: i64 %arg2

; The s_cmp_eq_u64 lift. The handler-emitted value-name `scmp64`
; appears verbatim, and the operand types are i64.
; CHECK: %scmp64 = icmp eq i64 %{{[^,]+}}, %{{[^,]+}}

; The s_cselect_b32 immediately after the compare reads SCC. The
; lift surfaces this as `select i1 %scmp64, i32 1, i32 0` — pins
; the SCC-writeback chain end-to-end.
; CHECK: select i1 %scmp64, i32 1, i32 0

; Negative pin: no narrowing of the 64-bit operands to i32.
; CHECK-NOT: %scmp64 = icmp eq i32

; Negative pin: no signed-compare regression (the SOPC_CMP_64
; family is unsigned-only per SOPInstructions.td).
; CHECK-NOT: %scmp64 = icmp sgt
; CHECK-NOT: %scmp64 = icmp sge
; CHECK-NOT: %scmp64 = icmp slt
; CHECK-NOT: %scmp64 = icmp sle
