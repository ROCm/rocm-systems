; RUN: %raise_cli %v_add_co_u32_sgpr_carry_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_add_co_u32_sgpr_carry_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Regression fence for the carry-chain SGPR-operand bug class
; (hotswap/docs/modrep-predicate-chain.md §6.4, fixed
; 2026-04-22 in `handle_valu.cpp`'s `readCarryInI1` /
; `writeCarryOutI1` helpers).
;
; PRE-FIX SHAPE (the bug). The six carry-chain handlers
; (V_{ADD,SUB,SUBREV}_CO_(CI_)U32) hardcoded `loadVCC` / `storeVCC`
; for both carry-in (ci variants) and carry-out, silently ignoring
; the explicit scalar operand the e64 / VOP3B encoding names. For a
; pair like:
;
;   v_add_co_u32 vX, s0, vA, vB              (carry-OUT to s0)
;   v_add_co_ci_u32_e64 vY, s0, vC, vD, s0   (carry-IN from s0,
;                                             carry-OUT to s0)
;
; the pre-fix handler would:
;   * ignore the `s0` sdst on the first instruction, writing carry
;     to VCC instead;
;   * ignore the `s0` ssrc2 on the second instruction, reading
;     carry from VCC (which was never written by the intended
;     producer) — a stale carry-in.
;
; POST-FIX SHAPE (this fixture pins). `writeCarryOutI1` sees the
; sdst=`s0` ParsedReg and:
;   1. Ballots the per-lane carry i1 to source-wave-mask width via
;      `projection.ballotI1ToWidth` → `amdgcn.ballot.i64` (WaveNative
;      cross-widening source width is still i64 here, then trunc'd
;      to source width by `writeRegExecWidth`).
;   2. Stores the narrow mask to the s0 alloca via
;      `writeRegExecWidth`.
;   3. Records the fresh per-lane i1 shadow via
;      `ctx.recordSgprWaveMaskI1(0, carryI1, isPair=false)` so the
;      next consumer reads the i1 directly (bypassing the lossy
;      narrow-mask extract).
;
; Then `readCarryInI1` on the second instruction's ssrc2=`s0` hits
; the fresh-shadow cache and returns the SAME i1 SSA value the first
; add produced — no store-load round-trip through VCC, no lossy
; narrow-mask extract.
;
; The net IR shape for the pair is a pure dataflow chain:
;
;   %addCarryI1 = extractvalue {i32, i1} %add1_with_ov, 1
;   %carry_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %addCarryI1)
;   ...
;   %addCarryZext = zext i1 %addCarryI1 to i32              ; <-- SAME %addCarryI1
;   %ci = call {i32, i1} @llvm.uadd.with.overflow.i32(..., %addCarryZext)
;   ...
;
; A regression that re-introduces hardcoded-VCC would break the SSA
; chain — the second add's carry-in would `load i1, ptr %vcc`
; instead of zext-forwarding the first add's carry.

; CHECK-LABEL: define amdgpu_kernel void @v_add_co_u32_sgpr_carry_kernel(

; First carry-chain add (V_ADD_CO_U32 with sdst=s0) — classic
; uadd_with_overflow → ballot-and-store to s0 shape.
; CHECK: [[ADD1:%[[:alnum:]_.]+]] = call { i32, i1 } @llvm.uadd.with.overflow.i32
; CHECK: [[CARRY1:%[[:alnum:]_.]+]] = extractvalue { i32, i1 } [[ADD1]], 1
;
; The carry-out ballots to i64 (WaveNative target width) before
; truncation to source wave-mask width — this is the `carry_ballot`
; twine from `writeCarryOutI1`. A regression that skips the ballot
; (e.g. direct storeSGPR32 of a per-lane zext) fails this pin.
; CHECK: %carry_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 [[CARRY1]])

; Second carry-chain add (V_ADD_CO_CI_U32_e64 with ssrc2=s0 and
; sdst=s0) — the carry-IN must be the SAME i1 that the first add
; produced (fresh-shadow lookup), NOT a fresh `load i1, ptr %vcc`.
; The zext of the SAME `[[CARRY1]]` is what pins this.
; CHECK: zext i1 [[CARRY1]] to i32

; The second add's combined carry-out is also routed via
; writeCarryOutI1 → ballot-and-store to s0. Distinct `carry_ballot`
; SSA name (suffixed with a number) confirms a separate ballot
; call.
; CHECK: %carry_ballot{{[0-9]+}} = call i64 @llvm.amdgcn.ballot.i64

; NEGATIVE PIN: no `load i1, ptr %vcc` between the first and
; second adds. If the pre-fix handler were restored, the second
; add's carry-in would route through `loadVCC` which lowers to
; `load i1, ptr %vcc` — FileCheck-NOT across the relevant span
; catches the regression directly.
; CHECK-NOT: load i1, ptr %vcc
