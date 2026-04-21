; RUN: %raise_cli %v_cmpx_ballot_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --enable-wave-native \
; RUN:     --emit-ir=v_cmpx_ballot_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; V_CMPX / V_CMP→SGPR ballot discipline under cross-wave (wave32 →
; wave64) lifts.  Pins the routing from a per-lane i1 compare result
; through `llvm.amdgcn.ballot.i64` into either EXEC (V_CMPX) or an
; SGPR (V_CMP→SGPR).
;
; Two routing arms are asserted:
;
;   * V_CMPX (writes EXEC).  Under `WaveNativeProjection` the EXEC
;     alloca storage width is the TARGET wave width (i64 for gfx942),
;     so the wave64 ballot flows directly into the EXEC update with
;     NO truncation.  The AND against the prior EXEC happens at i64.
;   * V_CMP with SGPR destination.  The destination is a 32-bit SGPR
;     and cannot hold a 64-bit mask, so the ballot is truncated from
;     i64 to the source wave width (i32).  This is the one residual
;     lossy path under `WaveNativeProjection`; kernels that rely on
;     the full 64-bit ballot being preserved across the SGPR round-
;     trip are caught separately by the obstruction classifier.
;
; Identifier pinning.  `cmpx_ballot` / `cmpx_exec` / `vcmp_ballot` /
; `vcmp_ballot_trunc` are the stable names emitted by the V_CMP(X)
; handler in `handle_valu_vcmp.cpp`; this test relies on them being
; stable so other lit checks can cross-reference the same shape.
;
; PROJECTION-TAG: `WaveNativeProjection` (wave32 source → wave64
; target).  If the cross-wave policy ever changes (e.g. switching
; to a same-wave same-target lower, or adding SPMDification), revisit
; this test together with the PROJECTION-TAG call sites and
; `wave_projection.{hpp,cpp}`.

; CHECK-LABEL: define amdgpu_kernel void @v_cmpx_ballot_kernel(

; V_CMPX — the compare feeds ballot.i64 directly into the EXEC
; update.  No `trunc i64 to i32` between the ballot and the AND;
; both operand and result of the AND are i64 because the EXEC
; alloca is now wave64-wide.  The first operand of the AND is the
; prior EXEC value; it may be an SSA load or a constant (e.g. -1
; when EXEC was just initialised), so allow both by matching "any
; non-comma up to the comma".
; CHECK:      %[[CMPX_CMP:[^ ]+]] = icmp ult i32 %{{[^ ,]+}}, 16
; CHECK-NEXT: %cmpx_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %[[CMPX_CMP]])
; CHECK-NEXT: %cmpx_exec = and i64 {{[^,]+}}, %cmpx_ballot

; V_CMP writing SGPR — ballot is produced at the target wave width
; (i64) and then truncated to the source wave width (i32) so it
; fits the SGPR destination.  Down-stream consumers of this SGPR
; see only the low 32 bits of the ballot; that residual lossy
; narrowing is documented in `wave_projection.cpp`'s
; `WaveNativeProjection::ballotI1ToWidth`.
; CHECK:      %[[VCMP:[^ ]+]] = icmp ult i32 %{{[^ ,]+}}, 8
; CHECK-NEXT: %vcmp_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %[[VCMP]])
; CHECK-NEXT: %vcmp_ballot_trunc = trunc i64 %vcmp_ballot to i32

; Negative assertion: the pre-fix shape used `sext i1` to widen the
; compare result directly into an EXEC-width integer.  If that comes
; back, the V_CMPX/V_CMP handlers have regressed to the old path.
; Note `CHECK-NOT` is cumulative across the whole file, so this
; applies to the entire module.
; CHECK-NOT: sext i1 %{{[^ ]+}} to i32

; Negative assertion: the PRE-WaveNative shape under this
; projection had the V_CMPX path truncate the ballot to i32 before
; AND-ing into a narrow EXEC alloca, which silently dropped target
; lanes 32..63 of every data-dependent v_cmpx (Cliff C in
; compare_correctness/RESULTS.md).  The fix makes the V_CMPX AND
; operate at i64; a `trunc i64 ... to i32` on the SAME SSA name as
; `%cmpx_ballot` would mean the regression is back.
; CHECK-NOT: trunc i64 %cmpx_ballot to i32
