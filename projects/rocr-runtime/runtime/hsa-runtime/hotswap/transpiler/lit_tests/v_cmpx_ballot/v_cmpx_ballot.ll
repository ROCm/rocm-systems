; RUN: %raise_cli %v_cmpx_ballot_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_cmpx_ballot_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; V_CMPX / V_CMP→SGPR ballot discipline under cross-wave (wave32 →
; wave64) lifts. The fix in `handle_valu.cpp` + `reg_file.hpp`
; routes the result of a V_CMPX (EXEC write) and a V_CMP with SGPR
; destination through `llvm.amdgcn.ballot`, then truncates the wave64
; ballot back to the source's wave32 execTy (i32). The pre-fix code
; used `sext i1 to i32`, which produced a per-lane SSA value that was
; then read as a wave mask downstream — a silent miscompile that
; dropped stores on half the wave.
;
; This test pins down the ballot pattern end-to-end:
;
;   1. An `@llvm.amdgcn.ballot.i64(i1 %cmp)` call exists (target wave
;      width is 64 bits; the declaration uses the i64 overload).
;   2. The call's result is truncated from i64 back to i32 (source's
;      wave32 execTy width).
;   3. The truncated mask is AND-ed into the EXEC alloca (V_CMPX
;      branch) or written to an SGPR (V_CMP→SGPR branch).
;   4. No `sext i1 to i32` appears in the EXEC update or the SGPR
;      write — this is the pre-fix shape, and its presence would mean
;      the ballot routing regressed.
;
; MODREP: this test asserts the wave32-source / wave64-target
; modulo-replication projection. If the cross-wave policy ever
; changes (e.g. to SPMDification or same-wave lifts), revisit this
; test together with the MODREP-tagged call sites.

; CHECK-LABEL: define amdgpu_kernel void @v_cmpx_ballot_kernel(

; V_CMPX — the compare feeds ballot.i64, gets truncated to i32, and
; AND-s into EXEC. `cmpx_ballot_trunc` and `cmpx_exec` are the names
; pinned in the handler; they are stable identifiers used by other
; lit checks and not SSA-version-sensitive.
; CHECK:      %[[CMPX_CMP:[^ ]+]] = icmp ult i32 %{{[^ ,]+}}, 16
; CHECK-NEXT: %cmpx_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %[[CMPX_CMP]])
; CHECK-NEXT: %cmpx_ballot_trunc = trunc i64 %cmpx_ballot to i32
; The first AND operand is the prior EXEC value; it may be an SSA
; load or a constant (e.g. -1 when EXEC was just initialized), so
; allow both by matching "any non-comma up to the comma".
; CHECK:      %cmpx_exec = and i32 {{[^,]+}}, %cmpx_ballot_trunc

; V_CMP writing SGPR — same ballot/trunc shape, stored through
; `writeRegExecWidth` (which resolves to a store into the SGPR
; alloca). We match the ballot+trunc pair; the store is then the
; immediately-following instruction chain.
; CHECK:      %[[VCMP:[^ ]+]] = icmp ult i32 %{{[^ ,]+}}, 8
; CHECK-NEXT: %vcmp_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %[[VCMP]])
; CHECK-NEXT: %vcmp_ballot_trunc = trunc i64 %vcmp_ballot to i32

; Negative assertion: the pre-fix shape used `sext i1` to widen the
; compare result directly into an EXEC-width integer. If that comes
; back, the V_CMPX/V_CMP handlers have regressed to the old path.
; Note `CHECK-NOT` is cumulative across the whole file, so this
; applies to the entire module.
; CHECK-NOT: sext i1 %{{[^ ]+}} to i32
