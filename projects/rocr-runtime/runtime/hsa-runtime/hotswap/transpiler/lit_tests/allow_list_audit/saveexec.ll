; RUN: %raise_cli %saveexec_co --emit-ir=saveexec_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Audit: `s_and_saveexec_b64` routes its EXEC write through
; `storeExec` (handle_sop1.cpp). This SemOp family is the one
; EXEC-writer that is simultaneously an SGPR-pair producer — the
; handler has to (a) save the OLD EXEC into the destination SGPR
; pair, and (b) write `old_exec AND src` to EXEC. Post-mem2reg we
; cannot see the SGPR-pair save directly (the saved value may fold
; away if the kernel never reads it), but we can see the new EXEC
; SSA value (`%new_exec`) being consumed by the SPE active-bit
; computation that wraps the subsequent side-effectful store.
;
; If the S_AND_SAVEEXEC_B64 handler regressed and no longer routed
; through storeExec, the `lshr i64 <exec>, %spe_lane_mod` before the
; store would still read `-1` (the initial EXEC) instead of
; `%new_exec`, and the single-lane-gated store would silently
; execute on every lane.

; CHECK-LABEL: define amdgpu_kernel void @saveexec_kernel(

; v_cmp_lt_u32_e64 produces the narrow mask in s[4:5]. Pre-mem2reg
; that's a store to the relevant sgpr alloca; post-mem2reg it's the
; SSA name for the comparison's sign-extended i64 representation.
; CHECK:       %vcmp = icmp ult i32 %tid, 16

; s_and_saveexec_b64 writes `old_exec AND src` to EXEC. The handler
; names the new EXEC SSA value `%new_exec` — this is the audit
; signal.
; CHECK:       %new_exec = and i64 -1, %{{[^ ]+}}

; The SPE lane-active computation that follows (wrapping the
; global_store_dword) keys off %new_exec, NOT -1. This proves the
; SSA graph of EXEC reflects the saveexec write.
; CHECK:       %[[AT_LANE:[^ ]+]] = lshr i64 %new_exec, %{{[^ ]+}}
; CHECK-NEXT:  %[[BIT:[^ ]+]] = and i64 %[[AT_LANE]], 1
; CHECK-NEXT:  %[[ACTIVE:[^ ]+]] = icmp ne i64 %[[BIT]], 0
; CHECK-NEXT:  br i1 %[[ACTIVE]], label %[[DO:[^ ,]+]], label %{{[^ ,]+}}

; The store under the narrowed EXEC is the observable side-effect
; that motivates the whole audit.
; CHECK:       [[DO]]:
; CHECK-NEXT:    store i32 {{.*}}, ptr addrspace(1) %{{[^ ]+}}, align 4
