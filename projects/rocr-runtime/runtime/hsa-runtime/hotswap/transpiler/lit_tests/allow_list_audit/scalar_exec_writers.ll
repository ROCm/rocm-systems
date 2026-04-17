; RUN: %raise_cli %scalar_exec_writers_co --emit-ir=divergent_exec_kernel \
; RUN:     2>/dev/null \
; RUN:   | %FileCheck %s
;
; Audit: V_CMPX, S_AND_B64→EXEC, and S_MOV_B64→EXEC all route their
; EXEC writes through `storeExec` (handle_valu.cpp / handle_sop2.cpp /
; handle_sop1.cpp respectively). Post-mem2reg, the observable is that
; the `lshr i64 <exec>, %spe_lane_mod` in the SPE active-bit
; computation following each writer consumes the NEW EXEC SSA value
; (not the previous one).
;
; Concretely the fixture produces, in order:
;
;   global_store_dword        (initial under EXEC = -1)
;   v_cmpx_lt_u32_e64 ...     <-- writes EXEC to `%cmpx_exec`
;   global_store_dword        (under %cmpx_exec)
;   s_mov_b64 exec, -1        <-- writes EXEC back to -1
;   v_cmpx_ge_u32_e64 ...     <-- writes EXEC to `%cmpx_exec64`
;   v_cmp_lt_u32_e64 s[4:5]   (SGPR mask — NOT EXEC)
;   s_and_b64 exec, exec, s   <-- writes EXEC to `%and64`
;   global_store_dword        (under %and64)
;   s_mov_b64 exec, -1        <-- writes EXEC back to -1
;
; We assert the lshr following each side-effectful store uses the
; right EXEC SSA value.

; CHECK-LABEL: define amdgpu_kernel void @divergent_exec_kernel(

; Initial store is under full EXEC. We match any `store i32 <val>`
; irrespective of whether <val> is a constant or an SSA name.
; CHECK:       lshr i64 -1, %{{[^ ]+}}
; CHECK:       store i32 {{.*}}, ptr addrspace(1) %{{[^ ]+}}, align 4

; After the first v_cmpx, the EXEC SSA value becomes %cmpx_exec.
; This is the most important signal: the V_CMPX handler routed its
; write through storeExec, mem2reg promoted the alloca to SSA, and
; the next SPE diamond correctly consumes the narrowed mask.
; CHECK:       %cmpx_exec = and i64 -1, %{{[^ ]+}}
; CHECK:       lshr i64 %cmpx_exec, %{{[^ ]+}}

; Store under the narrowed EXEC (valA = 0xAA = 170).
; CHECK:       store i32 {{.*}}, ptr addrspace(1) %{{[^ ]+}}, align 4

; Between region 1 and region 2 the fixture issues
; `s_mov_b64 exec, -1` to restore EXEC. We audit the S_MOV_B64 →
; EXEC handler path by observing the SECOND v_cmpx's `and i64 -1,
; %...` consumes the constant `-1` — meaning the SSA graph reset
; cleanly after the mov. If S_MOV_B64's handler did NOT route its
; write through `storeExec`, this would still read `%cmpx_exec`
; (the mask from region 1) and we'd see
; `%cmpx_exec64 = and i64 %cmpx_exec, %...` instead.
; CHECK:       %cmpx_exec{{[0-9]+}} = and i64 -1, %{{[^ ]+}}

; s_and_b64 exec, exec, s[4:5] writes EXEC to %and64. This proves
; the S_AND_B{32,64} handler on an EXEC destination routes through
; storeExec — same SSA-level contract as the V_CMPX path.
; CHECK:       %and64 = and i64 %{{[^ ]+}}, %{{[^ ]+}}
; CHECK:       lshr i64 %and64, %{{[^ ]+}}

; Final store under the %and64 mask (valB = 0xBB = 187).
; CHECK:       store i32 {{.*}}, ptr addrspace(1) %{{[^ ]+}}, align 4
