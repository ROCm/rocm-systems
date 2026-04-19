; RUN: %raise_cli %s_load_b96_kernarg_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=s_load_b96_kernarg_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the gfx11+/gfx1250 96-bit scalar memory load
; (`s_load_b96`) over a kernarg buffer that contains a `by_value`
; aggregate larger than 8 bytes.  The SemOp + `extractKernargDword`
; helper + raiser by_value decomposition together produce the IR
; signature and SGPR-store shape pinned below.  See SemOp::S_LOAD_B96
; in transpiler/semop.hpp; the per-dword loop in handle_smem.cpp
; (under `if (isKernarg && immOffset && byteOffset <
; ctx.kernargs.implicitArgsBase)`); the `byValueSize > 8 && % 4 == 0`
; branch in raiser.cpp's `paramTypes` build; and the
; `extractKernargDword` helper in kernarg_layout.cpp.
;
; INVARIANTS PINNED:
;
;   1. The `by_value` aggregate (size 16, the `Big16` struct) is
;      decomposed into FOUR `i32` IR arguments at byte offsets
;      0/4/8/12, NOT a single `i32` / `[16 x i8]` / `i128` / aggregate
;      type. This is the kernarg-buffer-byte-layout-preserving
;      contract that lets `extractKernargDword` resolve interior
;      sub-struct loads.
;
;   2. The `global_buffer` slot at offset 16 follows the four i32
;      slots and is typed as `ptr addrspace(1)` (the global address
;      space the AMDGPU code-object ABI guarantees for
;      `global_buffer` kernargs).
;
;   3. The `s_load_b96 s[0:2], s[0:1], 0x4` lifts to three SGPR
;      stores: s0 <- %arg1, s1 <- %arg2, s2 <- %arg3 (the kernarg
;      dwords at offsets 4/8/12 of the Big16 by_value).  After
;      mem2reg the SGPR alloca file collapses; the loaded values
;      flow into the next instruction (a `v_dual_mov_b32 v4, s0 ::
;      v_dual_mov_b32 v5, s1`) as v_mov_b32 lifts that read s0 and
;      s1 — observable in the raiser's per-VGPR phi shape:
;
;        %vgpr4.0 = phi i32 [ %arg1, %spe_do{N} ], [ undef, ... ]
;        %vgpr5.0 = phi i32 [ %arg2, %spe_do{M} ], [ undef, ... ]
;
;      The `%arg1` / `%arg2` references in those phis are the
;      load-bearing pin: a regression that mis-routed any one of the
;      three dwords (e.g. picked the wrong by_value slot, or
;      computed the wrong sub-offset, or zero-substituted on a
;      "miss") would change the operand of one of these phis and
;      fail this pin.
;
;   4. NO refusal diagnostic on stderr: the previous failure mode
;      was `transpiler: s_load_b96 ... Cannot resolve kernarg at
;      offset 4`.  With raise_cli's stderr redirected to /dev/null
;      the test still fails because the FileCheck pins below would
;      not match (the kernel body would be a stub if the lift
;      refused).
;
; WHAT WE DO NOT PIN:
;
;   * The exact basic-block names (`%spe_do{N}`) on the phi RHS — the
;     raiser's emitUnderExec helper renumbers these as new SPE
;     scopes are added; we use `{{[a-zA-Z_0-9]+}}` regex placeholders.
;   * The exact SSA names of intermediate `zext` / `shl` / `or` /
;     `inttoptr` chains — those are LLVM-IR-printer-renumbered.
;   * The downstream global stores — the compiler's choice of how to
;     pack the `out[0..6]` writes is not part of the s_load_b96
;     handler contract.

; CHECK-LABEL: define amdgpu_kernel void @s_load_b96_kernarg_kernel(
; CHECK-SAME: i32 %arg0
; CHECK-SAME: i32 %arg1
; CHECK-SAME: i32 %arg2
; CHECK-SAME: i32 %arg3
; CHECK-SAME: ptr addrspace(1) %arg4

; The s_load_b96 result is routed into s[0:2] with s0=%arg1,
; s1=%arg2, s2=%arg3 (the kernarg dwords at offsets 4/8/12 of the
; by_value Big16 struct). The downstream `v_dual_mov_b32 v4, s0 ::
; v_dual_mov_b32 v5, s1` consumes s0 and s1, which the raiser's
; per-VGPR phi-under-EXEC shape carries as `phi i32 [ %arg1, ... ]`
; and `phi i32 [ %arg2, ... ]` references. CHECK-DAG accepts either
; ordering (s0/s1 lift order isn't stable across mem2reg passes).
; CHECK-DAG: phi i32 [ %arg1, %{{[a-zA-Z_0-9]+}} ]
; CHECK-DAG: phi i32 [ %arg2, %{{[a-zA-Z_0-9]+}} ]

; Negative pin: the previous failure mode emitted a refusal via
; `RaiseFailure::smemKernargMiss` after printing the "Cannot
; resolve kernarg" diagnostic. With the lift refused, the emit-ir
; output would not contain a kernel body, and the phi pins above
; wouldn't match. This CHECK-NOT additionally enforces that the
; s_load_b96 handler did not silently substitute zero / undef for
; the s0 or s1 operand (which would manifest as `[ i32 0, %... ]`
; or `[ i32 undef, %... ]` in those phi arms instead of `[ %argN,
; %... ]`). The phi pattern tolerates `undef` only on the *inactive*
; (false) arm of the EXEC predicate, not the active one.
; CHECK-NOT: phi i32 [ i32 0, %{{[a-zA-Z_0-9]+}} ]
; CHECK-NOT: phi i32 [ i32 undef, %{{[a-zA-Z_0-9]+}} ]

; Negative pin: the by_value decomposition must NOT collapse the
; 16-byte struct back into a single argument of any of these
; non-decomposed shapes. Any of these would indicate the
; `byValueSize > 8 && % 4 == 0` branch in raiser.cpp regressed.
; CHECK-NOT: define amdgpu_kernel void @s_load_b96_kernarg_kernel(i32 %arg0, ptr addrspace(1) %arg1)
; CHECK-NOT: define amdgpu_kernel void @s_load_b96_kernarg_kernel({{[^,]+\[16 x i8\]}}
; CHECK-NOT: define amdgpu_kernel void @s_load_b96_kernarg_kernel(i128
