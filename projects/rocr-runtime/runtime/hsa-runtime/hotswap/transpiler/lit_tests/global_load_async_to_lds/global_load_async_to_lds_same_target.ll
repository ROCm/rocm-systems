; RUN: %raise_cli %global_load_async_to_lds_co --isa=gfx1250 \
; RUN:     --target-isa=gfx1250 --emit-ir=global_load_async_to_lds_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Lift fixture for FLAT `global_load_async_to_lds_b{32,64,128}` —
; the same-target (gfx1250 → gfx1250) intrinsic-emit path. Pins
; the principled lift in transpiler/handle_flat.cpp under
; `SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B{8,32,64,128}` when
; `ctx.targetIsa.hasTensorOps` is true. Companion fixture to
; `global_load_async_to_lds.ll`, which pins the cross-target
; (gfx942) loud refusal.
;
; The FLATInstructions.td `FLAT_Global_Load_LDS_Pseudo<IsAsync=1>`
; multiclass yields three operand-shape variants — plain
; (vdst:VGPR_32, vaddr:VGPR_64, off:imm, cpol:imm) and SADDR
; (vdst:VGPR_32, saddr:SGPR_64, vaddr:VGPR_32, off:imm, cpol:imm)
; for each width. The fixture's HIP source uses the clang
; builtins `__builtin_amdgcn_global_load_async_to_lds_b{32,64,128}`
; which compile to the plain VFLAT 0x60-0x62 reals (vdst=v1,
; vaddr=v0, saddr=s[0:1]/s[6:7]/s[4:5] — the kernel-arg pointers
; landed in the SGPR base + per-lane VGPR offset shape, which the
; AMDGPU disassembler prints as the SADDR variant).
;
; The matching LLVM intrinsic family
; (IntrinsicsAMDGPU.td:3939-3946, all sharing `AMDGPUAsyncGlobalLoadToLDS`
; on line 3904) is
;
;   void llvm.amdgcn.global.load.async.to.lds.b{8,32,64,128}(
;       ptr addrspace(1) %gaddr,
;       ptr addrspace(3) %laddr,
;       i32 immarg      %offset,
;       i32 immarg      %cpol)
;
; The handler:
;   * casts `vdst` (a per-lane VGPR_32 carrying the LDS i32 base)
;     via `inttoptr i32 %vgpr to ptr addrspace(3)` (named
;     `lds_ptr*` in the emitted IR);
;   * decodes the global address via the shared FLAT `decodeFlatAddr`
;     helper (plain → vaddr-only, SADDR → saddr+vaddr) into a
;     `ptr addrspace(1)`;
;   * threads the FLAT `offset` immediate and the `cpol` immediate
;     through as the trailing `i32 immarg` pair (cpol = 0x800 here
;     because the assembler emits `scale_offset` for the
;     scaled-saddr path);
;   * wraps the call in `ctx.emitUnderExec` so wave-divergent EXEC
;     is honoured before the side-effecting DMA.
;
; We pin the per-width call shape and the LDS-pointer cast it
; consumes. Drift indicators:
;   * If a future LLVM rename swaps the intrinsic name (e.g. drops
;     the `async.` infix) the IR check fails immediately and
;     pinpoints the rename rather than letting a silently mis-named
;     intrinsic reach the backend.
;   * If the LDS-base operand is lowered as a `<n x i32>` vector
;     (or any non-`ptr addrspace(3)` shape), the matching
;     `inttoptr i32 ... to ptr addrspace(3)` line goes missing and
;     FileCheck reports the exact divergence.

; b32: per-lane LDS i32 base via inttoptr i32 → ptr addrspace(3),
; then the b32 async DMA call.
; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: call void @llvm.amdgcn.global.load.async.to.lds.b32(
; IR-SAME: ptr addrspace(1)
; IR-SAME: ptr addrspace(3) %lds_ptr
; IR-SAME: i32 0
; IR-SAME: i32 {{-?[0-9]+}}

; b64: same shape, b64 intrinsic.
; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: call void @llvm.amdgcn.global.load.async.to.lds.b64(
; IR-SAME: ptr addrspace(1)
; IR-SAME: ptr addrspace(3) %lds_ptr
; IR-SAME: i32 0
; IR-SAME: i32 {{-?[0-9]+}}

; b128: same shape, b128 intrinsic.
; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: call void @llvm.amdgcn.global.load.async.to.lds.b128(
; IR-SAME: ptr addrspace(1)
; IR-SAME: ptr addrspace(3) %lds_ptr
; IR-SAME: i32 0
; IR-SAME: i32 {{-?[0-9]+}}
