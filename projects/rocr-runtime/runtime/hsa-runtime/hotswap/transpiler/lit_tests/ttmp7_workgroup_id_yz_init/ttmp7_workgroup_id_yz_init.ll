; RUN: %raise_cli %ttmp7_workgroup_id_yz_init_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=ttmp7_wg_id_y_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Regression-fence for the `ttmp7 = (workgroup_id_z << 16) |
; (workgroup_id_y & 0xFFFF)` raiser-entry init on gfx12+ (see
; `raiser.cpp` Phase 4; landed alongside the matmul_fp16_16x16 fix).
;
; Before the raiser initialised ttmp7 this kernel's `s_and_b32 sN,
; ttmp7, 0xffff` read an uninitialised SGPR, so downstream lifted-IR
; consumers of `workgroup_id_y` always saw 0 — a kernel launched
; with a 2D grid wrote only its Y=0 column of workgroups, leaving
; the rest of the output at whatever the destination memory held at
; dispatch (verified empirically on `matmul_kernel_16x16` under
; compare_correctness: cols 0..15 match the CPU reference, cols
; 16..31 retain the host's pre-launch `0xCD` poison fill).
;
; We assert:
;   1) The raise succeeds (no %not; the RUN line fails the test if
;      raise_cli returns non-zero).
;   2) The lifted IR contains both `@llvm.amdgcn.workgroup.id.y`
;      and `@llvm.amdgcn.workgroup.id.z` calls at kernel entry.
;      Checking for both simultaneously pins BOTH halves of the
;      packed ttmp7 layout — a regression that drops the Z-shift
;      (e.g. forgets to include Z) would pass a Y-only check and
;      silently miscompile 3D-grid kernels on the rarer Z>0
;      launches.
;   3) The IR packs them via `shl i32 {{.*}}, 16` (the Z<<16
;      portion), pinning the exact encoding the AMDGPU backend's
;      `AMDGPULegalizerInfo::loadInputValue` requires.

; IR-LABEL: define amdgpu_kernel void @ttmp7_wg_id_y_kernel(
;
; We pin the FULL `(Z << 16) | (Y & 0xFFFF)` dataflow from the two
; workgroup.id intrinsics to the `or` that combines them.  Capture
; variables tie the shift / mask operands to the specific intrinsic
; results so a regression that shifted Y (instead of Z) by 16, or
; masked Z (instead of Y) with 0xFFFF, would fail the check.
;
; Note: we don't CHECK for the `store i32 %ttmp7_val, ptr %ttmp_7`
; into the raiser's ttmp[7] alloca; that store is routinely deleted
; by LLVM's mem2reg pass that runs before `emit-ir` prints, so the
; SSA value flows directly from `or` into the consumer side of the
; kernel body (the inline-asm `s_and ttmp7, 0xffff` lift).  The
; combined `or` shape plus the `@llvm.amdgcn.workgroup.id.{y,z}`
; anchors above are the robust regression fence.
;
; The capture regex is anchored on the `ttmp7_wg_id_{y,z}` prefix
; produced by `raiser.cpp`'s `B.CreateCall(fn..., {}, "ttmp7_wg_id_y")`
; name.  Without the prefix anchor IR-DAG would match an EARLIER
; `@llvm.amdgcn.workgroup.id.y()` call the raiser emits for the s3
; SGPR alloca path (that one is named `%wg_id_y`), and the
; downstream `and [[WG_Y]], 65535` check would then fail because
; the actual AND consumes the SECOND call's result.  If a future
; refactor renames these SSA values, update the anchor here.
; IR-DAG: [[WG_Y:%ttmp7_wg_id_y[a-zA-Z0-9_.]*]] = call i32 @llvm.amdgcn.workgroup.id.y()
; IR-DAG: [[WG_Z:%ttmp7_wg_id_z[a-zA-Z0-9_.]*]] = call i32 @llvm.amdgcn.workgroup.id.z()
; IR:     [[Y_LO:%[a-zA-Z0-9_.]+]] = and i32 [[WG_Y]], 65535
; IR:     [[Z_HI:%[a-zA-Z0-9_.]+]] = shl i32 [[WG_Z]], 16
; IR:     {{%[a-zA-Z0-9_.]+}} = or i32 [[Y_LO]], [[Z_HI]]
