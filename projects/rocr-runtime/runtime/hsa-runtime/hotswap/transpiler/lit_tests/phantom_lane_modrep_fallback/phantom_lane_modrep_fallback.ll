; RUN: %raise_cli %phantom_lane_modrep_fallback_co \
; RUN:     --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=phantom_lane_modrep_fallback_kernel 2>&1 \
; RUN:   | %FileCheck %s
;
; Regression fence for the phantom-lane → MODREP fallback added in
; `raiser.cpp`.  The kernel's `__launch_bounds__(32)` drives
; `max_flat_workgroup_size` to 32 in the gfx1250 HSACO's
; amdhsa metadata; under the gfx942 (wave64) target the fallback
; triggers because 32 < 64.  With the fallback in place, raise_cli
; exits 0, stderr carries the "falling back to
; ModuloReplicationProjection" diagnostic, and the raised IR
; contains no `@llvm.amdgcn.init.whole.wave` call (that intrinsic
; is only emitted by `WaveNativeProjection::emitInitialExec`;
; MODREP's default `emitInitialExec` returns source-width -1 with
; no intrinsic call).  See the paired `wmma_phantom_lane_refuse/`
; fixture for the COMPLEMENT: same launch_bounds but with a WMMA
; op that DOES trigger the refusal gate in `handle_valu_vop3p.cpp`.
;
; Three independent signals the fallback is working correctly:

; 1. The fallback log message names the kernel, the offending
;    workgroup size, and the target wavefront width.  A regression
;    that silently drops the fallback would skip this log, and the
;    downstream signals (#2 + #3) would also change — this check
;    gives an early, attribution-friendly fence.  The CHECK-SAME
;    ordering below matches the actual `errs()` emission order
;    in `raiser.cpp` (kernel-name -> phantom-lane-regime banner ->
;    max_flat_workgroup_size -> target wavefront width -> falling
;    back message) so a future reordering is caught, not silently
;    rewritten.
; CHECK: phantom-lane regime
; CHECK-SAME: max_flat_workgroup_size=32
; CHECK-SAME: target wavefront width=64
; CHECK-SAME: falling back to ModuloReplicationProjection

; 2. Raise completes cleanly (no refusal, no lift failure).  The
;    IR-LABEL below catches that — if raise_cli refused, the IR
;    wouldn't be emitted and the CHECK wouldn't match.
; CHECK-LABEL: define amdgpu_kernel void @phantom_lane_modrep_fallback_kernel(

; 3. The raised IR must NOT contain `init.whole.wave` — that call
;    is the WaveNativeProjection's `emitInitialExec` entry-block
;    side effect, and its absence confirms the projection is MODREP
;    for this kernel.  A regression that flips the fallback back to
;    WaveNative would re-introduce the call and this CHECK-NOT
;    would fire.
; CHECK-NOT: call i1 @llvm.amdgcn.init.whole.wave
; CHECK-NOT: call {{.*}} @llvm.amdgcn.init_whole_wave
