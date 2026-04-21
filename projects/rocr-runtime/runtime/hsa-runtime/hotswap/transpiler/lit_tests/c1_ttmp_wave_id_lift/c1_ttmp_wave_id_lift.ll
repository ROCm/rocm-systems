; RUN: %raise_cli %c1_ttmp_wave_id_lift_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c1_ttmp_wave_id_lift_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Regression-fence for the Class-1 "wave_id lift" rescue
; (hotswap/docs/wave-size-translation.md §5.6.2, §6 Class-1).
;
; The canonical HIP-emitted pattern `s_bfe_u32 sDST, ttmp8, 0x50019`
; extracts `wave_id_in_workgroup` from bits [29:25] of ttmp8. Under
; cross-widening the generic BFE handler + SGPR alloca round-trip
; loses per-source-wave divergence (the formally-scalar BFE → SGPR
; shape collapses to `readfirstlane` during codegen, so all target
; lanes read one lane's wave_id). Before the lift this surfaced as
; a checkerboard miscompile in every Tensile / rocBLAS matmul (each
; Wave64 target wave covers two source waves but wrote to only one
; tile).
;
; The rescue in `handle_sop2.cpp` detects the exact
; `(src0 == ttmp8, src1 == imm 0x50019)` shape in the `S_BFE_U32`
; handler and emits the architectural expression directly from
; `@llvm.amdgcn.workitem.id.x` — a divergent-leaf intrinsic the
; backend keeps in a VGPR per lane. Downstream consumers of the
; destination SGPR see a divergent VGPR, so per-source-wave
; tile-offset arithmetic stays correct across cross-widening.
;
; We assert:
;   1) The raise succeeds (no %not; the RUN line errors the test
;      if raise_cli returns non-zero).
;   2) The kernel body is present (CHECK-LABEL anchors on it).
;   3) The lifted IR carries the canonical three-instruction shape
;      — `@llvm.amdgcn.workitem.id.x`, `lshr i32 ..., 5`
;      (log2(W_src=32) = 5), `and i32 ..., 31` — with 31 as the
;      five-bit mask from the 0x50019 immediate's width=5 field.
;      The three are asserted in decoded order; intermediate names
;      are regex-matched (%"..." or bare identifiers) to stay
;      stable across LLVM IR-printer versions.
;
; Paired with the lit_tests/c1_lane_id_leak v_mbcnt_hi fixture,
; which pins the *unrewritable* Class-1 leak (refuses). Together
; they pin both outcomes for Class-1 obstructions: rescue
; (this fixture, `isCanonicalWaveIdBfe == true` in
; `wave_size_obstruction.cpp` skips the refusal) vs. refuse
; (c1_lane_id_leak, no rescue path on paper).

; IR-LABEL: define amdgpu_kernel void @c1_ttmp_wave_id_lift_kernel(
; IR: call i32 @llvm.amdgcn.workitem.id.x()
; IR: lshr i32 {{.*}}, 5
; IR: and i32 {{.*}}, 31
