; RUN: %raise_cli %v_fma_mix_f32_bf16_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_fma_mix_f32_bf16_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Pins the VOP3P mixed-precision FMA family (V_FMA_MIX_F32 +
; V_FMA_MIX_F32_BF16).  Both variants share the op_sel / op_sel_hi
; parser and write-back shape in handle_valu_vop3p.cpp; only the
; narrow element type differs (f16 vs bf16).  The fixture exercises
; the exact shape the kerneldex `_attn_fwd` kernel emits
; (op_sel:[0,1,0] op_sel_hi:[1,1,0] — src0 LO-narrow, src1 HI-narrow,
; src2 full-f32) with both mnemonics back-to-back in one kernel.
;
; Target-isa is gfx942 (cross-target) to prove the bf16 narrow-half
; conversion does not require a cross-target refusal: the lift is
; `bitcast i16 -> bfloat` + `fpext bfloat -> float`, which every
; AMDGPU target handles natively.
;
; Invariants pinned below:
;
;   1. bf16 variant emits `bitcast i16 ... to bfloat` + a named
;      `fpext bfloat ... to float` (`%mix_cvt_bf16`) for BOTH the LO
;      (op_sel[i]==0) and HI (op_sel[i]==1) halves.
;   2. f16 variant emits `bitcast i16 ... to half` + a named
;      `fpext half ... to float` (`%mix_cvt`) for BOTH halves.  Two
;      independent fma-mix sites share the same per-op SSA root names
;      which LLVM auto-renumbers (`%mix_cvt` and `%mix_cvt<N>`, same
;      for the bf16 name).
;   3. Both variants feed an `llvm.fma.f32` call (the handler's
;      `Intrinsic::fma` emit); no cross-target refusal appears.
;   4. src2 (op_sel_hi[2]==0 in the fixture) stays as a full f32 —
;      no narrow-half conversion for the third source.

; CHECK-LABEL: define amdgpu_kernel void @v_fma_mix_f32_bf16_kernel(

; ----- v_fma_mix_f32_bf16 (narrow type = bfloat) -----
; LO-bf16 half for src0 (trunc -> bitcast to bfloat -> fpext).
; CHECK-DAG: trunc i32 %{{.*}} to i16
; CHECK-DAG: bitcast i16 %{{.*}} to bfloat
; CHECK-DAG: %mix_cvt_bf16 = fpext bfloat %{{.*}} to float

; HI-bf16 half for src1 (lshr 16 -> trunc -> bitcast -> fpext).
; CHECK-DAG: lshr i32 %{{.*}}, 16
; CHECK-DAG: %mix_cvt_bf16{{[0-9]+}} = fpext bfloat %{{.*}} to float

; ----- v_fma_mix_f32 (narrow type = half) -----
; LO-f16 half for src0.
; CHECK-DAG: bitcast i16 %{{.*}} to half
; CHECK-DAG: %mix_cvt = fpext half %{{.*}} to float

; HI-f16 half for src1.
; CHECK-DAG: %mix_cvt{{[0-9]+}} = fpext half %{{.*}} to float

; ----- Both variants feed llvm.fma.f32 -----
; CHECK-DAG: %fma_mix = call float @llvm.fma.f32(float %mix_cvt_bf16, float %mix_cvt_bf16{{[0-9]+}}, float %{{[0-9]+}})
; CHECK-DAG: %fma_mix{{[0-9]+}} = call float @llvm.fma.f32(float %mix_cvt, float %mix_cvt{{[0-9]+}}, float %{{[0-9]+}})

; Negative pins: no cross-target refusal, no BF16-specific MFMA/WMMA
; fallback — the BF16 narrow-half is handled by universal fpext, not
; a cross-target lowering.
; CHECK-NOT: unsupportedOpcode
; CHECK-NOT: llvm.amdgcn.cvt.pk
; CHECK-NOT: llvm.amdgcn.mfma
