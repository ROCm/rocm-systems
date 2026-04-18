; RUN: %raise_cli %decoder_madmk_v_fmamk_f32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=decoder_madmk_v_fmamk_f32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Decoder regression test for VOP2 MADMK form. Pins that the
; `driftCheckSrcN` MADMK exception (src0Idx < immIdx < src1Idx)
; survives. Without the exception, raise_cli would abort with
; "srcMap disagrees with OpName::srcN table for v_fmamk_f32" and
; this fixture would never produce IR for FileCheck to match.
;
; The decoder fix landed in commit ab25d0257a after the GPT-OSS
; `attn_fwd` corpus kernel hit the strict-check abort. This
; synthetic fixture isolates the regression detection from the
; corpus state — a future change that re-introduces the strict
; check would fail this lit test even if the corpus sweep wasn't
; run.

; CHECK-LABEL: define amdgpu_kernel void @decoder_madmk_v_fmamk_f32_kernel(

; The lifted IR must contain an FMA call with the 32-bit literal
; (K = 0x40490fdb = pi) as one of the operands. The handler
; emits llvm.fma.f32, with the K either as an explicit f32
; constant or as a bitcast of the i32 literal — match either
; form. The exact bit pattern 0x40490fdb (= float pi ~3.1415927)
; is what the inline-asm fixture encodes; if the imm extraction
; truncates or sign-extends, this assertion catches it.
;
; The IR shape is `call float @llvm.fma.f32(float %s0, float %k,
; float %s2)` per the V_FMAMK_F32 handler in handle_valu.cpp
; (srcF(0)=src0, srcF(1)=K, srcF(2)=src2 → fma(s0, k, s2)).
; CHECK:      call {{.*}}float @llvm.fma.f32({{.*}}, float {{.*}}0x400921FB60000000{{.*}}

; The intrinsic declaration must be present.
; CHECK:      declare {{.*}}float @llvm.fma.f32(float, float, float)
