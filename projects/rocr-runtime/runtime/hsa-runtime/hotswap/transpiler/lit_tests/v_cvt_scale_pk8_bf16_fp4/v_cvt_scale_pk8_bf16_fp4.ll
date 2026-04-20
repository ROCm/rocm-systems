; RUN: %raise_cli %v_cvt_scale_pk8_bf16_fp4_co --isa=gfx1250 \
; RUN:     --emit-ir=v_cvt_scale_pk8_bf16_fp4_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; RUN: %not %raise_cli %v_cvt_scale_pk8_bf16_fp4_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 \
; RUN:     --emit-ir=v_cvt_scale_pk8_bf16_fp4_kernel 2>&1 \
; RUN:   | %FileCheck --check-prefix=CROSS %s
;
; Pins the gfx1250-only VOP3 packed-8 FP4 -> BF16 scaled convert
; (V_CVT_SCALE_PK8_BF16_FP4_e64, VOP3Instructions.td:1788).
;
; Same-target lift (RUN #1): the handler emits the LLVM intrinsic
; `@llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32, i32, i32 immarg)` with
; immarg = 0 (the corpus-observed scale_sel value) and a
; <8 x bfloat> result; the write-back path bit-casts the <8 x bfloat>
; to i128 / four dwords before handing it to `writeRegVec`.  No
; cross-target refusal fires on the same-target path.
;
; Cross-target lift (RUN #2): the handler refuses loudly because
; `int_amdgcn_cvt_scale_pk8_bf16_fp4` is gated behind `isGFX125xOnly`
; in IntrinsicsAMDGPU.td:686 and gfx942 has no MX-FP4 scaling unit.
; The refusal message pins the ISA / intrinsic / isGFX125xOnly
; anchor so a future TableGen rename forces a visible test update.

; CHECK-LABEL: define amdgpu_kernel void @v_cvt_scale_pk8_bf16_fp4_kernel(

; The handler's exact emit shape:
;   %cvt_scale_pk8_bf16_fp4 = call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32 %src, i32 %scale, i32 0)
; Pinned with explicit return-width + element-type + immarg 0.
; CHECK: %cvt_scale_pk8_bf16_fp4 = call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32 %{{.*}}, i32 %{{.*}}, i32 0)

; The <8 x bfloat> result must be collapsed to a 128-bit int before
; `writeRegVec` splits it into four consecutive dword VGPRs.
; CHECK: bitcast <8 x bfloat> %cvt_scale_pk8_bf16_fp4 to i128

; The intrinsic declaration must carry the gfx1250 immarg range
; annotation (`range(i32 0, 16)`) and the exact bf16-fp4 name.
; CHECK-DAG: declare <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32, i32, i32 immarg range(i32 0, 16))

; Negative pins: no cross-target refusal on the same-target path,
; and no accidental fallback through the unscaled `cvt_pk_f32_fp8`
; / `cvt_f32_fp8` paths that share similar mnemonics.
; CHECK-NOT: cross-target lift to gfx942
; CHECK-NOT: amdgcn.cvt.pk.f32
; CHECK-NOT: amdgcn.cvt.f32.fp8

; CROSS: v_cvt_scale_pk8_bf16_fp4 is a gfx1250-only VOP3
; CROSS-SAME: isGFX125xOnly
; CROSS-SAME: no corpus kernel exercises today
