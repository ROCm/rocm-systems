; RUN: %raise_cli %v_cvt_scale_pk8_bf16_fp4_co --isa=gfx1250 \
; RUN:     --emit-ir=v_cvt_scale_pk8_bf16_fp4_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; RUN: %raise_cli %v_cvt_scale_pk8_bf16_fp4_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 \
; RUN:     --emit-ir=v_cvt_scale_pk8_bf16_fp4_kernel 2>/dev/null \
; RUN:   | %FileCheck --check-prefix=CROSS %s
;
; RUN: %not %raise_cli %v_cvt_scale_pk8_bf16_fp4_co --isa=gfx1250 \
; RUN:     --emit-ir=v_cvt_scale_pk8_bf16_fp4_kernel_sel2 2>&1 \
; RUN:   | %FileCheck --check-prefix=REFUSE-SAME %s
;
; RUN: %not %raise_cli %v_cvt_scale_pk8_bf16_fp4_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 \
; RUN:     --emit-ir=v_cvt_scale_pk8_bf16_fp4_kernel_sel2 2>&1 \
; RUN:   | %FileCheck --check-prefix=REFUSE-CROSS %s
;
; Pins the gfx1250-only VOP3 packed-8 FP4 -> BF16 scaled convert
; (V_CVT_SCALE_PK8_BF16_FP4_e64, VOP3Instructions.td:1873).  See
; `hotswap/docs/matrix-translation.md §7.4` for the design note on
; the cross-target dequant expansion and the declared support set
; (scale_sel == 0 only; the two captured corpus blobs use only that
; value across 128 instances combined).
;
; Same-target lift (RUN #1): the handler emits the LLVM intrinsic
; `@llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32, i32, i32 immarg)` with
; immarg = 0 (the corpus-observed scale_sel value) and a
; <8 x bfloat> result; the write-back path bit-casts the <8 x bfloat>
; to i128 / four dwords before handing it to `writeRegVec`.  No
; cross-target refusal fires on the same-target path.
;
; Cross-target lift (RUN #2): the handler emits the per-nibble
; dequantisation expansion from
; `handle_valu.cpp::emitCvtScalePk8Bf16Fp4CrossTargetExpansion` —
; bit-algebra FP4 -> BF16 + E8M0 exponent-bits add + priority-
; ordered NaN / ±0 / overflow / subnormal merge, producing an
; <8 x bfloat> that writeRegVec consumes identically to the
; same-target path.

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

; Negative pins on the same-target path: no cross-target refusal, and
; no accidental fallback through the unscaled `cvt_pk_f32_fp8` /
; `cvt_f32_fp8` paths that share similar mnemonics.  Also pin that
; we did NOT accidentally emit the cross-target bit-algebra
; expansion instead of the intrinsic on the same-target path — the
; expansion would route through `mxfp4_nibble` named temporaries.
; CHECK-NOT: scale_sel != 0
; CHECK-NOT: amdgcn.cvt.pk.f32
; CHECK-NOT: amdgcn.cvt.f32.fp8
; CHECK-NOT: mxfp4_nibble

; ── Cross-target expansion ──────────────────────────────────────────
;
; On gfx942 (no FeatureGFX1250Insts, no MX-FP4 scaling unit), the
; handler emits the per-nibble bit-algebra dequant.  The intrinsic
; MUST NOT appear anywhere in the IR — routing the cross-target arm
; through the gfx1250-only intrinsic would lower to an
; "UnsupportedOpcode" backend failure on gfx942.
; CROSS-LABEL: define amdgpu_kernel void @v_cvt_scale_pk8_bf16_fp4_kernel(
; CROSS-NOT: call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4

; The handler's named temporaries (from
; `handle_valu.cpp::emitCvtScalePk8Bf16Fp4CrossTargetExpansion`)
; uniquify in the final IR as `%mxfp4_<name>`, `%mxfp4_<name>5`,
; `%mxfp4_<name>12`, ... (one suffix bucket per lane).  We pin the
; first-lane form (no suffix) for each checkpoint — that implicitly
; verifies emission ORDER (`mxfp4_nibble` before `mxfp4_scale_byte`
; etc. was wrong, so the checkpoints are order-insensitive via
; CROSS-DAG where semantically commutative).
;
; E8M0 scale-byte extraction: low byte of the scale i32 (scale_sel ==
; 0 only).  Emitted ONCE (before the per-lane loop) so not suffixed.
; CROSS: %mxfp4_scale_byte = and i32 %{{.*}}, 255
; CROSS: %mxfp4_is_scale_nan = icmp eq i32 %mxfp4_scale_byte, 255

; Per-lane nibble extraction: `lshr` by lane*4 then mask 0xF.  Pin
; the first-lane form.
; CROSS-DAG: %mxfp4_nibble = and i32 %{{.*}}, 15

; Per-lane FP4 E2M1 field decomposition: sign (bit 3), exp (bits 2..1),
; mant (bit 0).
; CROSS-DAG: %mxfp4_sign = and i32 %{{.*}}, 1
; CROSS-DAG: %mxfp4_exp_fp4 = and i32 %{{.*}}, 3
; CROSS-DAG: %mxfp4_mant_fp4 = and i32 %mxfp4_nibble, 1

; FP4 -> BF16 field synthesis: normal-branch exp is exp_fp4 + 126.
; CROSS-DAG: %mxfp4_bf16_exp_norm = add i32 %mxfp4_exp_fp4, 126

; Subnormal-FP4 branch selector: exp_fp4 == 0.
; CROSS-DAG: %mxfp4_is_fp4_sub = icmp eq i32 %mxfp4_exp_fp4, 0

; Scaled exponent: bf16_exp + scale_byte - 127.
; CROSS-DAG: %mxfp4_exp_plus_scale = add i32 %mxfp4_bf16_exp, %mxfp4_scale_byte
; CROSS-DAG: %mxfp4_new_exp = sub i32 %mxfp4_exp_plus_scale, 127

; Overflow saturate to ±Inf: (sign << 15) | 0x7F80 (= 32640 decimal).
; CROSS-DAG: %mxfp4_is_overflow = icmp sge i32 %mxfp4_new_exp, 255
; CROSS-DAG: %mxfp4_inf_bits = or i32 %mxfp4_sign_field, 32640

; Subnormal shift: (0x80 | bf16_mant) right-shifted by (1 - new_exp),
; clamped to zero at shift >= 8.
; CROSS-DAG: %mxfp4_implicit_1_mant = or i32 128, %mxfp4_bf16_mant
; CROSS-DAG: %mxfp4_shift_amt = sub i32 1, %mxfp4_new_exp
; CROSS-DAG: %mxfp4_shift_too_big = icmp sge i32 %mxfp4_shift_amt, 8

; Per-lane priority merge lands as a nested select into a bfloat, then
; inserted into the <8 x bfloat> accumulator.
; CROSS-DAG: %mxfp4_lane_i16 = trunc i32 %mxfp4_lane_i32 to i16
; CROSS-DAG: %mxfp4_lane_bf16 = bitcast i16 %mxfp4_lane_i16 to bfloat
; CROSS: insertelement <8 x bfloat>

; The <8 x bfloat> result flows into the same writeRegVec -> i128
; split the same-target arm produces.  Pin the bitcast so any drift
; in writeRegVec's shape surfaces immediately.
; CROSS: bitcast <8 x bfloat> %{{.*}} to i128

; The LUT table's 16-entry constants aren't emitted by the bit-
; algebra form (no `constant` or `private global` [16 x i16] shows
; up); the expansion stays register-only.  Negative pin so a
; future LUT-based refactor surfaces as a visible diff.
; CROSS-NOT: private constant [16 x i16]

; Refusal-message sanity pin: the previous release's refusal-block
; diagnostic ("is a gfx1250-only VOP3" / "no corpus kernel exercises
; today") was replaced by the expansion.  If either ever reappears
; on the cross-target path, the refusal path has regressed.
; CROSS-NOT: v_cvt_scale_pk8_bf16_fp4 is a gfx1250-only VOP3
; CROSS-NOT: no corpus kernel exercises today

; ── scale_sel != 0 refusal, both arms ──────────────────────────────
;
; Pins the handler's "declared support set" boundary: the 4-bit
; `scale_sel` field's semantics for packed-8 FP4 aren't pinned
; in-tree (no AMD ISA spec in the tree; captured corpus uses only
; scale_sel == 0).  Widening the support set requires either the
; spec landing OR a corpus kernel + canary exercising the case.
; Until then, both handler arms refuse loudly with a diagnostic
; citing §7.4.
;
; Same-target arm refuses pre-intrinsic (no ` @llvm.amdgcn.cvt.scale.pk8.bf16.fp4`
; call should appear in the diagnostic-emitting output).
; REFUSE-SAME: scale_sel != 0 is outside the declared support set
; REFUSE-SAME-SAME: matrix-translation.md §7.4
; REFUSE-SAME-NOT: call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4

; Cross-target arm refuses pre-expansion (no `mxfp4_` named
; temporaries should appear in the diagnostic-emitting output).
; REFUSE-CROSS: scale_sel != 0 is outside the declared support set
; REFUSE-CROSS-SAME: matrix-translation.md §7.4
; REFUSE-CROSS-NOT: mxfp4_nibble
; REFUSE-CROSS-NOT: call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4
