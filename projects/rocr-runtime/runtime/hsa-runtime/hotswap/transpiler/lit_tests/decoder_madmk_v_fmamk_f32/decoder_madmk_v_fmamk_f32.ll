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

; The lifted IR must contain an FMA call with three float operands.
; The K=π literal (encoding 0x40490FDB) lands at the middle operand
; slot per the V_FMAMK_F32 handler convention (`srcF(0)=src0,
; srcF(1)=K, srcF(2)=src2` → `fma(s0, k, s2)`).
;
; We assert the IR shape (three float operands) and that the K
; literal appears SOMEWHERE in the IR, allowing for either of LLVM
; IR's two valid f32-constant renderings:
;   * decimal/hex-int form like `0x40490FDB` or `3.141593e+00`
;   * hex-double form like `0x400921FB60000000` (the f64 expansion
;     of pi-as-f32)
; LLVM has historically rendered f32 constants in either form
; depending on whether the value is exactly representable as a
; short decimal — pinning a single form would couple the test to
; the IR printer's heuristic. The shape check + K-literal-anywhere
; check is sufficient to catch:
;   * decoder regressions (no FMA call at all → fails CHECK)
;   * MADMK srcMap reordering (K imm in wrong operand slot →
;     fails the K-literal check below if K stops being inline)
;   * imm extraction truncation/sign-extension (K bit pattern
;     wrong → fails the K-literal check below)
;
; Match the FMA call with the K=π literal at the middle (second)
; operand slot — that's where the V_FMAMK_F32 handler routes the
; K-imm per its `srcF(0)=src0, srcF(1)=K, srcF(2)=src2` convention
; (handle_valu.cpp). The literal can render in either of LLVM IR's
; valid f32-constant forms:
;   * hex-int form like `0x40490FDB`
;   * f64-expanded hex-double form like `0x400921FB60000000` (the
;     f64 expansion of pi-as-f32; LLVM's IR printer chooses this
;     when the value isn't exactly representable as a short
;     decimal — which is the case for pi)
; A `3\.14159` decimal-float form is also accepted in case a future
; LLVM defaults to that. Pinning a single form would couple the
; test to the IR printer's heuristic; matching any of the three
; correct renderings is sufficient to catch:
;   * decoder regressions (no FMA call at all → CHECK fails)
;   * MADMK srcMap reordering (K imm in wrong operand slot → the
;     literal wouldn't be in the second-operand position → CHECK
;     fails because the constraint is positional)
;   * imm extraction truncation/sign-extension (K bit pattern wrong
;     → none of the three alternatives match → CHECK fails)
;
; CHECK: call {{.*}}float @llvm.fma.f32(float {{.*}}, float {{0x40490FDB|0x400921FB60000000|3\.14159[0-9]+}}, float {{.*}})

; The intrinsic declaration must be present.
; CHECK: declare {{.*}}float @llvm.fma.f32(float, float, float)
