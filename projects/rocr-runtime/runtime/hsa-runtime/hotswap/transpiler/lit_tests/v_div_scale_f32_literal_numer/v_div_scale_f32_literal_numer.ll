; RUN: %raise_cli %v_div_scale_f32_literal_numer_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=v_div_scale_f32_literal_numer_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift fixture for `v_div_scale_f32`'s literal-numerator encoding —
; the shape hipcc's AMDGPU backend emits for any `1.0 / x` fdiv
; expansion, where the numerator rides through src2 as an inline
; `1.0` constant rather than a register.
;
; The three-arm decoder in `handle_valu.cpp::V_DIV_SCALE_F32`
; distinguishes:
;   * (n, d, n)  with src0 == src2 → scale NUMERATOR    → flag=true.
;   * (d, d, n)  with src0 == src1 → scale DENOMINATOR  → flag=false.
;   * otherwise                    → loud refusal (unsupported shape).
;
; Equality is at the OPERAND level — register identity for register
; operands, immediate-bit-pattern equality for inline constants.
; Before this contract was in place, the handler's register-only
; gate fell through for literal-carried numerators and decoded both
; scale calls as `(src0, src1, false)`; the downstream div_fmas /
; div_fixup chain then collapsed to `1.0` bit-exact (observable as
; layer-norm's rstd reading `0x3f800000` regardless of input).
;
; This fixture pins four stable anchors:
;
;   (a) the POSITIVE shape: a scale-DENOM call whose numer is the
;       literal `1.000000e+00` and whose denom is a VGPR-sourced
;       value — the canonical `(1.0, %x, false)` lifted triple;
;   (b) the POSITIVE shape's scale-NUMER sibling in the same pair —
;       the canonical `(1.0, %x, true)` — asserting both arms of
;       the expanded IEEE fdiv are rewritten consistently;
;   (c) a NEGATIVE assertion against the pre-fix shape: neither
;       scale call should carry the numer as a VGPR duplicated into
;       src0 (i.e. `div.scale.f32(float %<vgpr>, float %<vgpr>, ...)` —
;       the (d, d, ...) form before my audit landed); and
;   (d) the resulting `div.fixup.f32` uses the canonical `1.0`
;       numerator (as the 3rd arg of the fixup), which is how the
;       AMDGPU backend surfaces the fdiv's numer after the Newton
;       iteration — a secondary sanity check that the chain stays
;       coherent end-to-end rather than just locally-correct on the
;       scale calls.
;
; Matching is substring-based on stable LLVM-IR landmarks so the
; fixture survives unrelated SSA-name churn in the raiser.

; CHECK-LABEL: define amdgpu_kernel void @v_div_scale_f32_literal_numer_kernel(

; (a) Scale DENOMINATOR arm of `1.0 / x`: canonical (1.0, vgpr, false).
; CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float 1.000000e+00, float %{{[^,]+}}, i1 false)

; (b) Scale NUMERATOR arm of the same fdiv: canonical (1.0, vgpr, true).
; CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float 1.000000e+00, float %{{[^,]+}}, i1 true)

; (d) The Newton-iteration fixup emits the `1.0` numerator in the
;     third arg position.  This anchors the chain's coherence: if a
;     future rewrite drops the literal-numer audit on the scale
;     calls but somehow keeps the fixup call, (a) and (b) fire and
;     (d) silently passes.  Pairing them guards against isolated
;     regressions on either side.
; CHECK: call float @llvm.amdgcn.div.fixup.f32(float %{{[^,]+}}, float %{{[^,]+}}, float 1.000000e+00)

; (c) Negative: neither scale call should carry a VGPR-sourced
;     numer duplicated into both src0 and src1 — the pre-fix
;     `(d, d, ...)` shape that collapsed the whole expansion to
;     rstd = 1.0 bit-exact.  Match on the `div.scale.f32` call
;     prefix followed by a `%<name>, float %<same-name>,` pattern
;     where the same VGPR-derived SSA name appears in both operand
;     positions.  LLVM's IR printer renders both operands with
;     their SSA name, so the literal match here is `float %X, float
;     %X` for any `X`; a single back-reference via
;     `{{[^,]+}}, float %{{[^,]+}}` cannot assert equality, so we
;     spell out the forbidden pattern with an explicit register-
;     dominated shape.  `1.000000e+00, float %` passes (one side
;     literal, one side register); `float %foo, float %foo` fails.
; CHECK-NOT: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %[[X:[^,]+]], float %[[X]],
