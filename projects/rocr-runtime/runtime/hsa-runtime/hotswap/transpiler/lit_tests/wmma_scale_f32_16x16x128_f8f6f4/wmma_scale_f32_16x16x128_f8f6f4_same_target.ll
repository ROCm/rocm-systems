; RUN: %raise_cli %wmma_scale_f32_16x16x128_f8f6f4_co --isa=gfx1250 \
; RUN:     --target-isa=gfx1250 --emit-ir=wmma_scale_f32_16x16x128_f8f6f4_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Lift fixture for v_wmma_scale_f32_16x16x128_f8f6f4 (gfx1250 RDNA4
; VOP3PX2 opcode 0x033, ScaledWMMA family) — same-target
; (gfx1250 -> gfx1250) intrinsic-emit path. Pins the principled lift
; in transpiler/handle_valu_vop3p.cpp under
; SemOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4 when
; `ctx.targetIsa.hasTensorOps` is true. Companion fixture to
; `wmma_scale_f32_16x16x128_f8f6f4.ll`, which pins the cross-target
; (gfx942) loud refusal.
;
; 18 MC pseudos collapse onto this single SemOp (9 mantissa pairs
; `{f4,f6,f8} A × {f4,f6,f8} B` × `_twoaddr`/`_threeaddr`), per
; `WMMA_F8F6F4_Profiles` in VOP3PInstructions.td:1908. The per-matrix
; dword count is encoded by the opcode's `_fA_fB_w32_*` suffix
; (f8 → 16 dwords, f6 → 12, f4 → 8) and the in-family element
; distinction (BF8 vs FP8 within f8; BF6 vs FP6 within f6) lives in
; the `matrix_a_fmt` / `matrix_b_fmt` named-immediate operands
; (`enum MatrixFMT { FP8=0, BF8=1, FP6=2, BF6=3, FP4=4 }`,
; SIDefines.h:1052-1058). The HIP fixture compiles to the
; `_f8_f8_w32_threeaddr` MC pseudo with `matrix_a_fmt:MATRIX_FMT_BF8`
; and `matrix_b_fmt:MATRIX_FMT_FP8` (default) — the same shape as the
; failing kerneldex GEMMs (B8F8 / F8B8 ID73f0 contractions).
;
; The native intrinsic `int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4`
; (IntrinsicsAMDGPU.td:4138, class
; `AMDGPUWmmaScaleIntrinsicModsC<llvm_i32_ty>`) takes 14 args:
;
;   <8 x float> llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4(
;       i32 matrix_a_fmt, <NA x i32> A,
;       i32 matrix_b_fmt, <NB x i32> B,
;       i16 c_mod, <8 x float> C,
;       i32 matrix_a_scale, i32 matrix_a_scale_fmt, i32 scale_src0,
;       i32 matrix_b_scale, i32 matrix_b_scale_fmt, i32 scale_src1,
;       i1 matrix_a_reuse, i1 matrix_b_reuse)
;
; Overloaded on D, A and B element vector types, so the f8_f8 form
; mangles to `.v8f32.v16i32.v16i32`. The handler decodes named
; operands via `AMDGPU::getNamedOperandIdx` (`matrix_a_fmt`,
; `matrix_b_fmt`, `matrix_a_scale`, `matrix_b_scale`,
; `matrix_a_scale_fmt`, `matrix_b_scale_fmt`, `scale_src0`,
; `scale_src1`, `matrix_a_reuse`, `matrix_b_reuse`,
; `src2_modifiers`) so any future TableGen reshuffle of the scaled-
; WMMA Ins64 layout flows in for free.
;
; INVARIANTS PINNED:
;
;   1. The native gfx1250 scaled-WMMA intrinsic is emitted (NOT a
;      fallback to MFMA / non-scaled WMMA / a different K-width).
;      The defining marker is the intrinsic name
;      `llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4` with mangled
;      types `.v8f32.v16i32.v16i32` reflecting the f8_f8 fragment
;      shape from the HIP fixture.
;
;   2. The `matrix_a_fmt` arg is `i32 1` (MATRIX_FMT_BF8) and
;      `matrix_b_fmt` is `i32 0` (MATRIX_FMT_FP8 default) — exactly
;      what the disassembled HIP fixture shows
;      (`matrix_a_fmt:MATRIX_FMT_BF8`, matrix_b_fmt omitted at default
;      0). The accumulator type is `<8 x float>` and the A/B fragment
;      types are `<16 x i32>` (the f8 family width).
;
;   3. The `scale_src0` and `scale_src1` slots carry the kernel's
;      runtime VGPR-loaded scale-source values, NOT immediates —
;      pinned via `i32 %{{.+}}` so any regression that hard-codes
;      scales to 0 surfaces immediately.
;
;   4. The reuse args use the canonical defaults: `i1 false` for
;      `matrix_a_reuse` / `matrix_b_reuse` (matches what the HIP
;      builtin emits when `_Constant bool` reuse args are passed
;      `false`).
;
; NEGATIVE PINS:
;
;   * NO call to `llvm.amdgcn.mfma.scale.*` — the cross-target gfx942
;     decomposition path is unimplemented and would mean the
;     same-target lift silently mis-dispatched.
;   * NO call to the non-scaled `llvm.amdgcn.wmma.f32.16x16x128.*`
;     intrinsic — a regression that drops the scale-source operands
;     would land here.
;   * NO call to a different K-width WMMA intrinsic
;     (`16x16x32`, `16x16x64`, `16x16x4`) — would indicate cross-K
;     dispatch confusion.

; CHECK-LABEL: define amdgpu_kernel void @wmma_scale_f32_16x16x128_f8f6f4_kernel(

; The native gfx1250 scaled-WMMA intrinsic, with the f8_f8 fragment
; shape reflected in the mangled types `.v8f32.v16i32.v16i32`.
; matrix_a_fmt = MATRIX_FMT_BF8 (1), matrix_b_fmt = MATRIX_FMT_FP8
; (0), C_mod = 0, scale {a,b}_scale = 0, scale {a,b}_scale_fmt = 0,
; scale_src0 / scale_src1 are runtime VGPR values, reuse a/b = false.
; IR: %wmma_scale{{[0-9]*}} = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(
; IR-SAME: i32 1, <16 x i32> %{{[^,]+}},
; IR-SAME: i32 0, <16 x i32> %{{[^,]+}},
; IR-SAME: i16 0, <8 x float> %{{[^,]+}},
; IR-SAME: i32 0, i32 0, i32 %{{[^,]+}},
; IR-SAME: i32 0, i32 0, i32 %{{[^,]+}},
; IR-SAME: i1 false, i1 false)

; Negative: no MFMA scale fallback (K=128 scaled-WMMA → MFMA
; decomposition is unimplemented in wmma_lowering.cpp).
; IR-NOT: @llvm.amdgcn.mfma.scale.

; Negative: no non-scaled K=128 WMMA dispatch (would drop the scale
; operands).
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4(

; Negative: no other-K WMMA dispatch (cross-K dispatch confusion).
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x32.
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x64.
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x4.
