; RUN: %raise_cli %wmma_f32_16x16x4_f32_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=wmma_f32_16x16x4_f32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Cross-target lift fixture for v_wmma_f32_16x16x4_f32 (gfx1250 RDNA4
; VOP3P opcode 0x05D) lowered to gfx942 (CDNA3) via
; `emitWMMAtoMFMA_F32_16x16x4` (transpiler/wmma_lowering.cpp).
; Companion fixture to `wmma_f32_16x16x4_f32_same_target.ll`, which
; pins the same-target (gfx1250 -> gfx1250) native-intrinsic-emit
; path.
;
; This SemOp stands ALONE from the K=32 (16-bit) and K=64 (8-bit)
; WMMA family (`SemOp::V_WMMA_F32_16x16x32_*`,
; `SemOp::V_WMMA_F32_16x16x64_*`, `SemOp::V_WMMA_I32_16x16x64_IU8`)
; because:
;
;   * The per-Wave32-lane A/B fragment is `<2 x f32>` (2 VGPRs, not
;     the 8-VGPR <16 x t> / <8 x i32> fragment used by the K=32 /
;     K=64 family).
;   * The target MFMA is `mfma_f32_16x16x4f32` with signature
;     `(float A, float B, <4 x float> C)` — so the per-lane MFMA
;     input is ONE f32 dword, not a `<4 x t>` / i64 pack.
;   * K=4 matches the MFMA K dimension exactly, so each Wave32
;     group dispatches ONE MFMA call (not 2 chained like the
;     K=32/K=64 path where each group splits into 2× K=16 calls).
;
; The rest of the lowering is shared: ds.bpermute-driven lane
; redistribution via the same `redistributeAcc` / `collectResult`
; helpers that drive the K=32/K=64 decomposition.
;
; LAYOUT CONTRACT (corpus-verifiable, not just IR-pinned).
; The per-lane (i, k) layout assumed for the gfx1250 V_WMMA_F32_16x16x4_F32
; A/B inputs is the natural extrapolation of the documented K=32/K=64
; pattern (see `wmma_lowering.cpp`):
;
;   i = lane % 16
;   k = 2*floor(lane/16) + GPR
;
; Validated against the hipBLASLt/Tensile `SS_SS_HA_Bias_SAV_UA`
; family of f32 GEMM kernels (macro-tile `MT32x32x16`, WMMA shape
; `MI16x16x1`, wave32) — the 4 corpus kernels that surfaced this
; gap in the cross-target kerneldex sweep. `tools/tensile_gold_verify/`
; compares the lifted gfx942 output against a gfx1250 gold reference,
; so a layout mismatch would show up as a numerical regression, not
; a silent wrong answer.
;
; INVARIANTS PINNED:
;
;   1. Cross-target lift dispatches to mfma_f32_16x16x4f32 (NOT
;      the native gfx1250 WMMA intrinsic, which would fail at
;      gfx942 codegen). Defining marker: the per-group MFMA
;      intrinsic name `mfma.f32.16x16x4f32`.
;
;   2. The MFMA signature is `(float, float, <4 x float>)` — NOT
;      the K=16 / K=32 shapes (which would use `<4 x t>` / i64 A/B
;      and would indicate cross-K dispatch confusion).
;
;   3. ONE MFMA call per Wave32 virtual group (2 total across
;      both groups). The K=32/K=64 family produces 2 MFMAs per
;      group (4 total); any extra `mfma.f32.16x16x4f32` call here
;      would indicate an incorrect K-decomposition.
;
;   4. The accumulator is packed into `<4 x float>` (standard
;      gfx942 MFMA C/D layout) from the WMMA's `<8 x float>`
;      per-Wave32-lane fragment via the shared `redistributeAcc`
;      path.
;
;   5. Each Wave32 group pass is wrapped in ONE
;      `@llvm.amdgcn.strict.wwm.v8i32` call that packs the 8
;      result dwords into a single `<8 x i32>` vector and fences
;      the entire redistribute -> MFMA -> collect chain in
;      Whole-Wave Mode. This guarantees all 64 W64 lanes execute
;      the MFMA pipeline regardless of the caller's EXEC mask,
;      so partial-wave Wave32 launches (blockDim == 32) do not
;      leave lanes 32-63 inactive and feed garbage into MFMA
;      (see "Whole-wave mode" section in wmma_lowering.cpp /
;      .hpp for the full correctness argument).
;
; NEGATIVE PINS:
;
;   * NO call to the native gfx1250 WMMA intrinsic
;     `llvm.amdgcn.wmma.f32.16x16x4.f32` — target is gfx942 which
;     does NOT have hasTensorOps, so any native emit here would
;     fail at codegen (and silently mis-lower if the codegen
;     somehow accepted it).
;   * NO `mfma.f32.16x16x16f16` / `mfma.f32.16x16x16bf16*` /
;     `mfma.f32.16x16x32_*` — K=16 / K=32 MFMA calls would mean
;     the K=4 SemOp fell through to the K=32/K=64 path.
;   * Exactly 2 `mfma.f32.16x16x4f32` calls (not 4) — the K=4
;     decomposition is 1 MFMA per Wave32 virtual group.
;   * Exactly 2 `@llvm.amdgcn.strict.wwm.v8i32` calls (one per
;     Wave32 virtual group). Any other count would mean the WWM
;     fencing is not per-group (too few) or got duplicated / is
;     wrapping per-dword rather than per-group (too many).

; CHECK-LABEL: define amdgpu_kernel void @wmma_f32_16x16x4_f32_kernel(

; First group pass (Wave32 group 0: W64 lanes 0-31).
; The MFMA call takes scalar `float` for A and B, and `<4 x float>`
; for the accumulator. Pin that shape explicitly.
; CHECK: %mfma = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %{{[^,]+}}, float %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)

; The first group's 8 result dwords are packed into a single
; `<8 x i32>` and handed to `@llvm.amdgcn.strict.wwm` so the
; whole redistribute -> MFMA -> collect chain runs in WWM.
; CHECK: call <8 x i32> @llvm.amdgcn.strict.wwm.v8i32(<8 x i32> %{{[^)]+}})

; Second group pass (Wave32 group 1: W64 lanes 32-63). LLVM
; uniquifies the value name because `%mfma` is in use, so we
; allow any integer suffix.
; CHECK: %mfma{{[0-9]+}} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %{{[^,]+}}, float %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)

; Second group's WWM fence (same <8 x i32> shape).
; CHECK: call <8 x i32> @llvm.amdgcn.strict.wwm.v8i32(<8 x i32> %{{[^)]+}})

; Exactly 2 MFMA calls (one per Wave32 virtual group) — NOT 4.
; Anchored AFTER the per-group positive checks above, so any extra
; call would surface here as an unexpected match.
; CHECK-NOT: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(

; Exactly 2 strict.wwm calls (one per Wave32 virtual group).
; CHECK-NOT: call <8 x i32> @llvm.amdgcn.strict.wwm.v8i32(

; Negative: no native gfx1250 WMMA intrinsic (we are on gfx942).
; CHECK-NOT: @llvm.amdgcn.wmma.f32.16x16x4.f32

; Negative: no cross-K MFMA intrinsics (would indicate dispatch
; confusion with the K=32/K=64 WMMA family).
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16f16
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16bf16
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x32_
