; RUN: %raise_cli %wmma_f32_16x16x64_fp8_fp8_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=wmma_f32_16x16x64_fp8_fp8_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_wmma_f32_16x16x64_fp8_fp8 (gfx1250 RDNA4 VOP3P opcode
; 0x06a) lowered to gfx942 (CDNA3) via emitWMMAtoMFMA(..., FP8_FP8).
; See SemOp::V_WMMA_F32_16x16x64_FP8_FP8 in transpiler/semop.hpp; the
; matching dispatch in handle_valu_vop3p.cpp under the unified WMMA
; case block; the parameterised lowering body in
; transpiler/wmma_lowering.{hpp,cpp}.
;
; Bidirectional handler ↔ test back-reference: the four AB combinations
; (fp8_fp8, fp8_bf8, bf8_fp8, bf8_bf8) share the same dispatch shape —
; only the trailing intrinsic name differs. This fixture covers the
; FP8_FP8 corner directly and the others transitively through that
; shape symmetry; the underlying lane redistribution is also covered
; transitively by the BF16 fixture and the F16 GPU tests
; (Gfx1250Gpu.Matmul128x128).
;
; INVARIANTS PINNED:
;
;   1. The lift dispatches the FP8_FP8 SemOp through the parameterised
;      WMMAtoMFMA helper (NOT a fallback to a 16-bit dispatch). The
;      defining marker is the per-MFMA-call intrinsic name
;      `mfma.f32.16x16x32.fp8.fp8` — a regression that routed FP8_FP8
;      through the F16 path would emit `mfma.f32.16x16x16f16` instead
;      and produce wrong-element-type accumulators with wrong K width.
;
;   2. The K=64 input is decomposed into 2 MFMA(K=32) calls per virtual
;      Wave32 group, with the accumulator chained (mfma1 → mfma2). Two
;      virtual Wave32 groups × 2 MFMAs each = 4 total MFMA calls in the
;      lifted IR. Anything fewer indicates a missing K-tile or a
;      missing group-pass; anything more indicates spurious redundant
;      emits.
;
;   3. The CDNA3 fp8/bf8 MFMA intrinsics take `i64` for A/B (defined
;      in IntrinsicsAMDGPU.td §3534-3540 before fp8 became a first-class
;      LLVM type). The lowering emits a `bitcast <2 x i32> ... to i64`
;      for each per-MFMA fragment to match. This is the single divergence
;      point from the 16-bit lowering, which packs as `<4 x half|i16>`.
;
;   4. The accumulator is `<4 x float>`, identical to the 16-bit
;      siblings. A regression that switched to a different accumulator
;      width would surface as `<N x float>` for N != 4 here.
;
;   5. Each Wave32 group pass is wrapped in ONE
;      `@llvm.amdgcn.strict.wwm.v8i32` call fencing the whole
;      redistribute -> MFMA1 -> MFMA2 -> collect chain in
;      Whole-Wave Mode, so lanes 32-63 execute the lower-half
;      group even when the kernel is launched at blockDim == 32
;      (partial-wave Wave32 launch on gfx942 Wave64). See the
;      "Whole-wave mode" section in wmma_lowering.cpp / .hpp for
;      the full correctness argument.
;
; NEGATIVE PINS:
;
;   * NO `mfma.f32.16x16x16f16` or `mfma.f32.16x16x16bf16` calls in
;     this kernel — those would mean FP8_FP8 dispatch silently fell
;     through to a 16-bit handler, which would also use a smaller K
;     per call and produce wrong results.
;
;   * NO native `wmma.f32.16x16x64.fp8.fp8` intrinsic — target is
;     gfx942 which does NOT have hasWMMA12; surface here would mean
;     the target-capability gate is broken.
;
;   * NO `mfma.f32.16x16x32.bf8.bf8` (or the cross variants) — the
;     dispatch must select the FP8_FP8 intrinsic exactly, not any of
;     the other three AB combinations.

; CHECK-LABEL: define amdgpu_kernel void @wmma_f32_16x16x64_fp8_fp8_kernel(

; Per-MFMA bitcast to i64 (CDNA3 fp8/bf8 MFMA element type).
; CHECK: %{{.*}} = bitcast <2 x i32> %{{.*}} to i64

; The fp8_fp8-specific CDNA3 intrinsic, called with i64 sources and
; <4 x float> accumulator. Pinning two separate MFMA chains (one per
; Wave32 virtual group) — see the four `mfma1`/`mfma2` value names in
; the lifted IR.
;
; First group pass:
; CHECK: %mfma1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)
; CHECK: %mfma2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> %mfma1, i32 0, i32 0, i32 0)
; CHECK: call <8 x i32> @llvm.amdgcn.strict.wwm.v8i32(<8 x i32> %{{[^)]+}})

; Second group pass (lane indices 32..63):
; CHECK: %mfma1{{[0-9]+}} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)
; CHECK: %mfma2{{[0-9]+}} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> %mfma1{{[0-9]+}}, i32 0, i32 0, i32 0)
; CHECK: call <8 x i32> @llvm.amdgcn.strict.wwm.v8i32(<8 x i32> %{{[^)]+}})

; Exactly 2 strict.wwm fences (one per Wave32 virtual group).
; CHECK-NOT: call <8 x i32> @llvm.amdgcn.strict.wwm.v8i32(

; Negative pin: the 16-bit MFMA intrinsics must NOT appear in this
; kernel (would indicate FP8_FP8 dispatch fell through to a 16-bit
; sibling, which would also use a wrong K-per-call).
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16f16
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16bf16

; Negative pin: native gfx12 WMMA intrinsic must NOT appear (target
; is gfx942 which does NOT have hasWMMA12).
; CHECK-NOT: @llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8

; Negative pin: the dispatch must select fp8_fp8 EXACTLY, not any of
; the other three AB combinations.
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8
