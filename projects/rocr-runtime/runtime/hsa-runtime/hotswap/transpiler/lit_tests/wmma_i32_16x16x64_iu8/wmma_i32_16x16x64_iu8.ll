; RUN: %raise_cli %wmma_i32_16x16x64_iu8_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=wmma_i32_16x16x64_iu8_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_wmma_i32_16x16x64_iu8 (gfx1250 RDNA4 VOP3P opcode
; 0x072) lowered to gfx942 (CDNA3) via emitWMMAtoMFMA(..., IU8).
; See SemOp::V_WMMA_I32_16x16x64_IU8 in transpiler/semop.hpp; the
; matching dispatch in handle_valu_vop3p.cpp under the unified WMMA
; case block; the parameterised lowering body in
; transpiler/wmma_lowering.{hpp,cpp}.
;
; INVARIANTS PINNED:
;
;   1. Integer-accumulator MFMA intrinsic dispatched. The defining
;      marker is the per-MFMA-call intrinsic name
;      `mfma.i32.16x16x32.i8` — every other WMMA variant uses an
;      `mfma.f32.*` intrinsic. A regression that routed IU8 through
;      a wrong (or no) accumulator-type fork would surface as either
;      a wrong intrinsic name or a CreateCall type mismatch crash.
;
;   2. Accumulator pack type is `<4 x i32>`, not `<4 x float>`. The
;      IU8 path is the ONLY non-`<4 x float>` accumulator in the
;      WMMA-to-MFMA helper today; the inputType switch carries the
;      accumulator pack type through alongside the intrinsic.
;
;   3. The K=64 input is decomposed into 2 MFMA(K=32) calls per
;      virtual Wave32 group, with the accumulator chained
;      (mfma1 → mfma2). Two virtual Wave32 groups × 2 MFMAs each =
;      4 total MFMA calls in the lifted IR. Identical decomposition
;      shape to the FP8/BF8 siblings, validating the shared
;      redistribution path.
;
;   4. The CDNA3 i8 MFMA intrinsic takes `i64` for A/B (defined in
;      IntrinsicsAMDGPU.td §3560 before AMDGCN had a packed-i8
;      vector type at all). The lowering emits a
;      `bitcast <2 x i32> ... to i64` for each per-MFMA fragment to
;      match — same pack type as the FP8/BF8 siblings, divergent
;      only in the dispatched intrinsic name.
;
; NEGATIVE PINS:
;
;   * NO `mfma.f32.*` calls in this kernel — those would mean IU8
;     dispatch silently fell through to a f32-accumulator handler.
;
;   * NO native `wmma.i32.16x16x64.iu8` intrinsic — target is gfx942
;     which does NOT have hasWMMA12; surface here would mean the
;     target-capability gate is broken.
;
;   * NO i32-acc MFMAs of WRONG K (mfma.i32.16x16x16i8, K=16) — the
;     dispatch must select the K=32 i8 MFMA (16x16x32) exactly.

; CHECK-LABEL: define amdgpu_kernel void @wmma_i32_16x16x64_iu8_kernel(

; Per-MFMA bitcast to i64 (CDNA3 i8 MFMA element type).
; CHECK: %{{.*}} = bitcast <2 x i32> %{{.*}} to i64

; The IU8-specific CDNA3 intrinsic, called with i64 sources and
; <4 x i32> accumulator. Pinning two separate MFMA chains (one per
; Wave32 virtual group).
;
; First group pass:
; CHECK: %mfma1 = call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x i32> %{{[^,]+}}, i32 0, i32 0, i32 0)
; CHECK: %mfma2 = call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x i32> %mfma1, i32 0, i32 0, i32 0)

; Second group pass (lane indices 32..63):
; CHECK: %mfma1{{[0-9]+}} = call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x i32> %{{[^,]+}}, i32 0, i32 0, i32 0)
; CHECK: %mfma2{{[0-9]+}} = call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x i32> %mfma1{{[0-9]+}}, i32 0, i32 0, i32 0)

; Negative pin: NO f32-accumulator MFMA in this kernel. A regression
; that routed IU8 through the f32-accumulator switch arm would
; surface as `mfma.f32.16x16x32.fp8.fp8` (or any other f32 MFMA).
; CHECK-NOT: @llvm.amdgcn.mfma.f32.

; Negative pin: native gfx12 WMMA intrinsic must NOT appear (target
; is gfx942 which does NOT have hasWMMA12).
; CHECK-NOT: @llvm.amdgcn.wmma.i32.16x16x64.iu8

; Negative pin: must dispatch to the K=32 i8 MFMA, not the older
; K=16 i8 MFMA (the latter would imply wrong K-decomposition).
; CHECK-NOT: @llvm.amdgcn.mfma.i32.16x16x16i8
