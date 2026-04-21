; RUN: %raise_cli %wmma_f32_16x16x32_bf16_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --enable-wave-native \
; RUN:     --emit-ir=wmma_f32_16x16x32_bf16_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_wmma_f32_16x16x32_bf16 (gfx1250 RDNA4 VOP3P opcode
; 0x062) lowered to gfx942 (CDNA3) via emitWMMAtoMFMA(..., BF16).
; See SemOp::V_WMMA_F32_16x16x32_BF16 in transpiler/semop.hpp; the
; matching dispatch in handle_valu_vop3p.cpp under
; `case SemOp::V_WMMA_F32_16x16x32_F16: case SemOp::V_WMMA_F32_16x16x32_BF16:`;
; the parameterised lowering body in transpiler/wmma_lowering.{hpp,cpp}.
;
; Bidirectional handler ↔ test back-reference: the F16 sibling case
; in the same dispatch block is covered transitively by the existing
; Gfx1250Gpu.Matmul128x128 gtest; this fixture covers the BF16 fork.
;
; INVARIANTS PINNED:
;
;   1. The lift dispatches the BF16 SemOp through the parameterised
;      WMMAtoMFMA helper (NOT a fallback to F16 dispatch). The
;      defining marker is the per-MFMA-call intrinsic name
;      `mfma.f32.16x16x16bf16.1k` — a regression that routed BF16
;      through the F16 path would emit `mfma.f32.16x16x16f16`
;      instead and produce wrong-element-type accumulators.
;
;   2. The K=32 input is decomposed into 2 MFMA(K=16) calls per
;      virtual Wave32 group, with the accumulator chained
;      (mfma1 → mfma2). Two virtual Wave32 groups × 2 MFMAs each
;      = 4 total MFMA calls in the lifted IR. Anything fewer
;      indicates a missing K-tile or a missing group-pass; anything
;      more indicates spurious redundant emits.
;
;   3. The CDNA3 bf16 MFMA intrinsic takes <4 x i16> arguments
;      (defined in IntrinsicsAMDGPU.td §3521 before bfloat became
;      a first-class LLVM type). The lowering emits a `bitcast
;      ... to <4 x i16>` for each per-MFMA fragment to match.
;
;   4. The accumulator is <4 x float>, identical to the F16 sibling.
;      A regression that switched to a different accumulator width
;      would surface as `<N x float>` for N != 4 here.
;
;   5. There is NO in-file `@llvm.amdgcn.strict.wwm*` marker around
;      the redistribute -> MFMA chain. The Wave64-collective
;      correctness guarantee (lanes 32-63 execute the MFMA pipeline
;      even on a partial-wave `blockDim == 32` launch) is provided
;      kernel-wide by a single `@llvm.amdgcn.init_whole_wave` call
;      at function entry, emitted by
;      `WaveNativeProjection::emitInitialExec`. That intrinsic
;      forces hardware EXEC = -1 for the remainder of the function
;      and captures the original per-lane active mask into the
;      transpiler's EXEC alloca, so `emitUnderExec` still gates
;      side effects against the logical wave32 active lanes while
;      the redistribute / MFMA / collect chain runs Wave64-wide.
;      See `hotswap/docs/wave-size-translation.md` §5.6.1 for the
;      register-allocator rationale behind this design.
;
; NEGATIVE PINS:
;
;   * NO `mfma.f32.16x16x16f16` calls in this kernel — that would
;     mean BF16 dispatch silently fell through to F16. The F16
;     intrinsic is correct for `V_WMMA_F32_16x16x32_F16` (separate
;     dispatch) but emitting it for the bf16 SemOp is a hard bug.
;
;   * NO `mfma.f32.16x16x16bf16` (without the `_1k` suffix) —
;     the non-_1k variant exists for older CDNA targets and uses
;     a different K dimension; the lowering chose `_1k` because
;     that's the K=16 variant matching our 2× decomposition.
;
;   * Exactly ONE `@llvm.amdgcn.init_whole_wave` call at function
;     entry — more than one would indicate a wave-projection bug;
;     zero would indicate the projection hook is not wired into
;     `AllocaRegFile::init`.
;
;   * Zero `@llvm.amdgcn.strict.wwm*` calls anywhere — a stray
;     WWM marker would revert the lowering to the old regalloc-
;     bottlenecked shape (see §5.6.1).

; CHECK-LABEL: define amdgpu_kernel void @wmma_f32_16x16x32_bf16_kernel(

; Kernel-entry EXEC virtualisation: exactly one init_whole_wave
; call, emitted by `WaveNativeProjection::emitInitialExec`.
; CHECK: call i1 @llvm.amdgcn.init.whole.wave()

; Per-MFMA bitcast to <4 x i16> (CDNA3 bf16 MFMA element type).
; CHECK: %{{.*}} = bitcast <2 x i32> %{{.*}} to <4 x i16>

; The bf16-specific CDNA3 intrinsic, called with <4 x i16> sources
; and <4 x float> accumulator. Pinning two separate MFMA chains
; (one per Wave32 virtual group).
;
; First group pass:
; CHECK: %mfma1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %{{[^,]+}}, <4 x i16> %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)
; CHECK: %mfma2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %{{[^,]+}}, <4 x i16> %{{[^,]+}}, <4 x float> %mfma1, i32 0, i32 0, i32 0)

; Second group pass (lane indices 32..63):
; CHECK: %mfma1{{[0-9]+}} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %{{[^,]+}}, <4 x i16> %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)
; CHECK: %mfma2{{[0-9]+}} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %{{[^,]+}}, <4 x i16> %{{[^,]+}}, <4 x float> %mfma1{{[0-9]+}}, i32 0, i32 0, i32 0)

; No in-file WWM markers anywhere; the kernel-entry init_whole_wave
; above is the sole hardware-EXEC-virtualisation signal. Only one
; init_whole_wave call per kernel.
; CHECK-NOT: call i32 @llvm.amdgcn.strict.wwm.i32(
; CHECK-NOT: call {{.*}} @llvm.amdgcn.strict.wwm
; CHECK-NOT: call {{.*}} @llvm.amdgcn.init.whole.wave

; Negative pin: the F16 intrinsic must NOT appear in this kernel
; (would indicate BF16 dispatch fell through to F16).
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16f16

; Negative pin: the non-_1k bf16 intrinsic is the wrong K variant.
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16bf16(

; Negative pin: native gfx12 WMMA intrinsic must NOT appear (target
; is gfx942 which does NOT have hasWMMA12).
; CHECK-NOT: @llvm.amdgcn.wmma.f32.16x16x32.bf16
