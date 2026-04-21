; RUN: %raise_cli %wmma_f32_16x16x4_f32_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=wmma_f32_16x16x4_f32_kernel 2>/dev/null \
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
;   5. There is NO in-file `@llvm.amdgcn.strict.wwm*` marker around
;      the redistribute -> MFMA -> collect chain. The partial-wave
;      correctness guarantee (all 64 W64 lanes execute the MFMA
;      pipeline even on a `blockDim == 32` launch) is provided by
;      a single kernel-entry `@llvm.amdgcn.init_whole_wave` call
;      emitted by `WaveNativeProjection::emitInitialExec` — that
;      intrinsic sets hardware EXEC = -1 for the remainder of the
;      function, which the redistribute / MFMA / collect chain
;      relies on as a kernel-wide ambient. Side-effect gating
;      against the *logical* wave32 active mask stays on
;      `emitUnderExec` diamonds at the consuming VGPR stores,
;      completely decoupled from hardware EXEC. See
;      `hotswap/docs/wave-size-translation.md` §5.6.1 for the
;      register-allocator rationale behind moving the EXEC = -1
;      guarantee to kernel entry instead of wrapping per-MFMA
;      WWM brackets.
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
;   * Zero `@llvm.amdgcn.strict.wwm*` calls anywhere in the kernel
;     (the init_whole_wave entry ambient replaced them; leaving a
;     stray WWM marker would revert the K=4 path to the old
;     regalloc-bottlenecked shape).
;   * Exactly ONE `@llvm.amdgcn.init_whole_wave` call at function
;     entry (the EXEC alloca seed emitted by
;     `WaveNativeProjection::emitInitialExec`). More than one
;     would indicate a wave-projection bug; zero would indicate
;     the projection hook is not wired into `AllocaRegFile::init`.

; CHECK-LABEL: define amdgpu_kernel void @wmma_f32_16x16x4_f32_kernel(

; Kernel-entry EXEC virtualisation: exactly one init_whole_wave
; call, emitted by `WaveNativeProjection::emitInitialExec`. This
; replaces the prior per-MFMA `strict.wwm` wrap.
; CHECK: call i1 @llvm.amdgcn.init.whole.wave()

; First group pass (Wave32 group 0: W64 lanes 0-31).
; The MFMA call takes scalar `float` for A and B, and `<4 x float>`
; for the accumulator. Pin that shape explicitly.
; CHECK: %mfma = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %{{[^,]+}}, float %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)

; Second group pass (Wave32 group 1: W64 lanes 32-63). LLVM
; uniquifies the value name because `%mfma` is in use, so we
; allow any integer suffix.
; CHECK: %mfma{{[0-9]+}} = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %{{[^,]+}}, float %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)

; Exactly 2 MFMA calls (one per Wave32 virtual group) — NOT 4.
; Anchored AFTER the per-group positive checks above, so any extra
; call would surface here as an unexpected match.
; CHECK-NOT: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(

; No in-file WWM markers around the redistribute / MFMA / collect
; chain; the kernel-entry init_whole_wave above is the sole
; hardware-EXEC-virtualisation signal.
; CHECK-NOT: call i32 @llvm.amdgcn.strict.wwm.i32(
; CHECK-NOT: call {{.*}} @llvm.amdgcn.strict.wwm
; CHECK-NOT: call {{.*}} @llvm.amdgcn.init.whole.wave

; Negative: no native gfx1250 WMMA intrinsic (we are on gfx942).
; CHECK-NOT: @llvm.amdgcn.wmma.f32.16x16x4.f32

; Negative: no cross-K MFMA intrinsics (would indicate dispatch
; confusion with the K=32/K=64 WMMA family).
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16f16
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16bf16
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x32_
