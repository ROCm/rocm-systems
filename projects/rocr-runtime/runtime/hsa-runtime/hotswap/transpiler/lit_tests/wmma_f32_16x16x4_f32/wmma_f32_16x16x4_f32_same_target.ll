; RUN: %raise_cli %wmma_f32_16x16x4_f32_co --isa=gfx1250 \
; RUN:     --target-isa=gfx1250 --emit-ir=wmma_f32_16x16x4_f32_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Lift fixture for v_wmma_f32_16x16x4_f32 (gfx1250 RDNA4 VOP3P
; opcode 0x05D) — same-target (gfx1250 -> gfx1250) intrinsic-emit
; path. Pins the principled lift in transpiler/handle_valu_vop3p.cpp
; under SemOp::V_WMMA_F32_16x16x4_F32 when
; `ctx.targetIsa.hasTensorOps` is true. Companion fixture to
; `wmma_f32_16x16x4_f32.ll`, which pins the cross-target (gfx942)
; loud refusal.
;
; This SemOp stands ALONE from the K=32 (16-bit) and K=64 (8-bit)
; WMMA family because (a) the per-lane A/B fragment is `<2 x f32>`
; (only 2 dwords) instead of `<16 x t>` (16-bit) or `<8 x i32>`
; (8-bit), and (b) `emitWMMAtoMFMA` in transpiler/wmma_lowering.cpp
; is parameterised on 16-/8-bit element packing and has NO K=4 f32
; codepath. The refusal sibling test pins the gfx942 contract; this
; fixture pins the same-target intrinsic-emit shape.
;
; The native intrinsic `int_amdgcn_wmma_f32_16x16x4_f32` is declared
; inside `AMDGPUWMMAIntrinsicsGFX1250` (gated by `isGFX125xOnly` in
; IntrinsicsAMDGPU.td:4113-4114). The matching call shape is
; `AMDGPUWmmaIntrinsicModsAllReuse` (8 args):
;
;   <8 x float> llvm.amdgcn.wmma.f32.16x16x4.f32(
;       i1 a_neg, <2 x float> a,
;       i1 b_neg, <2 x float> b,
;       i16 c_mod, <8 x float> c,
;       i1 a_reuse, i1 b_reuse)
;
; The handler emits the modifier args as `i1 false` / `i16 0` to
; match what the disassembler surfaces for the failing kerneldex
; kernels (clang's `_Constant` builtin args constrain modifiers to
; constants, and the failing GEMMs always emit them at default).
;
; INVARIANTS PINNED:
;
;   1. The native gfx1250 intrinsic is emitted (NOT a fallback to
;      MFMA). The defining marker is the intrinsic name
;      `llvm.amdgcn.wmma.f32.16x16x4.f32` with mangled types
;      `.v8f32.v2f32` reflecting the K=4 fragment shape.
;
;   2. The accumulator type is `<8 x float>` and the A/B fragment
;      type is `<2 x float>`. A regression that misroutes K=4 to a
;      K=32 / K=64 dispatch would emit `<16 x t>` or `<8 x i32>`
;      fragments instead.
;
;   3. The modifier args use the canonical defaults: `i1 false` for
;      a_neg/b_neg/a_reuse/b_reuse and `i16 0` for c_mod.
;
; NEGATIVE PINS:
;
;   * NO call to `llvm.amdgcn.mfma.*` — the MFMA fallback path
;     does not cover K=4 f32 and any such call here would mean the
;     handler silently mis-dispatched.
;   * NO call to a different K-width WMMA intrinsic
;     (`16x16x32` or `16x16x64`) — would indicate cross-K
;     dispatch confusion.

; CHECK-LABEL: define amdgpu_kernel void @wmma_f32_16x16x4_f32_kernel(

; The native gfx1250 WMMA intrinsic, with the K=4 f32 fragment
; shape reflected in the mangled types `.v8f32.v2f32`. Modifier
; args are all defaulted (i1 false / i16 0) to match what clang's
; `_Constant` builtin args produce for the failing kerneldex GEMMs.
; IR: %wmma{{[0-9]*}} = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x4.f32.v8f32.v2f32(
; IR-SAME: i1 false, <2 x float> %{{[^,]+}},
; IR-SAME: i1 false, <2 x float> %{{[^,]+}},
; IR-SAME: i16 0, <8 x float> %{{[^,]+}},
; IR-SAME: i1 false, i1 false)

; Negative: no MFMA fallback (K=4 f32 has no decomposition path).
; IR-NOT: @llvm.amdgcn.mfma.

; Negative: no other-K WMMA dispatch (cross-K dispatch confusion).
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x32.
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x64.
; IR-NOT: @llvm.amdgcn.wmma.i32.16x16x64.
