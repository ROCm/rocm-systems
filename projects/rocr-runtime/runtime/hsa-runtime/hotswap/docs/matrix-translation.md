# Matrix Translation — WMMA to MFMA (incl. MXFP scaled path)

> **Status:** one shape implemented end-to-end
> (`V_WMMA_F32_16x16x32_F16` → 2×`V_MFMA_F32_16x16x16_F16` via
> `wmma_lowering.cpp`), powering the F16 GEMM baseline. This doc
> generalises that lowering into a principled per-shape framework
> that covers the rest of the Gluon surface (BF16, I8, F8/F6/F4) and
> the MXFP scaled path, and specifies the gates that refuse shapes
> we don't cover rather than emitting something subtly wrong.
>
> **Scope:** gfx1250 WMMA/SWMMAC → gfx950 MFMA, including scaled
> (MXFP) variants. The design extends to any future matrix primitive
> with a K-decomposable shape.

---

## 1. Problem in one paragraph

gfx1250 matrix ops (WMMA) are **wave32 collectives**: 32 lanes
cooperate to compute a 16×16 output tile, each lane holding 8
fragment elements. gfx950 matrix ops (MFMA) are **wave64
collectives**: 64 lanes cooperate to compute the same-or-larger
tile, each lane holding 4 fragment elements. Three simultaneous
mismatches make simple intrinsic substitution impossible:

1. **Wave size.** The SPE projection runs the wave32 source kernel
   as two independent copies inside a wave64 (lanes 0..31 and
   lanes 32..63). An MFMA intrinsic naïvely placed in the IR
   couples those two copies — they become one matmul that mixes
   unrelated data.
2. **Fragment layout.** WMMA's lane-to-(row, col, k) mapping differs
   from MFMA's. The 256 output elements of a 16×16 tile land on
   different lanes after each primitive, so we cannot read-after-
   write directly between them.
3. **K-shape.** Some gfx950 MFMA shapes have a larger K than their
   gfx1250 WMMA counterpart (16×16×128 F8 MFMA vs. 16×16×64 F8 WMMA),
   which turns "replace WMMA with MFMA" into "fuse two WMMAs per
   MFMA" — a non-local rewrite.

The translation has to simultaneously (a) keep the two wave32
projections disjoint, (b) bridge the fragment layouts at the
boundary between source-visible VGPRs and the MFMA call, and (c)
handle K asymmetry. §3–§6 derive the general framework; §7 applies
it to MXFP; §8–§9 set the gates.

## 2. Why the baseline lowering works

`wmma_lowering.cpp` is the reference implementation. Its three
structural choices are reusable for every shape we add:

1. **Two-pass runGroupPass.** One pass lanes-0..31 drives the MFMA,
   one pass lanes-32..63 drives it. Each pass uses `ds_bpermute` to
   gather the wave32 group's operand fragments into the MFMA
   layout; MFMA runs on all 64 lanes but only its "owning" group's
   result is selected at the end. The top/bottom halves do not
   share matmul inputs across the boundary — they are two
   independent matmuls per source WMMA.
2. **ds_bpermute for layout bridging.** Reading from any lane
   regardless of EXEC, independent of the SPE predication story.
   Every fragment-layout translation reduces to a set of
   `ds_bpermute` calls plus a small `select` tree keyed on
   `laneId / 16` (the MFMA's lane-group ID).
3. **K-decomposition loop.** WMMA at K=32 unrolls into 2×MFMA at
   K=16 with accumulator chaining. The same structure generalises
   to K=N → (N/K_mfma)× MFMAs.

Cost per source WMMA: roughly `2 × (|A dwords| + |B dwords| + |C
dwords| + K/K_mfma) × ds_bpermute + 2 × K/K_mfma × MFMA +
|D dwords| × ds_bpermute`. For F16 16×16×32 this is ~116
ds_bpermute + 4 MFMAs. Matmul-bound kernels eat the overhead;
dot-product / reduction-over-matmul does not. That is a performance
consideration (out of scope for this doc); the policy here is
**correct output at any cost**.

## 3. Source model — gfx1250 WMMA / SWMMAC shapes

Captured corpus — the shapes we need to cover for GPT-OSS + Gluon:

| Shape | A dtype | B dtype | Out dtype | M×N×K | Source |
|---|---|---|---|---|---|
| `v_wmma_f32_16x16x32_f16` | F16 | F16 | F32 | 16×16×32 | F16 GEMM, F16 FA |
| `v_wmma_f32_16x16x32_bf16` | BF16 | BF16 | F32 | 16×16×32 | F16 GEMM (BF16 mode) |
| `v_wmma_i32_16x16x32_iu8` | I8 | I8 | I32 | 16×16×32 | (not in captured corpus, listed for completeness) |
| `v_wmma_f32_16x16x64_f8f6f4` | FP8/BF8/FP4 | same | F32 | 16×16×64 | MXFP GEMM, MXFP FA (via `wmma_scaled`) |
| `v_swmmac_f32_16x16x32_f16` | F16 | F16 (2:4 sparse) | F32 | 16×16×32 | Not observed; listed for refusal |

Only the first is today wired in `handle_valu_vop3p.cpp:145`; the
others exist in captured binaries we have not yet raised end-to-end.

Per-lane fragment sizes on gfx1250 (from AMD Matrix Instruction
Calculator):

| Shape | A VGPRs | B VGPRs | C/D VGPRs |
|---|---|---|---|
| 16×16×32 F16 | 8 (v16f16) | 8 (v16f16) | 8 (v8f32) |
| 16×16×32 BF16 | 8 (v16bf16) | 8 (v16bf16) | 8 (v8f32) |
| 16×16×32 IU8 | 8 (v16i8) | 8 (v16i8) | 8 (v8i32) |
| 16×16×64 F8F6F4 | 16 for FP8/BF8, 12 for FP6/BF6, 8 for FP4 | same | 8 (v8f32) |

SWMMAC adds a 4×packed-per-lane sparsity mask. Deferred (§8).

## 4. Target model — gfx950 MFMA / scaled MFMA shapes

Relevant gfx950 shapes (from `semop.hpp:213-241`; AMD ISA reference
chapter 14):

| Shape | A dtype | Out dtype | M×N×K | Per-lane A/B/C |
|---|---|---|---|---|
| `V_MFMA_F32_16x16x32_F16` | F16 | F32 | 16×16×32 | 4 (v4f16)/4/4 |
| `V_MFMA_F32_16x16x16_F16` | F16 | F32 | 16×16×16 | 2/2/4 |
| `V_MFMA_F32_16x16x32_BF16` | BF16 | F32 | 16×16×32 | 4/4/4 |
| `V_MFMA_F32_16x16x16_BF16_1K` | BF16 | F32 | 16×16×16 | 2/2/4 |
| `V_MFMA_I32_16x16x32_I8` | I8 | I32 | 16×16×32 | 4/4/4 |
| `V_MFMA_I32_16x16x16_I8` | I8 | I32 | 16×16×16 | 1/1/4 (packed) |
| `V_MFMA_F32_16x16x32_FP8_FP8` | FP8 | F32 | 16×16×32 | 4/4/4 |
| `V_MFMA_F32_16x16x32_BF8_BF8` | BF8 | F32 | 16×16×32 | 4/4/4 |
| `V_MFMA_SCALE_F32_16x16x128_F8F6F4` | FP8/BF8/FP6/BF6/FP4 (per-block scaled) | F32 | 16×16×128 | varies/varies/4 |
| `V_MFMA_SCALE_F32_32x32x64_F8F6F4` | same | F32 | 32×32×64 | 4 |

Key asymmetries vs. source:

- **Direct K match** for 16×16×32 F16/BF16/I8 → 1 MFMA per WMMA
  (modulo fragment-layout bridge).
- **K larger on target** for F8F6F4 — one
  `MFMA_SCALE_F32_16x16x128` covers 2 source `WMMA_16x16x64_F8F6F4`.
  Two adjacent WMMAs in a K-loop **must fuse** into one MFMA on
  target, or we emit half of a partial sum, which is semantically
  wrong.
- **No gfx950 SWMMAC** — 2:4 sparsity has no direct MFMA
  equivalent. Refuse.

## 5. Per-shape translation templates

### 5.0 Target-capability dispatch (native vs. decompose)

Every matrix SemOp has a target-capability precondition that selects
between two handler paths:

1. **Native path.** Target ISA exposes an LLVM intrinsic with the same
   fragment shape → emit the intrinsic directly. No `ds_bpermute`, no
   K-decomposition, no two-pass redistribution. This is the right answer
   for every same-family retargeting (gfx1251 → gfx1250, gfx1250 →
   gfx1251, gfx942 → gfx942) and for any future target that gains WMMA
   with the same shape.
2. **Decompose path.** Target lacks the shape → run one of the
   templates below (§5.1–§5.3).

The existing F16 handler already implements this for WMMA vs. MFMA:

```157:170:projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler/handle_valu_vop3p.cpp
    Value *result_val;
    if (ctx.targetIsa.hasWMMA12) {
      Function *wmmaFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::amdgcn_wmma_f32_16x16x32_f16,
          {v8f32Ty, v16f16Ty});
      result_val = ctx.B.CreateCall(wmmaFn, {
          ctx.B.getFalse(), a,
          ctx.B.getFalse(), b,
          ConstantInt::get(Type::getInt16Ty(ctx.C), 0), c,
          ctx.B.getFalse(), ctx.B.getFalse()
      }, "wmma");
    } else {
      result_val = emitWMMAtoMFMA(ctx, a, b, c);
    }
```

Each new shape added in §5.1–§5.3 generalises this branch: the native
path is just one `Intrinsic::getOrInsertDeclaration` + call when the
target capability bit is set; the decompose path is the template body.

#### 5.0.1 Capability bits per shape family

The dispatch key is a per-shape capability table, not a single
generation check. Extend `ISAProfile` with one bit per shape family so
`handle_valu_vop3p.cpp` branches on a flag, not a target triple:

| Source SemOp family | Native on target iff | Decompose template | Introduced by |
|---|---|---|---|
| `V_WMMA_F32_16x16x32_F16/BF16` | `targetIsa.hasWMMA12` (existing; `FeatureWMMA128bInsts`) | Template A (§5.1) | gfx12 (gfx1100+) |
| `V_WMMA_I32_16x16x32_IU8` | `targetIsa.hasWMMA12` | Template A | gfx12 |
| `V_WMMA_F32_16x16x64_F8F6F4` | `targetIsa.hasWMMA1250` (new; `FeatureGFX1250Insts`) | Template B (§5.2) onto `V_MFMA_F32_16x16x128_F8F6F4` when `targetIsa.hasMFMA` | gfx1250 |
| `V_WMMA_SCALED_F32_16x16x64_F8F6F4` | `targetIsa.hasWMMA1250` | Template C (§5.3) onto `V_MFMA_SCALE_F32_16x16x128_F8F6F4` when `targetIsa.hasScaledMFMA` | gfx1250 |
| `V_SWMMAC_F32_16x16x32_F16` | target has SWMMAC (no AMD target does today) | — (refuse, §R2) | gfx1250 |

The names line up with the TableGen `SubtargetFeature` definitions
(`FeatureWMMA128bInsts`, `FeatureGFX1250Insts`, etc.) so `ISAProfile::
fromSubtarget` stays a one-line read from `MCSubtargetInfo::hasFeature`
per bit.

`hasScaledMFMA` is a new bit for gfx950's `V_MFMA_SCALE_*` family. On
gfx942 it is false → Template C refuses (no software dequant fallback
per §5.4). On gfx1250 it is false too (native `V_WMMA_SCALED` is the
intended path) → use the native path above.

#### 5.0.2 Identity and same-family translation

When the source and target flags coincide for a given SemOp the native
path is always taken. Consequences:

- **gfx1251 → gfx1250** (same family, both have `hasWMMA12` and
  `hasWMMA1250`): every WMMA, including scaled, becomes a native WMMA
  intrinsic. No `ds_bpermute`, no K-doubling fuse, no fragment-layout
  tables consulted. Template A/B/C code is completely bypassed. The
  only remaining cost on this path is the LLVM IR roundtrip — which
  we already pay for the rest of the kernel.
- **Identity (gfx1250 → gfx1250)** is a degenerate same-family case
  and trivially uses the native path.

This is the concrete argument that a separate "fast code path" for
same-family translation is unnecessary: the capability branch above is
**the** fast path. It preserves the compiler's shape decisions exactly
because the intrinsic signature matches the source opcode.

#### 5.0.3 Relationship to the shape-registration gate (§G1)

`verifyMatrixShapeCoverage` (§G1) enforces that every WMMA SemOp has a
registered shape descriptor. This subsection adds the second column to
that table: for each `(sourceSemOp, targetIsa)` pair, exactly one of
{native intrinsic name, decompose template id, `refuse(reason)`} is
declared. The startup verifier fails loud if any row is missing. No
handler is allowed to pick between paths implicitly — the decision is
data, driven by the registered capability columns.

### 5.1 Template A: 1-to-1 K match, layout bridge only

Applies when the gfx950 shape has the same K as the gfx1250 shape
and the same M×N. **One MFMA per WMMA** via the two-pass pattern of
§2, with shape-specific dtype and fragment-size wiring.

```
emit<ShapeX>WMMAtoMFMA(ctx, a, b, c):
  aDwords[N_a] ← unpack(a)
  bDwords[N_b] ← unpack(b)
  cDwords[N_c] ← unpack(c)
  laneId ← emitLaneId()
  for groupBase in {0, 32}:
    mfmaA, mfmaB ← redistributeInput<ShapeX>(aDwords, bDwords,
                                             lane-group mapping for ShapeX,
                                             groupBase)
    mfmaC       ← redistributeAcc<ShapeX>(cDwords, groupBase)
    result      ← mfma<ShapeX>(mfmaA, mfmaB, mfmaC)
  resultDwords ← collectResult<ShapeX>(result0, result1, laneId)
  return pack(resultDwords)
```

The per-shape work is (a) the dtype tables for pack/unpack, (b) the
lane-group → WMMA-GPR index mapping inside `redistributeInput`, (c)
the `redistributeAcc` mapping, (d) the `collectResult` mapping, and
(e) the MFMA intrinsic ID.

**In-scope for T1-T3 below:** F16, BF16, I8 at 16×16×32. Each is
~150 LoC plus a per-shape lookup table, factored against the F16
baseline.

### 5.2 Template B: K-doubling fuse

Applies when gfx950's K is a multiple of gfx1250's K. Two adjacent
source WMMAs in the accumulator dependence chain must fuse into one
MFMA. Correct if and only if:

1. The two source WMMAs share the same accumulator C.
2. The two source WMMAs' operands are consecutive along K (A0
   covers K[0..31], A1 covers K[32..63], …).
3. No side effect (store, atomic, cross-lane) separates them.

Detection is a peephole over the instruction stream at raise time:
find pairs `wmma_{16x16x64}_f8f6f4(a0, b0, c0) → c1` followed
immediately by `wmma_{16x16x64}_f8f6f4(a1, b1, c1) → c2` where the
chain register matches. Fuse into one `mfma_{16x16x128}_f8f6f4` with
operands concatenated.

If the pair does not match, we have one WMMA that does not fuse.
Two options:

- (a) Refuse. Safest.
- (b) Emit a half-width MFMA by zero-padding the second operand.
  Semantically correct but costs 2× flops.

**Policy:** (a) refuse for the initial implementation. Expose a
`Shape` attribute whose canonical form demands a fuse-partner; if
none is found, refuse with `matrixShapeUnfusible`. Move to (b) only
when a real kernel demands it.

### 5.3 Template C: Scaled MFMA

Applies to `wmma_scaled` (MXFP) → `mfma_scale_*`. Structurally the
same as Template B (K-doubling fuse for the F8F6F4 shapes), with
two additional input streams — per-block scale values for A and B.

The Gluon `wmma_scaled` carries:
- `A_q` quantised tensor (FP8/FP4 etc.)
- `A_scale` per-block scale (one E8M0 per 32-element block)
- `B_q`, `B_scale`
- `dtype_a`, `dtype_b` metadata selecting the encoding

The gfx950 `V_MFMA_SCALE_F32_16x16x128_F8F6F4` has matching
operands: A, B, C, and a per-block scale carried in an extra SGPR
operand. The scale format (E8M0) is identical between architectures
— no dequantisation is required.

The translation:

1. Fuse the pair of source WMMA_scaled calls per §5.2.
2. Redistribute A, B, C fragments as in Template A.
3. Redistribute the scale fragments (new — their lane layout is
   defined in the ISA reference chapter 14.x). Scales live in VGPRs
   on both ISAs but with different per-lane packing.
4. Emit the scaled MFMA intrinsic
   (`@llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4`).

**The scale-layout redistribution is the one new mechanical piece
beyond Template B.** It is the same family of `ds_bpermute + lane-
group select` code, just on a different per-lane width.

### 5.4 Template D (rejected): software dequant + dense MFMA

A fallback option that dequantises FP8/FP4 to F16 in-place and runs
dense F16 MFMA. Correct, slow (~30× overhead), and conceptually
defeats the purpose of MXFP. Rejected per the "no fallback" policy.
If Template C refuses, we refuse the kernel.

**Scope note — distinguishing rejected-Template-D from the landed
§7.4 ISA-level dequant lift.** The Template-D rejection above is
scoped to *synthesis*: the raiser must not choose dequant-then-dense-
matmul as an automatic fallback for a scaled WMMA/MFMA shape Template
C refuses. §7.4 (`v_cvt_scale_pk8_bf16_fp4` cross-target lift) is a
different axis: when the *source* kernel itself already emits the
gfx1250 ISA-level dequant primitive (Triton's `tl.dot_scaled` on
gfx1250 does this before a dense BF16 WMMA), faithfully lifting that
primitive across targets is an individual-instruction lowering, not a
Template-D synthesis. The bit-exactness discipline in §7.4 (refuse
loudly on any input shape we can't prove bit-identical to the
hardware primitive) is precisely what separates a faithful lift from
a Template-D fallback. If the source emits it, we lift it; we do not
synthesise it.

## 6. Fragment-layout tables

The per-shape redistribution tables are **data**, not code. They
live in a new file `wmma_fragment_layouts.hpp`:

```cpp
struct FragmentLayout {
  // Source WMMA layout: for each source VGPR index g,
  // what (lane, k_or_row, m_or_col) does it hold?
  // Encoded as a table indexed by (shape, operand, gpr, lane).
  // Generated from AMD Matrix Instruction Calculator outputs.
  ...
};
```

The `wmma_lowering.cpp` comment block (lines 22-54) is the hand-
extracted table for F16 16×16×32. Generating all shapes mechanically
from the AMD tools avoids per-shape off-by-one bugs. **Do not
reverse-engineer by inspection.**

## 7. MXFP path, end to end

### 7.1 Source intent (Gluon)

```python
acc = gl.amd.gfx1250.wmma_scaled(a_q, a_scale, "e4m3",
                                 b_q, b_scale, "e4m3", acc)
```

Compiles to:

```asm
v_wmma_scaled_f32_16x16x64_f8f6f4  vD, vA_q, vB_q, vA_scale, vB_scale, vC
```

(No SemOp yet; today's raiser refuses the kernel at decode.)

### 7.2 Target shape

`v_mfma_scale_f32_16x16x128_f8f6f4`. K=128, so two adjacent source
WMMAs (K=64 each) fuse into one target MFMA. Scale granularity is
per 32-element block; K=128 holds 4 scale blocks per operand.

### 7.3 Refusal criteria

| Source pattern | Policy |
|---|---|
| `wmma_scaled` with unfusible neighbour | Refuse (`matrixShapeUnfusible`) |
| `wmma_scaled` with mixed dtype pair not in target's F8F6F4 table | Refuse (`mixedDtypeUnsupported`) |
| Per-block scale block size ≠ 32 | Refuse (current target only supports 32) |
| `wmma_scaled` output to non-F32 | Refuse (both platforms are F32-only here) |

### 7.4 ISA-level MXFP4 dequant primitive — `v_cvt_scale_pk8_bf16_fp4`

> **Status:** cross-target lift landed. `scale_sel == 0` bit-exact on
> gfx1250 → gfx942/gfx950; `scale_sel != 0` refuses loudly until the
> 4-bit field's semantics are pinned in-tree. This subsection is
> additive to §7.1–§7.3 (the Gluon `wmma_scaled` path); the two
> surfaces are orthogonal (see §5.4's scope note).

#### What it is

`v_cvt_scale_pk8_bf16_fp4` (VOP3 opcode 0x2a0 on gfx1250;
`VOP3Instructions.td:1873`) is an ISA-level primitive that converts a
single VGPR holding 8 packed FP4 E2M1 nibbles into 4 consecutive
VGPRs holding 8 BF16 values, applying a single E8M0 byte scale
selected from the 4 bytes of a second VGPR via the 4-bit `scale_sel`
immarg.

Operand layout (handler reads via `AMDGPU::getNamedOperandIdx` +
`OpName::scale_sel` — NOT by operand-index position, to survive a
future operand-table reorder):

| MC index | Operand | Width | Semantics |
|---|---|---|---|
| 0 | `vdst` | 128 bits (4 VGPRs) | `<8 x bfloat>` result |
| 1 | `src0` | 32 bits (1 VGPR) | packed-8 FP4; nibble 0 = lane 0, low nibble |
| 2 | `src1` | 32 bits (1 VGPR) | 4 E8M0 scale bytes |
| 3 | `scale_sel` | 4-bit ImmArg | byte-selector over `src1` |

The LLVM intrinsic
(`int_amdgcn_cvt_scale_pk8_bf16_fp4`, `IntrinsicsAMDGPU.td:688`,
gated by `isGFX125xOnly`) maps 1:1 to the MC instruction:

```
declare <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(
    i32 %src, i32 %scale, i32 immarg %scale_sel)
; ImmArg<ArgIndex<2>>, Range<ArgIndex<2>, 0, 16>
```

#### Same-target vs cross-target (side-by-side)

| Path | Condition | Emitted IR shape |
|---|---|---|
| Same-target (identity or gfx1250→gfx1250, or any future target with `hasTensorOps`) | `ctx.targetIsa.hasTensorOps` true | `call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32 %src, i32 %scale, i32 0)` — backend selects the hardware opcode directly. |
| Cross-target (gfx1250 → gfx942/gfx950, or any target without `hasTensorOps`) | `ctx.targetIsa.hasTensorOps` false | Per-nibble bit-algebra expansion: 8× {FP4 field decomposition, BF16 synthesis, E8M0 exponent-bits add, priority merge} + `insertelement <8 x bfloat>` chain. See `handle_valu.cpp::emitCvtScalePk8Bf16Fp4CrossTargetExpansion`. |

Both paths share the same operand-shape validation (`scale_sel`
immediate present; `scale_sel == 0`). A drift in the shared
validation surfaces on both targets identically.

#### 16-entry FP4 E2M1 → BF16 table

Pinned as constants in `transpiler/mxfp4_dequant.cpp` (see
`kMxfp4ToBf16Table`). Reproduced here in hex for visibility; the
authoritative source is the `.cpp` file, cross-checked by
`tests/mxfp4_dequant_test.cpp::OcpTableBitPatterns` on every CI run.

| FP4 nibble | Real value | BF16 hex (identity scale) |
|---:|---:|---:|
| `0b0000` | +0.0 | `0x0000` |
| `0b0001` | +0.5 | `0x3F00` |
| `0b0010` | +1.0 | `0x3F80` |
| `0b0011` | +1.5 | `0x3FC0` |
| `0b0100` | +2.0 | `0x4000` |
| `0b0101` | +3.0 | `0x4040` |
| `0b0110` | +4.0 | `0x4080` |
| `0b0111` | +6.0 | `0x40C0` |
| `0b1000` | -0.0 | `0x8000` |
| `0b1001` | -0.5 | `0xBF00` |
| `0b1010` | -1.0 | `0xBF80` |
| `0b1011` | -1.5 | `0xBFC0` |
| `0b1100` | -2.0 | `0xC000` |
| `0b1101` | -3.0 | `0xC040` |
| `0b1110` | -4.0 | `0xC080` |
| `0b1111` | -6.0 | `0xC0C0` |

FP4 ±0.5 maps to BF16 **normal** exp=126 mant=0 (real value 2^-1 is
representable as a BF16 normal, not a subnormal), per the subnormal-
FP4 branch in `mxfp4BitAlgebraBf16Bits`.

#### Corner-case semantics (cross-target expansion)

Per `handle_valu.cpp::emitCvtScalePk8Bf16Fp4CrossTargetExpansion` +
the C++ reference in `mxfp4_dequant.cpp`, all cases are bit-exact
against what the hardware primitive emits on bit-valid inputs in the
declared support set:

| Condition | BF16 result |
|---|---|
| `scale_byte == 0xFF` (E8M0 NaN) | `0x7FC0` (canonical qNaN); wins over FP4 ±0 (IEEE 0 × NaN = NaN) |
| FP4 ±0 × finite scale | `±0` preserving sign |
| `new_exp ≥ 0xFF` (overflow) | `±Inf = sign<<15 | 0x7F80`; BF16 supports Inf even though FP4 does not |
| `new_exp ∈ [1, 0xFE]` (normal) | `sign<<15 | (new_exp<<7) | bf16_mant` |
| `new_exp ≤ 0` (underflow) | Subnormal or ±0 via `(0x80 | bf16_mant) >> (1 - new_exp)`, clamped to 0 at shift ≥ 8 |

**Rounding mode:** N/A. The scale application is a multiplication
by a power of 2, which is exact in floating-point with zero bits of
rounding. We emit integer field manipulation instead of an fmul so
the lowering is bit-identical regardless of the target's float-mode
register state (FTZ / DAZ bits are irrelevant — no fmul runs).

#### Declared support set

**Supported today:** `scale_sel == 0` (i.e. scale byte = `src1 & 0xFF`).

**Refused loudly:** `scale_sel != 0`. The AMD ISA spec's definition
of the 4-bit `scale_sel` field's semantics for the packed-8 FP4 shape
is not reproduced in this tree. The captured gfx1250 corpus
(`scope_discovery/kernels/_matmul_ogs_{06d912ce88af,0af655e6ea2b}.hsaco`,
the two Triton GPT-OSS MoE matmul kernels `_matmul_ogs_NNT_bf16xbf16xmxfp4_*`)
uses only `scale_sel == 0` across 128 instances combined, which has
the unambiguous reading "scale byte is the low byte of the 32-bit
scale register". Widening the support set to `scale_sel != 0`
requires either (a) the AMD ISA spec for gfx1250 to land in-tree with
a precise semantics table, or (b) a new corpus kernel that exercises
the case and a canary extension whose native-gfx1250 run pins the
hardware behaviour.

#### What landing this unblocks

This is **standalone cross-target coverage** for the MXFP4 dequant
primitive; it narrows the kerneldex refusal surface for any future
gfx1250→gfx942 kernel that uses the cvt primitive without going
through scaled-WMMA.

End-to-end lift of the two corpus matmul_ogs kernels
(`_matmul_ogs_06d912ce88af`, `_matmul_ogs_0af655e6ea2b`) still blocks
on pending **Template A** (BF16 16×16×32 WMMA → MFMA; see §T2) and on
**TDM** (`global_load_async_to_lds_*`, landed separately). Those
kernels emit 64× `v_cvt_scale_pk8_bf16_fp4` + 64× `v_wmma_f32_16x16x32_bf16`
+ N× `global_load_async_to_lds_*` each; this work clears the first of
those three blockers, and the kernel raise will surface the other two
in sequence once retried.

#### Regression-guard pointers

| Layer | Path | Pins |
|---|---|---|
| Unit (C++) | `tests/mxfp4_dequant_test.cpp` | OCP table constants; bit-algebra ≡ LUT-via-double on all 4096 inputs; per-corner checks for NaN / ±0 / overflow / underflow |
| Lit (IR shape) | `lit_tests/v_cvt_scale_pk8_bf16_fp4/v_cvt_scale_pk8_bf16_fp4.ll` | Same-target RUN emits the intrinsic; cross-target RUN emits the bit-algebra expansion, no intrinsic, no LUT constant array |
| Canary (end-to-end) | `tools/compare_correctness/kernels/canary_cvt_scale_pk8_bf16_fp4.hip` + recipe entry in `compare_correctness.cpp` | `native` == `salmon` bit-exact BF16 output across 12-value scale-byte × pseudo-random packed-FP4 sweep; `legacy` expected to crash (SIG6) on the gfx1250-only opcode |

#### Landing commit SHA

_(filled in post-commit)_

### R1 — Unrecognised shape

Every WMMA SemOp has a registered shape descriptor. If an MC opcode
maps to a SemOp whose descriptor has no target mapping, refuse with
`matrixShapeUnsupported`.

### R2 — SWMMAC (2:4 sparsity)

No gfx950 equivalent. Refuse.

### R3 — Unfusible K-doubled pair (§5.2)

### R4 — Divergent EXEC at the WMMA site

WMMA assumes all 32 lanes are active. If SPE predication makes lanes
inactive at the WMMA, the lowering's two-pass model breaks (some
lanes have undefined fragment contents in the replicated copy). The
existing F16 lowering sidesteps this by documenting the assumption
("WMMA operations occur in non-divergent code — EXEC is all-ones",
`wmma_lowering.cpp:89`). Formalise it as a per-kernel gate: every
WMMA instruction must be uniformly reached (SPE's existing uniform-
reachability predicate, to be extended).

Note that "EXEC is all-ones at the WMMA" is an invariant on the
*source-modeled* EXEC (what `emitUnderExec` reads through the
alloca), not the *hardware* EXEC of the target gfx942/gfx950
wavefront. The wave-native projection forces hardware EXEC = -1
kernel-wide via `@llvm.amdgcn.init_whole_wave` at entry (see
`wave-size-translation.md` §5.6.1), so the Wave64-collective
semantics of `ds_bpermute` / MFMA / `v_cndmask` always work across
all 64 lanes regardless of the source kernel's partial-wave launch
shape. What R4 gates is that the source kernel author did not put
the WMMA inside an if-statement that predicates away source lanes
before the collective issues — that remains an IR-shape property
the uniform-reachability predicate can check without looking at
hardware EXEC at all.

### R5 — AGPR-only accumulator expectations

Some MFMA shapes on gfx950 require the accumulator in AGPRs. The
raised IR uses VGPRs; the target backend's AGPR allocator moves them
as needed. If a shape exists that cannot be serviced from VGPR
accumulators (has not been observed), refuse.

## 9. Principled fail-loudly gates

### G1 — Shape registration coverage (startup)

Every SemOp in the WMMA family must have an entry in
`wmma_fragment_layouts` **and** in a shape→target-MFMA mapping
table. Startup verifier
`verifyMatrixShapeCoverage(sourceIsa, targetIsa, opcMap)` fails loud
if any WMMA SemOp lacks a mapping.

### G2 — Per-kernel fusibility scan (pre-Phase-2)

For each Template-B / Template-C shape in the decoded instruction
stream, run the peephole matcher. If any Template-B/C-class WMMA
has no fusible partner, refuse.

### G3 — Uniform-EXEC gate at WMMA site (per-kernel)

At each `V_WMMA_*` decode, check SPE's uniform-reachability
predicate. Refuse on non-uniform.

### G4 — Dtype-table coverage (startup)

For F8F6F4: enumerate the cross-product of (A dtype, B dtype) Gluon
emits and the subset gfx950 supports. Refuse combinations not in
the intersection.

## 10. Engineering tasks

### T1 — Auto-generate fragment tables

Stand up a one-time generator from AMD Matrix Instruction Calculator
output (Python script under `hotswap/tools/`) producing
`wmma_fragment_layouts.inc` consumed by `wmma_lowering.cpp`. ~300
LoC; removes the hand-transcribed table as a source of bugs.

### T2 — BF16 and I8 16×16×32 shapes (Template A)

Extend `handle_valu_vop3p.cpp` to dispatch on new SemOps
`V_WMMA_F32_16x16x32_BF16`, `V_WMMA_I32_16x16x32_IU8`. Each is a
direct parallel of the F16 code with pack/unpack dtype substitutions
and the auto-generated layout table. ~200 LoC each.

### T3 — F8F6F4 shapes (Template B, no scale)

Add `V_WMMA_F32_16x16x64_F8F6F4` SemOp. Implement the
fusibility peephole in a new post-decode pass
`detectMatrixFusions(insts)` that annotates pairs. Handler emits one
`mfma_16x16x128_f8f6f4` per fused pair; solo WMMAs trigger G2
refusal. ~400 LoC (peephole ~150, handler ~250).

### T4 — Scaled WMMA (Template C) for MXFP

Add `V_WMMA_SCALED_F32_16x16x64_F8F6F4`. Extend the peephole from T3
to handle the scaled variant. Implement scale-layout redistribution
(new tables). Handler emits `mfma_scale_16x16x128_f8f6f4`. ~500 LoC.

### T5 — Uniform-reachability predicate for WMMA sites (G3)

Reuse SPE's uniform-reachability analysis (already required by
`wave-size-translation.md §5.1`). Extend `SemOpAttrs` with
`requiresUniformExec` and set it on every WMMA SemOp. Per-kernel
gate fires automatically once the attr is honoured. ~40 LoC.

Dependency order: **T1 first** (unblocks every other task). T2 and
T5 in parallel. T3, then T4.

## 11. Testing strategy

- Oracle: run the source gfx1250 kernel on a gfx1250 device (or in
  a reference CPU implementation) and compare tile outputs element-
  wise at tolerance appropriate for the dtype.
- Per-shape test harness: the existing `tests/mfma_gemm.hip` extends
  naturally. One .hip per shape, compiled by rocm-hipcc for gfx1250,
  translated, run on gfx950.
- Randomised inputs with deterministic seeding.
- Scale-path tests: round-trip an MXFP encoding through both
  architectures, compare bit-for-bit where feasible and
  element-wise otherwise.

## 12. Staging state — ModuloReplicationProjection-aware lowering (2026-04-22)

> **Status:** infrastructure landed, lowering staged-but-gated-off.
> Independent of the WMMA lowering, a prerequisite ABI bug in the
> raiser-entry `ttmp7` init was identified and fixed (2026-04-22;
> see §12.3 for the full bisection).  With that fix plus the MODREP
> refusal gate lifted in-session, `matmul_fp16_16x16` passes
> `compare_correctness` 5/5 end-to-end across all shapes
> (verified in-session; the test path is gated behind the WMMA
> refusal on `main`, so a fresh checkout does not exercise it
> without manually flipping the gate — see §12.2 gate-flip
> protocol).  `matmul_fp16` (the larger BLOCK=32 sibling with four
> parallel WMMAs sharing A / B operand vregs through
> `v_permlane16_swap_b32`) still exhibits a residual mode-5 /
> random-input divergence under the same enable — this is the
> remaining blocker to flipping the refusal gate (§12.4).

The WMMA → MFMA lowering in `wmma_lowering.cpp` now supports TWO
projections:

1. **`WaveNativeProjection`** — the original baseline. Kernel-entry
   `@llvm.amdgcn.init_whole_wave` provides HW `EXEC = -1` kernel-
   wide, so the `runGroupPass` pipeline runs with every target lane
   participating. `Gfx1250Gpu.Matmul128x128*` gtests pass end-to-end
   on this path; it is the verified production path today.

2. **`ModuloReplicationProjection`** (phantom-lane fallback for
   `max_flat_workgroup_size < targetWaveSize`) — NEW, staged. HW
   EXEC stays at the source-active mask kernel-wide, so target lanes
   past the source-wave width are HW-inactive for the rest of the
   kernel body (which is the whole point of the phantom-lane
   fallback — their undef VGPRs can't contaminate cross-lane ops).
   The WMMA lowering scopes a local HW-EXEC = -1 region around just
   the redistribute / MFMA / collect chain by wrapping the MFMA
   outputs and each collect-output dword in `@llvm.amdgcn.strict.wwm`,
   which the AMDGPU backend's `SIWholeQuadMode` pass expands into
   a proper scoped region (materialising as `s_or_saveexec_b64 sN,
   -1` entry / `s_mov_b64 exec, sN` exit in the final HSACO).

The MODREP path is correct on minimal repros (isolated and K-loop-
chained WMMAs, verified via `lit_tests/wmma_phantom_lane_f16_chain/`)
but an unexplained residual divergence remains on Triton's
`matmul_fp16_16x16` kernel at M>=32 through `compare_correctness`.
Until that residual is pinned, both WMMA arms in
`handle_valu_vop3p.cpp` keep the `!providesFullWaveExecInvariant()`
refusal gate. The infrastructure is landed additively so the next
investigation has a known-good baseline to compare against.

### 12.1 Infrastructure summary

| Symbol | Signature | Semantics |
|---|---|---|
| `WaveProjection::numSourceWavesPerTarget()` | `virtual unsigned` — pure virtual | Returns 1 under MODREP (phantom-lane = single source wave) or same-wave projections; 2 under `WaveNativeProjection` wave32→wave64 cross-widen; `report_fatal_error` under the unimplemented `ThreadLoopProjection`. Pure so every new projection class must answer the question explicitly. |
| `WaveProjection::wrapAsWWMValue(B, v)` | `Value *` helper | No-op under projections that guarantee HW EXEC=-1 kernel-wide (`WaveNativeProjection`); emits a per-value `@llvm.amdgcn.strict.wwm` call otherwise. Accepts any scalar/fixed-vector integer or floating-point type supported by the intrinsic's overload set; asserts on other types. |
| `emitWMMAtoMFMA*` (internal) | — | Now iterates `runGroupPass` for `numSourceWavesPerTarget()` passes and wraps MFMA outputs + collect-output dwords via `wrapAsWWMValue`. Release-build-safe `report_fatal_error` guards catch a refusal-gate regression that flips the gate without vetting the MODREP path. |

### 12.2 Gate-flip protocol

When the `matmul_fp16_16x16` residual is pinned and the MODREP path
is ready to enable:

1. **Prove correctness** on `compare_correctness --recipe=matmul_fp16
   {,_16x16}`: all shapes must match, not just `EXIT=2`. Run both
   recipes through the full sweep.
2. **Drop the refusal gates** in `handle_valu_vop3p.cpp` (both the
   K=4 f32 arm and the K=32/K=64 arm). The `!providesFullWaveExec
   Invariant()` check and its `RaiseFailure::unsupportedShape`
   branch are the only pieces removed; the rest of the dispatch
   (native `hasTensorOps` path, `hasMFMA` cross-target path, no-
   path-available refusal) stays exactly as-is.
3. **Flip the lit fixture**
   `lit_tests/wmma_phantom_lane_f16_chain/wmma_phantom_lane_f16_chain.ll`
   from refusal-CHECKs to affirmative IR-shape CHECKs. The fixture
   header spells out the specific anchors the flipped version
   should pin (strict.wwm markers, single-pass runGroupPass output,
   MFMA chain).
4. **Flip the lit fixture**
   `lit_tests/wmma_phantom_lane_refuse/wmma_phantom_lane_refuse.ll`
   similarly — the K=4 f32 variant.
5. **Drop the release-build-safe guards** in `emitWMMAtoMFMA*`
   (`if (numSrcWaves != 2) report_fatal_error(...)`) once the
   `numSrcWaves == 1` branch is live. Keep the `numSrcWaves ∉ {1,
   2}` guard — that's a contract check for new projection classes.

### 12.3 Prerequisite ABI fix — `ttmp7` init for `workgroup_id_y/z` (2026-04-22)

The AMDGPU backend's entry-function ABI for gfx12+ preloads
`workgroup_id_y` and `workgroup_id_z` into a packed `ttmp7` by
`SIMachineFunctionInfo`'s argument-descriptor table (see LLVM's
`AMDGPULegalizerInfo::loadInputValue`):

```cpp
WorkGroupIDY = ArgDescriptor::createRegister(TTMP7, 0xFFFFu);
WorkGroupIDZ = ArgDescriptor::createRegister(TTMP7, 0xFFFF0000u);
```

i.e. `ttmp7[15:0] = workgroup_id_y`, `ttmp7[31:16] = workgroup_id_z`.
Triton-generated gfx1250 kernels read the Y component via the
canonical idiom:

```
s_and_b32 sDST, ttmp7, 0xffff       ; wg_id_y
```

`raiser.cpp`'s Phase-4 entry init previously initialised `ttmp9`
(`workgroup_id_x`) and `ttmp8[29:25]` (`wave_id_in_workgroup`)
for the canonical Tensile / rocBLAS BFE pattern, but left `ttmp7`
uninitialised.  Under MODREP the uninitialised SGPR read as
whatever the alloca's initial pattern yielded — effectively
always zero — so every 2D-grid kernel's `workgroup_id_y` collapsed
onto the Y=0 column.

Bisection trace (against `matmul_fp16_16x16` M=32, all-ones ×
all-ones; expected output every cell = K = 32; host harness
`/tmp/matmul_test.cpp` in-session):

| Column range | Observed | Diagnosis |
|---|---|---|
| cols 0..15 | 32.0 (correct) | WG(\*, pid_n=0) writes |
| cols 16..31 | `0xCDCD` (poison) | WG(\*, pid_n=1) never writes — all WGs see `pid_n = 0` |

The pattern is unambiguous: the kernel body was running every
workgroup to completion but every WG wrote to its `pid_n=0`
destination region, so the Y=1 column of workgroups landed atop
the Y=0 column and the Y=1 output region retained the host's
pre-launch `0xCD` memset poison.

The fix (in `raiser.cpp`'s `if (AMDGPU::isGFX12Plus(...))` block):

```cpp
// ttmp7 = (workgroup_id_z << 16) | (workgroup_id_y & 0xFFFF)
Value *wgIdY = B.CreateCall(fnWorkgroupIdY, {}, "ttmp7_wg_id_y");
Value *wgIdZ = B.CreateCall(fnWorkgroupIdZ, {}, "ttmp7_wg_id_z");
Value *wgIdYLo = B.CreateAnd(wgIdY, B.getInt32(0xFFFF), "wg_id_y_lo16");
Value *wgIdZHi = B.CreateShl(wgIdZ, B.getInt32(16), "wg_id_z_hi16");
Value *ttmp7Val = B.CreateOr(wgIdYLo, wgIdZHi, "ttmp7_val");
B.CreateStore(ttmp7Val, regs.ttmp[7]);
```

Masking Y to 16 bits before the OR is safe on no-Z kernels (the
backend's mask there is `~0u`; the consumer's `s_and ttmp7,
0xffff` discards the upper bits regardless) and is the principled
all-cases shape for 2D/3D grid kernels alike.

**Regression fence:** `lit_tests/ttmp7_workgroup_id_yz_init/`
pins the `@llvm.amdgcn.workgroup.id.{y,z}` + `shl ..., 16`
IR shape at kernel entry.  A backwards-stride to pre-fix behaviour
would drop at least one of those three anchors.

### 12.4 Remaining `matmul_fp16` divergence — multi-WMMA + `v_permlane16_swap_b32`

With §12.3's fix applied and the refusal gate lifted locally,
`matmul_fp16_16x16` passes `compare_correctness` 5/5.  The sibling
`matmul_fp16` (BLOCK=32, four parallel WMMAs per WG, epilogue
without LDS round-trip) still diverges for non-uniform inputs.

Host-harness bisection with deterministic `mode ∈ {0, 3..6}` input
generators narrows the residual (harness not checked in — rebuilding
for a future investigation is the right step, not re-reading the
prior session's `/tmp` file):

| Mode | A layout | B layout | Expected | Observed | Verdict |
|---|---|---|---|---|---|
| 0 | all 1s | all 1s | C[i,j] = K = 32 | C[i,j] = 32 | ✓ match |
| 6 | A[i,k]=i/16 | all 1s | C[i,j] = 2i | C[i,j] = 2i | ✓ match |
| 5 | all 1s | B[k,j]=j/16 | C[i,j] = 2j | C[i,j] = 2(j±8 mod 16) | ✗ mismatch (cols 0..15: got=ref+16; cols 16..31: got=ref-16) |

The A-only-varying and all-uniform cases being bitwise correct
while the B-only-varying case is systematically off by the
sub-tile width strongly suggests the bug is specific to B's
data-flow path rather than the (A↔B-symmetric) redistribute
math, MFMA intrinsic choice, or collect mapping.  This is a
bisection hypothesis, not a formal proof — a data-dependent
redistribute bug that happens to thread the wrong SSA value as
the B fragment while A's SSA is correct would show the same
asymmetry.  Getting to a formal localisation needs either a
minimal synthetic repro kernel that isolates the B-path (in
progress; `quad_wmma_kernel` sketched in-session but didn't
cleanly reproduce the ±16 pattern) or an MIR-level dump after
`si-wqm` / regalloc to observe the B-operand vreg threading.

Two confounded differentiators between the working
`matmul_fp16_16x16` and the failing `matmul_fp16`:

  1. **`v_permlane16_swap_b32` presence.**  `matmul_fp16` emits
     8 of these (one per A-fragment dword pair); `matmul_fp16_16x16`
     emits zero.
  2. **Multi-WMMA per WG.**  `matmul_fp16` has 4 parallel WMMAs
     per workgroup (a 2×2 output sub-tile grid);
     `matmul_fp16_16x16` has 1.

With only one sample on each side I can't separate these
empirically.  The candidate root causes below apply under either.

The swap sequence emitted by Triton for `matmul_fp16`:

```
v_permlane16_swap_b32_e32 v186, v194   ; A fragment pair 0
v_permlane16_swap_b32_e32 v187, v195
...
v_permlane16_swap_b32_e32 v193, v201   ; A fragment pair 7
```

Each swap exchanges a register PAIR between lanes 0..15 and 16..31,
setting up the two-row-tile A fragment distribution.  The lift
decomposes each `v_permlane16_swap_b32 vdst, src0` into a pair of
EXEC-ungated `ds_bpermute` reads using `lane_id XOR 16` as the
partner selector — semantics are correct per the AMDGPU ISA spec,
and individual `v_permlane16_swap_b32` lit fixtures (`c2_permlane_swap`)
pass.  The bug surfaces only under the joint regime:

- MODREP phantom-lane projection (source-active lanes 0..31 only);
- Four PARALLEL WMMAs (not chained via accumulator PHIs) sharing
  A and B operand vregs after the swap;
- Non-uniform B input so the swap'd-vreg data matters.

### 12.4.1 Additional session findings (2026-04-22)

Session 2 ran several directed experiments; the residual is still
open, but the evidence pool has narrowed:

  * **Mode-3 mis-layout is NOT a uniform ±16 shift.**  With
    `A[i,k] = (i*32+k)/64` and `B = identity` (so the reference is
    `C[i,j] = A[i,j]`), row 0 of the output is bitwise correct (all
    32 cells), while row 1 shows a mixed pattern:
      * `C[1, 0]` is correct (`0.5`);
      * `C[1, 1..7]` return `0.0` (expected `A[1, 1..7]` ∈
        `[0.5156, 0.6094]`);
      * `C[1, 8..15]` return `A[1, j] + A[1, j+8]` (a SUM of two
        matrix cells);
      * `C[1, 16..23]` show the same values as `C[1, 8..15]`;
      * rows 2..31 all fail.
    This is not a single-axis shift; the interaction mixes values
    across BOTH the row and column axes, which rules out the
    simplest "fragment accidentally off by K" / "stride-by-one"
    hypotheses.

  * **MFMA-output `strict.wwm` wrap is NOT load-bearing for
    correctness on single-WMMA kernels.**  Dropping the
    `wrapAsWWMValue` calls on `mfma1` / `mfma2` results (keeping
    only the collect-dword wrap) — leaving `SIWholeQuadMode` to
    propagate WWM backward from the collect — keeps
    `matmul_fp16_16x16` bitwise correct across all five shapes AND
    does NOT change `matmul_fp16`'s mode-5 failure pattern (same
    1024/1024 mismatches, same ±16 offset).  This falsifies
    candidate #1's narrow form (MFMA-output WWM region collapse),
    though a broader "WWM region interacts badly with multi-WMMA
    regalloc" variant is still live.

  * **LDS reads sit INSIDE the WWM region in both kernels.**
    Disassembling the lifted HSACOs shows the `ds_read_b128` loading
    the B fragment lives between `s_or_saveexec_b64 sN, -1` and
    `s_mov_b64 exec, sN`.  Under `EXEC = -1`, all 64 target lanes
    (including phantom lanes 32..63) execute the LDS read; the
    phantom lanes compute LDS addresses from their own `lane_id` /
    `mbcnt` output, which point past the LDS segment the source-
    active lanes wrote — so those lanes read zero / uninitialised
    LDS, and their MFMA output is whatever that data computes to.
    This would naturally break target lanes 32..47 / 48..63, whose
    collect-stage bperm reads feed source-active lanes 16..31 (the
    second half of each wave32's output fragment).  `matmul_fp16_16x16`
    has the same pattern but passes, so the phantom-LDS-read
    hypothesis must have a mitigating factor on the 16×16 path —
    perhaps because its collect doesn't need the phantom lanes'
    MFMA output (the output tile is 16×16 so only the first 16 of
    every 32 per-lane outputs are consumed via the store pattern).
    VERIFY: trace `matmul_fp16_16x16`'s per-lane output STORE
    addressing; if it never reads source-lanes 16..31 of the
    collect (e.g. the Triton-emitted epilogue only stores lane 0..15
    output to C because BLOCK_M=16), the phantom-LDS-read corruption
    would land only in unused VGPR slots and not surface.

  * **Hand-rolled `quad_wmma_kernel` / `single_wmma_kernel` repros
    are INVALID.**  Each lane in my sketched repros loads `A0[0]`
    uniformly (not per-lane fragment-distributed), so every lane
    feeds the same <16 x half> into the WMMA and the intrinsic's
    cross-lane aggregation returns a non-meaningful result (the
    "all-15" / "all-0" per-lane output on mode-5 / mode-6 that I
    couldn't interpret earlier).  A valid repro needs to manually
    distribute the A/B matrix across lanes per the gfx12 wave32
    WMMA fragment layout — this is the next concrete step.

Candidate root causes to investigate next (ordered by suspicion,
all unverified — these are TODOs, not localisations):

1. **`strict.wwm` region collapse across multiple parallel MFMAs.**
   Each of the 4 source WMMAs lowers to a pair of chained MFMAs
   (K=32 → 2× K=16) wrapped in `@llvm.amdgcn.strict.wwm`.  The
   backend's `SIWholeQuadMode` pass may merge the 8 MFMA regions
   into a single large WWM scope; `SIPreAllocateWWMRegs`'s
   dedicated-physreg-per-WWM-vreg requirement may then produce a
   spill/reload pattern that crosses the swap's partner-lane read
   boundary.  Verify via MIR dump after `si-wqm` + `si-pre-
   allocate-wwm-regs` on the lifted IR.
2. **Redistribute-step CSE across WMMAs sharing operands.**  The
   B fragment for WMMA(0,0) and WMMA(1,0) derives from the same
   source vreg pair `b[170:177]`.  The LLVM mid-level CSE may
   collapse the redistribute's per-WMMA `ds.bpermute` calls into
   a single SSA value reused across both WMMAs.  If one WMMA's
   redistribute path happens to land before the swap and another
   after, the shared SSA values encode the wrong B fragment for
   one of them.  Verify via `-print-after-all` on the IR's mid-
   level pipeline, specifically GVN / CSE passes.
3. **Scoped-WWM alternative lowering.**  The file header comment
   in `wmma_lowering.cpp` (lines 152-173) documents an earlier
   design that was abandoned because `SIPreAllocateWWMRegs`
   exhausted the 256-VGPR pool on 128×128 f16 matmuls.  The
   abandoned design wrapped MFMA outputs in `strict.wwm`; the
   current code re-introduces `strict.wwm` via `wrapAsWWMValue`
   for MODREP.  A genuinely scoped `s_or_saveexec_b64 sN, -1`
   → MFMA → `s_mov_b64 exec, sN` sequence emitted directly by
   the raiser (no `strict.wwm` markers, no WWM-region allocator
   dependency) would sidestep both issues.

Until one of these is pinned, the K=32/K=64 refusal gate stays
in place — the principled outcome for `matmul_fp16` remains
EXIT=2 (refuse), and the gate's diagnostic in
`handle_valu_vop3p.cpp` now surfaces the `matmul_fp16_16x16` WIN
and the `matmul_fp16` OPEN items explicitly.

## 13. Relationship to other axes

- **SPE / wave-size** (`wave-size-translation.md`): WMMA sites require uniform
  reachability (G3). The two-pass lowering is itself the
  SPE-compatible decomposition — it runs both wave32 replicas of the
  source fragment independently, which is exactly the modulo-
  replication projection SPE uses.
- **TDM** (`tdm-translation.md`): MFMA operands arrive through LDS
  in most GEMMs. TDM-based LDS filling is a predecessor concern —
  matrix translation assumes the operands are in VGPRs at the
  moment of the MFMA call, however they got there.
- **Sync** (`sync-translation.md`): No matrix-specific sync concerns.
  The accumulator chain across a K-loop is an IR-level dataflow
  dependence; the backend emits the needed waitcnt.
- **ABI** (`abi-translation.md`): AGPR vs VGPR-only accumulator is
  declared at ABI level. The raiser emits VGPR accumulators; target
  backend's AGPR allocator is allowed to move them.
- **Cross-cutting capability dispatch:** §5.0 is the matrix-axis
  instance of the project-wide "emit native when the target supports
  it, decompose only when it does not" principle. See
  `target-capability-dispatch.md` for the shared design and the
  open implementation question (does LLVM already expose per-SemOp
  feature requirements we can reuse for the registration table?).
