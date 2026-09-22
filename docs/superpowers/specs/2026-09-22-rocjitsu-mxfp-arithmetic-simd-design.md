# RocJITsu MXFP Arithmetic SIMD Design

## Goal

Extend PR #12052 from gfx1250 MXFP pack/unpack conversions to the low-precision
decode work performed by gfx1250 MXFP WMMA instructions. Preserve architectural
results and the existing scalar oracle while reducing host execution time.

The arithmetic MAC core is already native-SIMD. The remaining targeted cost is
the element-by-element FP4, FP6, BF6, FP8, and BF8 decode used to populate the
F32 matrix staging buffers.

## Scope

This change covers the gfx1250 paths routed through:

- `exec_wmma_f32_mixed`
- `exec_wmma_f32_scaled_mixed`

That includes dense F8F6F4 WMMA, fixed FP4 WMMA, scale32 WMMA, and scale16
WMMA. CDNA4/gfx950 MFMA uses the same low-precision primitives but remains a
follow-up so PR #12052 keeps its gfx1250 architecture boundary.

## Considered Approaches

### 1. Batch only the low-precision decode (selected)

Keep the existing scalar layout gather because matrix layouts can permute lanes
and cross DWORD boundaries. Gather one native-width batch of raw codes, decode
the batch with the SIMD primitives added by PR #12052, and store F32 values in
the existing A/B staging buffers.

This is a small shared-executor change, reuses established layout logic, and
does not alter FMA ordering or generated instruction bodies.

### 2. Add per-format specialized WMMA kernels

Generate a kernel for every A/B format pair and shape, fusing layout extraction,
decode, and multiplication. This may ultimately be faster, but it multiplies
code size and validation surface across 25 format pairs and four scaled/dense
forms. It is not justified before measuring the staged-decode implementation.

### 3. Vectorize across emulated GPU lanes

Reuse the conversion instruction's lane-oriented register algorithm directly.
WMMA layouts gather values across lanes into logical matrices, so this fights
the existing mapping and would duplicate matrix-layout logic. It is rejected.

## Design

Add a helper in `shared/mma_exec.h` that recognizes the five MX extractor
types. For a requested logical element range it:

1. Uses the caller's existing location function and `read_packed` to gather raw
   codes from a snapshotted register region.
2. Groups codes into `util::native<uint32_t>` batches.
3. Calls the matching bit-exact SIMD decoder.
4. Stores decoded F32 lanes into the caller's existing staging buffer.
5. Falls back to the supplied scalar extractor for unsupported extractor types
   or builds without `<experimental/simd>`.

The helper is used only inside the existing native-SIMD branches of
`exec_wmma_f32_mixed` and `exec_wmma_f32_scaled_mixed`. The scalar branches are
unchanged. `RJ_FORCE_SCALAR=1` therefore remains the architectural oracle.

The fast path must continue reading through `read_mixed_matrix_fast_path_regions`
before any destination write. This preserves source/destination overlap,
register ownership behavior, and plugin-visible access semantics.

## Correctness Validation

- Exhaustively compare each SIMD low-format decoder against its scalar decoder;
  PR #12052 already supplies this primitive-level coverage.
- Extend WMMA exactness coverage to every FP4/FP6/BF6/FP8/BF8 A/B format pair.
- Cover dense, scale32, scale16, and fixed 32x16x128 FP4 forms.
- Exercise random data, all low-format codes, exceptional FP8/BF8 values,
  E8M0 edge scales, accumulator modifiers, and destructive register overlap.
- Run focused SIMD tests in default and AVX2 builds, then the full RocJITsu test
  suite.

## Performance Method

Produce matched before/after binaries from the same source revision, compiler,
Release flags, CPU affinity, and benchmark inputs. The only source difference
between binaries is the arithmetic staging patch.

Measure decoded instruction execution for:

- Dense 16x16x128 F8F6F4 across representative and aggregate format pairs.
- Scaled 16x16x128 F8F6F4 for scale32 and scale16.
- Scaled 32x16x128 FP4 for scale32 and scale16.

Report medians from alternating runs. The primary table compares SIMD-before
against SIMD-after, which isolates the new decode staging. A scalar column is
reported as context but is not used to attribute the incremental gain because
`RJ_FORCE_SCALAR` also disables the pre-existing SIMD matrix core.

No performance claim is accepted unless outputs match the scalar oracle before
timing and the patched binary is faster over repeated process runs.
