# Coverage per architecture

Approximate coverage of distinct library operations used by Rocjitsu, with
default modifiers and FP mode (not the entire GPU ISA). GPU mnemonics that map
to the same operation count once: FMA, FMAC, FMAMK and FMAAK share one FMA entry.
These tables cover the matrix, FP, DOT and math families;
conversion and graphics operations are not included in these percentages.

## RDNA3

| Instructions | Loose | Bit-exact | FP status |
|---|---:|---:|---:|
| Matrix (WMMA) | 100% | 100% | N/A |
| FP arithmetic | 100% | 70% | 3% |
| Math (RCP, RSQ, SQRT, etc.) | 100% | 93% | 7% |
| Dot products | 100% | 100% | 0% |

## RDNA4

| Instructions | Loose | Bit-exact | FP status |
|---|---:|---:|---:|
| Matrix (WMMA, SWMMAC) | 100% | 100% | N/A |
| FP arithmetic | 100% | 80% | 10% |
| Math (RCP, RSQ, SQRT, etc.) | 100% | 93% | 20% |
| Dot products | 100% | 100% | 50% |

Percentages are rounded to the nearest 10%, or nearest 1% below 10% and above 90%.
Bit-exact coverage excludes loose fallbacks. FP status measures bit-exact
execution with reporting, counting only instructions that can produce FP status;
N/A means none in that family do. Support for nondefault modifiers and FP modes
varies by instruction.

## FP arithmetic and DOT counts

These counts group the APIs called by generated handlers by library operation, combining
Wave32/Wave64 and encoding variants. Bit-exact means mode 2 succeeds with default
controls. FP status counts only APIs with a `trapsts` or `excp_flag_user` output;
mode 3 succeeding on an API without that output is not FP-status coverage.

| Architecture/family | Bit-exact mnemonics | FP-status mnemonics |
|---|---:|---:|
| RDNA3 FP | 43/62 | 2/59 |
| RDNA4 FP | 59/78 | 11/75 |
| RDNA3 DOT | 8/8 | 0/4 |
| RDNA4 DOT | 12/12 | 2/4 |

The FP-status numerators are RDNA3 `DIV_FIXUP_F32/F64`; RDNA4 adds
`DIV_FIXUP_F16`, `DIV_FMAS_F32/F64`, `FMA_F16`, `FMA_MIXLO/HI_F16`, `MUL_F32`,
`PK_FMA_F16`, and `PK_FMAC_F16`. `DIV_SCALE_F32/F64` and `FMA_DX9_ZERO_F32`
have no FP-status output and are excluded from these denominators.

For DOT, integer instructions and FP8/BF8 DOT4 have no FP-status output and are
excluded. RDNA4 FP32-output `DOT2_F32_F16/BF16` accepts mode 3 and leaves flags
unchanged, matching its documented gfx1201 behavior; true16-output DOT2 rejects
mode 3. The 50% figure describes these two supported contracts out of four,
not the ability to raise a flag on half of all DOT mnemonics.

## CDNA and gfx1250

### Matrix instructions

CDNA matrix support covers twenty gfx942 MFMA forms (FP16/BF16, XF32 and FP8/BF8),
fourteen gfx950 half-input MFMA forms plus eight OCP FP8/BF8 forms, and five
gfx1250 half-input WMMA forms plus sixteen FP8/BF8 K64/K128 forms.
gfx942 FP8/BF8 uses FNUZ; gfx950 and gfx1250 use OCP.
All have scalar, AVX2 and AVX-512-v4 implementations. Loose, allow-Loose and
Compatibility are supported; strict bit-exact modes are not advertised.
Comparison and register-routing instructions also use rj-arith. Other families
are migrating independently; an entry point's header documents its supported
semantics and controls.

### Floating-point selection

CDNA3/4 legacy MIN/MAX and MIN3/MAX3/MED3 support FP16/FP32, with binary
FP64 MIN/MAX as well. gfx950 also has propagating FP32 MINIMUM3/MAXIMUM3.
gfx1250 supports numeric and propagating binary/ternary selection, including
packed FP16 forms. Loose and Compatibility retain their respective numerical
and output-control policies, with scalar, AVX2 and AVX-512 paths.

FP-status reporting is not implemented for these selection entry points, so
explicit mode 3 stops execution. Mode 2 also rejects effective OMOD scaling on
gfx950 propagating FP32 ternary selection. Selected NaN inputs are also rejected
for gfx950/gfx1250 packed FP16 MINIMUM3/MAXIMUM3 and gfx1250 packed FP16
MINIMUM/MAXIMUM, pending target qualification. Rejection leaves
the destination unchanged; Compatibility continues to support these cases.
