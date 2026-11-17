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

gfx1250 FP32 ADD/SUB/SUBREV use their `_gfx1250` entry points with no
Compatibility FP causes. MUL retains inherited input-denormal/range reporting
without adding gfx12 INVALID causes; OMOD retains its legacy cause suppression.
Both explicit strict modes remain unqualified for these operations.

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

Gfx1250 binary FP32 selection uses `_gfx1250` entry points to preserve its
numeric-only Compatibility policy separately from gfx12's reporting policy.
FP-status reporting is not implemented for these CDNA selection entry points, so
explicit mode 3 stops execution. Mode 2 also rejects effective OMOD scaling on
gfx950 propagating FP32 ternary selection. Selected NaN inputs are also rejected
for gfx950/gfx1250 packed FP16 MINIMUM3/MAXIMUM3 and gfx1250 packed FP16
MINIMUM/MAXIMUM, pending target qualification. Rejection leaves
the destination unchanged; Compatibility continues to support these cases.


## CDNA integer operations

CDNA vector 32-bit integer arithmetic, carry/borrow, 64-bit shifts and widening
MAD/QSAD use the same library kernels on gfx942, gfx950 and gfx1250. Their
architecture-specific spellings are resolved in the simulator binding. These
integer operations support all semantics and reuse the existing scalar and SIMD
backends. Dedicated AVX-512 coverage varies by operation.
Halfword and packed-integer operations also reuse the library, preserving CDNA
low-half destination zero-extension. DOT2/4/8 use [integer_dot.cpp](src/dot/integer_dot.cpp); CDNA-specific
PRNG, BITOP3, rotate, funnel-shift and bit-count forms also route through
the library.

## Comparison Compatibility ownership

Gfx12 signaling FP comparisons classify Compatibility causes inside rj-arith,
including the inherited whole-wave nonfinite gate. The VM owns operand views,
VCC/EXEC writes and trap delivery. Ordinary Compatibility comparisons remain
numeric-only; explicit reporting qualification is unchanged. Gfx10 signaling
policy remains in the partially migrated RDNA2 adapter until that migration.

## Scalar floating point

gfx12 scalar arithmetic, comparisons, rounding and conversions use rj-arith in
every semantics, including Compatibility. Compatibility retains the inherited
scalar FP-status model; arithmetic and conversions still reject explicit mode 3
where reporting is not qualified. Scalar comparisons and rounding support mode 3.
The VM owns SGPR/SCC writes and trap delivery. gfx1150 retains its legacy scalar
FP implementation pending migration; gfx1250 has its own documented library policy.

## Older native FP16 FMA

Gfx908/gfx90a native and legacy FMA and gfx908 fused MAC, plus gfx1010/gfx1030/
gfx1150 FMA/FMAC and literal FMA, use the inherited gfx11 numeric model in
Compatibility and the existing scalar/AVX2/AVX-512 paths. Source/destination
half selection, partial EXEC, guest rounding/denormals, IEEE NaN quieting and
round-before-OMOD remain profile-specific bindings. They do not gain new
empirical qualification or FP-status reporting; active explicit strict decoded
requests are rejected. Loose and allow-Loose remain supported.

Gfx90a separately rounded MAD/MAC uses its own rj-arith model, described below;
it is not folded into native FMA. Packed/mixed half arithmetic is also migrated.
Graphics/division still have live legacy users.

## gfx908 FP16 MAD

Ordinary/legacy MAD and literal MADMK/MADAK reuse the existing CDNA3/4 rounded
FP32-intermediate half-MAD model, including scalar/AVX2/AVX-512 dispatch,
source/destination halves, output modifiers, FP16_OVFL, aliases and partial EXEC.
This inherited approximation supports Compatibility/Loose/allow-Loose, not
explicit bit-exact or FP-status semantics. Gfx90a's separately rounded half
MAD/MAC remains a different model and is not enabled through these entry points.

### gfx90a half MAD/MAC Compatibility

MAD/MAD_LEGACY, MAC and literal MADMK/MADAK use the inherited two-half-rounding
model in Compatibility, sharing existing scalar/AVX2/AVX-512 half MUL and ADD.
Loose uses FP32-intermediate MAD. Guest rounding, forced intermediate flushing,
IEEE NaN quieting, FP16_OVFL and final output modifiers are retained. This is
software-model reuse, not new empirical qualification; strict modes reject and
FP status is unchanged. The native half FMA migration is independent.

## Older Radeon FP64 arithmetic

Rocjitsu uses rj-arith for ADD, MUL, FMA and LDEXP on gfx1010, gfx1030 and
gfx1150, for both wave widths. These bindings retain the inherited guest
rounding, denormal, NaN and output-modifier policies. gfx1030 FMA keeps its
existing Radeon model; the other bindings reuse the CDNA-named entry points
because their preexisting numerical models agree. This reuse does not establish
new hardware bit-exact qualification. Compatibility and Loose support these
bindings, with scalar, AVX2 and AVX-512-v4 paths.

## gfx1251 packed FP64

ADD/MUL/FMA and MIN_NUM/MAX_NUM use the inherited MODE-aware model in
Compatibility, Loose and allowed-fallback semantics. Their arithmetic reuses
existing scalar/AVX2/v4 FP64 paths; packed selection fixups and masked stores
are also SIMD. Directed AVX2 arithmetic uses its inherited integer fallback.
Strict numeric and required FP-status semantics remain unsupported. All float
values, per-element signs, CLAMP, scalar carriers, EXEC and overlap are supported.
This is inherited behavior, not new empirical hardware qualification.
