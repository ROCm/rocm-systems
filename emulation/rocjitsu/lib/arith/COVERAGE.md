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

Gfx1150 floating relational predicates use scalar/AVX2/AVX-512 numeric and
classification kernels on both waves. Compatibility preserves the inherited
DX10_CLAMP-gated INVALID policy (CLAMP: any NaN; otherwise signaling NaNs).
Explicit numeric modes leave FP status unused; strict status mode is unsupported.
No new empirical hardware qualification is claimed for this inherited model.

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

## gfx1251 packed U64

The three packed U64 forms (ADD_NC, SUB_NC, LSHL_ADD) use scalar, AVX2 and
AVX-512 paths for all semantics. ADD/SUB include independent half-element
negation and mathematical U64 saturation. LSHL_ADD retains the inherited
active count range0..4, checked before writes by Rocjitsu; no expanded hardware
claim is made for other counts. Scalar carriers replicate the appropriate
U64 data or U32 count. Existing decoder and simulator coverage is retained.

### gfx908 floating comparisons

All FP16/FP32/FP64 relational CMP/CMPX forms, including F/TRU, use shared
scalar, AVX2 and AVX-512 numeric predicates. Wave64-only gfx908 APIs preserve
Compatibility's CLAMP-only INVALID classification, including quiet NaNs and
constant predicates. Loose and numeric-only modes leave status untouched;
mode 3 remains unsupported. This carries the existing emulator model without
claiming new empirical qualification. Other older profiles migrate separately.

### gfx10 native floating comparisons

FP32/FP64 relational CMP/CMPX, including F/TRU, use the library on gfx1010
and gfx1030 for both waves. Shared gfx908/gfx10 status composition preserves
Compatibility's CLAMP-only INVALID policy. F/T predicates retain strict
FP-status support for this simple classification; other predicates still reject
that mode. Loose skips reporting. Scalar, AVX2 and AVX-512 paths remain wide.
FP16 migration is separate; its existing VM-side signaling path remains live.

Gfx1010/gfx1030 FP16 relational comparisons now use the same shared CLAMP-only
Compatibility status composition as native precisions, on both wave sizes.
The VM preserves ignored VOP3 OPSEL and normalized SDWA source halves; strict
FP-status mode remains unsupported. Gfx1150 retains its separate signaling policy.

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

## gfx12 dual MUL FP status

Gfx12 dual MUL Compatibility classification is library-owned and retains the
legacy FMA-style guest-policy model, including input-denormal causes beside
NaNs. Other dual operations retain their existing Compatibility policy. This
ownership change does not expand strict empirical qualification.

## Older ordinary FP32 binary arithmetic

ADD, SUB, SUBREV and MUL on gfx90a/gfx1010/gfx1030 use the existing numerical
models and scalar/AVX2 backends; Compatibility also uses the existing AVX-512
paths where eligible. Loose retains the AVX2 arithmetic path. Compatibility retains
the inherited MUL exception classifier; its qualification is separate from explicit FP-status
semantics. SDWA retains output scaling, clamp and destination-byte placement.
DPP can suppress all stores while Compatibility MUL still accumulates its
execution-participating causes. Gfx908 ADD/MUL and gfx1150 arithmetic have
additional status policies and remain separate migrations.

## Older native FP32 FMA

Native FMA and FMAC on gfx908/gfx90a/gfx1010/gfx1030 use the existing FP32
FMA implementation, as do gfx1010/gfx1030 FMAMK/FMAAK literal forms. These
bindings preserve the inherited Compatibility model and provide Loose paths,
using scalar, AVX2 and AVX-512-v4 arithmetic. Reusing the model does not establish
new hardware qualification. Separately rounded MAC/MAD and gfx1150's additional
FP-status policy remain separate migrations.

## Older Legacy/DX9 arithmetic

Legacy FP32 MUL on gfx908/gfx90a/gfx1010/gfx1030 and DX9 MUL on gfx1150
share the inherited multiply model: guest denormal preparation, positive-zero
products and unconditional NaN quieting. Both wave APIs support Loose, allow-Loose and Compatibility, with scalar,
AVX2 and AVX-512 paths; strict exact modes are unsupported. gfx1150 DX9 FMA/FMAC reuse the existing
DX9 FMA APIs, retaining forced denormal flushing and IEEE-dependent NaN quieting.
Legal SDWA MUL forms retain their selected destination placement.

## Older FP32 power-of-two scaling

FP32 LDEXP on gfx908/gfx90a/gfx1010/gfx1030/gfx1150 reuses the gfx942 API's
inherited always-quiet NaN, guest rounding and older output-modifier policy.
Scalar, AVX2 and AVX-512 paths support arbitrary encodings; SIMD repairs only
subnormal/boundary lanes. Compatibility and allow-Loose use the inherited model;
explicit exact modes remain unqualified.
The integer exponent does not receive floating-point source modifiers.

## Older native math

Gfx908/gfx90a/gfx1010/gfx1030/gfx1150 FP32 RCP, RCP_IFLAG, RSQ and SQRT use
rj-arith's inherited scalar and SIMD models. Compatibility preserves their
existing numeric and FP-status policy; explicit empirical semantics remain
unqualified. The VM retains operand access, destination placement and trap delivery.

The gfx942/gfx950 FP32 and FP16 native math APIs also own SDWA output controls
in scalar, AVX2 and AVX-512 paths. FP32 SIN/RCP_IFLAG retain guest-mode output;
other FP32 math uses nearest-output scaling. FP16 finalizes an already-scaled
half without a second multiplication. These controls do not change ordinary
encodings, and gfx1250 does not accept the SDWA selector. See
[the API contract](include/rj_arith/detail/rj_arith_math.h) for details.

The gfx942 integral/decomposition APIs (TRUNC/CEIL/FLOOR/RNDNE, FREXP_MANT
and FRACT, FP16/FP32) also own inherited SDWA rounded-output scaling and clamp.
They share the native-unary scalar/AVX2/v4 output passes. Ordinary arithmetic
and its existing strict-mode/FP-status support are unchanged.

Older FP64 native-math migration also routes TRIG_PREOP on gfx908/gfx90a,
gfx1010/gfx1030 and gfx1150 through the inherited 1201-bit table and existing
scalar/AVX2/AVX-512 implementations. Compatibility/Loose remain the supported
semantics; strict empirical requests are not newly qualified.

## Older-profile integer dot products

Rocjitsu routes the integer DOT2/DOT4/DOT8 forms present on gfx908, gfx90a,
gfx1030 and gfx1150 through the existing scalar, AVX2 and AVX-512-v4 backends.
This includes the old-destination DOTC forms where present, half selectors,
signedness and saturation. Gfx1010 has no such integer-dot instructions.
Floating-point DOT2 forms migrate as described below.

## Older host-model floating dots

gfx908, gfx90a and gfx1030 FP16-input FP32-output DOT2/DOT2C use the same
pair-sum-then-accumulate model as gfx942, including packed source selectors,
modifiers, destination overlap and partial EXEC. Compatibility and Loose use
the existing scalar, AVX2 and AVX-512-v4 implementations; strict bit-exact modes
remain unsupported. This transfers the inherited software model without new
empirical qualification. gfx1150 uses the distinct model described below.
The superseded HostF32 packed-dot and accumulating emitters have no remaining
production users and are removed.

## gfx1150 floating dots

FP16/BF16 DOT2 with FP32 or true16 output and the FP16 DOT2ACC use the
existing gfx11 arithmetic in both waves. Compatibility retains rocjitsu's
integer-grid accumulation, NaN and packed narrowing policies; Loose uses
the existing scalar, AVX2 and AVX-512 arithmetic. Packed-output Compatibility
retains the inherited scalar grid model. Guest rounding/denormal controls
remain ignored; FP16 overflow, inline replication, modifiers, selected
C/D halves and DPP continue through the shared adapters. This inherits the
existing software model, not a new gfx1150 hardware qualification. Explicit
strict decoded modes 2 and 3 reject active VOP2, VOP3 and VOP3P requests
without changing D; Compatibility/Loose/allow-Loose support the inherited model.
This qualification boundary does not cover VOPD: its FP16/BF16 dual DOT2ACC
forms retain their existing execution policy and can write both destinations
in modes 2 and 3.

The old true16-dot emitter loses its last production users and is deleted.
Its semantic classes now route directly to the shared library binding, along
with the migrated HostF32 packed and accumulating forms on CDNA1/2 and RDNA2.

## BF16 SDWA widening

BF16-to-FP32 widening accepts `CVT_SDWA` for inherited output scaling and
clamp, including normalization of tiny inputs before scaling. Scalar, AVX2 and
AVX-512 share this model. Compatibility, Loose and allow-Loose support it; strict
empirical modes reject this unqualified encoding policy. Ordinary widening
retains its existing qualification. The VM stages inputs and places output bytes.

Unsuffixed packed RTZ conversion also accepts `CVT_SDWA`. It finalizes already-
rounded halves and applies inherited clamp without multiplying again; scalar,
AVX2 and AVX-512 share the finalizer. The gfx12 API keeps rejecting this selector.

## gfx1150 scalar floating point

All 58 scalar FP forms used by Rocjitsu support Loose and Compatibility.
Compatibility preserves the inherited software model, including guest rounding,
IEEE-dependent NaNs and FP status. The 30 gfx1150-specific arithmetic, selection,
rounding and conversion entry points do not advertise new empirical qualification;
comparisons share the existing raw-bit predicates. Scalar operations have one
SGPR value per operand, so there is no lane-vector SIMD backend to select.

## Older CDNA ordinary FP16 arithmetic

Rocjitsu routes gfx908/gfx90a ADD, SUB, SUBREV, MUL and LDEXP through the existing
CDNA half-arithmetic APIs, including VOP2/VOP3, DPP and SDWA. Compatibility keeps
the inherited guest-rounding, FP16 overflow and output-modifier policies; Loose
uses the same scalar/AVX2/AVX-512-v4 backends with its established freedoms.
This reuse adds no empirical bit-exact qualification. Older Radeon arithmetic,
FP16 FMA/MAD, packed/mixed arithmetic and selection remain separate migrations.
The shared legacy binary executors remain live on Radeon until that migration.

## gfx1150 ordinary FP16 arithmetic

ADD/SUB/SUBREV/MUL/LDEXP use the existing gfx942 numeric variant in both waves:
Compatibility keeps the inherited destination-rounding tininess test and
unconditional signaling-NaN quieting, guest rounding, FP16 overflow and
round-before-OMOD order. Loose uses the established approximate arithmetic.
Both share scalar, AVX2 and AVX-512-v4 backends, including guest modes.
The gfx1150 adapters preserve selected halves, inline operands and DPP.
Strict empirical modes remain unsupported; no new qualification is claimed.
Legacy gfx1150 did not report FP status for these operations, so this
migration introduces none. The shared legacy half executors remain live on
CDNA1/2 and gfx10 until their separate migrations; their gfx1150 inline VOP3
bodies and legacy calls are removed.

## gfx10 ordinary FP16 arithmetic

gfx1010/gfx1030 ADD/SUB/SUBREV/MUL use explicit model variants, with LDEXP
reusing the gfx942 variant. Both waves and VOP2/VOP3/DPP/SDWA use rj-arith.
gfx1010 retains encoded-result tininess and quiets input signaling NaNs;
gfx1030 retains its existing gfx11 numeric model. Guest rounding, denormal
controls, saturation and output ordering stay with each inherited model.
Scalar/AVX2/AVX-512 numeric paths are shared; Loose skips status classification.

Compatibility carries the legacy TRAPSTS classifier into the library:
gfx1010 ADD reports causes, while gfx1030 reports all four binary operations.
The corresponding Compatibility status pointers must be valid; other supported
semantics leave them unused. Strict modes remain unsupported rather than
claiming a new empirical qualification. Operand staging and destination-field
placement remain in the VM; the old VM classifiers and rounded SDWA arithmetic
adapter lose their final users and are deleted. CDNA1/2 and gfx1150 still need
the shared legacy half numeric executors on this independent branch.

### Older-target half selection

Gfx908/gfx90a MIN/MAX/MIN3/MAX3/MED3 reuse the existing CDNA selection model;
gfx1010/gfx1030/gfx1150 reuse Radeon NaN priority and median zero ordering.
Gfx1150 also routes MINMAX/MAXMIN. Compatibility, Loose and allow-Loose use
scalar/AVX2/AVX-512 kernels with decoded half selection, aliases and EXEC.
SDWA retains rounded-result scaling and byte placement. These migrations add
no target empirical qualification or FP status; strict decoded requests reject.

## Legacy selection SDWA output

Unpacked legacy MIN/MAX FP16/FP32, unsuffixed and gfx942, accept `ALU_SDWA`
for inherited scaling of the rounded result and subsequent clamp. Scalar, AVX2
and AVX-512 preserve guest rounding, selected halves, aliases and EXEC. Numerical
policy lives in the library; byte placement and observations remain in the VM.
Ordinary selection and its FP-status/qualification behavior are unchanged.

## gfx908 and gfx90a integer matrix instructions

Each target's five signed-byte MFMA forms have scalar, AVX2 and AVX-512-v4
implementations. All semantics use wrapping integer arithmetic and ignore FP
controls. gfx90a retains its adjacent M-lane A broadcasts; gfx908 retains the
inherited full-wave mapping. The target suffix also distinguishes the ignored
FP16_OVFL control from the newer MFMA saturation contract. This migration does
not qualify the older targets' floating-point MFMA models.

## gfx90a FP64 MFMA

Both gfx90a FP64 MFMA instructions use the existing two dense FP64 library
operations, with scalar, AVX2 and AVX-512 implementations. Loose, allow-Loose
and Compatibility are supported; strict bit-exact modes remain unsupported.
The binding ignores BLGP on gfx90a instead of forwarding the newer targets'
sign controls. EXEC and guest FP controls do not affect this inherited model.
Compatibility retains legacy FP32 widening for inline accumulators.

## gfx1150 dense WMMA

All six gfx1150 dense WMMA forms, in Wave32 and Wave64, use the existing gfx11
library operations: FP16/BF16 factors with FP32 or packed output, and IU8/IU4
factors with INT32 output. Compatibility retains the legacy ordered DOT2 model
and packed narrowing, or integer group saturation. Loose uses the existing
faster arithmetic. The gfx1150 binding preserves its operand-tuple remapping.
This reuses the inherited gfx11 model; it is not a new gfx1150 hardware
qualification. Existing scalar, AVX2 and applicable AVX-512 paths are shared.
Integer gfx11 WMMA also uses the shared AVX-512 VNNI kernel at the Zen4 CPU
level, in both wave sizes and with staged saturation.

## gfx1251 FP64 WMMA

The Wave32 FP64 WMMA instruction uses the library's shared dense FP64 scalar,
AVX2 and AVX-512-v4 kernels. Compatibility and Loose retain the inherited fixed
nearest-even fused arithmetic with preserved subnormals; guest FP controls do
not alter it. Explicit strict modes remain unsupported pending qualification.
The decoder still requires a complete Wave32 EXEC and validates tuples before
operand reads. The whole-tile library API has no EXEC parameter.

## gfx908 and gfx90a FP32 matrix instructions

The five FP32-input MFMA forms on each older CDNA target use rj-arith in
Loose, allow-Loose and Compatibility semantics, with scalar, AVX2 and
AVX-512-v4 paths. Strict empirical modes are not qualified. gfx90a retains
its adjacent-block A broadcasts and independent guest input/output denormal
controls; gfx908 retains full-wave broadcasts and preserves denormals.
Both ignore overflow saturation. This does not qualify their FP16/BF16 or
FP64 forms, which remain separate instruction families.

### gfx90a floating comparisons

FP16/FP32/FP64 relational CMP/CMPX, including F/TRU, use the existing shared
comparison APIs and scalar, AVX2 and AVX-512 backends. Compatibility carries
raw ordering, guest input flushing and the existing absence of FP-status
reporting; CLAMP does not add a classifier on this target. Mode 3 remains
unsupported. The VM owns explicit SGPR/VCC and EXEC commits, including both
mask writes for CMPX. This migration does not newly qualify a numerical model.
