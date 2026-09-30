# Shared x86-64-v4 provider for EXP/LOG and CDNA5 WMMA

This supersedes the EXP/LOG-held-constant packaging comparison in
[the focused WMMA design](2026-09-29-cdna5-wmma-v4-provider-design.md). The
five WMMA forms and their qualification remain unchanged.

## Images and build modes

In `RJ_CDNA5_WMMA_V4_LAYOUT=dso`, `librocjitsu.so` contains scalar/v3
execution and one sibling `librocjitsu_x86_v4.so` contains both the EXP/LOG v4
kernels and the five focused CDNA5 WMMA v4 callbacks. These are the two
participating core DSOs; unrelated RocJITsu plugins and hooks are unchanged.
The existing `off` mode retains its current EXP/LOG behavior and has no WMMA
overlay. `combined` and `dso` use the same v4 source objects and compile
options, placing them in the main library or the provider respectively. Do not
add an independent math-DSO layout switch. Keep an off-mode adapter so this
packaging change does not remove the existing explicit EXP/LOG v4 path.

The v4 EXP/LOG wave callbacks, including their eight-lane loops, live beside
the v4 bit kernels in the provider object. The main library does not call a
v4 bit kernel through an indirect pointer on every pack. Split baseline FP
selection from those callbacks so the main DSO has no direct v4 references.
In both measured modes, remove the original v4 math object from the main
library's transitive `rocjitsu_isa` composition; `combined` gets it only
through the isolated provider object. Standalone object-composed CLI, test,
fuzz, and hook images retain their current embedded EXP/LOG v4 behavior by
explicitly linking the v4 object and an embedded selection variant. Their
CDNA5 WMMA callbacks remain baseline as in the existing experiment. Do not
blanket-remove v4 from every `rocjitsu_isa` consumer. Test-only direct x8/x16
kernel coverage does not validate the shared-library path.

## Provider contract and runtime selection

Use one renamed, versioned private C getter, `rj_x86_v4_provider_v1`, and a
same-build descriptor containing the five typed WMMA callbacks, two typed
EXP/LOG wave kernels, the WMMA backend binding hook, an ABI version, and a
build digest.
The descriptor uses `F32UnaryWave`/`UnaryF32Kernel`, not raw VGPR pointers or
native SIMD types. Its digest includes the math and WMMA execution sources,
ABI headers, compiler identity, and effective v4 flags. Export only the
getter from the provider; retain `-z defs` and no duplicate core objects.
Flatten v4 COMDAT groups before symbol localization so final linking cannot
coalesce v3 and v4 helper definitions. Apply the same isolated object to
`combined` and `dso`.

One baseline-compiled provider resolver owns CPU/OS AVX-512 qualification and
descriptor validation. In `dso` mode it also owns sibling-path resolution,
`dlopen`, and the process-lifetime handle; `combined` calls the same getter
directly. No v4 code or initializer runs before the capability check.
The independent EXP/LOG and WMMA selectors call it only when their own policy
needs v4; neither selector relies on the other initializing first. Preserve
`RJ_MATH_BACKEND`, `RJ_CDNA5_WMMA_BACKEND`, `RJ_FORCE_SCALAR`, and the existing
gfx1201 VOP1-only EXP/LOG qualification. EXP/LOG `auto` remains disabled by
its performance gate and must not load the DSO merely to report a v4-capable
host. Missing, incompatible, or unsupported v4 falls back for `auto`; an
explicit `v4` request reports an error. Resolve failures during provider
initialization, not in `FpProvider::unary_f32()`, which is `noexcept`.

## Validation and comparison

Final-link every affected shared and object-composed image. Confirm the main
DSO has no EXP/LOG v4 definitions or AVX-512 code from either focused family,
the provider exports only its getter, and selected EXP/LOG and WMMA callbacks
resolve to the provider DSO in `dso` mode. Add a shared-image EXP/LOG
probe/driver beside the WMMA probe; the current object-composed FP test is not
this proof. Exercise both paths through `librocjitsu.so`, including bit-exact
scalar-oracle cases, force-scalar/v3 fallback, missing and mismatched provider,
and explicit-v4 errors. Direct object-linked tests alone are insufficient.

For `combined` versus `dso`, report stripped installed bytes for the main
library and provider, combined bytes, `.text`, and `.rodata`. Use at least
seven interleaved warmed shared-library runs for qualified EXP/LOG and all
five decoded WMMA forms, plus cold provider-load time. Keep input, compiler,
flags, and selected v4 callbacks matched; force `RJ_MATH_BACKEND=v4` because
EXP/LOG `auto` is disabled. These are host-simulator timings, not GPU
performance. Report any unavailable v3-host or whole-workload test as a
limitation rather than extrapolating from a microbenchmark.
