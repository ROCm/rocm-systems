## What

Adds `__amd_rocclr_copyBufferNT` - `__amd_rocclr_copyBuffer` with `__builtin_nontemporal_store` on the store - behind a new `DEBUG_CLR_BLIT_NONTEMPORAL` flag, default **false**.

It keeps `ulong2` and tests `aligned_size == sizeof(ulong2)`, so it reaches its own wide path.

Worth stating plainly, because the ticket assumed otherwise: **adding the hint does not cost access width.** `__builtin_nontemporal_store` rejects HIP's `ulong2` because that type is a struct, not because 128 bits is too wide - the builtin accepts a pointer to any native vector, and HIP reaches 128-bit non-temporal stores through `ext_vector_type(2)`. Here the question does not arise at all: `blitcl.cpp` is OpenCL, where `ulong2` is already a native vector.

## Why, measured

gfx1250 (MI450 A0). Paired against the same kernel without the hint: one sample per arm per iteration in shuffled order, with the noise floor measured in the same run from a duplicate slot per arm.

| scenario | effect |
|---|---|
| isolated 1 GiB copy | -0.47% [-0.83, -0.11] - below the +/-0.74 pp resolution limit, so **no effect** |
| isolated 96-192 MiB copy | roughly **-3% to -5%**, significant |
| co-running cache-sensitive kernel's runtime | **-2.4% to -4.7%**, significant for an 8-128 MiB working set, peak near 32 MiB |

The concurrent case is the only one with a mechanism on this part: gfx1250 does not retain GL2 across a dispatch boundary, so a copy cannot leave anything behind for a later kernel to lose - which also means the hint cannot hurt a sequential consumer. 27 adversarial cases, at footprints from half of GL2 to 2.7x GL2, found none where the hint is measurably worse.

For scale: narrowing the access instead of keeping 128-bit costs **+77%** (64-bit) and **+220%** (32-bit) on a 1 GiB copy - two orders of magnitude more than any temporal hint is worth.

## Blocker before this is enabled anywhere

**Not architecture-gated.** On MI200 and Navi the same builtin emits *coherence* bits, not a cache hint, and nothing here was measured on those parts. Default-false keeps the default path unchanged, but a user who sets the flag on a non-gfx12 part today gets coherence semantics they did not ask for.

## Validation

- ISA, against the real `BlitLinearSourceCode` blob extracted from `blitcl.cpp`: the NT kernel emits `global_store_b128 ... th:TH_STORE_NT` at full width, its load carries no hint, the baseline kernel is unchanged.
- Byte-exact: `hipMemcpyAsync` D2D over 10 sizes x 3 offsets, including sizes that hit the scalar remainder tail and offsets that force the unaligned `uint` path, plus a guard byte past the end. 0 failures with the flag off and on.
- Kernel selection confirmed from `AMD_LOG_LEVEL=4`.
- hip-tests MemoryTest1/2 and DeviceMemoryTest: identical pre-existing failure sites with the flag off and on.
- PAL mirror is compile-consistent but untested (Windows-only).

## Measurements and full write-up

On [`users/victzhan/AIRUNTIME-28-investigation`](https://github.com/ROCm/rocm-systems/tree/users/victzhan/AIRUNTIME-28-investigation/airuntime28), under `airuntime28/`. **Not for merging.** Start at `REPORT.md`; `METHOD.md` covers what is and is not controlled; `CHANGELOG.md` lists every claim an earlier revision of this work made and this one withdraws.

Two side findings worth knowing independently of this change: gfx1250 does not retain GL2 across a kernel dispatch (not the fence scope, not the memory type - mechanism still open, overlaps AIRUNTIME-2), and `hipDeviceProp_t::l2CacheSize` reports 4 MiB where GL2 measures 96-128 MiB, because the KFD record behind that figure is an unpopulated stub rather than a wrong number.

Ticket: AIRUNTIME-28
