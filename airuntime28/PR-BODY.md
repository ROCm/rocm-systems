## What

Adds `__amd_rocclr_copyBufferNT` — `__amd_rocclr_copyBuffer` with
`__builtin_nontemporal_store` on the store — behind a new `DEBUG_CLR_BLIT_NONTEMPORAL` flag,
default **false**.

It keeps `ulong2` and tests `aligned_size == sizeof(ulong2)`, so it reaches its own wide path.

Worth stating plainly, because the ticket assumed otherwise: **adding the hint does not cost
access width.** `__builtin_nontemporal_store` rejects HIP's `ulong2` because that type is a
struct, not because 128 bits is too wide — the builtin accepts a pointer to any native vector,
and HIP reaches 128-bit non-temporal stores through `ext_vector_type(2)`. Here the question
does not arise at all: `blitcl.cpp` is OpenCL, where `ulong2` is already a native vector.

## Why, measured

gfx1250 (MI450 A0 engineering sample). Every figure is a paired comparison against the same
kernel without the hint: one sample per arm per iteration in shuffled order, with the noise
floor measured in the same run from a duplicate slot per arm.

| scenario | effect on the metric that matters |
|---|---|
| isolated 1 GiB copy | -0.47% [-0.83, -0.11] — below the ±0.74 pp resolution limit, so no effect |
| isolated 96–192 MiB copy | roughly -3% to -5%, significant |
| co-running cache-sensitive kernel's runtime | -2.4% to -4.7%, significant for an 8–128 MiB working set, peak near 32 MiB |

The concurrent case is the only one with a mechanism on this part. gfx1250 does not retain GL2
across a dispatch boundary, so a copy cannot leave anything behind for a later kernel to lose —
which also means the hint cannot hurt a sequential consumer. 27 adversarial cases, spanning
footprints from half of GL2 to 2.7x GL2, found none where the hint is measurably worse.

For scale: narrowing the access instead of keeping 128-bit costs **+77%** (64-bit) and
**+220%** (32-bit) on a 1 GiB copy — two orders of magnitude more than any temporal hint is
worth. A variant that buys the hint by narrowing the store is strictly worse than doing nothing.

## Blocker before this is enabled anywhere

**It is not architecture-gated.** On MI200 and Navi the same builtin emits *coherence* bits
rather than a cache hint, and nothing here was measured on those parts. Default-false keeps the
default path unchanged, but a user who sets the flag on a non-gfx12 part today gets coherence
semantics they did not ask for.

## Validation

- **ISA** — the NT kernel emits `global_store_b128 ... th:TH_STORE_NT` at full width, its load
  carries no hint, and the baseline kernel is unchanged. Checked against the real
  `BlitLinearSourceCode` blob extracted from `blitcl.cpp`, so the check cannot drift from what
  ships.
- **Byte-exactness** — `hipMemcpyAsync` D2D over 10 sizes x 3 base offsets, including sizes that
  exercise the scalar remainder tail (1, 7, 255, 4 KiB+3, 3 MiB+17), offsets that force the
  unaligned `uint` path, and a guard byte past the end to catch overruns. 0 failures with the
  flag off and on.
- **Kernel selection** — confirmed from `AMD_LOG_LEVEL=4`: `__amd_rocclr_copyBuffer` with the
  flag off, `__amd_rocclr_copyBufferNT` with it on.
- **hip-tests** — MemoryTest1, MemoryTest2 and DeviceMemoryTest abort at the same pre-existing
  points with the flag off and on, and MemoryTest2's failing sites are identical. Six runs each
  under flag-off, flag-on and stock ROCm give 0 failures every time.
- **PAL** — the mirror is compile-consistent but untested; PAL is Windows-only.

## Not my call

Nobody has shown that 100–300 MiB device-to-device copies, or concurrent cache-sensitive
kernels, are hot for a workload we care about. That is what decides whether this is ever worth
enabling.

## Where the measurements live

The full investigation - measurement core, eight experiments, the inspection scripts, and the
result set every figure above is pinned to - is on
[`users/victzhan/AIRUNTIME-28-investigation`](https://github.com/ROCm/rocm-systems/tree/users/victzhan/AIRUNTIME-28-investigation/airuntime28),
under `airuntime28/`. **That branch is not for merging**; it exists so these numbers can be
re-derived rather than taken on trust.

Start at `airuntime28/REPORT.md`. `METHOD.md` covers the controls and, more usefully, what is
not controlled. `CHANGELOG.md` lists every claim an earlier revision of this work made and this
one withdraws. Re-run everything with `airuntime28/remote/run_all.sh`, which writes a
timestamped result set with provenance and fails loudly if any variant stops compiling to the
instruction it claims.

## Side findings from the investigation
Two things worth knowing independently of this change:

1. **gfx1250 does not retain GL2 across a kernel dispatch.** Not the fence scope (forcing
   system scope everywhere changes nothing, and gfx1250 does not take PR #966's gfx12 path), not
   the allocation's memory type (none of six allocation kinds retains). Mechanism still open;
   this overlaps AIRUNTIME-2 and is a considerably larger lever than this change if it is not
   intended.
2. **`hipDeviceProp_t::l2CacheSize` reports 4 MiB on this part; GL2 measures 96–128 MiB.** The
   KFD record behind that figure has all its geometry fields zeroed, so it is an unpopulated
   stub rather than a wrong number. It is worth treating as absent, because sizing a working set
   against it produces measurements that look fine and mean nothing.

## Related

`PR 2616` never executes its own 64-bit path: its kernel branches on
`aligned_size == sizeof(ulong)` (8) while `shaderCopyBuffer` still passes `kMaxAlignment` (16),
so every aligned copy falls through to the `uint` branch and runs a *32-bit* non-temporal copy —
3.2x slower than the baseline on a 1 GiB copy. Any measurement of that PR as-is would have shown
non-temporal stores to be catastrophic, for reasons unrelated to non-temporal stores.

Ticket: AIRUNTIME-28
