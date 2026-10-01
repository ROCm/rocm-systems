# AIPROFCOMP-865 — Health workloads: pinned host + alloc-once

**Date:** 2026-10-01
**Related:** [TCC series affinity + coverage](aiprofcomp-865-tcc-series-affinity-coverage.md) §6
**Evidence:** `validation-artifacts/spp-health-261001-gfx942-500med/reports/delta_ge100_investigation.md`

## Goal

For gfx942 health cases (**vcopy**, **nbody**/mini-nbody, **mega_kernel**):

1. Use **pinned host** staging where H2D/D2H is on the hot path (`hipHostMalloc`).
2. **Do not** `hipMalloc` / `hipFree` (or host alloc/free) **inside** the iteration loop — allocate once, reuse for all health iters (`-i 500`, nbody `131072 500`, mega `-n 500`).

Stable **device VAs within one process** keep L2 channel hashing steadier across the 500 profiled dispatches in a single PMC pass. This does **not** remove cross-pass remapping when each `pmc_perf_*.yaml` restarts the app (see affinity doc §6.2).

## Before → after

| Workload | Before | After | Files |
|----------|--------|-------|-------|
| **vcopy** | Device alloc once ✓; host `malloc` (pageable); no per-iter alloc/free ✓ | Host H2D/D2H buffers **pinned**; device still once outside `-i` loop | `sample/vcopy.cpp` |
| **nbody** | Upstream HIP-Examples: device once ✓; host `malloc`; **`nIters` hardcoded 10** (ignored argv[2]) | Vendored under `sample/mini-nbody/`: **argv[2] iters**, host **pinned**, device once | `sample/mini-nbody/hip/nbody-block.cpp`, `timer.h`, `LICENSE`, `README.md` |
| **mega_kernel** | Device alloc once ✓; host `malloc` for I/O; per-iter only `hipMemcpy`/`hipMemset` ✓ | Host I/O buffers **pinned**; device still once outside `-n` loop | `sample/mega_kernel/main.cpp` |

| Check | vcopy | nbody | mega_kernel |
|-------|:-----:|:-----:|:-----------:|
| Pinned host staging | yes | yes | yes |
| No per-iter alloc/free | yes (was already) | yes (was already) | yes (was already) |
| Health CLI compatible | `-i 500` | `131072 500` | `-b 65536 -n 500` |

## What we did **not** do

- **Zero-copy / mapped host as kernel args** — would route through IO and change TCC/EA health fingerprints; staging pin only.
- **Cross-pass fixed device VAs** — would need a persistent allocator or single long-lived process across PMC YAML passes; out of scope.
- Packing / affinity implementation (separate design).
