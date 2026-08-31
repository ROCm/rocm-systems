# Finding: on gfx1250, nothing in GL2 survives a kernel dispatch

Found while trying to measure AIRUNTIME-28, but it is not about AIRUNTIME-28 and deserves its
own ticket. It overlaps AIRUNTIME-2 (GL2 residency), and if it is not intended behaviour it is
a considerably larger lever than the change AIRUNTIME-28 is about.

**Summary.** Cache state does not persist across a kernel dispatch boundary on this part. It is
not the fence scope, it is not the allocation's memory type, and it is not a small cache. The
mechanism is still open and needs the CP/firmware team.

Measured on `heliosr-1b114-a07-4`, gfx1250 (MI450 A0 engineering sample, `REV_ID 0x00`), result
set `results/20260828_062133`.

## The measurement

Dependent-load latency over a footprint, three ways. **cold**: the previous dispatch streamed
1 GiB through the cache. **warm**: the previous dispatch walked the exact same addresses.
**same**: both walks inside one dispatch, per hop. If cold equals warm, nothing survived the
boundary; the third column is the control that separates "the cache does not work" from "the
cache is not retained".

| footprint | cold ns/hop | warm ns/hop | same-dispatch ns/hop | cold/warm |
|---|---|---|---|---|
| 32 KiB | 334.0 | 333.6 | 155.7 | 1.00x |
| 1 MiB | 413.8 | 413.8 | 326.2 | 1.00x |
| 4 MiB | 593.8 | 595.4 | 402.7 | 1.00x |
| 16 MiB | 715.9 | 719.5 | 596.9 | 0.99x |
| 32 MiB | 742.3 | 738.9 | 676.5 | 1.00x |
| 96 MiB | 758.4 | 762.9 | 734.7 | 0.99x |
| 256 MiB | 763.4 | 764.9 | 759.5 | 1.00x |

Largest cold/warm ratio at any footprint within GL2: **1.012x**. Meanwhile a walk repeated
inside a single dispatch runs up to **2.1x faster per hop** (155.7 against 334.0 at 32 KiB).
Caching demonstrably works. It does not survive leaving a kernel.

The result is insensitive to how hard the "cold" arm tries. Sweeping the flush from 128 MiB to
2 GiB — 1.3x to 21x measured GL2 — moves cold latency by **0.06%** (745.49 to 745.92 ns/hop),
and every value sits on the warm reference. The flush is not doing anything, because the
dispatch boundary has already done it.

That insensitivity is now asserted rather than merely observed: if cold latency starts tracking
flush size, `flush_sensitivity` fails the run, because it would mean residency had appeared and
every flush constant in the suite needed re-picking.

## It is not the fence scope

The obvious suspect was [PR #966](https://github.com/ROCm/rocm-systems/pull/966), which makes
gfx12 use a system-scope acquire on every dispatch — and a system-scope acquire issues
`GL2_INV`. Two measurements rule it out.

**gfx1250 does not take that code path.** The predicate is
`versionMajor()==12 && versionMinor()==0 && stepping in {0,1}` (`rocvirtual.cpp:2338`), which
matches gfx1200/gfx1201 (Navi44/Navi48) but not gfx1250, which is minor 5. Reading the decoded
packet headers back out of the runtime confirms it: at `AMD_LOG_LEVEL=5 AMD_LOG_MASK=8` a run
submits 448 dispatches with `acquire=1` (AGENT) and 272 with `acquire=2` — not system scope
everywhere.

**Forcing system scope changes nothing.** `AMD_OPT_FLUSH=0` moves all 720 dispatches to
`acquire=2, release=2`, verified in the same log, and the latency scan reproduces row for row
(339.3 vs 340.7 ns/hop at 32 KiB, 760.9 vs 768.4 at 256 MiB). If the fence scope were what
evicted GL2, agent scope would have retained it.

`remote/fence_check.sh` runs both halves.

## It is not the allocation's memory type

AMD's coherence documentation says the CP's kernel-boundary invalidate is selective —
`INVL2_NC` drops `MTYPE_NC` lines "while leaving MTYPE_CC and MTYPE_RW data in the cache" — and
the [Cache coherence](https://amd.atlassian.net/wiki/spaces/AMDGPU/pages/777540643) page states
MI450 maps **local memory RW** (the A0 workaround changed the *remote* AID mapping, not local).
A plain `hipMalloc` buffer should therefore survive.

Six allocation kinds, and none of them does (`residency` experiment, 1 MiB footprint; same at
32 MiB):

| allocation | cold ns/hop | warm ns/hop | same-dispatch | warm/cold | retains? |
|---|---|---|---|---|---|
| `hipMalloc` | 416.4 | 419.4 | 334.0 | 1.007 | no |
| `hipMalloc` fine-grained | 415.7 | 416.2 | 327.8 | 1.001 | no |
| `hipHostMalloc` | 1069.6 | 1077.5 | 1070.6 | 1.007 | no |
| `hipHostMalloc` NonCoherent | 1084.9 | 1079.5 | 1058.5 | 0.995 | no |
| `malloc` + `hipHostRegister` | 1072.2 | 1077.1 | 1063.9 | 1.005 | no |
| `hipMallocManaged` | 908.9 | 893.4 | 889.8 | 0.983 | no |

The same-dispatch column is the point: for device memory it is clearly faster than a single
walk (334.0 against 416.4 ns/hop), so caching works within a dispatch for exactly the
allocation that should have survived one. For the host-visible kinds it does not — they are not
cached in GL2 at all — which is consistent, but it means device memory is the row that carries
the argument.

## And GL2 is real, and 24x larger than the driver claims

A capacity sweep — repeated reads of a fixed footprint inside one dispatch, total traffic held
constant — finds a genuine cache:

| footprint | read GB/s | reading |
|---|---|---|
| 256 KiB - 12 MiB | 1742 -> 31019 | near cache (L0 + GL1), peak 31.0 TB/s; below ~4 MiB the grid is underfilled, so those points are launch-bound floors rather than cache limits |
| 16 - 96 MiB | ~11400 - 12100 | **GL2 plateau**, a 2.6x drop entering 16 MiB |
| 128 - 192 MiB | 8047 -> 4937 | GL2 capacity being exceeded |
| 256 MiB - 1 GiB | ~4360 - 4490 | HBM floor |

Capacity is bracketed at **96-128 MiB**: the plateau runs to 96 MiB and the curve crosses the
midpoint between plateau and HBM floor at 128 MiB. That matches the architecture notes' 96 MB
per AID and contradicts `l2CacheSize` = 4 MiB, which `hipDeviceProp_t`, `rocminfo` and
`amd-smi` all report — they read the same KFD topology record, whose geometry fields
(`cache_line_size`, association, latency) are all zero. It is an unpopulated stub, so **treat
the driver figure as absent rather than as a small number** (`remote/kfd_cache.sh` dumps the
record).

That 24x error is not harmless: it is what sized every footprint in the first round of this
work, and it is why the concurrency benefit first came out at 1%. See
[CHANGELOG.md](CHANGELOG.md).

## What the documentation does and does not permit

Worth being exact, because a full flush is not itself out of spec. The coherence page says the
CP does "an invalidate (`INVL2_NC`) **or** writeback/invalidate (`WBINVL2`) on this local cache
when performing a system-scope acquire fence". `WBINVL2` wipes RW lines too, so a complete
flush is explicitly allowed, and the measurement is consistent with MI450's CP choosing it over
the selective `INVL2_NC`.

Two things still do not fit:

- Those operations are documented as tied to **system scope**, yet most dispatches here are
  agent scope and forcing everything to system scope changes nothing. Scope is not gating the
  invalidation, and the documentation does not account for that.
- If the CP were using the selective `INVL2_NC`, local memory is `MTYPE_RW` on MI450 and would
  survive. It does not.

So the open question is narrower than "this contradicts the spec". It is whether an
unconditional, scope-independent flush is a deliberate choice on this part. If it is, the answer
to AIRUNTIME-2 style residency work is "not available on MI450", which is worth knowing
explicitly. If it is not, recovering cross-kernel residency is a much larger lever than
AIRUNTIME-28. Either way it needs the CP/firmware team, not more measurement from here.

## Context for whoever picks this up

[ROCM-11953](https://amd-hub.atlassian.net/browse/ROCM-11953) (SWDEV-548770) is where the gfx12
system-acquire came from: a P1/S1 7.1 branch blocker found on **Navi48**, not MI450. A CLR
change ("Track last used command") stopped placing marker packets for blocking commands like
`hipMemcpyWithStream`, which exposed a latent hazard — `hipHostMalloc` memory is `MTYPE_NC`, so
L2 could hold a stale copy of data the host had since overwritten. A host write of 0 was
invisible to the *next H2D copy*, breaking PyTorch's `test_sharded_grad_scaler_found_inf`, with
no workaround short of inserting sleeps. The hardware semantics quoted there, for Navi4x:

```
AcquireFence==1  GLK_WB, GLK_INV, GLV_INV, GL1_INV
AcquireFence==2  GLK_WB, GLK_INV, GLV_INV, GL1_INV, GL2_INV, GL2_WB
ReleaseFence==1  nothing
ReleaseFence==2  GL2_WB
```

The reporter found corruption only with `hipHostMalloc`; plain `malloc` or `malloc` +
`hipHostRegister` did not reproduce. That is what makes "scope the fence to dispatches that can
touch host-coherent allocations" attractive *on parts where the fence is actually the cost* —
but as measured above, MI450 is not one of them.

[ROCM-29191](https://amd-hub.atlassian.net/browse/ROCM-29191) is worth reading before touching
scope handling, though not for the reason its title suggests: it was opened as a CP/cache-op
scope defect and then root-caused to a compiler bug (a bad `LCOMPILER-2448` prefetch workaround
reading kernarg memory before CLR filled it), fixed via ROCM-28799. The durable fact from it is
a statement from the compiler side that in DPX mode a device-scope acquire does not invalidate
`MTYPE_NC` lines out of GL2, and that CLR deliberately sends a system-scope acquire when
reusing a kernarg slot for exactly that reason.

## Reproducing

```bash
./build.sh cache_capacity residency
./build/cache_capacity --iters 15 --warmup 4   # capacity sweep + residency scan
./build/residency       --iters 25 --warmup 5  # six allocation kinds
./fence_check.sh                               # fence scope; needs the patched CLR build
./kfd_cache.sh                                 # the empty KFD cache descriptor
```
