# Method

How the numbers in [REPORT.md](REPORT.md) were obtained, and what they are and are not
evidence of. The measurement core lives in `src/common/`; each experiment is a thin main over
it, so the controls described here apply uniformly rather than per program.

## Design principle: change one thing

The question "does a non-temporal store hint help" is easy to answer wrongly, because the
obvious way to add the hint also changes the access width, and width turns out to matter about
two hundred times more. So the variant table (`src/common/variants.h`) is built so that every
interesting comparison has a control differing in exactly one respect:

| to isolate | compare | against |
|---|---|---|
| the store hint at full width | `nt-store-128` | `plain-128` |
| the load hint | `nt-both-128` | `nt-store-128` |
| the store hint at 64-bit width | `nt-store-64` | `plain-64` |
| the store hint at 32-bit width | `pr2616-actual-32` | `plain-32` |
| access width alone | `plain-64`, `plain-32` | `plain-128` |
| hand-writing the store, hint aside | `asm-plain-128` | `plain-128` |
| the `TH_STORE_NT_RT` hint | `ntrt-store-128` | `asm-plain-128` |

The last two are the ones that changed a conclusion. `TH_STORE_NT_RT` looks like a 1.1% win
against the production baseline; against a hand-written store carrying the *default* hint it is
worth +0.09%, because the gain was codegen. Without that control a codegen artefact is
indistinguishable from a cache effect.

Every variant is a faithful transcription of `__amd_rocclr_copyBuffer`: same argument list,
same grid-stride loop, same `end_ptr` termination, same scalar tail. `aligned_size` is kept as
a runtime value rather than folded away, because which branch it selects is itself part of what
is under test.

## Dispatch geometry

A copy kernel's performance depends as much on how it is dispatched as on what it stores, so
every experiment dispatches the way the runtime does: element count rounded up, global work
size clamped to `limit_blit_wg_` (the CU count) times a 512-thread local size then aligned up,
`end_ptr` excluding the scalar tail. One implementation, in `src/common/geometry.h`, mirroring
`KernelBlitManager::shaderCopyBuffer` argument for argument. Getting any of it wrong changes
the number of grid-stride iterations per thread, which is a larger effect than anything under
test.

## Establishing that a result is not noise

Two rules, both in `src/common/harness.h`, applied by every experiment.

**Arms are interleaved, not blocked.** Each arm is measured once per iteration, and the order
is reshuffled every iteration. Measuring all iterations of one arm and then the next credits
each arm with whatever the machine happened to be doing during its block; the effects being
chased here are around 1%, which is well inside the drift a GPU produces over a few seconds.
Differences are then computed **paired** — sample *i* of one arm against sample *i* of another,
taken seconds apart under the same conditions — so drift cancels instead of being averaged
over and hoped about.

**Every arm occupies two slots.** Both slots run identical code, so the difference between them
is the resolution limit of the rig: measured, per run, rather than assumed. A result counts
only if its 95% bootstrap interval excludes zero **and** its magnitude exceeds the widest
same-arm-twice gap in that run. The second test is what distinguishes a real 1% from a
confidently-measured artefact — a tight interval says a measurement is repeatable, not that it
is measuring the intended thing. Results failing it are printed with `(ns)` and must not be
quoted as effects.

Intervals come from a bootstrap on the median of the paired differences: 3000 resamples, fixed
seed, one definition in `src/common/stats.h`. That last detail is not cosmetic — the previous
tree had this copy-pasted seven times and it had already drifted to 2000 resamples in one
program, so two tables in the same report were computed with different estimators.

Reporting the resolution limit alongside every result also bounds the negative claims. "No
adversarial case found" means nothing without knowing the smallest regression the suite could
have detected, so the adversarial experiment prints its own coarsest resolution (8.5 pp, at
one insensitive case) next to its verdict.

## Timing

Events are recorded inside the stream, not around the host call, so the interval is GPU
execution and excludes launch overhead — which at 16 KiB is several times the copy itself.
Setup work is enqueued *before* the start event and only the thing under test between the
events. That split matters: the first version of the adversarial suite timed the copy together
with the thing the copy was supposed to damage, so a win on the copy masked any loss on the
victim.

Where a case enqueues on several streams, all of them are drained before the elapsed time is
read, so queue depth cannot accumulate across iterations.

## Confounders and what removes each

| confounder | how it would show up | control |
|---|---|---|
| Drift, thermal or otherwise, during a run | credited to whichever arm ran later | one sample per arm per iteration, shuffled order, paired differencing |
| Inline asm changing codegen | a codegen win credited to the temporal hint | `asm-plain-128`: same asm, default hint |
| Access width | a width penalty credited to the hint | `plain-64`, `plain-32` |
| Cache state left by the previous measurement | a warm start reads as a speedup | 1 GiB flush before every timed region (10.7x measured GL2) |
| Host launch overhead | dominates small copies, unrelated to the kernel | events recorded inside the stream; an empty-dispatch reference printed alongside small-copy results |
| Two kernel objects rather than one | code layout differences read as a hint effect | `small_copy` runs a single kernel object with the store selected by an argument, alongside the two-object form |
| The compiler eliminating or hoisting the work | measures nothing at all | byte-exactness check before timing, memory clobbers in the sweep loop, never-true sink stores |
| A variant not compiling to what it claims | measures the wrong instruction entirely | `remote/isa_check.sh` against expectations declared in `variants.h`; run first by `run_all.sh` |
| Support kernels carrying hints of their own | the probe is not neutral | the same ISA check asserts no `th:` modifier in `sweepReadKernel` or `chaseKernel` |
| First-write cost on a fresh allocation | the first size to reach new pages pays for the allocator | source and destination fully written before any timing |
| Other tenants on the GPU | random interference | machine state captured into `provenance.txt` per run |
| Clock state | unpinnable on this part | measured instead: `clock_watch.sh` samples throughout each run and the distribution is recorded |

## Probes

Three, chosen to make different questions large enough to read.

**Bandwidth sweep** (`sweepReadKernel`, one pass over a footprint larger than cache) is used
for the flush and for the concurrency victim, where the question is throughput.

**Dependent-load chase** (`chaseKernel`, a randomised cycle over 64-byte-strided lines, one
lane) is used for every residency question. Each hop's address comes from the previous load, so
the runtime is steps x memory latency and nothing else. That converts "is this buffer still
cached" from a bandwidth question, where prefetch and queueing blur the answer, into a
directly readable time difference. Randomisation is what stops the prefetcher hiding it.

**Repeated read inside one dispatch**, total traffic held constant, is how cache capacity is
measured. Everything stays inside a single dispatch deliberately: cross-dispatch retention is
itself under question, so relying on it would confound the measurement.

## Footprint sizing

Every footprint is expressed relative to `kGL2Bytes` in `src/common/config.h`, the *measured*
capacity, never the driver's `l2CacheSize`. The two disagree by 24x and the first round of this
work sized everything against the driver figure, which is what made the concurrency result come
out at 1% instead of 5% and made the adversarial suite's "no case found" untested. Nothing in
this suite may reintroduce that by hardcoding a size.

## What is not controlled

- **One machine, one GPU, an A0 engineering sample.** No claim here transfers to production
  silicon without re-measurement.
- **The concurrency victim is synthetic.** A sweep loop over a cache-resident buffer is a
  plausible stand-in for a cache-sensitive kernel, not a workload.
- **Absolute times in the 16-48 MiB band are not a smooth function of size** and are bimodal
  across process invocations. Paired comparisons are unaffected because both arms always sit on
  the same plateau, but absolute figures in that band describe a run. Cause not established.
- **The end-to-end test compares whole processes**, because the flag is read once at runtime
  init and cannot be switched inside a run. That comparison is far noisier than everything else
  here, which is why it carries a null control — two runs at the same setting — and is treated
  as a check that the plumbing works and the sign is right, not as an effect size.
- **PAL is not measured at all.** PAL is Windows-only; the mirror change is
  compile-consistent only.
