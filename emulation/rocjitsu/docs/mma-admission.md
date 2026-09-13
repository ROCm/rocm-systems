# Cached MMA lookahead admission

This optional prototype refines the [MMA scoreboard](async-instructions.md):
K32 WMMA is offloaded only when bounded lookahead finds another independent
MMA that the issuing thread can execute. Instruction size alone cannot predict
profitability. An isolated or dependent MMA pays helper handoff and completion
costs without useful overlap.

## Implementation

Each optional async CU adapter owns two caches: immutable speculative decode
objects keyed by encoding bytes, and admission plans keyed by PC, VMID and
register-bank bounds. Both successful and rejected plans are cached. The scan
stops at a dependency, unsupported instruction, control-flow boundary or the
edge of the current code page. It never executes instructions, changes the PC,
notifies observers or emits a diagnostic. The runtime scoreboard continues to
check dependencies and available helper capacity.

A plan collects an independent group, offloads its prefix and reserves the
final MMA for the issuer. Restricting the scan to pairs sacrificed throughput
on wider independent groups. The earliest outstanding issuer reservation wins,
so subsequent admissions cannot keep moving all useful work into the pool.

I-cache invalidation increments an epoch. After an epoch change, a plan checks
its saved code bytes before reuse; unchanged code keeps the decoded objects
and plans across dispatches. Changed bytes rebuild the plan. Debugger and
fetchability checks remain in the ordinary issue path. Cached objects use heap
allocation independent of the decoder's thread-local execution pool, and their
backing encoding bytes remain stable for their full lifetime.

**Only the additional speculative decoding is cached.** Ordinary execution
still decodes each issued instruction and retains its existing mutable state
and ownership. For a fixed scan bound and unchanged code, speculative decoding
and plan construction scale with distinct code examined per participating CU,
not dynamic loop iterations. Byte validation repeats after I-cache invalidation.
The cache currently retains old code variants until CU destruction; eviction
and sharing immutable entries among CUs are not implemented.

## Controls

All async execution remains disabled by default. K32 also remains an explicit
opt-in. The original large-WMMA selection and memory execution are unchanged.

| Variable | Meaning | Default |
|---|---|---|
| `RJ_MMA_ADMISSION` | 0 disabled; 1 observe K32 plans without gating; 2 gate K32; 3 gate every selected MMA family | 0 |
| `RJ_MMA_LOOKAHEAD` | Maximum following instructions examined, clamped to 1–16 | 8 |

For example, combine these with the existing scoreboard controls:

```sh
RJ_MATRIX_COEXEC=4 RJ_ASYNC_WMMA_MIN_K=32 RJ_MMA_ADMISSION=2 \
  RJ_MMA_SHARED_HELPERS=4 RJ_MMA_HELPERS=7 \
  agent-reserved-run taskset -c 80-95 \
  /path/to/rocjitsu --config /path/to/throughput-config.json -- /path/to/workload
```

`RJ_ADMISSION` counters distinguish speculative decodes, decode-cache hits,
plan construction, plan hits, epoch validation, accepted/rejected admissions
and reserved issuer MMAs. Counters flush at wave halt; the printed entry count
is the cache's current size, so summing it across flushes overcounts storage.

## Validation and measurements

The cache tests cover repeated loops, unchanged and changed code after I-cache
invalidation, positive and negative caching, register hazards and bounds,
page boundaries, decode failures and wider groups.
The existing queue, matrix, instruction-cache, debugger, fetchability and wait
counter tests exercise the integration in eight admission/capacity/lookahead
configurations. The scaling benchmark checks that 100 warm traversals add no
speculative decodes or plans:

```sh
agent-reserved-run taskset -c 80-95 /path/to/rocjitsu_tests \
  --gtest_filter=MmaAdmissionBenchmark.ColdAndWarmProgramSizeScaling
```

The complete local experiment and raw commands are under
`/home/jakub/rocjitsu/misc/async-scoreboard-benchmark/admission/`.
See `report.md` there for commands, frozen-runtime hashes, raw results,
profiles, CPU counters, timing ranges and controls. The initial campaign contains
554 successful timed application processes, including the full 32-case
f16/bf16 and 80-case FP8 Gluon runtime GEMM selections. Numerical checks and
complete throughput signatures agree throughout. Fifty focused tests pass in
16 configurations; the cache/matrix/queue test translation unit also passes
17 tests instrumented with ASan and UBSan, linked to the existing runtime.
This is not full-runtime sanitizer or hardware numerical qualification.

### Results on reserved host cores (2026-09-13)

Six rotated rounds, Clang 23 `-O2 -g -DNDEBUG`, LTO off, fixed TheRock SDKs.
All timed process trees use the reserved-core wrapper on CPUs 80–95, with
unused reserved SMT siblings. Gluon calls its original CPU reference; IREE
uses the source-built end-to-end matmul runner. The throughput plugin measures
active-dispatch wall time; process wall time includes startup and validation.

With two CU workers and four shared helpers, gated K32 returns the IREE
1024-cubed f16 kernel from 3.631 seconds unrestricted async to 3.024 seconds,
matching ordinary execution at 3.026 seconds. Its process wall time returns
from 4.110 to 3.470 seconds (ordinary 3.475). Admission rejects 59,392 candidates
and submits 3,584 jobs, versus roughly 65,800 unrestricted submissions.

The original 1024-cubed Gluon f16/bf16 GEMMs improve 8.8%/10.3% in dispatch time
and 4.8%/5.8% in process wall time versus the same two ordinary CU workers.
Independent HIP K32 chains gain 55% in dispatch time. The dependent chain
submits no jobs; its +1.9% timing delta is not resolved beyond the host's
measurement variability, while retired instructions increase about 0.22%.
An early cached-rejection bypass retired fewer instructions but did not improve
wall time, so it was dropped.

At a matched budget of twelve execution workers, the measured tradeoff is:

| Workload | 12 ordinary CUs: dispatch / wall s | 8 CUs + 4 helpers: dispatch / wall s | Dispatch / wall change |
|---|---:|---:|---:|
| IREE f16 K32 | 0.9625 / 1.440 | 0.9748 / 1.455 | +1.3% / +1.0% |
| Gluon f16 K32 | 1.3407 / 3.970 | 1.2347 / 3.860 | -7.9% / -2.8% |
| Gluon FP8 K128 | 1.1837 / 3.800 | 0.9514 / 3.580 | -19.6% / -5.8% |

K128 retains the existing executor's benefit; mode 2 performs no speculative
decoding for that family. With only six execution workers, allocating all six
to ordinary CUs beats two CUs plus four helpers on these real kernels. The
fixed-issuer speedups therefore do not establish the best division of cores.
Whole-machine scaling remains unmeasured.

The broad f16/bf16 suite has no aggregate win: about +2.5% dispatch time and
+0.6–1.4% process wall time versus the existing large-only policy. Aligned
64-by-64 tiles improve, while ragged small tiles do not consistently benefit.
An isolated ragged case rejects all offloads and does not reproduce the suite
slowdown. A large apparent FP8-suite regression disappears when restricting
both policies to one L3 domain (80–87): 14.520 versus 14.482 seconds dispatch.
Profiles and the locality control support placement-sensitive reader-lock
contention; the exact contended lock was not isolated.

Warm cache lookups cost about 7–8 ns per site for 100–10,000 candidate sites;
initial construction costs roughly 0.7–0.9 microseconds per site in the cache
microbenchmark. One hundred warm traversals create no new decodes or plans.
Repeated real Gluon launches also stop constructing decodes/plans once the
participating CUs warm up, while revalidating bytes after I-cache invalidation.
The four-launch K32 case improves 6.3% in process wall time.

Lookahead 1 / 8 / 16 takes 3.834 / 3.369 / 3.232 seconds on the large Gluon f16
kernel. A one-instruction scan misses partners and fragments wider groups.
Sixteen is useful for tuning this workload; eight remains the default bound.
Keep K32 admission opt-in: the experiment demonstrates selective profitability,
with meaningful workload and CPU-budget limitations.


### K64 and K128 admission follow-up

A further 108 timed runs explicitly enable admission mode 3 for both large
FP8 shapes. The earlier K128 control used mode 2 and did not run lookahead for
that shape. This follow-up uses the same frozen runtime, original Gluon
1024-cubed FP8 GEMM, tile 64x64x128, six rotated rounds and reserved CPUs 80–95.
All numerical checks and complete instruction signatures pass.

With twelve execution workers, ordinary execution uses twelve CUs; both async
policies use eight CUs plus four helpers. Entries are median dispatch / process
wall seconds:

| Instruction K | Ordinary | Unrestricted offload | Cached admission |
|---|---:|---:|---:|
| 64 | 1.1835 / 3.810 | 0.9902 / 3.615 | 1.0025 / 3.640 |
| 128 | 1.2106 / 3.830 | 0.9743 / 3.600 | 0.9522 / 3.585 |

Cached admission improves dispatch / wall time by 15.3% / 4.5% for K64 and
21.3% / 6.4% for K128 versus ordinary execution. Most of that gain comes from
MMA offload: relative to unrestricted offload, admission changes dispatch time
by +1.2% for K64 and -2.3% for K128. With two fixed CU issuers plus four helpers,
both Gluon shapes retain about 26–27% dispatch-time gains versus two ordinary
CUs; admission changes little compared with unrestricted offload.

The eight-chain HIP controls show a cost: admission slows dispatch by 8.7%
for K64 and 12.9% for K128 versus unrestricted offload, while CPU time rises
only 1.1% and 2.0%. Helper submissions fall about 9–10%, consistent with less
overlap from reserving the final MMA for the issuer. These results support
keeping large WMMA on the existing unrestricted policy by default.

Full tables, host counters and reproduction commands are in
`/home/jakub/rocjitsu/misc/async-scoreboard-benchmark/admission/large-report.md`.
