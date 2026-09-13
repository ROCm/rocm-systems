# Experimental asynchronous instruction execution

This opt-in experiment extends [adjacent WMMA batching](matrix-coexecution.md)
with a register scoreboard and bounded execution windows. The issuing CPU
thread can decode and execute independent instructions while persistent helpers
run expensive handlers. The queue abstraction separates issue order from
completion policy; instruction eligibility and operand resolution belong to
the ISA adapter.

This is functional simulator acceleration, not a GPU timing model or a memory
hazard validator. It remains disabled by default.

## Queue and completion contract

Instructions enter each queue in program order. Independent MMA callbacks can
finish in either order. Polling retires every completed MMA, releasing that
instruction's register dependencies and helper slot. A consumer waits only for
the outstanding jobs that conflict with its reads or writes (RAW, WAR, WAW).
Shared read-only inputs are allowed. Source operands are not copied, so a later
overwrite must wait while a helper still reads them.

The issuing thread calls the helper's blocking completion primitive when it
needs an unfinished result. Linux private futexes are the default; standard
`std::atomic::wait` is also supported. A bounded 512-pause warm period precedes
blocking on both the idle helper and the issuer waiting for completion. Setting
`RJ_MMA_SPINS=0` restores immediate blocking; each side can also be tuned
separately. Helpers block after the warm period when no work arrives.
The completion word includes sleep intent to avoid missed wakeups and needless
wake syscalls. Both completion and job publication synchronize with
release/acquire operations. No coroutine or condition-variable mutex is needed.

Capacity acquisition never waits. A pair of atomic bitmap words covers up to
128 helpers. An exhausted pool takes at most two loads and no atomic RMW.
Reservation makes at most two CAS attempts per bitmap word; contention can
produce a conservative miss. Slot choice rotates to spread work across helpers.
On a miss, the issuing thread executes synchronously after resolving true
dependencies. It does not wait for an unrelated earlier MMA or enqueue work
behind a busy worker. A completed helper becomes reusable when its owner polls
or joins it; the worker does not recycle its own completion record.

Instruction destruction stays on the issuing thread. Decoder allocation is
thread-local; memory pipelines retain their existing ownership and accounting.
Exceptions from unexpected callback failures are collected after outstanding
work is joined; precise asynchronous exception rollback is not implemented.

## Memory accesses

Memory offloading has been removed. Its measured host access costs did not
amortize publication, synchronization and retirement, even in the synthetic
case with independent ALU work. The executor has a single MMA queue; loads and
stores always use the existing memory pipelines and wait-counter handling.

The scoreboard still checks memory instruction operands against pending MMA
reads and writes. Supported ordinary memory instructions may execute on the
issuer while independent MMAs run. LDS, atomics, direct-to-LDS transfers and
other excluded instructions drain the MMA window before executing. This does
not introduce background cache access or change memory completion ordering.

## Scope

The `ASYNC_MMA_WAVE_SIZE` ISA property selects eligible CU step/issue adapters at
compile time: zero disables the adapter, 32 enables gfx1250/gfx1201 wave32, and
64 enables gfx942/gfx950 wave64. `HAS_WMMA_K64` retains its separate meaning.
The factory creates an async `step()` override only when the target's instruction
family is selected. Enabling only the original K>=64 experiment still leaves
gfx1201/gfx942/gfx950 on their ordinary entry. Ordinary CUs inherit that entry;
unsupported ISAs and clocked execution have no async environment lookups, queue
checks, helper initialization or per-wave statistics. The active window is constructed lazily on the issuing stack after decoding
an eligible MMA and checking helper availability. Ineligible instructions skip
queue construction, polling and draining. The window is passed only to the
async specialization, preserving the ordinary CU object layout. Adding an ISA requires opting into the property
and supplying its ISA adapter.

The prototype requires functional execution, the adapter's wave size, full EXEC,
no VGPRMSB or GPRIDX addressing, and no active debugger, trap handler or architectural observer
plugin. Only the throughput plugin is admitted. Its instruction counts and
dispatch wall times remain useful; its per-handler timing does not include
background execution and must not be used to estimate concurrent CPU work.

The original arithmetic selection is f32-output FP8/BF8 16x16x64/128 WMMA and
32x16x128 FP4 WMMA. Separate controls add f32-output f16/bf16 K=32/K=16 WMMA,
multi-block MFMA, scaled MFMA, and regular f16 MFMA controls. Multi-block means
one ISA instruction computes several matrices, such as `32x32x4_2B`; it does
not mean submitting several decoded instructions as one helper job.

Keep the default arithmetic allowlist limited to the large gfx1250 shapes.
This is a conservative instruction-cost policy: the host handler must be long
enough to amortize publication, completion and lost locality. K alone is not a
cost estimate across instruction families, and a long handler still needs
independent work to overlap. K32 stays synchronous unless explicitly enabled
for an experiment; its synthetic gain did not generalize to the source-built
IREE kernel. K16 and MFMA extensions likewise remain separate opt-ins. Bounded
decode lookahead could refine profitability within an eligible family later;
it is not required to exclude K32 from the current selection.

Supported scalar/vector arithmetic can run between these operations. The
scoreboard resolves actual ISA register selectors, including inline constants
and packed-half aliases; the operand's `is_vgpr()` capability flag alone is
insufficient. VGPRs and AccVGPRs occupy distinct 256-register dependency banks.
MFMA scale VGPRs are reads, including the entire register containing a selected
scale byte. Immutable inline accumulators and scales are allowed; SGPR and
special-register MMA sources are excluded. Both lazy register banks are
materialized before publication on CDNA3/4.

Unknown instructions, control flow, barriers, EXEC/MODE changes and register
indexing operations drain the window. All work also drains at the instruction
window limit and before returning to CU scheduling. Jobs therefore do not
outlive the current wave's issue window. This bounds lifetime and keeps wave
storage valid, at the cost of missed overlap across scheduling boundaries.
Lazy register storage is materialized before publication to avoid races with
inline instructions allocating new chunks.

The process-wide helper pool is reused across issuers and XCDs. The existing
CU dispatch pool still serializes complete command-processor submissions.
Fork, concurrent independent VM instances, hardware numerical qualification,
and architectural tracing with jobs in flight remain unqualified.

## Controls

| Variable | Meaning | Default |
|---|---|---|
| `RJ_MATRIX_COEXEC` | 0 ordinary; 1 scan; 2 serial adjacent batch; 3 parallel adjacent batch; 4 scoreboard MMA | 0 |
| `RJ_MMA_SHARED_HELPERS` | Process-wide capacity, 0–128; zero forces synchronous execution | 4 in mode 4; unset means private helpers in mode 3 |
| `RJ_MMA_HELPERS` | Outstanding background MMAs per wave, 0–7 | 1 |
| `RJ_ASYNC_WINDOW` | Maximum issued instructions before draining, 1–256 | 32 |
| `RJ_ASYNC_WMMA_MIN_K` | Additional f16/bf16 WMMA selection: 32 enables K32; 16 enables K16 too; original large shapes remain eligible | 64 |
| `RJ_ASYNC_MFMA` | Bitmask: 1 multi-block f16/f32; 2 scaled f8f6f4; 4 regular f16 control shapes | 0 |
| `RJ_MMA_WAIT` | 0 atomic wait; 1 private Linux futex | 1 on Linux |
| `RJ_MMA_SPINS` | Bounded pause iterations before blocking | 512 |
| `RJ_MMA_IDLE_SPINS` | Override the helper's idle warm period | `RJ_MMA_SPINS` |
| `RJ_MMA_COMPLETION_SPINS` | Override the issuer's completion warm period | `RJ_MMA_SPINS` |

`RJ_ASYNC` counters report submitted MMA jobs, ordinary instructions issued
while work remains pending, dependency waits, boundary drains, capacity
fallbacks and owner-thread retirements. Pending does not prove that the worker
was actively computing for the entire overlap interval.

## Reproduction and validation

Build the simulator using [the benchmarking protocol](benchmarking.md). Use
the same frozen runtime for all modes, alternate their order, validate numerical
outputs, and require identical dispatch and instruction-family signatures.
Measure dispatch wall time and process wall time separately. Compare with both
adjacent batching and ordinary execution using more CU workers: helpers consume
additional host cores, and a speedup alone does not establish that helpers use
those cores better than ordinary CU parallelism.

The existing HIP matrix benchmark supports independent and dependent chains.
`tests/async_memory_hip_benchmark.cpp` supplies global B128 loads, optional
same-address store/load pairs, and either zero or 32 independent arithmetic
steps. It checks every output using an exact unsigned-integer host reference.
The explicit global address space prevents lowering to excluded FLAT accesses;
compiler barriers keep each loop iteration's access and the vector result live.

```sh
"$ROCM_PATH/bin/hipcc" -O2 --offload-arch=gfx1250 \
  tests/async_memory_hip_benchmark.cpp -o memory-hip
RJ_MATRIX_COEXEC=4 RJ_MMA_SHARED_HELPERS=4 RJ_MMA_SPINS=0 \
  agent-reserved-run taskset -c 80-95 /path/to/rocjitsu \
  --config /path/to/throughput-config.json -- ./memory-hip 32 0 256 256
```

This memory-only workload must submit no async jobs; it now checks the
synchronous fallback with the MMA adapter enabled.

Unit tests cover independent versus ordered completion, immediate capacity
fallback, helper reuse and failures, real interleaved memory/MMA execution,
register hazards, and inline-constant operand resolution. Extended MMA cases
compare both register banks bit-for-bit across independent execution, RAW, WAR,
WAW and scale-register reuse. The separate queue
stress test instruments the queue and helpers with ThreadSanitizer; it does not
constitute full-simulator ThreadSanitizer qualification.

`tests/async_mma_hip_benchmark.cpp` covers four independent or dependent chains
of smaller WMMA, multi-block MFMA, regular MFMA controls, and MXFP4 MFMA. Its
one-wave launch bounds prevent the large multi-block outputs from spilling.
Check code-object scratch metadata when changing the compiler or launch shape.

```sh
"$ROCM_PATH/bin/hipcc" -O2 --offload-arch=gfx950 -DMMA_TARGET=950 \
  tests/async_mma_hip_benchmark.cpp -o mma-hip
RJ_MATRIX_COEXEC=4 RJ_ASYNC_MFMA=1 RJ_MMA_HELPERS=7 \
  RJ_MMA_SHARED_HELPERS=4 agent-reserved-run taskset -c 80-95 \
  /path/to/rocjitsu --config /path/to/gfx950-throughput-config.json -- \
  ./mma-hip 2 128 512 0
```

Arguments are shape, block count, iterations, and dependent-chain flag. For
gfx942/gfx950, shapes 2/4/16 select `32x32x4_2B`/`16x16x4_4B`/`4x4x4_16B` f16;
8/116 select regular `32x32x8`/`16x16x16` f16. On gfx950, 128/64 select
`16x16x128`/`32x32x64` MXFP4. Compile separately with `MMA_TARGET=1250` or 1201
and the corresponding target for shape 32 or 16 WMMA.

## Measurements (2026-09-13)

The following tables preserve the earlier experiments, including the memory
offload modes that have since been removed. Modes 5 and 6 no longer select an
async adapter. Current policy and fallback measurements follow the tables.

Four alternating rounds on the reserved 16 physical host cores, with numerical
validation and identical throughput-plugin dispatch/instruction signatures.
Gluon/IREE use eight CU workers; HIP uses two. Async modes add four shared
helpers, allow seven MMAs per wave, and block without extra spins. Medians below
compare modes of the same frozen runtime; negative changes mean less time.

| Workload | Async mode | Ordinary/async dispatch s | Dispatch change | Ordinary/async process wall s | Wall change |
|---|---|---:|---:|---:|---:|
| Gluon 1024³ FP8 GEMM, K128 | MMA | 1.2217/0.9702 | -20.6% | 3.840/3.595 | -6.4% |
| Gluon FP8 GEMM suite, 80 cases | MMA | 14.7432/14.8528 | +0.7% | 19.980/19.975 | -0.0% |
| HIP K128, eight independent MMA chains | MMA | 2.0323/0.9594 | -52.8% | 2.390/1.325 | -44.6% |
| HIP load, immediate consumer | Memory | 0.0965/0.3641 | +277.5% | 0.420/0.695 | +65.5% |
| HIP load + 32 independent ALU steps | Memory | 0.8701/1.0478 | +20.4% | 1.235/1.425 | +15.4% |
| HIP same-address store/load + 32 ALU steps | Memory | 1.0423/1.5336 | +47.1% | 1.415/1.910 | +35.0% |
| IREE f16 matmul, WMMA K32 | Memory | 2.5008/2.5324 | +1.3% | 3.125/3.135 | +0.3% |
| IREE f32 softmax | Memory | 0.1671/0.1864 | +11.5% | 0.520/0.530 | +1.9% |

Adjacent batching takes 0.9776 s on the large Gluon GEMM and 1.0426 s on the HIP
MMA case. The scoreboard therefore adds little for the already adjacent Gluon
WMMAs, but reduces synthetic dispatch time a further 8.0%. The broad Gluon suite
does not establish a win; its dispatch ranges overlap and its wall medians agree.
An earlier two-worker Gluon configuration regressed 20.2%, with many boundary
drains and little independent work between eligible instructions.

At equal configured worker capacity, large GEMM with eight CU workers plus four
helpers takes 0.9702 s, versus 1.1917 s with twelve ordinary CU workers. HIP MMA
with two CU workers plus four helpers takes 0.9594 s, versus 1.0217 s with six
ordinary workers. Eight helpers reduce HIP MMA to 0.5365 s (0.890 s process wall),
using a larger CPU budget. These controls do not establish full-machine scaling.

Memory offload loses even with real overlap: the load/ALU case issues about 13.4
ordinary instructions per offloaded load while work remains pending. Its short
accesses do not amortize queue bookkeeping, publication, wakeups and retirement.
With zero helpers, memory mode takes 0.9777 s versus ordinary 0.8701 s; four
helpers raise it to 1.0478 s. Six ordinary CU workers take only 0.3697 s. These results motivated removing memory offloading and retaining MMA as the
acceleration path.

The ISA gate adds no executor checks or storage to unsupported CUs, confirmed
in emitted code. Whole-binary timing still depends on layout. The final gfx1201
parent control is 0.621669 → 0.626718 s dispatch (+0.8%) and 0.755 → 0.765 s wall
(+1.3%). It retires fewer instructions; perf shows residual frontend/branch costs.
Matching code-segment alignment mitigated an earlier 4–5% slowdown but did not
eliminate it. This is not a claim of exact performance neutrality. Final gfx1250
mode-zero parent controls are +0.2% for the large GEMM and −3.0% for load/ALU.

All comparisons use Clang 23, `-O2 -g -DNDEBUG`, no LTO, and mold with
`-Wl,-z,separate-code,-z,max-page-size=65536`; parent objects were relinked with the
same flags. These flags are experiment configuration, not a proposed global
build policy. The final implementation passed 736 focused correctness tests.

Raw data, frozen binaries/source, commands, numerical checks, ranges, perf
profiles and the code-layout investigation are under
`/home/jakub/rocjitsu/misc/async-scoreboard-benchmark/`. `report.md` documents the
complete experiment; `gluon/aligned*.json` and runtime `0018` supply the final
tables. Earlier negative results remain available.

### Smaller WMMA and multi-block/scaled MFMA extension

The extension uses frozen runtime `0020`, the same toolchain/reserved-core
protocol, four balanced rounds, numerical checks and complete instruction
signatures. New families are explicitly enabled. Main comparisons retain two
ordinary CU workers for HIP or eight for real kernels and add four helpers.
Negative changes below mean less time.

| Workload | Dispatch change | Process wall change |
|---|---:|---:|
| HIP gfx950 two-block / four-block f16 MFMA | -42.9% / -32.4% | -37.8% / -26.7% |
| HIP gfx950 MXFP4 16x16x128 / 32x32x64 | -60.8% / -55.6% | -52.3% / -50.3% |
| Gluon 1024³ K32 f16 / bf16 | -6.1% / -3.7% | -2.1% / -1.1% |
| Gluon 1024³ K16 f16 | +6.2% | +3.2% |
| Triton MXFP4 1024³, MFMA non-K dimension 16 / 32 | -19.5% / +4.2% | -5.4% / +1.8% |
| Original Gluon f16/bf16 suite, 32 cases | +5.2% | +3.5% |
| IREE f16 matmul, 14 dispatches | +3.5% | +3.1% |

The equal-worker-budget controls qualify these gains. HIP MXFP4 16x16x128
takes 0.8072 s with two CU workers plus four helpers, versus 1.0691 s with six
ordinary workers (-24.5%). The multi-block HIP cases favor ordinary workers.
The real K32 f16 GEMM takes 1.2688 s with eight plus four, versus 1.3177 s with
twelve ordinary workers (-3.7%). Both real MXFP4 variants favor twelve ordinary
workers: the 16x16 variant takes 0.7398 s ordinarily versus 0.7959 s asynchronously.
Useful same-wave overlap therefore depends on the kernel and available CU work.

For K16, the zero-helper async path already adds 2.6%; handoffs and dependency
waits raise the loss to 6.2%. The 32x32 MXFP4 kernel encounters register
dependencies on roughly three quarters of submitted MMAs; offloading loses even
though the zero-helper async entry is faster. Limiting each wave to one job does
not remove either slowdown. Small 32x32 Gluon tiles explain most of the suite
loss; their standalone zero-helper control reproduces the bookkeeping cost.
IREE's wide, overlapping timing ranges do not establish a stable change.

These results support keeping the new selections opt-in. The wider correctness
run passes 998/999 tests; the remaining DBT byte-identity failure reproduces in
the earlier build and with async disabled. All extended register-hazard cases
pass across eighteen mode/capacity/wait combinations, and both queue/helper
ThreadSanitizer stress variants pass. Emitted-code checks confirm the original
ordinary entries and the four intended optional ISA adapters.

The full report, all eleven HIP shapes, real-workload adaptations, zero-helper
controls, ranges, profiles and frozen provenance are in
`/home/jakub/rocjitsu/misc/async-scoreboard-benchmark/small-mma/report.md`.
The gfx950 GPU kernels are original Gluon/Triton kernels; their preparation and
reference checks run on CPU because the installed PyTorch failed to load a
gfx950 reference kernel. Hardware numerical qualification remains outstanding.

### Source-built IREE kernels with CPU headroom

To avoid crowding the reserved 16 cores, this follow-on explicitly uses two CU
workers, one simulation engine and four MMA helpers. IREE task workers and
OpenMP/MKL/OpenBLAS pools are capped at one. It uses the same frozen runtime
`0020`, four balanced rounds, throughput signatures and numerical checks.
The eight-worker/eight-engine results above describe a different configuration;
they do not establish deployment limits on a machine with more available cores.

IREE's repository generators supply the batch-MFMA presets and ordinary/scaled
matmul tests. All fourteen measured workloads were compiled from generated MLIR
and their emitted matrix instructions and zero-scratch metadata verified. The
end-to-end runner retains its default sampled numerical check of up to 10,000
output elements. Aggregate dispatch time includes generated reference-conversion
and packing kernels; process wall time includes the entire runner.

| IREE workload | Ordinary / async dispatch s | Change | Ordinary / async process wall s | Change |
|---|---:|---:|---:|---:|
| gfx950 MXFP4, 1024³, MFMA 16x16x128 | 2.4658 / 1.3233 | -46.3% | 2.855 / 1.795 | -37.1% |
| gfx950 MXFP4, 4096x1024x256 | 2.4502 / 1.6070 | -34.4% | 2.905 / 2.010 | -30.8% |
| gfx950 MXFP4, 1024³ with data tiling | 2.8220 / 2.7384 | -3.0% | 3.285 / 3.290 | +0.2% |
| gfx950 f16, 1024³, MFMA 16x16x32 | 1.6205 / 1.5472 | -4.5% | 2.140 / 2.045 | -4.4% |
| gfx1201 f16, 1024³, WMMA K16 | 2.5702 / 2.0662 | -19.6% | 2.790 / 2.305 | -17.4% |
| gfx1250 f16, 1024³, WMMA K32 | 3.0101 / 3.6750 | +22.1% | 3.455 / 4.155 | +20.3% |

Eight helpers lower the plain MXFP4 case to 1.1312 s dispatch / 1.550 s wall,
versus 1.8691 / 2.180 with two helpers. Six ordinary workers take 2.5932 / 2.920:
the kernel has only sixteen workgroups across eight XCDs, limiting independent
CU work. The rectangular case has the same multiply count and more workgroups;
six ordinary workers take 1.5466 / 1.940, slightly faster than two plus four
helpers. Extra cores can be useful through either form of parallelism, depending
on the kernel's grid and same-wave dependencies. Full-machine scaling is untested.

The actual two-/four-block batch-MFMA kernels regress 18–52% in dispatch time;
larger per-wave tiles reduce those losses to 10–17%. The emitted code reuses
accumulators and overwrites MMA inputs, producing true scoreboard dependencies.
LDS accesses and `v_perm_b32` also drain the current conservative window. The
gfx1250 K32 kernel frequently changes VGPRMSB and accesses LDS between WMMAs.
Those boundaries explain why these cases differ from the profitable kernels;
they are prototype restrictions, not intrinsic requirements to serialize every
LDS access or permutation with an MMA.

All 168 final/control samples passed numerical checking and complete instruction
signature comparisons. A larger-tile TileAndFuse preset failed IREE compilation
before simulator execution and was excluded; its diagnostics are retained. The
full tables, zero-helper controls, emitted ISA, generator snapshot, compiler and
artifact hashes, ranges and CPU-use counters are in
`/home/jakub/rocjitsu/misc/async-scoreboard-benchmark/iree-source/report.md`.

### MMA-only policy and synchronous fallback

Memory offloading has been removed. The default large-WMMA allowlist keeps K32
synchronous. A non-filling peek of the instruction cache sends cached words
outside that allowlist through ordinary issue, preserving its fetchability,
debugger and decode checks. Cache misses use full decoding. An MMA window is
constructed only after an eligible instruction is decoded and helper capacity
is observed. A busy pool still falls back immediately.

Six balanced IREE K32 rounds with runtime `0023` give 3.050 s ordinary versus
2.964 s with the large-only adapter, with zero offloads; process wall time is
3.510 versus 3.425 s. The ranges overlap, so this supports no observed wall-time
regression. The filter is not free: retired instructions rise 0.49% and user
cycles 1.17%. Merely making the window lazy, without bypassing the async issue
body, had retained a 4.3% dispatch slowdown in its separate comparison.

Four Gluon K128 rounds retain a 28.1% dispatch improvement (3.266 to 2.350 s)
and 15.3% wall improvement (5.860 to 4.965 s). Both workloads use two CU workers,
one engine and four helpers on the reserved cores, with numerical checks and
identical throughput signatures. Final cache/fetchability/queue/memory testing
passes 55 focused tests.

`MatrixHandlerBenchmark.DenseAndSparseAdjacentPairs` measures actual decoded
callbacks with one helper MMA and one issuer MMA, without intervening ALU.
Four rounds put sparse f32-output K64 f16/bf16 callbacks at 145-151 us and K128
FP8/BF8 callbacks at 195-199 us. Their independent pairs improve 47-49%.
Serialized offload adds roughly 9-11 us: K32 rises from 31.94 to 41.38 us
(29.6%). Large dense and sparse handlers amortize that cost better.

Passing the pair calibration alone is insufficient: K32 also improves 37% on
independent pairs, yet regresses in the IREE kernel. Sparse remains a measured
candidate, outside the runtime allowlist; no end-to-end sparse-kernel gain or
hardware qualification is claimed. Full data, failed intermediate comparisons,
test logs and reproduction scripts are in
`/home/jakub/rocjitsu/misc/async-scoreboard-benchmark/eligibility/report.md`.

### Bounded warm waits

An empty job using the actual helper/pool protocol takes 10.51 us with
immediate blocking and 1.00 us with 512 pause iterations before blocking,
a 10.5x reduction over six balanced rounds. Its payload contains instruction
and wave pointers plus the floating-point environment, with no matrix copy.
Saving/restoring the floating-point environment costs about 0.137 us. The burst
also reduces whole-process CPU time from 196 to 52 ms and context switches
from approximately 42,000 to four; the large cost was sleeping and waking.

The selected budget is bounded. Over a 100 ms idle interval, the helper consumes
about 0.061 ms of CPU, including surrounding wakeup work. A cold submission
still pays wakeup latency. An issuer immediately joining a long MMA may also
exhaust its completion budget and sleep: dense K128 serialized overhead falls
from 9.2 to 3.4 us and sparse K128 from 11.5 to 5.8 us. Independent pairs keep
the issuer busy and hide more of that cost. A 1 us transport measurement is not
a promise of 1 us total overhead for every instruction schedule.

Final runtime `0026` uses 512 pauses on each side and retains helper rotation.
Against immediate-blocking async, six-round medians improve synthetic K128
dispatch by 15.9% (0.9150 to 0.7691 s) and Gluon K128 by 0.9% (2.3433 to
2.3213 s). Their process wall times fall from 1.285 to 1.130 s and 4.985 to
4.960 s respectively. Longer spinning and last-helper reuse were tested but
not selected. All 48 final application runs passed numerical and complete
instruction-signature checks; final default settings passed 55 focused tests.
The queue/helper stress passed ThreadSanitizer across eight experimental
wait/budget/selection configurations.

The K32 control still submits no jobs: wall medians are 3.485 s ordinary and
3.500 s with the adapter (+0.4%), with overlapping ranges. The small eligibility
filter cost remains; the new wait code is not reached on this workload.
Raw data, CPU-cost comparisons, policy controls and reproduction scripts are in
`/home/jakub/rocjitsu/misc/async-scoreboard-benchmark/eligibility/warm-report.md`.
