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
`std::atomic::wait` is also supported. Additional spinning defaults to zero.
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

Instruction destruction, architectural memory writeback, and wait-counter
updates stay on the issuing thread. Decoder allocation is thread-local.
Exceptions from unexpected callback failures are collected after outstanding
work is joined; precise asynchronous exception rollback is not implemented.

## Memory experiment

Ordinary global/buffer loads and stores can use the same executor pool, with an
ordered completion policy. ISA execution captures addresses, masks and store
bytes on the issuer. The helper performs the cache/memory access; the issuer
writes load results and releases the ISA wait counter at completion.

Only **one data-memory operation per CU** may be outstanding in this prototype.
Every subsequent memory instruction drains it, including instructions that
will run synchronously. This preserves the CU's mutable L1 cache ownership and
same-wave store-to-load ordering. A store's captured source can be overwritten
after issue. Existing outstanding wait counters prevent starting an offload.
This stronger ordering does not model separate GPU load/store pipelines.

CDNA5 ISA sections 5.7 and 5.7.1 distinguish data writeback from ordered counter
completion. They also require same-wave, same-address store/load ordering, and
describe separate counters for flat, LDS, asynchronous and tensor operations.
Supporting more memory concurrency needs explicit cache ownership, ordered
counter accounting and cross-queue address dependencies. Simply offloading all
memory handlers would violate these requirements.

Excluded memory operations include scalar memory, flat/scratch, atomics,
global-to-LDS loads, LDS operations, tensor DMA, transposed loads and partial
D16 destinations. They execute through the existing path after draining the
window. The large Gluon GEMM's input transfers go directly to LDS, so its memory
offload coverage consists of output stores. The IREE f16 GEMM exercises ordinary
loads and K=32 WMMAs; the latter require the separate smaller-WMMA opt-in below.
The memory adapter remains restricted to gfx1250.

## Scope

The `ASYNC_MMA_WAVE_SIZE` ISA property selects eligible CU step/issue adapters at
compile time: zero disables the adapter, 32 enables gfx1250/gfx1201 wave32, and
64 enables gfx942/gfx950 wave64. `HAS_WMMA_K64` retains its separate meaning.
The factory creates an async `step()` override only when the target's instruction
family is selected. Enabling only the original K>=64 experiment still leaves
gfx1201/gfx942/gfx950 on their ordinary entry. Ordinary CUs inherit that entry;
unsupported ISAs and clocked execution have no async environment lookups, queue
checks, helper initialization or per-wave statistics. The active window lives
on the issuing stack and is passed only to the async specialization, preserving
the ordinary CU object layout. Adding an ISA requires opting into the property
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
| `RJ_MATRIX_COEXEC` | 0 ordinary; 1 scan; 2 serial adjacent batch; 3 parallel adjacent batch; 4 scoreboard MMA; 5 scoreboard memory; 6 both | 0 |
| `RJ_MMA_SHARED_HELPERS` | Process-wide capacity, 0–128; zero forces synchronous execution | 4 in modes 4–6; unset means private helpers in mode 3 |
| `RJ_MMA_HELPERS` | Outstanding background MMAs per wave, 0–7 | 1 |
| `RJ_ASYNC_WINDOW` | Maximum issued instructions before draining, 1–256 | 32 |
| `RJ_ASYNC_WMMA_MIN_K` | Additional f16/bf16 WMMA selection: 32 enables K32; 16 enables K16 too; original large shapes remain eligible | 64 |
| `RJ_ASYNC_MFMA` | Bitmask: 1 multi-block f16/f32; 2 scaled f8f6f4; 4 regular f16 control shapes | 0 |
| `RJ_ASYNC_MEM_MIN_BYTES` | Minimum full-wave bytes for a memory offload | 512 |
| `RJ_MMA_WAIT` | 0 atomic wait; 1 private Linux futex | 1 on Linux |
| `RJ_MMA_SPINS` | Optional bounded spin before blocking | 0 |

`RJ_ASYNC` counters report submitted MMA/load/store jobs, ordinary instructions
issued while work remains pending, dependency waits, boundary drains, capacity
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
RJ_MATRIX_COEXEC=5 RJ_MMA_SHARED_HELPERS=4 RJ_MMA_SPINS=0 \
  agent-reserved-run taskset -c 80-95 /path/to/rocjitsu \
  --config /path/to/throughput-config.json -- ./memory-hip 32 0 256 256
```

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
helpers raise it to 1.0478 s. Six ordinary CU workers take only 0.3697 s. Keep
memory experimental; these results support MMA as the useful acceleration path.

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
