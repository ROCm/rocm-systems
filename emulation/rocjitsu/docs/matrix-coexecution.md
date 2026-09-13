# Experimental large-WMMA co-execution

This opt-in prototype executes adjacent independent gfx1250 WMMA instructions
on persistent CPU helpers. Helpers can belong to each issuing host thread or
to a bounded pool shared by all issuers and XCDs. The issuer executes work
itself when the shared pool is busy. All finish before the CU returns to
its scheduler. No coroutines are needed: these handlers perform synchronous
CPU arithmetic, so actual overlap requires multiple CPU threads.

The [scoreboard experiment](async-instructions.md) adds independent completion
and overlap across supported non-matrix instructions. The measurements below
describe the earlier adjacent-batch experiment.

## Controls

| Environment variable | Meaning | Default |
|---|---|---|
| `RJ_MATRIX_COEXEC` | 0 ordinary issue; 1 scan only; 2 serial batch; 3 parallel batch | 0 |
| `RJ_MMA_HELPERS` | Maximum batch width minus one, clamped to 0-7; also the private helper limit | 1 |
| `RJ_MMA_SHARED_HELPERS` | If set, process-wide helper capacity, clamped to 0-128; 0 forces inline fallback | unset: private pools |
| `RJ_MMA_WAIT` | 0 standard atomic wait; 1 private Linux futex protocol | 1 on Linux |
| `RJ_MMA_SPINS` | Additional CPU pause iterations before sleeping, on either side | 0 |

Helpers are created lazily. A helper has one atomic job slot; publication and
completion synchronize with release/acquire operations. The Linux protocol
records sleep intent in the job word and issues `FUTEX_WAKE_PRIVATE` only when
that bit was set. Standard atomic waiting is available for comparison and as
the fallback on other systems. The caller's floating-point environment is
transferred with each job.

Choose N together with `cpu_dispatch_threads` and `num_threads` in the simulator
configuration. These are limits, not counts of continuously busy cores. The
shared CU pool currently serializes complete command-processor submissions;
small grids can leave much of the host idle despite a large pool. Conversely,
adding helpers to a busy host can oversubscribe it. `RJ_MMA_SHARED_HELPERS`
bounds the total helpers independently of the batch width. There is no
automatic tuning in this prototype.

The shared pool now uses two atomic free-slot bitmaps and the same persistent
helper protocol. Submission checks availability without scanning individual
slots and makes bounded reservation attempts. If every slot is busy,
the issuer immediately executes that instruction inline; it does not wait for
capacity or enqueue work. The final instruction always stays on the issuer.
Claims remain held through completion and joining, so another issuer cannot
overwrite a job before its owner reads the result. This favors simple ownership
over immediately recycling a completed helper. All submitted jobs are joined
before propagating an unexpected callback exception.

`RJ_COEXEC` counters distinguish accepted-batch coverage (`covered`) from
instructions actually offloaded (`shared_submitted`), inline execution
(`shared_inline`, including the issuer's normal final instruction), failed
claim attempts (`shared_full`) and slots probed (`shared_probes`). The current
CU dispatch pool serializes complete XCD submissions; helper reuse across XCDs
does not by itself remove that scheduling limit. The process-wide pool is an
opt-in experiment; fork and multiple-VM lifecycle integration remain unqualified.

## Scope and semantics

The fast path requires functional execution, gfx1250 wave32, full EXEC, valid
low-bank VGPR operands and no debugger or trap handler. It accepts same-opcode
runs of f32-output FP8/BF8 16x16x64/128 WMMAs and 32x16x128 FP4 WMMA. It stops at
other instructions, register hazards, page boundaries or the configured limit.
MFMA, mixed opcodes and lookahead across scalar instructions are not included.

Each new instruction is checked against every pending instruction for RAW,
WAR and WAW hazards. Shared read-only inputs are allowed. The issuer allocates
lazy destination storage before publishing jobs. The existing generated
arithmetic callbacks execute directly; decoding and instruction destruction
remain on the original thread. All helpers are joined even if a callback
throws. General asynchronous retirement and failure recovery remain outside
this feasibility prototype.

Observer plugins disable batching except for the throughput plugin. Its
instruction counts and dispatch wall times remain usable, but the joined
batch's handler time is attributed to its first instruction. Per-instruction
architectural snapshots and summed concurrent CPU time are not preserved.
Use dispatch wall time, not family execution time, to compare parallel runs.

## HIP benchmark

`tests/matrix_coexecution_hip_benchmark.cpp` uses the actual gfx1250 WMMA
builtins with independent accumulator chains or a dependent control. It checks
every output exactly. Build from the RocJITsu source directory with the local
TheRock SDK selected by `ROCM_PATH`:

```bash
"$ROCM_PATH/lib/llvm/bin/amdclang++" -x hip -O2 -std=c++20 \
  -DMATRIX_CHAINS=8 --offload-arch=gfx1250 \
  --rocm-path="$ROCM_PATH" --hip-path="$ROCM_PATH" \
  tests/matrix_coexecution_hip_benchmark.cpp \
  -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 \
  -o matrix-hip8
```

Arguments are `SHAPE BLOCKS ITERATIONS REPEATS DEPENDENT`. Shape 64 means
16x16x64 FP8; 128 means 16x16x128 FP8; 32 means 32x16x128 FP4. Each block has one
wave. `DEPENDENT=1` must accept no batches. Compile with `MATRIX_CHAINS=4` for
the four-chain case. Inspect the generated ISA: eight FP4 accumulators can
spill, and scalar loop bookkeeping splits an otherwise independent WMMA run.

For example, after building RocJITsu and setting `RJ_BUILD` and `RJ_CONFIG`
(to a gfx1250 functional config with the throughput plugin):

```bash
agent-reserved-run taskset -c 80-95 env \
  RJ_MATRIX_COEXEC=3 RJ_MMA_HELPERS=7 RJ_MMA_WAIT=1 RJ_MMA_SPINS=0 \
  /usr/bin/time -v "$RJ_BUILD/tools/rocjitsu/rocjitsu" \
  --config "$RJ_CONFIG" -- ./matrix-hip8 128 32 128 1 0
```

Follow [benchmarking.md](benchmarking.md) for the matching runtime/SDK setup,
reservation, counters and alternating runs. Compare mode 0, mode 2 and mode 3
with identical binaries, inputs and configurations. Validate the instruction
counts as well as numerics. Measure process wall time separately from dispatch
wall time; startup and copies matter for short kernels.

## Initial measurements

On sixteen reserved physical CPU cores, portable Clang 23 `-O2` without LTO,
with exact HIP output checks and the throughput plugin, four repeats of the
final prototype gave these median times:

| Workload/configuration | Ordinary dispatch | Parallel dispatch | Ordinary process | Parallel process |
|---|---:|---:|---:|---:|
| K128 FP8, 8 chains, 32 blocks x 128 iterations; 2 issuers, N=7 | 1.975 s | 0.561 s | 2.335 s | 0.895 s |
| K64 FP8, 4 chains, 256 blocks x 256 iterations; 16 issuers, N=1 | 1.561 s | 1.367 s | 1.905 s | 1.705 s |

The first loop has seven adjacent WMMAs: it uses six helpers per issuer,
although the configured limit is seven. In a separate four-repeat comparison
on the same K128 workload, increasing the ordinary pool from 8 to 16 threads
left dispatch time near 1.02 s; two issuers with helpers took 0.565 s.

A standard-library wait protocol initially regressed the busy K64 case.
Replacing it with private futex waiting removed the shared waiter registry and
yield loop and reversed the slowdown. An empty handoff microbenchmark showed
roughly 9-11 us sleeping round trips versus 1-2 us with bounded spinning, but
spinning offered little kernel benefit with spare cores and hurt crowded
configurations. Private futex waiting with zero extra spins is the initial
choice; the best N depends on available instruction and host parallelism.

These are synthetic simulator results, not full-machine or production GEMM
qualification. The separate experiment report retains the complete N=0/1/2/3/7
sweep, three WMMA shapes, dependent controls, perf profiles, hashes, commands
and raw samples. Validation includes 100 related tests, dedicated hazard/lazy
storage/clocked-fallback tests, an eight-way overlap test and helper-only TSan
stress with both wait protocols. The complete simulator has not been checked
under TSan for this experiment.

## Gluon GEMM and the shared pool

The follow-up uses the unmodified `test_runtime_gemm -k float8` subset of
Triton's `test_gluon_cdna5.py` at `24bb8414f7c34625c8cd8490e5d6e2d66f5cdce3`:
80 passed, 48 skipped (block K smaller than instruction K), and 48 deselected
on every run. It covers 250/256-cubed inputs, FP8/BF8 combinations, K64/K128
WMMA, and several tile sizes. A larger input calls the same test function with
M=N=K=1024, FP8 inputs and a 64x64x128 tile. All runs retain its Torch CPU
reference and `rtol=atol=1e-4` check, with matching instruction signatures.

Four balanced repeats, throughput plugin, warm Triton cache, sixteen reserved
physical cores and four shared helpers give these medians. Negative changes
mean less time than ordinary issue in the same binary and configuration.

| Workload | CU workers / engine threads | Ordinary dispatch | Shared dispatch | Dispatch change | Process change |
|---|---:|---:|---:|---:|---:|
| Original FP8 subset | 2 / 1 | 12.067 s | 13.320 s | +10.4% | +7.1% |
| Original FP8 subset | 8 / 8 | 17.011 s | 15.067 s | -11.4% | -8.6% |
| Same GEMM, 1024-cubed | 8 / 8 | 1.218 s | 0.979 s | -19.6% | -6.2% |

For the last row, process time falls from 3.840 to 3.600 s. Private helpers take
1.026 s dispatch and 3.640 s process; four shared helpers improve on that while
roughly one third of attempted offloads execute inline because the pool is
busy. Eight shared helpers offload more work but do not improve median time.
The small subset covers only 30.1% of WMMAs with accepted batches; the larger
input covers 75%. The eight-worker subset remains slower in absolute terms than
ordinary execution with two workers, despite its gain within that configuration.

The preceding private-helper comparison with two CU workers improved the
larger GEMM by 13.2% in dispatch and 7.1% in process time, but regressed the
original subset by 3.5% and 2.7%. These results support co-execution for larger
GEMM and some concurrency configurations, not a universal default.

Separate profiles localize most extra user cycles in the two-worker shared-pool
slowdown to existing reader/writer locking. Sampled unlock callers primarily
resolve to `RequestMtypeResolver::at()` and
`GpuMemory::reacquire_page_table_request()`, which release/reacquire the request
lease and VMID registry lock for memory chunks. Slot search is 0.02% of sampled
user cycles. Restricting the same two-worker comparison to one eight-core L3
domain does not remove the slowdown. The precise scheduling interaction and a
safe reduction of that memory-lock traffic remain unresolved; the experiment
does not justify removing mapping-lifetime or reentrancy protection.

The shared implementation passes 103 related tests and nine dedicated tests
across both wait protocols and capacities 0/1/2/4/8. Helper-only TSan stress
also passes both protocols, with and without spinning, using eight concurrent
issuers and disjoint non-atomic payloads. The separate Gluon report retains
full timing ranges, paired changes, perf profiles, caller samples, commands and
frozen binaries. Gluon uses Torch and matching TheRock libraries from the
20260822 nightly consistently across all variants; its absolute process times
should not be mixed with the earlier HIP experiment's runtime setup.
