# Experimental large-WMMA co-execution

This opt-in prototype executes adjacent independent gfx1250 WMMA instructions
on persistent CPU helpers. Each issuing host thread owns at most N helpers;
the issuer also executes one instruction. All finish before the CU returns to
its scheduler. No coroutines are needed: these handlers perform synchronous
CPU arithmetic, so actual overlap requires multiple CPU threads.

## Controls

| Environment variable | Meaning | Default |
|---|---|---|
| `RJ_MATRIX_COEXEC` | 0 ordinary issue; 1 scan only; 2 serial batch; 3 parallel batch | 0 |
| `RJ_MMA_HELPERS` | Maximum helpers per issuing thread, clamped to 0-7 | 1 |
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
adding helpers to a busy host can oversubscribe it. There is no global helper
budget or automatic tuning in this prototype.

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
