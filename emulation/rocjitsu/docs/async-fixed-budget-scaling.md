# Best allocations at fixed execution-thread budgets

For paired measurements after instruction-fetch caching landed upstream, see the
[cache rebase comparison](async-cache-rebase-scaling.md).

Measured 2026-09-15 on async-scoreboard implementation `efc055a3e79`, rebased onto
`origin/develop` at `cc17ebc55df`. These are the best measured allocations for two
1024 x 1024 x 1024 FP8 GEMMs in functional simulation, ranked by median dispatch time.

## Best measured configurations

**Budget = E + (D - 1) + H**, where:

- E is `num_threads`, the XCD engine-thread count.
- D is `cpu_dispatch_threads`, including the submitting engine; it adds D-1 pool workers.
- H is `RJ_MMA_SHARED_HELPERS`, the process-wide async helper count.

Every row below sums to its exact budget. Launcher, doorbell and other runtime threads
are outside this execution budget. E/D/H gives the actual settings, with D inclusive.

| Budget | gfx950 MFMA E/D/H | Dispatch s | gfx1250 WMMA E/D/H | Dispatch s |
|---:|---:|---:|---:|---:|
| 4 | 1/3/1 | 1.5123 | 1/4/0 | 2.1871 |
| 8 | 1/3/5 | 0.8958 | 1/8/0 | 1.3532 |
| 16 | 2/7/8 | 0.5685 | 1/16/0 | 0.8910 |
| 24 | 1/9/15 | 0.5121 | 2/16/7 | 0.7649 |
| 32 | 3/12/18 | 0.4910 | 4/16/13 | 0.7556 |

Medians use three fresh confirmation processes per finalist. The complete report
includes all three finalists per workload/budget and their min/max ranges. Small differences
with overlapping ranges remain unresolved; these measurements do not establish a global
optimum for other workloads or execution policies.

## Confirmation ranges and process cost

Process wall time includes startup and transfers. All confirmation runs omit numerical
reference computations and output comparisons, as requested, and synchronize after launch.

| Budget | Workload | E/D/H | Dispatch median [min,max] s | Process wall median s |
|---:|---|---:|---:|---:|
| 4 | mfma | 1/3/1 | 1.5123 [1.4845,1.5233] | 3.470 |
| 4 | wmma | 1/4/0 | 2.1871 [2.1069,2.1999] | 4.400 |
| 8 | mfma | 1/3/5 | 0.8958 [0.8678,0.9057] | 2.860 |
| 8 | wmma | 1/8/0 | 1.3532 [1.3524,1.3985] | 3.570 |
| 16 | mfma | 2/7/8 | 0.5685 [0.5681,0.6001] | 2.540 |
| 16 | wmma | 1/16/0 | 0.8910 [0.8817,0.8929] | 3.100 |
| 24 | mfma | 1/9/15 | 0.5121 [0.5072,0.5149] | 2.480 |
| 24 | wmma | 2/16/7 | 0.7649 [0.7631,0.7752] | 2.970 |
| 32 | mfma | 3/12/18 | 0.4910 [0.4838,0.5031] | 2.460 |
| 32 | wmma | 4/16/13 | 0.7556 [0.7520,0.7649] | 2.970 |

## Method and interpretation

All processes use the same 64 reserved physical cores (32-95), with SMT siblings unused,
on a Threadripper PRO 9995WX. CPU affinity remains 64 cores while allocated execution
threads are limited to the stated budget. A shared lock serializes measured processes.
The frozen runtime uses Clang 23, -O2 without LTO, and the existing TheRock SDKs.

The search covers every integer E/D/H split at budgets 4 and 8. Budgets 16, 24 and 32
start with E=1,2,4,8 and a coarse helper grid, then explore nearby helper counts and
adjacent engine counts around the leaders. The initial 32-thread sweep contains 28
allocations, with three repetitions each. Refinement uses one process per new candidate;
the strongest three allocations per workload/budget receive three fresh, shuffled runs.

Both workloads use 64x64 output tiles and 256 workgroups. gfx950 uses the original Triton
16x16x128 MFMA kernel; gfx1250 uses the original Gluon K128 WMMA kernel. H=0 selects
ordinary mode 0. H>0 selects async mode 4, seven outstanding jobs per wave, 512-pause
warm waits, lookahead 8 and target-default admission. The same GPU inputs and kernels
are retained after removing the numerical comparisons during screening.

In the runtime measured here, D>1 uses one SoC-wide dispatch pool that serializes
complete submissions from the XCDs,
so at most D CUs plus H helper MMAs can advance concurrently. Other engines can do
scheduling work but cannot submit concurrent CU batches. At D=1 the pool is absent,
and all E engines can independently advance CUs. This explains why increasing E can
consume budget without increasing the CU batch width. These are capacity bounds,
not measured CPU utilization.

## Full 32-thread allocation table

The original coarse sweep completed before numerical checks were disabled. It uses
the same GPU kernels; the dispatch values below are medians of three runs per workload.
The final choices above also consider later local refinements.

| XCD threads E | Dispatch width D | Pool workers D-1 | Helpers H | gfx950 dispatch s | gfx1250 dispatch s |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 31 | 0 | 0.8413 | 0.9124 |
| 1 | 28 | 27 | 4 | 0.6591 | 0.8243 |
| 1 | 24 | 23 | 8 | 0.6015 | 0.7845 |
| 1 | 20 | 19 | 12 | 0.5474 | 0.7790 |
| 1 | 16 | 15 | 16 | 0.5261 | 0.7655 |
| 1 | 12 | 11 | 20 | 0.5226 | 0.9924 |
| 1 | 8 | 7 | 24 | 0.6031 | 1.0771 |
| 2 | 31 | 30 | 0 | 0.8383 | 0.9087 |
| 2 | 27 | 26 | 4 | 0.6545 | 0.8176 |
| 2 | 23 | 22 | 8 | 0.5910 | 0.7835 |
| 2 | 19 | 18 | 12 | 0.5299 | 0.7704 |
| 2 | 15 | 14 | 16 | 0.5210 | 0.9131 |
| 2 | 11 | 10 | 20 | 0.4970 | 0.9851 |
| 2 | 7 | 6 | 24 | 0.5821 | 1.3010 |
| 4 | 29 | 28 | 0 | 0.8270 | 0.9036 |
| 4 | 25 | 24 | 4 | 0.6576 | 0.8064 |
| 4 | 21 | 20 | 8 | 0.5855 | 0.7899 |
| 4 | 17 | 16 | 12 | 0.5474 | 0.7664 |
| 4 | 13 | 12 | 16 | 0.5182 | 0.9373 |
| 4 | 9 | 8 | 20 | 0.5169 | 0.9958 |
| 4 | 5 | 4 | 24 | 0.7625 | 1.5051 |
| 8 | 25 | 24 | 0 | 0.8392 | 0.9112 |
| 8 | 21 | 20 | 4 | 0.6399 | 0.8186 |
| 8 | 17 | 16 | 8 | 0.5838 | 0.7842 |
| 8 | 13 | 12 | 12 | 0.5430 | 0.9135 |
| 8 | 9 | 8 | 16 | 0.5162 | 0.9705 |
| 8 | 5 | 4 | 20 | 0.7640 | 1.5090 |
| 8 | 1 | 0 | 24 | 0.6215 | 1.4344 |

## Evidence and reproduction

The campaign completed 555 timed processes covering
351 workload/allocation pairs, including 90 fresh
confirmation processes. Completion, exact budgets, instruction signatures, async
retirement, source/runtime hashes and recorded configurations passed the metadata checks.
Numerical checks are disabled in all confirmation runs. Earlier completed checked samples
are retained and labeled; whole-process timings across those two modes are not comparable.

Local evidence: `/home/jakub/rocjitsu/misc/async-scoreboard-fixed-budgets-20260915`.
`report.md` contains every explored allocation and every finalist range; `summary.csv`
and `best.json` provide structured results. Phase plans and results, per-sample commands,
configs, environment, throughput, counters and resource use are retained. `provenance.json`
and `validation.json` record the inputs and completed metadata checks.

From that evidence directory, resume or summarize the campaign with:

```sh
python3 search.py screen
python3 search.py refine
python3 search.py confirm
python3 analyze.py
python3 validate.py
```

Completed samples are reused. Refinement may append further nearby candidates; new
independent timing samples need a fresh evidence directory. The frozen runtime and
original workload paths are recorded in the harness. See
[CPU dispatch, XCD and async-helper scaling](async-thread-scaling.md) for the earlier
unconstrained sweep and rebase validation.
