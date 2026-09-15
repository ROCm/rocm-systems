# Concurrent XCD submissions to the shared CPU dispatch pool

The prototype removes whole-submission serialization while retaining one worker
set per SoC. The measurements below compare it with the serialized pool after the
latest cache rebase, using the same launch-only GEMMs and reserved CPU cores.

In the original 4-32-thread campaign, the selected new 32-thread allocation cuts
dispatch time by
**21.1% on gfx950** and **19.8% on gfx1250** versus the previous heuristic on the
serialized runtime. Process wall falls **3.9%** and **5.1%**, respectively. Times
are medians of four fresh adjacent pairs with equal total thread budgets;
percentages compare those medians.
At four threads the change is small; the larger budgets benefit more from
concurrent XCD engines. See the fixed-budget comparison below for every budget.

The E=1 controls show 0.3-2.9% higher dispatch medians. At E=8 and D=8, dispatch falls
43.5%/35.7% without helpers and 22.7%/29.1% with eight helpers on gfx950/gfx1250.
Those fixed-D/H controls isolate cross-XCD concurrency; their total thread budgets
rise with E and should not be confused with the equal-budget comparison.

## Scaling from one to 64 execution threads

This fresh pass measures the concurrent runtime at every budget, including the
one-thread baseline E/D/H = 1/1/0. The 4-32-thread allocations are the previous
selections; the 48/64-thread allocations are the winners of the extended search
described below. Each row is the median of four fresh processes, with all sixteen
target/budget combinations shuffled within each round. Screening and confirmation
samples are excluded from these medians.

E/D/H denotes XCD engines / inclusive dispatch width / shared async helpers, with
**B = E + D - 1 + H**. Each speedup is that target's one-thread median divided by
the row's median. Both times and speedups are listed as **dispatch / process wall**.

| Budget | gfx950 E/D/H | Dispatch / wall s | Speedup vs 1 thread (dispatch / wall) | gfx1250 E/D/H | Dispatch / wall s | Speedup vs 1 thread (dispatch / wall) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1/0 | 4.2137 / 6.165 | 1.00x / 1.00x | 1/1/0 | 6.0565 / 8.240 | 1.00x / 1.00x |
| 4 | 2/3/0 | 1.4863 / 3.445 | 2.83x / 1.79x | 1/4/0 | 2.1705 / 4.380 | 2.79x / 1.88x |
| 8 | 2/7/0 | 0.8411 / 2.800 | 5.01x / 2.20x | 2/5/2 | 1.3092 / 3.510 | 4.63x / 2.35x |
| 16 | 4/5/8 | 0.5141 / 2.475 | 8.20x / 2.49x | 4/11/2 | 0.8476 / 3.045 | 7.15x / 2.71x |
| 24 | 8/9/8 | 0.4497 / 2.425 | 9.37x / 2.54x | 8/15/2 | 0.6781 / 2.890 | 8.93x / 2.85x |
| 32 | 8/13/12 | 0.4099 / 2.380 | 10.28x / 2.59x | 8/21/4 | 0.6061 / 2.820 | 9.99x / 2.92x |
| 48 | 8/16/25 | 0.3940 / 2.360 | 10.69x / 2.61x | 8/32/9 | 0.5411 / 2.750 | 11.19x / 3.00x |
| 64 | 8/34/23 | 0.3774 / 2.355 | 11.17x / 2.62x | 8/32/25 | 0.5484 / 2.755 | 11.04x / 2.99x |

The gain from 48 to 64 threads is small. gfx950 dispatch falls 4.2%, while process
wall falls only 0.2%. gfx1250 dispatch rises 1.4% and wall rises 0.2%; its dispatch
ranges overlap (48: 0.5311-0.5485 s, 64: 0.5175-0.5526 s), so these measurements
do not establish an additional benefit at 64. For these workloads, 48 threads is
a reasonable stopping point when total wall time matters. Full ranges for this
fresh pass are in `scaling-summary.csv` in the extension evidence directory.

## Why concurrent submissions are feasible

The old pool held `run_mutex_` across an entire batch and its join. Its task spans,
results, task index, exception and completion state belonged to the pool, so a
second XCD could not submit until the first finished.

Command processors already own separate active-CU and result arrays. They retain
the queue-structure lock while workers execute, and workers append completion
records rather than performing the CP's stateful retirement actions. Each CP
applies those actions after its own batch joins. XCDs own disjoint CUs; the D=1
path already allows their engine threads to execute concurrently.

The prototype therefore puts execution and completion state in a stack-owned
submission. A short mutex protects an intrusive queue of worker assignments and
worker ownership. Workers acquire an assignment, drop the mutex, and claim CUs
with that submission's atomic task index. Every caller also drains its own work.
When its task index is exhausted, the caller cancels unused assignments and waits
only for workers assigned to its submission. Results and exceptions stay local.
The last worker notifies under the mutex before the caller can destroy the
submission. Publication and cancellation allocate no queue nodes.

Workers rotate pending submissions when taking assignments. An assigned worker
still drains that submission's remaining CUs; the pool is not a preemptive
scheduler. Busy workers may delay a new submission's access to workers from the
CU pool, but its own caller can always make progress. Simulation-engine barriers,
queue ordering and the process-wide async MMA helper pool are unchanged.

## Execution-thread budget

**B = E + (D - 1) + H**, where E is `num_threads`, D is inclusive
`cpu_dispatch_threads`, and H is `RJ_MMA_SHARED_HELPERS`.

The pool retains D-1 workers across the entire SoC, and each submission uses its
caller plus at most D-1 workers. Before this change, D>1 limited concurrent CU
execution to D threads because other engines waited on the pool-wide lock. The
prototype can use up to E+D-1 CU threads, plus H async helpers. These are capacity
bounds: active CUs, engine synchronization, admission and workload size determine
how many threads are busy. Runtime launcher and doorbell threads are outside B.

## Measurement method

Both variants use functional simulation on the same 64 reserved physical cores
(32-95), with SMT siblings unused, on a Threadripper PRO 9995WX. A shared workspace
lock serializes builds, tests and measured processes. B constrains the execution
allocation; CPU affinity stays at 64 cores for every row.

The frozen serialized runtime is `711f78cc05b`, matching the execution code at
prototype parent `07272a1d9a2`; the frozen concurrent runtime is the implementation
committed as `af9d3f4f349`. The review follow-up changes tests, comments and a private
condition-variable name, preserving the measured execution algorithm. Both builds
use Clang 23, RelWithDebInfo (-O2), no LTO and the existing local TheRock SDKs.

Both workloads launch an FP8 E4M3 GEMM of size 1024 cubed, with 64x64 output tiles
and 256 workgroups. gfx950 uses the original Triton 16x16x128 MFMA kernel; gfx1250
uses the original Gluon K128 WMMA kernel. H=0 selects mode 0; H>0 selects mode 4,
seven outstanding jobs per wave, 512-pause waits, lookahead 8 and target-default
admission. The same compiled GPU kernels and cache directories are used throughout.

Numerical references and output comparisons are disabled as requested. Every
process must finish successfully, preserve the full instruction-count signature,
and retire every submitted async operation. Dispatch time is GEMM wall time from
the throughput plugin. Process wall includes framework startup, transfers,
synchronization, assembly dumping and shutdown.

Screening covers every integer E/D/H split at budgets 4 and 8. At 16, 24 and 32,
E is 1, 2, 4 or 8 and H uses a coarse grid including zero, small counts, multiples
of four and budget fractions. Prior heuristic allocations are also eligible as
finalists, using their repeated concurrent-control measurements. The two fastest
screened allocations per target and budget receive four fresh, balanced
serialized/concurrent pairs. Selection uses median dispatch time. Four fresh pairs also cover the prior heuristic and E=1/E=8
controls at D=8. Pilot and screening samples are excluded from confirmation medians.
These are the best allocations in this search, not a global optimum for other
workloads. Small differences with overlapping ranges should be treated as ties.

The extension to budgets 48 and 64 screens 150 target/configuration combinations,
then 31 nearby splits. The initial grid uses E in {1,2,4,8}, D in
{1,8,16,24,28,32,target cap}, and additional D values derived from H in
{8,16,24,32,40,48,56}, retaining valid exact-budget combinations. The dispatch cap
is 36 on gfx950 and 32 on gfx1250; requests above these limits would be clamped
and would not provide the claimed execution budget. Refinement tries E within one
and D within two of each screening leader. Three finalists per target/budget
receive four fresh balanced serialized/concurrent pairs. The 4-32-thread
confirmation rows below are from the original campaign; the 48/64-thread rows
are from this extension.

CPU cycles per task-clock millisecond dropped between the early samples and the
main campaign. Fourteen early screening measurements were therefore replaced
with fresh measurements before selecting finalists; their originals remain in
the raw evidence. This avoids favoring allocations that happened to run before
the shift. Finalist pairs use consecutive processes at the same settings and
alternate which implementation runs first. Control pairs use the same alternating
order, with screening processes interleaved under the shared lock.

## Retuning within the same thread budget

The previous heuristic on the serialized runtime is compared with the best
confirmed allocation on the concurrent runtime. These rows include both the
implementation change and retuning, using four fresh adjacent pairs with
alternating implementation order. The old and new triples differ while their
total budgets match.

| Target | Budget | Previous E/D/H | New E/D/H | Previous dispatch s | New dispatch s | Change | Previous wall s | New wall s | Change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gfx950 | 4 | 1/3/1 | 2/3/0 | 1.5131 | 1.4781 | -2.3% | 3.475 | 3.450 | -0.7% |
| gfx950 | 8 | 1/3/5 | 2/7/0 | 0.8941 | 0.8596 | -3.9% | 2.870 | 2.835 | -1.2% |
| gfx950 | 16 | 1/7/9 | 4/5/8 | 0.5911 | 0.5018 | -15.1% | 2.570 | 2.475 | -3.7% |
| gfx950 | 24 | 1/9/15 | 8/9/8 | 0.5092 | 0.4365 | -14.3% | 2.485 | 2.400 | -3.4% |
| gfx950 | 32 | 2/12/19 | 8/13/12 | 0.4943 | 0.3898 | -21.1% | 2.460 | 2.365 | -3.9% |
| gfx1250 | 4 | 1/4/0 | 1/4/0 | 2.1283 | 2.1217 | -0.3% | 4.305 | 4.315 | +0.2% |
| gfx1250 | 8 | 1/8/0 | 2/5/2 | 1.3947 | 1.2885 | -7.6% | 3.605 | 3.495 | -3.1% |
| gfx1250 | 16 | 1/16/0 | 4/11/2 | 0.9149 | 0.8171 | -10.7% | 3.120 | 3.025 | -3.0% |
| gfx1250 | 24 | 1/16/8 | 8/15/2 | 0.7787 | 0.6671 | -14.3% | 2.990 | 2.875 | -3.8% |
| gfx1250 | 32 | 2/16/15 | 8/21/4 | 0.7473 | 0.5991 | -19.8% | 2.960 | 2.810 | -5.1% |

## Best confirmed allocations at exact budgets

Each row uses four fresh samples per implementation at the **same E/D/H triple**.
The faster confirmed candidate is selected by median concurrent dispatch time.

| Target | Budget | E/D/H | Serialized dispatch s | Concurrent dispatch s | Change | Serialized wall s | Concurrent wall s | Change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gfx950 | 4 | 2/3/0 | 1.8151 | 1.4819 | -18.4% | 3.780 | 3.445 | -8.9% |
| gfx950 | 8 | 2/7/0 | 1.0574 | 0.8515 | -19.5% | 3.025 | 2.810 | -7.1% |
| gfx950 | 16 | 4/5/8 | 0.6795 | 0.5138 | -24.4% | 2.640 | 2.485 | -5.9% |
| gfx950 | 24 | 8/9/8 | 0.5661 | 0.4427 | -21.8% | 2.550 | 2.415 | -5.3% |
| gfx950 | 32 | 8/13/12 | 0.5311 | 0.3905 | -26.5% | 2.505 | 2.365 | -5.6% |
| gfx950 | 48 | 8/16/25 | 0.5582 | 0.3694 | -33.8% | 2.530 | 2.340 | -7.5% |
| gfx950 | 64 | 8/34/23 | 0.5473 | 0.3583 | -34.5% | 2.520 | 2.335 | -7.3% |
| gfx1250 | 4 | 1/4/0 | 2.1335 | 2.1213 | -0.6% | 4.345 | 4.340 | -0.1% |
| gfx1250 | 8 | 2/5/2 | 1.5441 | 1.3172 | -14.7% | 3.755 | 3.520 | -6.3% |
| gfx1250 | 16 | 4/11/2 | 1.0928 | 0.8359 | -23.5% | 3.300 | 3.050 | -7.6% |
| gfx1250 | 24 | 8/15/2 | 0.9658 | 0.6879 | -28.8% | 3.180 | 2.895 | -9.0% |
| gfx1250 | 32 | 8/21/4 | 0.8066 | 0.6061 | -24.9% | 3.010 | 2.825 | -6.1% |
| gfx1250 | 48 | 8/32/9 | 0.7821 | 0.5508 | -29.6% | 2.985 | 2.760 | -7.5% |
| gfx1250 | 64 | 8/32/25 | 0.7503 | 0.5337 | -28.9% | 2.960 | 2.735 | -7.6% |

## Isolating the effect of concurrent XCD callers

Four balanced pairs per row. E=1 controls have no cross-XCD submission contention.
Holding D and H fixed while raising E also raises B; these rows explain the mechanism,
while the preceding table constrains the total execution budget.

| Target | E/D/H | Budget | Serialized dispatch s | Concurrent dispatch s | Change | Serialized wall s | Concurrent wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| gfx950 | 1/8/0 | 8 | 1.0906 | 1.0962 | +0.5% | 3.055 | 3.060 |
| gfx950 | 8/8/0 | 15 | 1.0802 | 0.6106 | -43.5% | 3.055 | 2.580 |
| gfx950 | 1/8/8 | 16 | 0.6138 | 0.6316 | +2.9% | 2.585 | 2.610 |
| gfx950 | 8/8/8 | 23 | 0.5933 | 0.4588 | -22.7% | 2.570 | 2.440 |
| gfx1250 | 1/8/0 | 8 | 1.3865 | 1.3909 | +0.3% | 3.595 | 3.605 |
| gfx1250 | 8/8/0 | 15 | 1.3828 | 0.8894 | -35.7% | 3.600 | 3.100 |
| gfx1250 | 1/8/8 | 16 | 1.0085 | 1.0268 | +1.8% | 3.205 | 3.235 |
| gfx1250 | 8/8/8 | 23 | 1.0053 | 0.7131 | -29.1% | 3.220 | 2.920 |

## Confirmation ranges

All finalists are shown: two per target/budget at 4-32 threads and three at
48/64 threads. Ranges span four fresh processes; close overlapping ranges do not
establish a stable ordering. CPU seconds include the whole process.

| Target | Budget | E/D/H | Concurrent dispatch median [min,max] s | Concurrent wall median [min,max] s | CPU seconds |
|---|---:|---:|---:|---:|---:|
| gfx1250 | 4 | 1/4/0 | 2.1213 [2.1092,2.2150] | 4.340 [4.320,4.440] | 12.84 |
| gfx1250 | 4 | 2/3/0 | 2.2335 [2.1916,2.8021] | 4.430 [4.390,5.010] | 12.45 |
| gfx1250 | 8 | 2/5/2 | 1.3172 [1.3157,1.3286] | 3.520 [3.510,3.530] | 12.06 |
| gfx1250 | 8 | 2/6/1 | 1.5239 [1.2904,1.5397] | 3.735 [3.490,3.750] | 12.62 |
| gfx1250 | 16 | 4/9/4 | 0.8666 [0.8482,0.8850] | 3.075 [3.060,3.100] | 13.48 |
| gfx1250 | 16 | 4/11/2 | 0.8359 [0.8056,0.8447] | 3.050 [3.010,3.060] | 13.84 |
| gfx1250 | 24 | 4/15/6 | 0.7208 [0.6856,0.7445] | 2.905 [2.890,2.950] | 15.56 |
| gfx1250 | 24 | 8/15/2 | 0.6879 [0.6780,0.6916] | 2.895 [2.880,2.910] | 16.23 |
| gfx1250 | 32 | 8/17/8 | 0.6065 [0.6034,0.6133] | 2.815 [2.810,2.830] | 17.50 |
| gfx1250 | 32 | 8/21/4 | 0.6061 [0.6040,0.6143] | 2.825 [2.810,2.830] | 18.29 |
| gfx1250 | 48 | 8/24/17 | 0.5706 [0.5465,0.5767] | 2.780 [2.750,2.780] | 20.37 |
| gfx1250 | 48 | 8/26/15 | 0.5573 [0.5409,0.5610] | 2.760 [2.760,2.780] | 20.57 |
| gfx1250 | 48 | 8/32/9 | 0.5508 [0.5335,0.5532] | 2.760 [2.740,2.770] | 22.05 |
| gfx1250 | 64 | 8/30/27 | 0.5490 [0.5409,0.5542] | 2.755 [2.750,2.770] | 22.22 |
| gfx1250 | 64 | 8/31/26 | 0.5447 [0.5345,0.5504] | 2.755 [2.740,2.770] | 22.65 |
| gfx1250 | 64 | 8/32/25 | 0.5337 [0.5200,0.5493] | 2.735 [2.720,2.760] | 22.79 |
| gfx950 | 4 | 1/3/1 | 1.4931 [1.4224,1.5367] | 3.470 [3.400,3.520] | 8.92 |
| gfx950 | 4 | 2/3/0 | 1.4819 [1.4724,1.4925] | 3.445 [3.430,3.460] | 8.82 |
| gfx950 | 8 | 2/7/0 | 0.8515 [0.8249,0.8647] | 2.810 [2.800,2.840] | 8.62 |
| gfx950 | 8 | 3/5/1 | 0.8562 [0.8392,0.8604] | 2.820 [2.820,2.840] | 8.44 |
| gfx950 | 16 | 4/5/8 | 0.5138 [0.4975,0.5269] | 2.485 [2.470,2.510] | 8.79 |
| gfx950 | 16 | 4/9/4 | 0.5280 [0.5135,0.5331] | 2.490 [2.480,2.500] | 9.34 |
| gfx950 | 24 | 8/5/12 | 0.4567 [0.4561,0.4654] | 2.430 [2.410,2.440] | 9.84 |
| gfx950 | 24 | 8/9/8 | 0.4427 [0.4324,0.4502] | 2.415 [2.390,2.420] | 10.53 |
| gfx950 | 32 | 4/25/4 | 0.4217 [0.4187,0.4275] | 2.385 [2.380,2.390] | 12.45 |
| gfx950 | 32 | 8/13/12 | 0.3905 [0.3810,0.4004] | 2.365 [2.350,2.380] | 11.32 |
| gfx950 | 48 | 4/28/17 | 0.3786 [0.3663,0.3910] | 2.360 [2.350,2.380] | 13.66 |
| gfx950 | 48 | 8/16/25 | 0.3694 [0.3663,0.3888] | 2.340 [2.340,2.360] | 12.85 |
| gfx950 | 48 | 8/32/9 | 0.3834 [0.3624,0.3983] | 2.350 [2.330,2.380] | 15.29 |
| gfx950 | 64 | 8/34/23 | 0.3583 [0.3457,0.3705] | 2.335 [2.310,2.350] | 16.31 |
| gfx950 | 64 | 8/35/22 | 0.3728 [0.3690,0.3763] | 2.350 [2.330,2.360] | 16.76 |
| gfx950 | 64 | 8/36/21 | 0.3756 [0.3712,0.3767] | 2.355 [2.350,2.370] | 17.25 |

## Previous heuristic at the same settings

Four balanced pairs at every previously selected target-aware allocation.
This separates implementation gains from choosing a different triple.

| Target | Budget | E/D/H | Serialized dispatch s | Concurrent dispatch s | Serialized wall s | Concurrent wall s |
|---|---:|---:|---:|---:|---:|---:|
| gfx950 | 4 | 1/3/1 | 1.5250 | 1.4902 | 3.485 | 3.465 |
| gfx1250 | 4 | 1/4/0 | 2.1439 | 2.1533 | 4.365 | 4.355 |
| gfx950 | 8 | 1/3/5 | 0.8924 | 0.8907 | 2.860 | 2.860 |
| gfx1250 | 8 | 1/8/0 | 1.3865 | 1.3909 | 3.595 | 3.605 |
| gfx950 | 16 | 1/7/9 | 0.5821 | 0.6232 | 2.560 | 2.590 |
| gfx1250 | 16 | 1/16/0 | 0.9069 | 0.9044 | 3.115 | 3.115 |
| gfx950 | 24 | 1/9/15 | 0.5079 | 0.5129 | 2.475 | 2.490 |
| gfx1250 | 24 | 1/16/8 | 0.7914 | 0.7920 | 3.005 | 3.005 |
| gfx950 | 32 | 2/12/19 | 0.4897 | 0.4495 | 2.460 | 2.420 |
| gfx1250 | 32 | 2/16/15 | 0.7528 | 0.6665 | 2.950 | 2.885 |


## Validation and evidence

The two gated concurrency regressions fail against the serialized pool and pass
against the prototype. They verify simultaneous use of shared workers and
independent completion. The exception test keeps a failing submission open after
its first CU throws, then requires an unrelated submission to complete before the
failing caller joins. A deliberately broken pool with one shared exception slot
fails this test by both leaking and losing the exception. Additional tests enforce
widths 1 and 2 with idle workers available, repeatedly reuse submissions from eight
callers, and exercise XCD fanout completion and barrier ordering with a shared pool.

The current focused suite passes all 237 tests with async mode 4 and shared
helpers enabled. The initial 236-test focused suite also passes in a complete
ThreadSanitizer build without race reports. After the review changes, the updated
14-test pool/XCD slice passes ten ThreadSanitizer repetitions. The initial five
new concurrency/XCD tests also passed 100 ordinary repetitions. The full CTest
suite was not repeated for this prototype.

Local evidence directory:
`/home/jakub/rocjitsu/misc/async-dispatch-concurrent-benchmark-20260915`.

- `screening.csv`: all 286 retained target/configuration screens, including all
  integer splits at budgets 4 and 8; these are single samples.
- `confirmation.csv` and `best.csv`: both finalists and their confirmed medians,
  ranges, process wall time and CPU seconds, with matching serialized controls.
- `retuned.csv`: fresh adjacent comparisons of each selected new allocation
  against the previous heuristic at the same total budget.
- `controls-summary.csv`: repeated fixed-D/H controls and the prior heuristic.
- `*-plan.json`, per-phase JSON results, and `samples/`: every command, config,
  environment, perf counter record, timing log, instruction signature and output.
- `provenance.json`, both runtime manifests, `validation.json`, and build/test logs:
  source versions, binary hashes and validation evidence.
- `run.py`, `small-budgets.py`, `rescreen.py`, `controls.py`,
  `make-confirm-plan.py`, `final-retune.py`, and `summarize.py`: the runners and
  analysis. The runners cache completed samples; use a new phase/directory for
  fresh measurements. `python3 summarize.py` regenerates the summary tables.

The original campaign retains 286 screening samples after replacing 14 early
samples, 136 repeated control samples, 160 finalist confirmation samples, and
80 final retuning-comparison samples. The 16 pilot samples and 14 replaced screens
are retained as evidence and excluded from the reported comparison medians.

Extension evidence directory:
`/home/jakub/rocjitsu/misc/async-dispatch-concurrent-large-budgets-20260915`.
All 345 processes completed successfully: four warmups, 181 candidate screens,
96 paired confirmation samples and 64 fresh scaling samples. Every instruction
signature matched and every submitted async operation retired. Numerical checks
remain disabled. This extension changes documentation only and uses the same
previously reviewed runtime binaries, verified against their SHA-256 manifests.

- `scaling-table.md` and `scaling-summary.csv`: the full 1-64-thread table,
  per-target speedups, medians, ranges and CPU seconds.
- `screening.csv`, `confirmation.csv`, `finalists.json` and `winners.json`:
  the candidate search and repeated selection evidence.
- `*-plan.json`, per-phase JSON results and `samples/`: complete ordered plans,
  configurations, commands, environments, timings, perf counters and output.
- `provenance.json`, `validation.json` and `clock-evidence.csv`: runtime and
  input hashes, exact-budget and signature checks, and frequency-counter evidence.
- `run.py`, `analyze.py` and `validate.py`: runners and analysis. Run
  `python3 analyze.py summarize` and `python3 validate.py` in this directory
  to regenerate and audit the results.
