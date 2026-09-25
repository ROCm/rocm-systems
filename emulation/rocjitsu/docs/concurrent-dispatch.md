# Concurrent XCD dispatch and configured thread allocations

Functional simulation uses engine threads to advance whole XCDs and a shared
CPU worker pool per SoC to execute CU quanta. Previously the pool held a lock
across a complete submission and join, so XCD engine threads waited for one
another even when pool workers were idle.

## Submission ownership

Each `CpuDispatchPool::run()` now owns its task span, result span, atomic task
index, exception and completion condition. A short mutex protects an intrusive
queue of worker assignments. Workers claim an assignment under the mutex, then
drain its CUs without holding that mutex. Pending submissions rotate when a
worker takes an assignment. An assigned worker finishes its current submission;
this is not preemptive scheduling.

Each caller executes only its own submission and cancels unused assignments
when its task index is exhausted. It joins only workers holding that submission.
The last worker notifies before dropping the mutex, protecting the stack-owned
submission's lifetime. There is no per-submission queue allocation.

Independent callers must own disjoint CUs and result storage. Command processors
already maintain their own active-CU/result arrays, keep queue structure stable
while their batch executes, and retire completion records after their own join.
XCDs own disjoint CUs. Existing queue and engine-barrier ordering stays in place.

For E engine threads and inclusive dispatch width D on each GPU, the retained
execution-thread allocation is **E + sum(D - 1)**. Every caller can make progress;
workers are shared, with D-1 persistent workers per SoC. Runnable CUs and workload
distribution determine how much of that capacity is used. Launcher, doorbell and
other runtime service threads are outside this allocation.

## Configuration policy

`thread_allocations` contains preferred `num_threads`/`cpu_dispatch_threads`
pairs. `resolve_execution_threads()` is a pure function: apply explicit overrides
and topology clamps, calculate each entry's total retained thread count, and
choose the largest fitting entry. Later entries break ties. It does not fill
unused budget between entries. The default budget is CPU affinity capped at 32;
`cpu_thread_budget` overrides that default. A larger budget can select larger
entries from a custom table. Explicit knob settings take priority over a budget.

The synchronous granules measured below stop at 32. The eight-XCD tables include a 24-thread entry;
otherwise their granules are powers of two. At 32 with async helpers disabled, gfx950 and gfx1250 select
8 engine threads and inclusive dispatch width 25. The [async MMA extension](async-instructions.md)
adds helper allocations and uses E + sum(D - 1) + H for the total budget. Desktop presets select one
engine and dispatch width 32. See [the allocation table](configuration.md#thread-accounting-and-preferred-allocations).

The initial entries were screened in the previous study, then checked on this
standalone branch with the synchronous kernels. The final tables below report
medians of three randomized, interleaved rounds over 64 reserved physical cores
(CPUs 32-95). A shared lock excludes overlapping builds and tests. These are CPU
simulator measurements with a RelWithDebInfo Clang 23 build, not physical GPU
measurements. Numerical comparisons are disabled; all return codes and
per-target instruction signatures were checked.

The server workloads are 1024-cubed FP8 GEMMs with 256 workgroups: Triton MFMA
on gfx950 and Gluon WMMA on gfx1250. Desktop workloads use 2,048 workgroups with
32 iterations of independent FP16 WMMA. There are 144 timed processes across
48 configurations, plus five excluded warmups.

The data supports mixed engine/dispatch allocations over engine-only or
single-engine alternatives. Close cases are intentionally modest claims:
gfx950 at 32 threads is effectively tied between 8/25 and 4/29, and gfx1250 at
16 has overlapping ranges for 8/9 and 4/13. Keep eight engines in those cases
so each XCD has an engine. The smaller granules avoid oversubscribing narrow
budgets. Desktop 16-to-32 dispatch speedups are 1.69x, 1.74x and 1.42x for
gfx1100, gfx1151 and gfx1201 respectively. Small grids in the earlier desktop
study plateaued around four threads; target-only defaults cannot predict grid size.

MI210 and CDNA3 also enable dispatch workers, based on the integer smoke study
below. Multi-GPU presets default to E=1/D=1 for RCCL. Explicit D=0 opts into
granules that divide each budget's remaining slots among the per-GPU pools;
positive D overrides also remain available. Mirage uses the same defaults and
derives the multi-GPU granules from single-GPU tables embedded in the RocJITsu
backend. The hardware agent model contains no host tuning. The ceiling remains 32, with
larger custom entries and explicit knob overrides available.

### Synchronous server kernels

Seconds are dispatch / process wall; speedup is dispatch versus 1/1.

| Budget | gfx950 E/D | Seconds | Speedup | gfx1250 E/D | Seconds | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 5.3145 / 7.240 | 1.00x | 1/1 | 6.7014 / 8.900 | 1.00x |
| 2 | 1/2 | 3.1671 / 5.110 | 1.68x | 1/2 | 3.7704 / 5.980 | 1.78x |
| 4 | 2/3 | 1.8693 / 3.860 | 2.84x | 1/4 | 2.3584 / 4.570 | 2.84x |
| 8 | 2/7 | 1.0352 / 2.900 | 5.13x | 2/7 | 1.4089 / 3.620 | 4.76x |
| 16 | 8/9 | 0.6527 / 2.620 | 8.14x | 8/9 | 0.9119 / 3.110 | 7.35x |
| 24 | 8/17 | 0.5240 / 2.500 | 10.14x | 8/17 | 0.7012 / 2.900 | 9.56x |
| 32 | 8/25 | 0.5162 / 2.480 | 10.30x | 8/25 | 0.6345 / 2.860 | 10.56x |

### Desktop wide matrix grid

Each row is E=1, D=budget. Seconds are dispatch / process wall.

| Budget | gfx1100 seconds | Speedup | gfx1151 seconds | Speedup | gfx1201 seconds | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 34.2160 / 34.320 | 1.00x | 34.3292 / 34.420 | 1.00x | 4.2655 / 4.380 | 1.00x |
| 2 | 17.5324 / 17.630 | 1.95x | 17.3905 / 17.480 | 1.97x | 2.2888 / 2.400 | 1.86x |
| 4 | 8.9779 / 9.080 | 3.81x | 9.1107 / 9.200 | 3.77x | 1.2971 / 1.400 | 3.29x |
| 8 | 4.8693 / 4.980 | 7.03x | 4.8432 / 4.940 | 7.09x | 0.7278 / 0.830 | 5.86x |
| 16 | 2.6846 / 2.790 | 12.75x | 2.7469 / 2.840 | 12.50x | 0.4504 / 0.550 | 9.47x |
| 32 | 1.5921 / 1.700 | 21.49x | 1.5758 / 1.670 | 21.79x | 0.3180 / 0.420 | 13.42x |

### Same-budget alternatives

Positive delta means the selected entry was slower; overlapping ranges do not establish an ordering.

| Target | Budget | Selected E/D | Selected dispatch [min,max] s | Best alternative E/D | Alternative dispatch [min,max] s | Selected delta |
|---|---:|---:|---:|---:|---:|---:|
| gfx950 | 2 | 1/2 | 3.1671 [3.0882,3.3681] | 2/1 | 3.6866 [3.6498,3.7005] | -14.1% |
| gfx950 | 4 | 2/3 | 1.8693 [1.8450,1.8818] | 1/4 | 2.1794 [2.0911,2.1925] | -14.2% |
| gfx950 | 8 | 2/7 | 1.0352 [1.0326,1.0445] | 4/5 | 1.0884 [1.0679,1.1396] | -4.9% |
| gfx950 | 16 | 8/9 | 0.6527 [0.6448,0.6688] | 4/13 | 0.6837 [0.6818,0.6929] | -4.5% |
| gfx950 | 24 | 8/17 | 0.5240 [0.5230,0.5407] | 4/21 | 0.5858 [0.5845,0.5948] | -10.5% |
| gfx950 | 32 | 8/25 | 0.5162 [0.4882,0.5343] | 4/29 | 0.5126 [0.5120,0.5165] | +0.7% |
| gfx1250 | 2 | 1/2 | 3.7704 [3.7001,3.8178] | 2/1 | 5.0649 [4.9432,5.0779] | -25.6% |
| gfx1250 | 4 | 1/4 | 2.3584 [2.3220,2.3593] | 2/3 | 2.4836 [2.4745,2.5074] | -5.0% |
| gfx1250 | 8 | 2/7 | 1.4089 [1.3906,1.4187] | 4/5 | 1.4631 [1.4481,1.4645] | -3.7% |
| gfx1250 | 16 | 8/9 | 0.9119 [0.9033,0.9231] | 4/13 | 0.9415 [0.9084,0.9559] | -3.1% |
| gfx1250 | 24 | 8/17 | 0.7012 [0.6938,0.7032] | 4/21 | 0.7470 [0.7404,0.7827] | -6.1% |
| gfx1250 | 32 | 8/25 | 0.6345 [0.6245,0.6655] | 4/29 | 0.6820 [0.6780,0.7092] | -7.0% |

## Comparison with the alternative implementation

[ROCm/rocm-systems#11669](https://github.com/ROCm/rocm-systems/pull/11669), inspected
at `6a39ebb6ecf5a863a78f83e440966fe3cc56b3c6`, independently uses submission-local
state and a shared queue. Its sparse-submission benchmark is included here,
along with an adapted regression for its useful caller-isolation guarantee.

The scheduling contracts differ. That implementation allows only one caller to
execute CU work at a time, bounding active execution by D. This implementation
lets every engine caller execute, with the total E + sum(D - 1) controlled by the
configuration selector. It also queues one intrusive submission with assignment
counts instead of allocating a vector of lane nodes for every call. Neither
implementation lets a caller steal another submission's work: foreign work can
be waiting for the caller's own submission to finish.

## Validation and reproduction

Focused tests cover overlapping submissions, independent completion, exception
ownership, per-submission width limits, repeated reuse, caller isolation and XCD
queue/barrier ordering. Policy tests cover affinity, overrides, multiple GPUs,
topology clamps, discrete granules, preset tables and checkpoint persistence.
Mirage tests exercise generated configurations and the real CLI override path.

The sparse benchmark is excluded from default CTest runs. Run it explicitly:

```sh
build/tests/rocjitsu_tests --gtest_filter=CpuDispatchPoolBenchmark.SparseConcurrentSubmissions
```

Inspect a preset's effective allocations without constructing the simulated GPU:

```sh
build/tools/rocjitsu/rocjitsu --config configs/gfx950_mi355x.json --thread-budget-table
```

The [configuration reference](configuration.md#thread-accounting-and-preferred-allocations)
describes the override rules. The tables above record the measured allocations,
medians and observed ranges used to choose the shipped presets.

## Integer compute and multi-GPU smoke checks

An additional smoke study runs 1,024 workgroups of 64 lanes on every simulated
GPU. Each lane performs 512 integer mixing iterations; every output is checked
against a host reference. The timed region includes launching and synchronizing
all GPUs, after a small warmup. Three randomized rounds cover 25 configurations
(75 successful processes), followed by four runs using automatic granule
selection. Multi-GPU presets now require D=0 to opt into that selection.
These use the same Clang 23 runtime and 64 reserved physical
cores as the earlier measurements. No async MMA is enabled.

Each entry is E/D followed by dispatch speedup over E=1/D=1 on that same
topology. Speedups are medians of ratios paired by round. The budget is a
ceiling; multi-GPU pools retain E + GPUs*(D-1) threads.

| Budget | MI210 | CDNA3 | MI355X, 2 GPUs | MI455X, 4 GPUs |
|---:|---:|---:|---:|---:|
| 1 | 1/1: 1.00x | 1/1: 1.00x | 1/1: 1.00x | 1/1: 1.00x |
| 2 | 1/2: 1.88x | 1/2: 1.92x | 1/1: 1.00x | 1/1: 1.00x |
| 4 | 1/4: 3.50x | 2/3: 3.18x | 1/2: 1.73x | 1/1: 1.00x |
| 8 | 1/8: 6.50x | 2/7: 5.78x | 1/4: 2.87x | 1/2: 1.77x |
| 16 | 1/16: 11.01x | 8/9: 10.10x | 1/8: 4.22x | 1/4: 3.17x |
| 24 | 1/16: 11.01x | 8/17: 13.54x | 1/12: 6.83x | 1/6: 4.12x |
| 32 | 1/32: 17.85x | 8/25: 15.99x | 1/16: 6.86x | 1/8: 5.98x |

At the 32-thread ceiling, median dispatch / process wall times are
0.056 / 0.314 s for MI210, 0.063 / 0.364 s for CDNA3,
0.267 / 0.715 s for two MI355X GPUs, and 1.080 / 2.417 s for four MI455X GPUs.
The multi-GPU cases retain 31 and 29 execution threads respectively; with one
engine, each batch uses its owning GPU pool. The data supports dispatch
parallelism, without implying all GPU pools execute simultaneously.

RCCL AllReduce, Broadcast, AllGather and ReduceScatter pass with both serial
dispatch and the parallel widths above on two and four GPUs. Two-rank SendRecv
also passes at both widths. These are 18 collective cases and 52 rank
processes, with three randomized input sizes per case. Each case uses a fresh
daemon, matching the existing test fixture. The SendRecv test is specific to
two ranks and is not used with four.

**Small collectives do not share the compute speedups.** In these smoke runs,
two-rank AllReduce takes 4.53 s with D=1 and 12.44 s with D=16; four-rank
AllReduce takes 18.31 s with D=1 and 33.67 s with D=8. These are individual
correctness-test timings, including communicator setup and three 64-1024
element operations, not steady-state collective throughput. Multi-GPU defaults
therefore retain E=1/D=1 for RCCL; parallel granules are an explicit opt-in for
compute-heavy runs. The smoke study establishes correctness and useful
compute scaling, not a universal speedup or optimal granules for all kernels.

A separate serial-control run that reused a daemon across successive client
groups hung on Broadcast after AllReduce completed. Fresh-daemon runs pass;
the reuse issue remains separate from this granule qualification.
