# CPU dispatch, XCD and async-helper scaling

Measured 2026-09-15 with implementation `efc055a3e79`, rebased onto `origin/develop` at
`cc17ebc55df`. The experiment uses **64 reserved physical CPU cores** (32-95) on a Threadripper
PRO 9995WX, with SMT siblings unused.

## Results

Dispatch width 16 and 32 have similar timings with four helpers on these GEMMs. Larger helper
budgets improve the width-32 configuration further. The highest thread count does not
consistently give the shortest time.

- gfx950 MFMA, eight engine threads and dispatch width 32: adding 16 helpers reduces median
  dispatch from 0.8331 to 0.5248 s (37.0%), and process wall from 3.210 to 2.930 s (8.7%).
- gfx1250 WMMA, eight engine threads and dispatch width 32: adding 24 helpers reduces median
  dispatch from 0.9091 to 0.7572 s (16.7%), and process wall from 3.580 to 3.400 s (5.0%).

These are medians of three runs per configuration in each campaign. Differences between 16 and
24 helpers are small; consult the full ranges before selecting a budget.

## Method

One simulated GPU, functional execution, shipped eight-XCD configurations. Threadripper PRO
9995WX; 64 reserved physical CPU cores (32-95), SMT siblings unused. Each process runs through
`agent-reserved-run`, serialized under the workspace benchmark lock. Clang 23, RelWithDebInfo
(`-O2`), no LTO, existing TheRock SDKs. Three repetitions per configuration in each campaign, in
reproducibly shuffled order; eight pilot runs are excluded. The helper campaign has fresh
zero/four-helper controls interleaved through the same lock, so its comparisons do not depend on
timings from the preceding dispatch campaign.

Both cases multiply 1024 x 1024 x 1024 FP8 matrices, with 64 x 64 output tiles (256 workgroups).
gfx950 uses the original Triton matmul kernel with 16x16x128 MFMA and an exact full-output CPU
reference. gfx1250 calls the original Gluon GEMM test with K128 WMMA and its full reference
check. Every sample must pass numerics, retain the complete instruction-count signature, and
retire every offload.

Zero helpers selects ordinary mode 0. Nonzero helpers select async mode 4, seven outstanding
jobs per wave, shared helper budget H, 512-pause warm waits, and lookahead 8. Target defaults
apply: cached admission for gfx950 MFMA; unrestricted large WMMA on gfx1250. All timed policies
use the same rebased binary.

Dispatch time is throughput-plugin wall time for the GEMM. Whole-process wall time also includes
startup, compilation/cache loading, transfers, CPU reference and validation. CPU seconds sum
process-tree user and system time from `/usr/bin/time`. Values are medians; all ranges are
retained in the local report and CSV. These timings measure functional GPU simulation on the
host. Small differences with overlapping ranges should not be treated as reliable improvements.

## Thread accounting

`E = num_threads` is the engine-thread count, capped at eight XCDs for these configurations. `D
= cpu_dispatch_threads` includes the submitting engine thread, so it adds D-1 retained pool
workers. `H = RJ_MMA_SHARED_HELPERS` adds H process-wide helpers. Execution-thread slots total E
+ D - 1 + H, excluding launcher, doorbell and other runtime threads. At D=1, up to E CUs execute
concurrently; at D>1, same-SoC batches serialize and at most D CUs execute concurrently, plus at
most H offloaded MMAs. These are capacity bounds, not measured utilization.

The proposed E=8,D=32 settings allocate 39 execution slots without helpers, 43 with H=4, 55 with
H=16, and 63 with H=24. D=32 reaches the shipped gfx1250 per-CP CU cap; gfx950 permits D=36,
although these GEMMs have only 32 workgroups per XCD.

## gfx950 / CDNA4 MFMA

Median dispatch seconds. Each entry compares the same checked workload.

| Dispatch width D | E=1, H=0 | E=1, H=4 | E=8, H=0 | E=8, H=4 |
|---:|---:|---:|---:|---:|
| 1 | 4.2093 | 1.7981 | 1.2482 | 0.7447 |
| 2 | 2.5513 | 1.2293 | 2.6710 | 1.2069 |
| 4 | 1.7408 | 0.9287 | 1.6901 | 0.9007 |
| 8 | 1.0842 | 0.6773 | 1.0914 | 0.6548 |
| 16 | 0.8165 | 0.6596 | 0.8001 | 0.6420 |
| 32 | 0.8292 | 0.6461 | 0.8200 | 0.6441 |

Helper scaling with eight engine threads. Entries are **dispatch / process wall seconds**.

| Shared helpers H | D=8 | D=32 |
|---:|---:|---:|
| 0 | 1.0865 / 3.460 | 0.8331 / 3.210 |
| 1 | 0.8906 / 3.290 | 0.7319 / 3.100 |
| 2 | 0.7884 / 3.180 | 0.7035 / 3.100 |
| 4 | 0.6660 / 3.050 | 0.6333 / 3.020 |
| 8 | 0.5772 / 2.970 | 0.5803 / 2.970 |
| 16 | 0.6007 / 2.990 | 0.5248 / 2.930 |
| 24 | 0.6281 / 3.030 | 0.5429 / 2.930 |

Fastest measured dispatch median: E=8, D=32, H=16: 0.5248 s. Fastest process-wall median: E=8,
D=32, H=16: 2.930 s.

## gfx1250 / CDNA5 WMMA

Median dispatch seconds. Each entry compares the same checked workload.

| Dispatch width D | E=1, H=0 | E=1, H=4 | E=8, H=0 | E=8, H=4 |
|---:|---:|---:|---:|---:|
| 1 | 6.1319 | 4.7495 | 1.9875 | 1.3953 |
| 2 | 3.3193 | 2.4621 | 3.3871 | 2.4620 |
| 4 | 2.1789 | 1.4540 | 2.1238 | 1.4489 |
| 8 | 1.3774 | 1.0680 | 1.3840 | 1.0464 |
| 16 | 0.9061 | 0.8167 | 0.9031 | 0.8111 |
| 32 | 0.9137 | 0.8154 | 0.9045 | 0.8109 |

Helper scaling with eight engine threads. Entries are **dispatch / process wall seconds**.

| Shared helpers H | D=8 | D=32 |
|---:|---:|---:|
| 0 | 1.3788 / 4.050 | 0.9091 / 3.580 |
| 1 | 1.2351 / 3.880 | 0.8575 / 3.520 |
| 2 | 1.1330 / 3.780 | 0.8333 / 3.490 |
| 4 | 1.0439 / 3.690 | 0.8167 / 3.470 |
| 8 | 1.0165 / 3.690 | 0.7861 / 3.450 |
| 16 | 0.9965 / 3.650 | 0.7606 / 3.420 |
| 24 | 1.0435 / 3.700 | 0.7572 / 3.400 |

Fastest measured dispatch median: E=8, D=32, H=24: 0.7572 s. Fastest process-wall median: E=8,
D=32, H=24: 3.400 s.

## Reproduction and validation

All **228 timed processes** (68 distinct settings, three repetitions in each campaign) passed
numerical, full instruction-signature and offload-retirement checks. The helper campaign
includes fresh zero/four-helper controls interleaved with the other budgets. The eight initial
pilot processes are excluded from the tables.

The complete build and 158 focused async/dispatch/lifetime tests pass. ISA/DBT regeneration
reproduces the checked-in sources exactly. The standard CTest selection has 20 failures out of
4,489 enabled tests; all 20 reproduce at identical assertions in a separately built pristine
upstream checkout with the same compiler and SDK. They concern DBT layout and instrumentation
setup/execution. One further test is disabled.

Local evidence: `/home/jakub/rocjitsu/misc/async-scoreboard-scaling-20260915`. Its `report.md`
includes all ranges, CPU seconds and offload counts; `summary.csv` is the machine-readable
table. `validation.json`, `host.json`, `runtime/manifest.json`, `range-diff.txt`, and
`upstream-failure-comparison.json` record checks, source/binary hashes, rebase adaptations and
upstream failures. Per-sample commands, configs, environments, assembly, numerics and counters
live in `samples/`.

From the evidence directory, reproduce with:

```sh
python3 run.py main --rounds 3
python3 run.py helpers --rounds 3 &
python3 controls.py
wait
python3 analyze.py
python3 validate.py
```

Existing completed samples are checked and reused; independent measurements require a fresh
evidence directory. The runtime and original workload paths are recorded in the harness.
