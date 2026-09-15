# Async scoreboard after the instruction-fetch cache lands upstream

The branch was rebased from `origin/develop` `cc17ebc55df` to `7fa475e0226`. The previous
branch already contained the instruction-fetch mapping cache as prototype commit
`efa5d2768a3`. Upstream `f5b05f815b0` contains that cache plus earlier VMID epoch
invalidation and additional tests. The duplicate prototype was dropped; all other 21
patches are unchanged according to `range-diff.txt`. The runtime source difference
from the previous branch is the VMID invalidation-order fix in `gpu_memory.h`.

Both binaries contain instruction-fetch mapping caching. This comparison measures the
rebase and its VMID invalidation fix.

## Outcome

The equal-weight geometric mean of configuration medians improved by 0.24% for dispatch
and 0.23% for process wall. Individual dispatch medians range from 2.65% faster to 1.96%
slower. This rerun shows essentially unchanged overall performance; the fetch-mapping
cache was already included in the earlier measurements.

## Matched old/new results

Median seconds, four fresh samples per binary and configuration. Negative changes mean
less time. E/D/H is XCD engine count / inclusive CPU dispatch width / shared helper count.
Every allocation satisfies B = E + D - 1 + H. Dispatch is the throughput-plugin GEMM wall
time. Process wall also includes Python/framework startup, transfers, synchronization,
assembly dumping and process shutdown. Neither binary computes or checks numerical references.

| Target | Budget | E/D/H | Old dispatch s | New dispatch s | Change | Old process wall s | New process wall s | Change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gfx1250 | 4 | 1/4/0 | 2.1647 | 2.1072 | -2.7% | 4.380 | 4.310 | -1.6% |
| gfx950 | 4 | 1/3/1 | 1.5059 | 1.4954 | -0.7% | 3.475 | 3.465 | -0.3% |
| gfx1250 | 8 | 1/8/0 | 1.3773 | 1.3660 | -0.8% | 3.600 | 3.580 | -0.6% |
| gfx950 | 8 | 1/3/5 | 0.8970 | 0.8805 | -1.8% | 2.860 | 2.850 | -0.3% |
| gfx1250 | 16 | 1/16/0 | 0.8951 | 0.8973 | +0.3% | 3.100 | 3.115 | +0.5% |
| gfx950 | 16 | 1/7/9 | 0.5923 | 0.6021 | +1.7% | 2.565 | 2.570 | +0.2% |
| gfx1250 | 24 | 1/16/8 | 0.7863 | 0.7853 | -0.1% | 2.990 | 2.985 | -0.2% |
| gfx950 | 24 | 1/9/15 | 0.5157 | 0.5258 | +2.0% | 2.490 | 2.490 | +0.0% |
| gfx1250 | 32 | 2/16/15 | 0.7478 | 0.7546 | +0.9% | 2.960 | 2.965 | +0.2% |
| gfx950 | 32 | 2/12/19 | 0.5064 | 0.5018 | -0.9% | 2.470 | 2.465 | -0.2% |

Equal-weight geometric mean change across configuration medians: dispatch_seconds -0.24%, process_wall_seconds -0.23%.

## Pairwise variability

Each old/new pair uses the same settings and runs consecutively. Each configuration has
two old-first and two new-first pairs. Pairs are shuffled reproducibly across workloads
and budgets. These small samples characterize timing variability; close medians with
overlapping ranges do not establish a reliable performance change.

| Target | Budget | Dispatch paired change median [min,max] | Process-wall paired change median [min,max] | New faster pairs: dispatch/wall |
|---|---:|---:|---:|---:|
| gfx1250 | 4 | -2.5% [-9.6,+0.4] | -1.6% [-5.2,+0.2] | 3/3 of 4 |
| gfx950 | 4 | -1.4% [-5.6,+0.8] | -0.7% [-2.6,+0.3] | 3/3 of 4 |
| gfx1250 | 8 | +0.2% [-2.5,+0.6] | -0.6% [-0.6,+0.3] | 2/3 of 4 |
| gfx950 | 8 | -1.6% [-2.8,-0.8] | -0.5% [-1.4,+0.4] | 4/3 of 4 |
| gfx1250 | 16 | +0.4% [-0.3,+1.7] | +0.6% [+0.3,+0.6] | 2/0 of 4 |
| gfx950 | 16 | +1.7% [-5.5,+6.1] | +0.4% [-1.9,+1.2] | 2/1 of 4 |
| gfx1250 | 24 | +0.2% [-0.4,+0.6] | +0.0% [-0.7,+0.3] | 2/1 of 4 |
| gfx950 | 24 | +2.4% [-1.9,+5.2] | +0.2% [-0.8,+1.2] | 1/1 of 4 |
| gfx1250 | 32 | +0.7% [-2.2,+3.9] | -0.2% [-0.3,+1.0] | 2/2 of 4 |
| gfx950 | 32 | -1.0% [-1.5,-0.6] | +0.0% [-0.8,+0.4] | 4/1 of 4 |

## Method and artifacts

Ten unchanged configurations from the target-aware heuristic, for budgets 4, 8, 16, 24
and 32. Four balanced old/new pairs per configuration give 80 measured processes.
Four warmup processes, one per binary/workload at budget 32, are excluded. Both variants
use the same 64 reserved physical cores (32-95), with SMT siblings unused. A shared
benchmark lock serializes builds, tests and measured processes.

The old frozen runtime is implementation `efc055a3e79`; the newly built runtime is
`711f78cc05b`. Both use Clang 23, RelWithDebInfo (-O2), no LTO and the same existing
TheRock SDKs. Both launch the identical unchecked FP8 GEMMs: 1024 cubed, 64x64 output
blocks, 256 workgroups. gfx950 uses 16x16x128 MFMA; gfx1250 uses K128 WMMA. GPU code
comes from the same cached original Triton/Gluon kernels.

Async policy remains mode 4 for H>0 and mode 0 for H=0, seven outstanding jobs per wave,
512-pause warm waits, lookahead 8 and target-default admission. All samples must finish
successfully, preserve the full instruction signatures and retire every async submission.

The build succeeded. Focused cache/dispatch/async/plugin tests passed: 161 with the default
environment plus six async-only tests rerun with mode 4 and shared helpers enabled. The
full CTest suite was not repeated for this measurement task.

`summary.csv` includes medians, ranges, paired changes, CPU seconds and the historical
pre-rebase timings. Fresh old/new pairs are the primary comparison; historical samples
are retained for context. `plan.json`, `results.json`, `warm.json`, `samples/`, runtime
manifests, `provenance.json`, `validation.json`, build/test logs and rebase evidence
provide the complete inputs and outputs.

Local evidence: `/home/jakub/rocjitsu/misc/async-scoreboard-cache-rebase-20260915`.

See [fixed-budget scaling](async-fixed-budget-scaling.md) for the original search and
[the heuristic evaluation](/home/jakub/rocjitsu/misc/async-scoreboard-heuristic-20260915/report.md)
for the unchanged configuration selection.
