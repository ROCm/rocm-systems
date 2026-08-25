# 7. Experiment matrix for a single AMD GPU node

Everything here is runnable with the two PR branches built and no additional tooling. Commands use
only flags that exist on `users/vkale/kernel-replay-rocprofv3`: `--kernel-replay-beta-enabled`,
repeatable `--pmc`, `--kernel-include-regex`, `--kernel-exclude-regex`,
`--kernel-iteration-range`, `--log-level`, `--kokkos-trace`.

## 7.0 Protocol, applied to every experiment

Four observables, all available today:

| Observable | How to get it | What it tells you |
|---|---|---|
| Did replay happen? | counter output rows carry `replay_pass`; a replayed dispatch id appears once per pass | distinguishes A/B from D |
| Why was it declined? | `--log-level warning` — the one-shot strings are `HIP graph launches are not supported`, `only single-packet dispatch submissions are replayed`, `snapshot capture failed` | identifies which gate fired |
| How big was the snapshot? | `--log-level info` — `kernel-replay snapshot: captured {N} regions` and `restore: restored {N}/{M} regions` | the $F$ term of §1.3, in region count; byte count needs R11 |
| Was the replay input-identical? | put one **canary counter in every `--pmc` group** and compare its value across `replay_pass` for the same `dispatch_id` | the only available empirical test of §6.3 O1/O3 today |

The canary is the single most informative thing to run tomorrow, and it needs no code change:

```bash
rocprofv3 --kernel-replay-beta-enabled \
  --pmc "SQ_WAVES SQ_INSTS_VALU" \
  --pmc "SQ_WAVES TCC_HIT_sum"   \
  --pmc "SQ_WAVES TCC_MISS_sum"  \
  -d out -o run -- ./app
```

`SQ_WAVES` is a function of the launch geometry, so for a correctly replayed dispatch it must be
**identical in all three passes**. If it is not, the passes did not execute the same work, and
nothing else in that row is trustworthy. Post-process with:

```bash
python3 - <<'PY'
import csv, collections
rows = list(csv.DictReader(open('out/run_counter_collection.csv')))
by = collections.defaultdict(dict)
for r in rows:
    if r['Counter_Name'] == 'SQ_WAVES':
        by[(r['Kernel_Name'], r['Dispatch_Id'])][r.get('Replay_Pass','0')] = r['Counter_Value']
bad = {k: v for k, v in by.items() if len(set(v.values())) > 1}
print(f"dispatches checked: {len(by)}   canary-inconsistent: {len(bad)}")
for k, v in list(bad.items())[:20]:
    print(k, v)
PY
```

Column names may need adjusting to the actual CSV header; the check itself is the point.

Baseline discipline for every run: (1) a clean run with no profiler, saving application output;
(2) a run with `--pmc` and no replay (application replay, one run per group) as the counter
reference; (3) the replay run. Compare (1) vs (3) for application correctness and (2) vs (3) for
counter agreement. Three repetitions of each; report medians and spreads, because §1.4 predicts
pass-dependent bias that will look like noise if you only run once.

## 7.0b Preflight: two checks that decide whether any result is meaningful

Run these before anything else. Both are one-liners and either can invalidate an entire day of
measurements.

**P1 — Which allocator does the application actually use?** For Kokkos codes this is decisive
(§4.1): on HIP < 7.0.0, `KOKKOS_ENABLE_IMPL_HIP_MALLOC_ASYNC` defaults **on**, every `View` is a
stream-ordered allocation, and replay is silently unsound. Check the Kokkos configuration dump that
every Kokkos application prints, or grep the build:

```bash
grep -r KOKKOS_ENABLE_IMPL_HIP_MALLOC_ASYNC "$KOKKOS_BUILD/KokkosCore_config.h"
hipconfig --version   # < 7.0.0 means the default is ON
```

If it is on, either rebuild with `-DKokkos_ENABLE_IMPL_HIP_MALLOC_ASYNC=OFF` or reclassify
experiments 2.1–2.3 from Tier 2 to Tier 3 negative controls. For PyTorch the equivalent is
`PYTORCH_HIP_ALLOC_CONF`; for JAX, `XLA_PYTHON_CLIENT_PREALLOCATE`; for RAPIDS, the RMM mode.

**P2 — What is $B$ on this machine?** Every cost figure in §1 and §4 scales inversely with it, the
in-tree floor is 4 GB/s, and the only public AMD measurements of the same operation on real workloads
are 1.3–1.7 GB/s (§11.2). Take the number from experiment 0.1 and use it, rather than the study's,
for every break-even estimate. If it is below 2 GB/s, the Tier 2 wall-times in this matrix are
underestimates by 2–3×.

## 7.1 Tier 0 — mechanism validation (run first, ~30 min)

These use the in-tree tests and synthetic cases and answer "does the mechanism behave as documented
on this machine" before any real application is involved.

| # | What | Command | Expected | Oracle |
|---|---|---|---|---|
| 0.1 | snap/restore round trip and bandwidth | `ctest -R kernel_replay --output-on-failure` in the SDK build | pass; `[kr-perf]` lines printed | `restore_gbps` ≥ 4; record the actual number — it calibrates $B$ for §1.3 on this host |
| 0.2 | measured $B$ vs pinned link rate | run 0.1 with `ROCPROFILER_KR_MIN_SNAP_GBPS=20` | **expected to fail** | confirms §1.3.1: the unpinned `std::vector` destination, not the link, is the limit. This is the evidence for R14 |
| 0.3 | concurrency isolation | the in-tree `tests/kernel-replay-concurrency/` client | pass | non-replayed concurrent dispatch output intact |
| 0.4 | per-pass service control | `tests/kernel-replay-local-context/` | pass | contexts toggle per pass, global state unchanged |
| 0.5 | canary on a trivial kernel | any HIP sample, canary command above | `SQ_WAVES` identical across passes | establishes that the canary methodology works before applying it to real apps |

## 7.2 Tier 1 — expected Outcome A (sound and cheap)

Small footprint, pure-function kernels, single stream, `hipMalloc` provenance.

| # | Workload | Command sketch | Target kernel | Expected | Oracle |
|---|---|---|---|---|---|
| 1.1 | rocBLAS SGEMM | `rocprofv3 --kernel-replay-beta-enabled --pmc "SQ_WAVES SQ_INSTS_VALU" --pmc "SQ_WAVES TCC_HIT_sum" --kernel-include-regex "Cijk.*" -d out -o gemm -- rocblas-bench -f gemm -r s -m 4096 -n 4096 -k 4096 -i 20` | Tensile `Cijk_*` | A | canary constant; `C` matrix bit-identical to a clean run (deterministic for a fixed algorithm); snap region count small |
| 1.2 | BabelStream | `--kernel-include-regex "(dot|triad|mul|add|copy).*"` on `hip-stream -s 268435456 -n 20` | all five | A | canary constant; reported bandwidth in the clean run unchanged |
| 1.3 | Reduction with FP atomics | any atomic-add reduction sample | the reduction kernel | A for correctness, **canary constant but other counters may vary** | demonstrates the O6 distinction: correct replay, nondeterministic measurement. Record the spread — it is the reference for what "acceptable variation" means on this hardware |
| 1.4 | rocRAND-consuming kernel | a Monte Carlo sample with generator state in `hipMalloc`'d memory | the path kernel | A, and *more* reproducible than the app | every pass draws identical random numbers because restore rewinds the generator state; verify by dumping the result buffer per pass |

## 7.3 Tier 2 — real applications, single GPU

Ordered by expected difficulty. For each, run the canary variant and compare application output
against a clean run.

| # | Workload | Command sketch | Expected | What it tests |
|---|---|---|---|---|
| 2.1 | LAMMPS, Lennard-Jones, KOKKOS on one GPU | `rocprofv3 --kernel-replay-beta-enabled --kokkos-trace --pmc "SQ_WAVES SQ_INSTS_VALU" --pmc "SQ_WAVES TCC_MISS_sum" --kernel-include-regex "<pair kernel regex>" --kernel-iteration-range 5-6 -d out -o lmp -- lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lj` | A **only if preflight P1 passes** | the canonical Kokkos case — *provided* Kokkos was built without `hipMallocAsync`. On a stock ROCm 6.x build this is a Tier 3 silent-corruption control, not a Tier 2 success case (§4.1) |
| 2.2 | LAMMPS EAM (metal) | same, `-in in.eam` | A | larger per-atom footprint; compare snap region count and wall time against 2.1 to measure the $F$ scaling directly |
| 2.3 | LULESH | `--kernel-include-regex "Calc.*"` on `lulesh -s 45 -i 100` | A | multi-kernel timestep; verify final origin energy against the clean run (LULESH prints it, which makes this the cleanest numerical oracle in the whole matrix) |
| 2.4 | PyTorch eager forward+backward (ResNet-50 or a small transformer) | `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False rocprofv3 --kernel-replay-beta-enabled --pmc ... --kernel-include-regex "Cijk.*" -- python train_step.py --steps 3` | A **only with the allocator pinned to the segment allocator** | tests §2.2 directly: run it again with `expandable_segments:True` and it becomes 3.5 below |
| 2.5 | An MD or docking code with a long setup and a short kernel of interest | e.g. a docking scoring kernel after system preparation | A, and the **best value case** | this is the regime where kernel replay beats application replay decisively (§1.3.2): setup cost is paid once instead of $P$ times. Measure both and report the ratio — it is the feature's headline number |
| 2.6 | GROMACS short `mdrun` | `--kernel-include-regex "nbnxm.*"` | A or C depending on build | if the build uses GPU-resident mode with direct GPU communication, expect isolation issues; run single-rank, no direct GPU comm first |
| 2.7 | A JAX/XLA or `torch.compile`-without-graphs workload | fused kernels only | A, with unusually good eligibility | XLA-style whole-graph fusion produces few, large, self-contained kernels — structurally the best fit for per-dispatch replay |

## 7.4 Tier 3 — negative controls (run these deliberately)

These are the experiments that produce the evidence for §9. Each is expected to fail in a *specific*
way; the value is confirming the failure mode is the predicted one.

| # | Control | How to trigger | Predicted outcome | Confirms |
|---|---|---|---|---|
| 3.1 | HIP graph launch | any app using `hipGraphLaunch` (or PyTorch `--compile` with graph capture / an inference server in graph mode) | **D** — warning `HIP graph launches are not supported`, zero counter rows for graph-resident kernels | the §5.1A argument: measure *what fraction of dispatches* are lost, using `--kernel-trace` to count graph vs non-graph dispatches |
| 3.2 | Multi-packet submission | a library that submits several packets in one write (multi-kernel BLAS solutions, MIOpen fused solutions) | **D** — warning `only single-packet dispatch submissions are replayed` | how much real work is excluded by the packet-count gate |
| 3.3 | Managed memory | rebuild a Tier-2 case with `hipMallocManaged`, or run with `HSA_XNACK=1` on a managed-memory app | **C** — replay proceeds, app output wrong | O1 violation, silently. This is the experiment that most needs running, because the outcome is invisible without an output comparison |
| 3.4 | Stream-ordered pool allocator | an app using `hipMallocAsync` | **C** | same, different allocator |
| 3.5 | PyTorch expandable segments | rerun 2.4 with `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` | **C** (VMM-mapped memory is untracked) or a large drop in snapshot region count | the §2.2 point that an environment variable flips soundness |
| 3.6 | Concurrent async copy | two threads: one replays a long kernel, one `hipMemcpyAsync`es into a tracked buffer | **C** — the copy is reverted by restore | the SDMA hole (R17); ~30 lines of test code |
| 3.7 | ABA address reuse | during a window, another thread frees and reallocates the same size | **C** — stale bytes written into the new allocation | R1; expected to reproduce because HIP's allocator reuses addresses |
| 3.8 | Two-GPU collective | any 2-GPU RCCL allreduce (`rccl-tests`, or a 2-rank PyTorch DDP step) with replay enabled and the collective kernel not excluded | **E** — hang, then `ROCP_FATAL` after ~60 s | §3.2; run it once, deliberately, and record the log. It is the strongest argument for R2 and R4 |
| 3.9 | GPU-aware MPI, 2 ranks | a halo-exchange mini-app with GPU-aware MPI, replaying the pack kernel | **C** and possibly **E** | §3.3 |
| 3.10 | Large pre-reserved arena | any framework that pre-reserves most of HBM (arena/pool allocators, XLA's BFC-style allocator) | **B** — enormous snapshot, or `snapshot capture failed` and silently no data | §1.3.3 and R6; measure the wall time per dispatch and compare to the §1.3 prediction |
| 3.11 | Peer-accessible buffer, 2 GPUs, one process | enable peer access, have agent B write agent A's buffer while A replays | **C** | the $U_{peer}$ hole in §3.1b |
| 3.12 | Reentrant submission from a tool callback | custom tool whose `PASS` callback launches a kernel on the replaying agent | **hang, no timeout** | §6.4 L2 and R3; needs an external watchdog to run safely |
| 3.13 | **Kokkos on ROCm 6.x defaults** | build Kokkos LULESH or LAMMPS-KOKKOS with stock ROCm 6.x (`KOKKOS_ENABLE_IMPL_HIP_MALLOC_ASYNC` on), replay a `Calc*` kernel | **C** — replay proceeds, snapshot region count near zero, LULESH's printed final origin energy differs from a clean run | the highest-priority experiment in the matrix. LULESH prints a numerical result, so this is a *self-checking* demonstration of §4.1 and the whole justification for R0 |
| 3.14 | Monte Carlo with the accumulator moved to a pool | FinanceBench Monte Carlo (or the Tier 1.4 sample) with the result buffer switched to `hipMallocAsync` | **C** — printed option price wrong by roughly a factor of $P$ | a one-line, five-minute, numerically obvious reproduction of the central blind spot. Best candidate to attach to the PR |
| 3.15 | AutoDock-GPU `MAPPED_COPY` build | build with `MAPPED_COPY` so `pMem_gpu_evals_of_runs` is managed, replay the generate-and-evaluate kernel | **C** — evaluation count multiplied by $P$, GA terminates early, docking score changes | a real production code where one `#define` separates a correct replay from a corrupted scientific result (§4.4) |
| 3.16 | JAX/XLA default preallocation | any JAX workload without `XLA_PYTHON_CLIENT_PREALLOCATE=false` | **B** — ~75% of HBM in one region; snapshot either fails or takes tens of seconds per dispatch | 3.10 with a specific, extremely common trigger; rerun with preallocation disabled to show the fix (§4.7) |
| 3.17 | CPX partition mode | put an MI300X in CPX, run two workloads on two partitions, replay on one | **C** and contaminated counters | §3.1c and R7; eight HSA agents share one HBM and cache hierarchy, so the per-agent lock admits eight writers |

## 7.5 What to measure across the matrix, and how to present it

One table, one row per experiment, with: tracked region count, snapshot wall time, restore wall time
per pass, passes, total added wall time, application-output comparison result, canary consistency,
and outcome class (A/B/C/D/E). Two derived numbers are the ones worth putting in front of anyone
deciding the feature's future:

1. **Measured $B$** from Tier 0, versus the pinned link rate on the same host. The gap is R14's
   business case.
2. **The eligibility fraction** per Tier-2/3 workload: replayed dispatches ÷ total dispatches
   (`--kernel-trace` gives the denominator). For a graph-heavy inference workload this number will be
   near zero, which is R15's business case; for a Kokkos HPC code it should be near one.

Those two numbers, from real applications, settle more of the roadmap argument than any further
analysis.
