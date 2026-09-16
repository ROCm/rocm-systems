# Desktop execution-thread scaling

Keep the one-XCD desktop tables through **1/32/0** (E/D/H), with entries
1/1/0, 1/2/0, 1/4/0, 1/8/0, 1/16/0 and 1/32/0. Small jobs stop scaling early,
but a blanket four- or eight-thread cap leaves substantial matrix parallelism
unused. No async helpers are enabled by these desktop tables.

Four interleaved rounds on simulated gfx1100 (W7900), gfx1151, and gfx1201
(R9700), using reserved physical CPU cores 32-95. Every configuration uses E=1
and H=0, so its inclusive dispatch width D is also the total execution-thread
allocation. These are simulator measurements, not execution on physical GPUs.

## Workloads and interpretation

- `matrix`: 512 one-wave workgroups, 128 iterations of four independent FP16
  WMMA chains. This grid stops gaining at about 16 threads.
- `valu_memory`: 512 one-wave workgroups, 256 iterations combining volatile
  global loads/stores and vector arithmetic. Gains beyond 8-16 are small.
- `small_matrix`: four one-wave workgroups, 64 matrix iterations, eight launches.
  Four workers capture nearly all available parallelism.
- `wide_matrix`: 2,048 workgroups and 32 iterations, keeping total WMMA work
  equal to `matrix` while exposing more concurrent workgroups.
- `wide_valu_memory`: 2,048 workgroups and 64 iterations, likewise keeping total
  loop iterations equal to `valu_memory`.

The wider grids distinguish a workload's parallelism limit from a target's
thread-scaling limit. From 16 to 32 threads, the within-round median matrix
speedup is **1.81x on gfx1100, 1.81x on gfx1151 and 1.58x on gfx1201**.
The wider vector/memory case gains about 7-13%. Therefore these data support
retaining 32 as the automatic desktop ceiling, while budgets of 4, 8 or 16 can
choose the corresponding smaller table entry. The dispatch pool limits each
submission to its runnable CU count, so a small grid does not execute useful
work on all 32 workers.

Times below are medians of four fresh processes per configuration. Dispatch
time sums the workload kernels' throughput-plugin wall times; process wall
includes simulator/HIP startup and memory setup. The two grid sizes must be
compared within their own tables. No numerical output checks were performed;
all 312 timed processes succeeded and instruction counts matched across widths.

## matrix

| Threads | gfx1100 dispatch / wall s | gfx1151 dispatch / wall s | gfx1201 dispatch / wall s |
|---:|---:|---:|---:|
| 1 | 13.0228 / 13.232 | 13.0382 / 13.232 | 3.7100 / 3.919 |
| 2 | 6.6709 / 6.873 | 6.7261 / 6.923 | 2.1141 / 2.292 |
| 4 | 3.5327 / 3.744 | 3.5620 / 3.769 | 1.0820 / 1.291 |
| 8 | 2.0382 / 2.267 | 2.0202 / 2.217 | 0.5082 / 0.715 |
| 16 | 1.0886 / 1.316 | 1.0725 / 1.265 | 0.2698 / 0.489 |
| 32 | 1.0892 / 1.315 | 1.0734 / 1.265 | 0.2905 / 0.514 |

## valu_memory

| Threads | gfx1100 dispatch / wall s | gfx1151 dispatch / wall s | gfx1201 dispatch / wall s |
|---:|---:|---:|---:|
| 1 | 1.3790 / 1.616 | 1.3835 / 1.566 | 1.3437 / 1.541 |
| 2 | 0.8056 / 1.015 | 0.7928 / 0.965 | 0.8206 / 1.015 |
| 4 | 0.5638 / 0.765 | 0.5516 / 0.765 | 0.6146 / 0.815 |
| 8 | 0.4540 / 0.665 | 0.4420 / 0.665 | 0.5093 / 0.715 |
| 16 | 0.3840 / 0.614 | 0.3940 / 0.615 | 0.4501 / 0.665 |
| 32 | 0.3922 / 0.615 | 0.4046 / 0.615 | 0.4514 / 0.665 |

## small_matrix

| Threads | gfx1100 dispatch / wall s | gfx1151 dispatch / wall s | gfx1201 dispatch / wall s |
|---:|---:|---:|---:|
| 1 | 0.5220 / 0.715 | 0.5209 / 0.715 | 0.1404 / 0.314 |
| 2 | 0.2647 / 0.464 | 0.2605 / 0.439 | 0.0711 / 0.264 |
| 4 | 0.1337 / 0.314 | 0.1334 / 0.314 | 0.0376 / 0.239 |
| 8 | 0.1334 / 0.340 | 0.1342 / 0.314 | 0.0368 / 0.214 |
| 16 | 0.1338 / 0.339 | 0.1340 / 0.339 | 0.0367 / 0.214 |
| 32 | 0.1340 / 0.339 | 0.1338 / 0.314 | 0.0365 / 0.214 |

## wide_matrix

| Threads | gfx1100 dispatch / wall s | gfx1151 dispatch / wall s | gfx1201 dispatch / wall s |
|---:|---:|---:|---:|
| 4 | 3.3983 / 3.695 | 3.3868 / 3.645 | 1.1256 / 1.391 |
| 8 | 1.9524 / 2.242 | 2.0073 / 2.267 | 0.6134 / 0.890 |
| 16 | 1.1510 / 1.416 | 1.1677 / 1.416 | 0.3577 / 0.615 |
| 32 | 0.6305 / 0.890 | 0.6455 / 0.915 | 0.2260 / 0.514 |

## wide_valu_memory

| Threads | gfx1100 dispatch / wall s | gfx1151 dispatch / wall s | gfx1201 dispatch / wall s |
|---:|---:|---:|---:|
| 4 | 0.5621 / 0.840 | 0.5591 / 0.815 | 0.5881 / 0.865 |
| 8 | 0.4544 / 0.715 | 0.4572 / 0.715 | 0.5070 / 0.765 |
| 16 | 0.3790 / 0.665 | 0.3924 / 0.665 | 0.4397 / 0.715 |
| 32 | 0.3563 / 0.615 | 0.3530 / 0.614 | 0.4001 / 0.665 |

## Matrix dispatch scaling from one thread

| Threads | gfx1100 | gfx1151 | gfx1201 |
|---:|---:|---:|---:|
| 1 | 1.00x | 1.00x | 1.00x |
| 2 | 1.95x | 1.94x | 1.75x |
| 4 | 3.69x | 3.66x | 3.43x |
| 8 | 6.39x | 6.45x | 7.30x |
| 16 | 11.96x | 12.16x | 13.75x |
| 32 | 11.96x | 12.15x | 12.77x |

## Evidence and reproduction

Evidence directory:
`/home/jakub/rocjitsu/misc/desktop-thread-scaling-20260916`.

- `desktop.hip`, `compile.log`, `desktop`: the launch-only HIP benchmark, built
  with the workspace TheRock SDK for all three targets.
- `run.py`, `wide.py`, `summarize.py`: launch plans and analysis; repeated runs
  reuse completed samples.
- `main.json`, `wide.json`, `summary.json`, `validation.json`: 216 initial and
  96 wider-grid measurements, medians/ranges, CPU time and validation.
- `pilot.json`: 27 untimed-for-report pilot processes; the matrix pilot used
  1,024 iterations and is excluded from the tables.
- `samples/`: exact commands/configs, selected environment, stdout/stderr,
  throughput logs and process timing for every process.
- `runtime/`, `manifest.json`: frozen binaries and SHA-256 hashes. The runtime
  was built from the concurrent-dispatch branch with the initial per-VM helper
  ownership changes. All E/D/H values were explicit; no heuristic/table
  selection or async offload was exercised by these measurements.

The production tables stop at the largest measured desktop width of 32.
An explicit `cpu_dispatch_threads` still overrides that table, subject to the
target's CU capacity. These synthetic workloads support the chosen ceiling;
they do not claim that 32 is optimal for every desktop workload.
