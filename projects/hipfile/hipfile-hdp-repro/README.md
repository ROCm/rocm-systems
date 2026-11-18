# hipFile visibility test bundle

`run-all.sh` runs every test and probe from the HDP / L2 visibility investigation on one node and
writes a one-line-per-run `summary.txt`, per-section logs, and before/after health snapshots.

## What you need on the node

- ROCm with `amdclang++`, CMake, `python3`, `gdb` (the ctest entries run under gdb).
- A checkout of the branch `users/john00003/hipfile-read-gpu-visibility-test` (Kurt's PR
  `users/kumcmill/test-hipfile-read-visible-on-cpu` plus John's two commits). The script builds
  hipFile in `<checkout>/projects/hipfile/build-visibility` unless you pass `--build-dir`.
- A directory on an AIS-capable mount (ext4 with `data=ordered`, or xfs, on a local NVMe that is
  not the boot drive) for test files. Without fastpath, the script runs fallback only and says so.
- Optional: passwordless `sudo` to read the kernel log in the health snapshots (read-only).

## Run it

```
scp -r ~/scratch/hipfile-hdp-repro/node-bundle NODE:
ssh NODE
git clone --filter=blob:none --sparse https://github.com/ROCm/rocm-systems.git && cd rocm-systems && git sparse-checkout set projects/hipfile && git checkout users/john00003/hipfile-read-gpu-visibility-test && cd ..
~/node-bundle/run-all.sh --hipfile-src ~/rocm-systems/projects/hipfile --dir /mnt/<ais-mount>/<you> --build
```

If the branch isn't on GitHub yet, copy the worktree's hipFile project instead and point
`--hipfile-src` at it:

```
rsync -a --exclude 'build*' ~/code/rocm-systems-hipfile-gpu-visibility/projects/hipfile/ NODE:hipfile-src/
~/node-bundle/run-all.sh --hipfile-src ~/hipfile-src --dir /mnt/<ais-mount>/<you> --build
```

Useful options: `--quick` (smoke run, about 1/10 the iterations), `--sections gpu,ctest` (subset),
`--gpu N` (one GPU of several; sets `ROCR_VISIBLE_DEVICES`), `--allow-hdp-flush` (see below),
`--dry-run`. If `amdclang++` cannot find libstdc++ (GCC install without matching `-dev`
package), set `EXTRA_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/<ver>`.

Results go to `./results-<host>-<timestamp>/`: `summary.txt` (read this first), `logs/<section>.log`,
`logs/health-before.log`, `logs/health-after.log`, `run.log` (everything), `bin/` (probe builds).

## Sections

| Section | What runs | Touches |
|---|---|---|
| `health` | `health.sh` before and after: AER counters for each AMD GPU and its bridges, D-state processes, KFD processes, kernel-log trouble markers | read-only |
| `ctest` | `ctest -R 'hipfileread_(cpu\|gpu)_visibility\|hipfilewrite_visibility'`: Kurt's 2 entries and John's 12 | CPU BAR reads (Kurt's) |
| `kurt` | `hipfileread-cpu-visibility`, fastpath and fallback × `--cpus` 2/8/24/all × `--buffer-size` 64K/1M/4M × 3 reps, 2000 iterations | CPU BAR reads |
| `gpu` | `hipfileread-gpu-visibility`, fastpath and fallback × `--kernel after` (no prime; `--prime --prime-wait sync/flag/sleep`; each also with `--read async`) and `--kernel during` (±`--prime`), 1 MiB × 2 reps; plus 256 MiB for three configs | GPU reads only |
| `write` | `hipfilewrite-visibility`, fastpath and fallback × producers `kernel` (each `--wait`; `--kernel during` in fine, uncached and coarse memory), `memcpy` (4K to 1M), `memset`, `copy`, `read`, with `--prime write`/`store`, `--background-writers` and `--threads`; then the AQL packet headers HIP submits for eight configs | GPU and drive reads only |
| `aqllog` | the `--prime-wait flag` case with `--read sync` and `async` under `AMD_LOG_LEVEL=5 AMD_LOG_MASK=8`, printing the AQL packet headers HIP submits | GPU reads only |
| `hipbar` | HIP-only Failure-1 repro (`hip-bar-read-stale`, no hipFile): `READERS` 1/2/7/20 × `MEM` coarse/fine/uncached × `KICK` off/on (unrelated memset + sync) | CPU BAR reads |
| `a1` | a1-cpu-side's `barprobe` with its 9 config files (sync/event/AQL-barrier/memset interventions, writers, sizes, reader layouts) × 3 reps; per-reader timing idle vs background GPU load; Kurt's test instrumented to count stale watcher events under interventions | CPU BAR reads, CPU BAR writes for some configs |
| `a2` | a2-gpu-side's `stale-probe` batches b01–b11 (load/acquire variants, memory pools, kicks, timing, after-mode residency, `AMD_OPT_FLUSH`); gfx10/gfx11 only (inline assembly) | CPU BAR writes; GPU reads |
| `c` | c-hypotheses' `f1probe` (Failure 1 discriminators) and `f2probe` (Failure 2 discriminators) configs | CPU BAR reads and writes |
| `hdpprobe` | only with `--allow-hdp-flush`: John's GPU test with a write-only HDP flush between `hipFileRead` and the checking kernel | HDP register write |

### HDP register

Configs that write the HDP flush register (`HDP_MEM_COHERENCY_FLUSH_CNTL`, `HDP_REG_FLUSH_CNTL`)
are skipped unless you pass `--allow-hdp-flush`; the summary lists each skipped line. In this
bundle every HDP access is **write-only**: the copies of a1's `interventions.h` and c's
`f1probe.hip`/`f2probe.hip` had their register read-backs removed, because upstream amdgpu
(a5cb344033c7) stopped reading this remapped register back after it caused AER error storms
and a host-bridge freeze. Consequently a1's `hdpmem`/`hdpreg` and c's `hdpflush`/`hdpreg` behave
like their write-only variants here.

### CPU access to VRAM through the BAR

`kurt`, `hipbar`, `a1` and `c` read VRAM from the CPU through the large BAR (that is what
Failure 1 is about); `a2`, `c` and some `a1` configs also write it. To avoid all of that, run
`--sections health,ctest,gpu,write,aqllog` (and note that the `ctest` section includes Kurt's CPU test).

## What rowlet (RX 6800 XT, gfx1030, ROCm 7.15, amdgpu-dkms 7.1.3) showed

| Test | fastpath | fallback |
|---|---|---|
| Kurt's test, 64 KiB, 8/24/46 CPUs | pass (2000 it) | **fail at iteration 0–3** (HDP read cache) |
| Kurt's test, 1 MiB (ctest setting) | pass | pass |
| GPU test `after`, no prime / `--prime --prime-wait sync` | pass | pass |
| GPU test `after --prime --prime-wait flag` / `sleep` | **fail at iteration 0, every word stale** (GPU L2) | pass |
| GPU test `--read async`, every variant | pass | pass |
| GPU test `during` ±`--prime` (with `HSA_FORCE_FINE_GRAIN_PCIE=1`, set by the program) | pass | pass |
| ctest, all 6 | pass | pass |
| `hipbar` READERS ≥ 2 | n/a | 30–95% of iterations stale; `KICK=1` → 0 |
| GPU test `--prime-wait sleep` + write-only HDP flush (`hdpprobe`) | **still fails** | pass |

`hipfilewrite-visibility` on the same machine, 1 MiB unless a size is given. Everything
passes with fallback. With fastpath:

| Configuration | Result | Cause |
|---|---|---|
| `--producer kernel --wait sync` / `query` / `event` / `callback`, `--write async` | pass | HIP enqueues a system-scope barrier |
| `--producer kernel --wait sleep` / `flag` | **fail at iteration 0, every word stale**; passes with `--memory uncached` or `fine` | GPU L2 |
| `--producer memset` (4K to 64M) | **fail at iteration 0 or 1**; with `--memory uncached` only the start of the buffer is stale and a second write is right | GPU L2, and `hipMemsetD32` returns before its kernel finishes |
| `--producer copy --buffer-size 4K` | **fail at iteration 0, every word stale** | GPU L2 |
| `--producer copy` | **fail at iteration 59 of 1000, 20 KiB stale**, and a second write is right | `hipMemcpy` between GPU buffers returns before its kernel finishes |
| `--producer memcpy` 4K to 16K with `--background-writers` 1 to 8 | **fail within 6 iterations** in coarse, fine and uncached memory; passes from 32K | HDP read cache |
| `--producer kernel --kernel during --buffer-size 4K --background-writers 4`, fine or uncached | **fail within 9 iterations** | HDP read cache |
| `--producer read --prime store --wait sleep --threads 4 --buffer-size 4M` | **fail at iteration 82: the file and the buffer hold the priming kernel's data** | GPU L2 written back over `hipFileRead`'s data |
| `--producer kernel --kernel during`, `memcpy` 4K to 16M, `read` 4K to 16M, with `--prime write`, `--threads` up to 16 | pass | |

On an MI300X node (ctr-smc-mi300x-cx67-25, gfx942, ROCm 7.14, amdgpu-dkms 6.19.14) the `gpu`
and `write` sections ran on three GPUs with the test files on one NVMe drive: the GPU behind
the same PCIe switch as the drive, one on another root complex of the same socket, and one on
the other socket. The three gave the same results. Every `gpu` configuration passes, and so
does every `write` configuration with fallback. With fastpath, none of the GPU L2 or HDP read
cache failures above occur, including in 150 s soaks of thirteen configurations on the three
GPUs at once. The two that fail are the synchronous calls that return before their kernel
finishes: `--producer memset --buffer-size 4K` at iteration 128 to 404 of 1000 on each GPU, and
`--producer copy` at iteration 2 on one GPU. In both, a second write is right.

Failure 1 (CPU reads, fallback) is the HDP read cache: only an HDP flush issued after the copy,
or an AQL packet with an acquire fence, clears it. Failure 2 (GPU reads, fastpath) is the GPU L2:
only an AQL packet with a system-scope acquire clears it; the HDP flush does not. The detailed
reports are in `~/scratch/hipfile-hdp-repro/reports/` on cgy-decidueye.
