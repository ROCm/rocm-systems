# Nightly tarball test runner

Validates `rocprof-sys-*` against a ROCm (TheRock) tarball using the latest
`develop` tests from `ROCm/rocm-systems`.

**Entry point:** `../run-nightly-tarball-tests.py`
**Needs:** Python 3.8+, `git`, `cmake`, `wget` or `curl`, and a GPU for the test phase.

## 1. Default run (test binaries shipped in the tarball)

```bash
cd /path/to/rocprofiler-systems/scripts
python3 run-nightly-tarball-tests.py --tier quick
```

What this does:

1. Download/extract the latest nightly ROCm tarball → `<work-dir>/rocm`
2. Sparse-checkout `projects/rocprofiler-systems` from `develop`
3. Create a pytest venv
4. Build examples + stage tests into the ROCm prefix
5. Run pytest in **install mode** against the tarball’s `bin/rocprof-sys-*`

Default work dir: `./rocprofiler-systems-tests` (created next to your cwd).

## 2. Common options

```bash
# Smaller per-GPU tarball (or auto-detect from KFD)
python3 run-nightly-tarball-tests.py --variant gfx90a
python3 run-nightly-tarball-tests.py --variant auto

# Pick archive kind: nightly (default), release RC, or nightly + -tests- data
python3 run-nightly-tarball-tests.py --tarball release --variant gfx94X-dcgpu
python3 run-nightly-tarball-tests.py --tarball tests --variant gfx94X-dcgpu

# Pin a version / branch / work dir
python3 run-nightly-tarball-tests.py --rocm-version 20260817 --work-dir /scratch/$USER/rocm-test

# Include MPI tests
python3 run-nightly-tarball-tests.py --tier standard --run-labels mpi
```

Tiers: `quick` → `standard` → `comprehensive` → `full`.

## 3. Build from source (test freshly built binaries)

Uses the tarball only for ROCm runtime/toolchain; builds and tests
rocprofiler-systems from `develop`:

```bash
python3 run-nightly-tarball-tests.py --build-from-source --tier standard
```

## 4. Air-gapped / two-phase cluster

Shared `--work-dir` on a filesystem both nodes can see (not home if compute is RO).

```bash
# Phase 1 — networked node (login): download + clone + venv, then stop
python3 run-nightly-tarball-tests.py \
  --work-dir /scratch/$USER/rocm-test --variant gfx90a --prepare-only

# Phase 2 — GPU compute node: no network
python3 run-nightly-tarball-tests.py \
  --work-dir /scratch/$USER/rocm-test --offline --tier standard
```

On login nodes with no GPU, pass an explicit `--variant` (do not rely on `auto`).

### Notes: sample Slurm scripts

This folder ships sample `.sbatch` wrappers you can copy and adapt per cluster
(partition, account, GPU GRES, time/memory, modules, `WORK_DIR`, `VARIANT`, etc.):

- `nightly-networked-gpu.sbatch` — single-phase GPU job (networked compute node)
- `nightly-offline-two-phase.sbatch` — prepare on login, then submit an offline GPU job

They are starting points only; edit the `#SBATCH` header and env defaults for your site
before submitting.

## 5. Results

Under `<work-dir>`:

| File | Meaning |
| --- | --- |
| `summary-*.log` / `RESULT-*.md` | Pass/fail + counts |
| `detailed-*.log` | Full run log |
| `failures-*.log` | Failing tests (on failure) |
| `latest-*.txt` | Pointers to the newest artifacts |

```bash
python3 run-nightly-tarball-tests.py --help
```
