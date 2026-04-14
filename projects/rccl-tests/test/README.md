# RCCL-Tests Unit Test Suite

The goal of RCCL-Tests Unit Test Suite is to ensure that changes in RCCL-Tests source code do not break RCCL-Tests functionality. This is not a well-rounded regression suite for RCCL functionality/performance, although we do provide some level of RCCL `regression` testing.

## Contents

- [Prerequisites](#prerequisites)
- [Running Tests](#running-tests)
- [Test Structure](#test-structure)
- [Collectives Registry](#collectives-registry)
- [Test Markers](#test-markers)
- [GPU Detection](#gpu-detection)
- [Filterability Cheat Sheet](#filterability-cheat-sheet)
- [Regression Profile](#regression-profile)
- [Coverage Notes and Known Gaps](#coverage-notes-and-known-gaps)
- [CI Integration](#ci-integration)
- [Additional Options](#additional-options)

## Prerequisites

1. Install Python test dependencies:
   ```shell
   pip install -r requirements.txt
   ```

2. Build the perf binaries (see the main [README.md](../README.md)). The test
   runner expects them in `../build/` relative to the `test/` directory.

3. Set `LD_LIBRARY_PATH` to include the RCCL library:
   ```shell
   export LD_LIBRARY_PATH=/path/to/rccl-install/lib:$LD_LIBRARY_PATH
   ```

4. For MPI tests, set `PATH` and `LD_LIBRARY_PATH` to include MPI:
   ```shell
   export PATH=/path/to/mpi-install/bin:$PATH
   export LD_LIBRARY_PATH=/path/to/mpi-install/lib:$LD_LIBRARY_PATH
   ```

## Running Tests

Run all precheckin tests — standalone + MPI (requires `mpirun` on `PATH`):
```shell
python3 -m pytest -m "not regression"
```

Run standalone tests only (no MPI runtime required):
```shell
python3 -m pytest -m "not mpi and not regression"
```

Run the regression sweep:
```shell
python3 -m pytest -m regression
```

Dry-run to see what would be collected without executing:
```shell
python3 -m pytest --collect-only
```

## Test Structure

The suite is organized into independent parametrized test files. Every
parameter combination is a separate pytest item with a filterable node ID —
no subtests, no loops, no hidden state.

| File | Groups | What it covers |
|------|--------|----------------|
| `collectives.py` | — | `Collective` dataclass registry (not a test file) |
| `test_code_paths.py` | 2–13 | CLI flag coverage: step modes, correctness, blocking, placement, HIP graph, threading, rooted collectives, iteration control, output formats, memory types, env vars, MPI, internal timeout |
| `test_all_sweep.py` | 14 | `-o all` / `-d all` / `-r all` sweep modes |
| `test_ops.py` | 15 | Individual reduction ops parametrized (`sum`, `prod`, `min`, `max`, `avg`, `mulsum`) |
| `test_dtypes.py` | 16 | Individual datatypes parametrized (all 12 types) |
| `test_env_vars.py` | 17 | Env-var-driven code paths: NIC counter collection (`RCCL_TESTS_NET_COUNTER_ENABLE`, `NCCL_IB_HCA`, `RCCL_TESTS_NIC_COUNTER_LIST`, `RCCL_TESTS_NET_COUNTER_NIC_PREFIX`) and GPU split (`NCCL_TESTS_SPLIT`, `NCCL_TESTS_SPLIT_MASK`) |
| `test_regression.py` | — | Full regression sweep: collectives × datatypes × byte ranges × GPU grid |

## Collectives Registry

All collectives are defined in `collectives.py` as a `Collective` dataclass:

```python
Collective(name, executable, has_ops, has_root, arch_gate)
```

| Name | Executable | has_ops | has_root | arch_gate |
|------|-----------|---------|----------|-----------|
| allreduce | all_reduce_perf | yes | no | — |
| allgather | all_gather_perf | no | no | — |
| broadcast | broadcast_perf | no | yes | — |
| reduce | reduce_perf | yes | yes | — |
| reducescatter | reduce_scatter_perf | yes | no | — |
| alltoall | alltoall_perf | no | no | — |
| alltoallv | alltoallv_perf | no | no | — |
| scatter | scatter_perf | no | yes | — |
| gather | gather_perf | no | yes | — |
| sendrecv | sendrecv_perf | no | no | — |
| hypercube | hypercube_perf | no | no | — |
| allreduce_bias | all_reduce_bias_perf | yes | no | gfx942/gfx950 |

To add a new collective, add one line to `collectives.py` — all test files
pick it up automatically via the `COLLECTIVES` parametrize list.

## Test Markers

| Marker | Description |
|--------|-------------|
| `mpi` | Tests requiring MPI runtime (`mpirun` on `PATH`) |
| `regression` | Full regression sweep (datatypes × byte ranges × GPU counts) |
| `gfx942_gfx950` | Tests requiring gfx942 or gfx950 GPU architecture (auto-skipped on other hardware) |

```shell
python3 -m pytest -m "not regression"             # unit tests only (typical precheckin run)

python3 -m pytest -m mpi                          # MPI tests only
python3 -m pytest -m "not mpi"                    # non-MPI tests only

python3 -m pytest -m regression                   # regression sweep only

python3 -m pytest -m gfx942_gfx950                # arch-gated tests only
python3 -m pytest -m "not gfx942_gfx950"          # skip arch-gated tests
```

Arch-gated tests are skipped at **collection time** on non-matching hardware, so the
binary is never launched. On matching hardware (gfx942/gfx950), if the RCCL build
predates the feature (e.g. an older ROCm stack without `ncclAllReduceWithBias`), the
test is skipped at **runtime** with a clear reason rather than failing.

## GPU Detection

GPU count is detected automatically at session start:

1. `rocm_agent_enumerator -gpu` (fast, <50ms) — preferred
2. `rocminfo` fallback (1-5s) if `rocm_agent_enumerator` is unavailable

If `ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` is set, the suite
validates that every requested index is within the hardware range and
uses `len(visible_indices)` as the GPU count. Range syntax is supported:

```shell
# 4-GPU node, restrict to 2 GPUs:
ROCR_VISIBLE_DEVICES=0,2 python3 -m pytest       # NGPUS=2, runs on physical 0,2
ROCR_VISIBLE_DEVICES=0-3 python3 -m pytest       # NGPUS=4, runs on physical 0-3

# Invalid index → session exits immediately with a clear error:
ROCR_VISIBLE_DEVICES=0,9 python3 -m pytest       # exits: index 9 out of range
```

The detected GPU count is reported in the pytest header and used as `-t`
for all single-process tests.

## Filterability Cheat Sheet

Every test combination has a unique, filterable node ID. Use `-k` to target
specific combinations or `pytest <nodeid>` to rerun exact failures from CI.

### By arch gate

```shell
pytest -m gfx942_gfx950               # all arch-gated tests (skipped on non-matching hw)
pytest -m "not gfx942_gfx950"         # exclude arch-gated tests entirely
pytest -k "allreduce_bias"            # same effect via name filter
```

### By collective

```shell
pytest -k "allreduce"          # all allreduce tests across all files
pytest -k "broadcast"          # all broadcast tests
pytest -k "allreduce_bias"     # arch-gated tests only (auto-skipped on non-gfx942/950)
pytest -k "not allreduce_bias" # skip allreduce_bias entirely
```

### By message size

```shell
pytest -k "1K"                 # all tests at 1K
pytest -k "1G"                 # all tests at 1G (the slow ones)
pytest -k "allreduce and 1K"   # allreduce at 1K only
```

### By test file (group)

```shell
pytest test_code_paths.py      # Groups 2-13: CLI flags and execution modes
pytest test_all_sweep.py       # Group 14: -o/-d/-r all sweep modes
pytest test_ops.py             # Group 15: individual reduction ops
pytest test_dtypes.py          # Group 16: individual datatypes
pytest test_env_vars.py        # Group 17: env-var-driven code paths (NIC counters, GPU split)
pytest test_regression.py      # Regression sweep
```

### By CLI flag / code path

```shell
pytest -k "stepfactor"         # -f multiplicative size stepping
pytest -k "stepbytes"          # -i additive size stepping
pytest -k "correctness_check"  # -c correctness verification
pytest -k "check1"             # -c 1 specifically
pytest -k "check2"             # -c 2 specifically
pytest -k "blocking"           # -z 1 blocking mode
pytest -k "null_stream"        # -y 1 NULL stream
pytest -k "placement"          # -O in-place and out-of-place
pytest -k "in_place"           # -O 0 in-place only
pytest -k "out_of_place"       # -O 1 out-of-place only
pytest -k "hip_graph"          # -G HIP graph capture and replay
pytest -k "thread_per_gpu"     # -t N -g 1 threading model
pytest -k "multi_gpu_per_thread" # -t 1 -g N threading model
pytest -k "root_explicit"      # -r 0 explicit root
pytest -k "root_all"           # -r all root rotation sweep
pytest -k "custom_iters"       # -n non-default iteration count
pytest -k "aggregated_iters"   # -m batched operations
pytest -k "multi_cycle"        # -N outer cycle loop
pytest -k "cpu_time"           # -C CPU timing
pytest -k "timestamps"         # -S timestamp output
pytest -k "average_mode"       # -a all modes
pytest -k "avg_min"            # -a 2 MIN averaging specifically
pytest -k "json_output"        # -J JSON file output
pytest -k "memory_report"      # -M memory usage reporting
pytest -k "memory_type"        # -Y all non-default types (fine, host, managed)
pytest -k "mem_fine"           # -Y fine specifically
pytest -k "mem_host"           # -Y host specifically
pytest -k "mem_managed"        # -Y managed specifically
pytest -k "rotating_tensor"    # -E rotating tensor pattern
pytest -k "rccl_reporter"      # -Z/-X RCCL reporter CSV
pytest -k "local_register"     # -R local buffer registration
pytest -k "algo_proto"         # -A algo/proto/channel reporting
pytest -k "env_device"         # NCCL_TESTS_DEVICE override
pytest -k "env_min_bw"         # NCCL_TESTS_MIN_BW threshold
pytest -k "internal_timeout"   # -T internal timeout mechanism
```

> `parallel_init` (`-p 1`) is currently disabled. See
> [Coverage Notes and Known Gaps](#coverage-notes-and-known-gaps).

### By op or dtype

```shell
pytest -k "op and prod"        # just the prod reduction op
pytest -k "op and sum"         # just the sum reduction op
pytest -k "dtype and fp8"      # both fp8 variants
pytest -k "dtype and bfloat16 and 1G"  # bfloat16 at 1G
```

### By MPI / single-process

```shell
pytest -k "mpi"                # all MPI tests
pytest -k "not mpi"            # all single-process tests
pytest -k "test_split"         # MPI split env-var tests
pytest -k "test_env_"          # env-var coverage tests
```

### Combined filters

```shell
pytest -k "blocking and 1G"          # blocking mode at 1G only
pytest -k "hip_graph and 1K"         # HIP graph at 1K only
pytest -k "memory_type and 1M"       # all memory types at 1M
pytest -k "mem_fine and 1G"          # fine-grain memory at 1G
pytest -k "allreduce and not mpi"    # allreduce single-process only
```

### Regression filters

```shell
pytest test_regression.py -k "allreduce and float and 1M"
pytest test_regression.py -k "gpus8"   # full-machine tests only
pytest test_regression.py -k "gpus2"   # smallest multi-GPU regression point
pytest test_regression.py -k "not allreduce_bias"
```

### Rerun exact CI failures by node ID

```shell
pytest "test_code_paths.py::test_blocking[1G]"
pytest "test_code_paths.py::test_placement[out_of_place-1M]"
pytest "test_code_paths.py::test_memory_type[mem_fine-1G]"
pytest "test_code_paths.py::test_correctness_check[check2-1K]"
pytest "test_ops.py::test_op[prod-1M]"
pytest "test_dtypes.py::test_dtype[fp8_e5m2-1G]"
pytest "test_regression.py::test_regression_single[allreduce-float-1M-4G-gpus8]"
```

## Regression Profile

The regression suite sweeps all collectives × all datatypes × three byte
ranges × a machine-appropriate GPU count grid.

The GPU grid is computed at import time from the detected `NGPUS` using
powers-of-2 starting at 2, always including `NGPUS` itself:

| Machine | NGPUS | GPU grid |
|---------|-------|----------|
| 8-GPU node | 8 | `[2, 4, 8]` |
| 6-GPU node | 6 | `[2, 4, 6]` |
| 4-GPU node | 4 | `[2, 4]` |
| 2-GPU dev box | 2 | `[2]` |
| 1-GPU dev box | 1 | `[1]` |

On hosts with ≥2 GPUs the regression grid is strictly multi-GPU; on a
1-GPU host it falls back to `[1]` so the `regression` marker still has at
least one item to schedule. Single-GPU (`1`) functional coverage is
provided by the non-regression suites (`test_code_paths.py`, `test_ops.py`,
`test_dtypes.py`, `test_all_sweep.py`, `test_env_vars.py`), which already
run at `-t NGPUS -g 1` for standalone tests and skip multi-GPU items on a
1-GPU host via `pytest.skip(...)`.

Estimated runtime at 2s/test on an 8-GPU node: ~3.8 hours (within nightly
CI budget). Ops and memory types are not swept in regression — those are
covered individually by `test_ops.py` and `test_code_paths.py::test_memory_type`.

## Coverage Notes and Known Gaps

### Intentionally Not Covered

The suite focuses on framework wiring and code-path activation. The
following CLI flags and env vars are not exercised here on purpose,
either because they require live hardware/topology to be meaningful or
because their behaviour is verified elsewhere:

- `-x / --enable_xccl` — vendor library selection; covered by integration
  builds, not by these unit tests.
- `-x_dev_id / --hsa_xgmi_dev_id` — XGMI device-ID overrides; require
  specific multi-die hardware.
- `-u / --cumem` and `-q / --quiet` — output/UX-only flags with no
  observable correctness side-effect.
- `NCCL_TESTS_GPU_GRAPH` / `NCCL_TESTS_GPU_GRAPH_PATH` — graph-tracing
  helpers; covered by the HIP-graph code path in `test_hip_graph` via
  `-G`.

### Disabled tests / TODO

- `test_parallel_init` (`-p 1` threaded NCCL/RCCL init) is intentionally
  commented out in `test_code_paths.py`. Preliminary investigation suggests
  concurrent `ncclInitKernelsForDevice()` calls race on
  `cudaFuncSetAttribute()` with no lock held, corrupting kernel
  shared-memory configuration and causing GPU page faults during
  subsequent collectives. Re-enable once the upstream fix lands (tracked
  in prior PR discussion as `#6159`).

## CI Integration

**Precheckin** (standalone + MPI, requires `mpirun` on `PATH`):
```shell
python3 -m pytest -m "not regression" \
  --junitxml=testreport.xml --test-timeout=300
```

**Nightly regression**:
```shell
python3 -m pytest -m regression \
  --junitxml=testreport.xml --test-timeout=600
```

`pytest-rerunfailures` is installed but retries are not enabled by default.
Use `--reruns=N --reruns-delay=SECONDS` in CI or local reruns when needed.
Because tests are independent parametrized items (not subtests), retries are
scoped to the exact failing combination — not the entire test file.

## Additional Options

| Option | Default | Description |
|--------|---------|-------------|
| `--test-timeout=N` | 300 | Per-test subprocess timeout in seconds — propagated to every `run_rccl_perf` and `run_rccl_mpi` call |
| `--hostfile=FILE` | — | MPI hostfile — passed as `-hostfile FILE` to every `mpirun` invocation; omitted from the command if not set |
| `--mpi-args=STR` | — | Extra args inserted into every `mpirun` invocation between `-np N` and the executable; shell-tokenized with `shlex.split`. Example: `--mpi-args='--bind-to none --mca pml ucx'` |
| `--junitxml=FILE` | — | JUnit XML report for CI |
| `-v --tb=short` | on | Verbose output with short tracebacks (set in `pytest.ini`) |
| `--reruns=N` | off | Retry failed tests N times (provided by `pytest-rerunfailures`) |

### Passing extra mpirun flags

`--mpi-args` is the supported way to inject arbitrary `mpirun` flags such as
`--bind-to`, `--map-by`, `--mca`, `--allow-run-as-root`, or `--oversubscribe`
into every MPI test:

```shell
python3 -m pytest -m mpi \
    --hostfile=hosts.txt \
    --mpi-args="--bind-to none --map-by ppr:1:numa --mca pml ucx"
```

Tokens are split with `shlex.split` so quoting works the same way as on the
shell. The active value (if any) is shown in the pytest report header.
Tunables that OpenMPI honors via environment variables (`OMPI_MCA_*`,
`UCX_*`, `NCCL_*`, `RCCL_*`) can also be exported directly — they're
inherited by every `mpirun` child.
