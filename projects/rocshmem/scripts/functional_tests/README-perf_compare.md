# rocSHMEM Performance Scripts

Scripts for building, running, and comparing rocSHMEM performance across configurations.

## Quick-start: compare current branch vs develop

The baseline is automatically set to the **merge-base** of your current branch with `origin/develop` (i.e. the common ancestor, not develop HEAD), so results reflect only the changes on your branch.

```bash
# From the rocm-systems/projects/rocshmem directory:
./scripts/functional_tests/run_perf_compare.sh --suite heatmap --iterations 5

# Output:
#   plots-heatmap-<branch>/heatmap_summary.png   summary heatmap (green = faster)
#   plots-heatmap-<branch>/heatmap_data.csv       raw percentage data (CSV)
#   plots-heatmap-<branch>/heatmap_summary.txt     summary % table (text)
#   plots-heatmap-<branch>/per_test/*.png         per-test latency-vs-size curves
#   plots-heatmap-<branch>/per_test/*.txt         per-test salient metrics + IQR stats
```

Run `--suite all` for the full functional test suite (slower):

```bash
./scripts/functional_tests/run_perf_compare.sh --suite all --iterations 5
```

## Skip rebuild on second run

After the first full run, use `--skip-build` to re-run tests without rebuilding:

```bash
./scripts/functional_tests/run_perf_compare.sh --suite heatmap --iterations 20 --skip-build
```

Skip only the baseline (build and test runs) when the baseline data is already available:

```bash
./scripts/functional_tests/run_perf_compare.sh --suite heatmap --skip-baseline
```

Skip everything and just regenerate plots from existing logs:

```bash
./scripts/functional_tests/run_perf_compare.sh --suite heatmap --skip-build --skip-baseline --skip-branch
```

## Multiple named variants (e.g. feature flag comparison)

Use `--variant-args NAME:ENV=VAL,...:extra-cmake-args` to compare multiple configurations of the same branch. Each spec produces its own build directory and named column in the heatmap.

```bash
# Example: compare SDMA off/disabled/enabled against develop baseline
./scripts/functional_tests/run_perf_compare.sh \
  --branch-args "-DUSE_SDMA=ON" \
  --variant-args "sdma-off::-DUSE_SDMA=OFF" \
  --variant-args "sdma-disabled:ROCSHMEM_SDMA_ENABLED=0:" \
  --variant-args "sdma-enabled:ROCSHMEM_SDMA_ENABLED=1:"
```

Fields in each spec (separated by `:`):
1. **Name** — label in plots and build directory suffix
2. **Env vars** — comma-separated `KEY=VALUE` pairs set at runtime (empty = none)
3. **CMake args** — extra cmake flags for this variant's build (empty = none)

## Compare a GitHub PR

```bash
./scripts/functional_tests/run_perf_compare.sh --pr 4574 --suite heatmap --iterations 5
# Fetches PR #4574, builds it vs origin/develop, outputs plots-heatmap-pr4574/
```

## Generate the heatmap from existing logs (no rebuild, no re-run)

If you already have log directories from a previous run:

```bash
python3 scripts/functional_tests/perf_compare.py \
  --baseline "path/to/build-baseline/logs-heatmap-*" \
  --variants "mybranch:path/to/build-branch/logs-heatmap-*" \
  --outdir plots/

# With multiple variants:
python3 scripts/functional_tests/perf_compare.py \
  --baseline "build-develop/logs-heatmap-*" \
  --variants \
    "sdma-off:build-sdma-off/logs-heatmap-*" \
    "sdma-disabled:build-sdma-on/logs-heatmap-disabled-*" \
    "sdma-enabled:build-sdma-on/logs-heatmap-enabled-*" \
  --outdir plots/
```

The script auto-detects multiple iteration directories (glob with `*`) and aggregates them with IQR outlier removal before computing medians.

## Docker-based PR comparison

> **Note:** the Docker baseline is the HEAD of `develop` at the time the image was
> built, not the PR's merge-base. For a fresh image against a recent PR the
> difference is usually negligible. For a precise merge-base comparison, use the
> local workflow instead.

### Option A: one-shot (build image, run, get results on host)

```bash
# From the projects/rocshmem directory:
# Build the image — compiles both develop (/app/build) and PR (/app/build4574)
docker build -f docker/Dockerfile.ubuntu --build-arg PR_NUM=4574 --tag $USER/rocshmem-pr4574 docker/

# Run comparison; results appear on the host at ./pr4574-results/
mkdir -p pr4574-results
docker run --rm \
  --shm-size 64G --network host --device /dev/dri --device /dev/kfd \
  --ipc host --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  -v "$(pwd)/pr4574-results:/results" \
  $USER/rocshmem-pr4574 perf-compare --suite heatmap --iterations 5
# Plots: ./pr4574-results/heatmap_summary.png
```

### Option B: interactive session (build once, re-run quickly)

```bash
# Start a persistent container with a bind-mount for results
docker run -d --name rocshmem-pr4574 \
  --shm-size 64G --network host --device /dev/dri --device /dev/kfd \
  --ipc host --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  -v "$(pwd)/pr4574-results:/results" \
  $USER/rocshmem-pr4574 sleep infinity

# Jump interactive, do some modifications, rebuild
docker exec -it rocshmem-pr4574 bash

# Run as many times as needed (e.g. vary --iterations or --suite)
docker exec rocshmem-pr4574 perf-compare --suite heatmap --iterations 5

# Teardown
docker stop rocshmem-pr4574 && docker rm rocshmem-pr4574
```

## CI integration (heatmap artifact generation)

`functional_tests/driver.sh` accepts `--artifact-dir DIR`. When run on a heatmap suite with this flag, it generates performance plots into DIR after the tests complete and prints `CI ARTIFACT:` lines for each output file.

### Single-build CI job (per-test latency plots, no baseline comparison)

```bash
cd $BUILD_DIR
$ROCSHMEM_DIR/scripts/functional_tests/driver.sh \
  tests/functional_tests/rocshmem_functional_tests \
  heatmap \
  logs-heatmap \
  --artifact-dir $CI_ARTIFACTS_DIR
# Produces: $CI_ARTIFACTS_DIR/per_test/*.png
```

### Two-build CI job (heatmap comparison: branch vs develop)

Set `PERF_BASELINE_DIR` and `PERF_BRANCH_DIR` so driver.sh knows where both pre-built binaries live:

```bash
cd $BRANCH_BUILD_DIR
PERF_BASELINE_DIR=$DEVELOP_BUILD_DIR \
PERF_BRANCH_DIR=$BRANCH_BUILD_DIR \
  $ROCSHMEM_DIR/scripts/functional_tests/driver.sh \
    tests/functional_tests/rocshmem_functional_tests \
    heatmap \
    logs-heatmap \
    --artifact-dir $CI_ARTIFACTS_DIR
# Produces: $CI_ARTIFACTS_DIR/heatmap_summary.png + heatmap_data.csv + per_test/
```

Alternatively, call `run_perf_compare.sh` directly which handles the full build+run+compare pipeline:

```bash
$ROCSHMEM_DIR/scripts/functional_tests/run_perf_compare.sh \
  --suite heatmap \
  --iterations 3 \
  --outdir $CI_ARTIFACTS_DIR
```

## Reference: all `run_perf_compare.sh` options

| Option | Default | Description |
|--------|---------|-------------|
| `--iterations N` | 10 | Iterations per config |
| `--suite SUITE` | heatmap | Test suite (`heatmap`, `all`, `rma`, …) |
| `--build-config CFG` | all_backends | Build config under `scripts/build_configs/` |
| `--cmake-args ARGS` | — | Extra cmake args for all builds |
| `--branch-args ARGS` | — | Extra cmake args for branch build only |
| `--variant-args SPEC` | — | Additional variant (repeatable) |
| `--pr NUM` | — | Compare GitHub PR NUM vs develop |
| `--base-branch NAME` | origin/develop | Branch to compare against |
| `--baseline-dir PATH` | auto | Pre-built baseline directory |
| `--branch-dir PATH` | auto | Pre-built branch directory |
| `--skip-build` | — | Skip all builds |
| `--skip-baseline` / `--skip-develop` | — | Skip baseline build and test runs (reuse existing logs) |
| `--skip-branch` | — | Skip branch/PR build and test runs (reuse existing logs) |
| `--outdir DIR` | auto | Output directory for plots |
