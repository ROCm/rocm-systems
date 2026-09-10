# Automated benchmark suite

The in-tree benchmark suite runs a fixed workload matrix through rocjitsu on
`gfx950` and `gfx1250`. It is intended for nightly performance tracking and
manual evaluation of changes. The first version writes local JSON artifacts;
the CI proof of concept below converts those artifacts to the dashboard's
per-run JSON layout. Historical selection, production scheduling, and retention
policy remain follow-on work.

Use the controls and sampling protocol in
[Benchmarking rocjitsu](benchmarking.md) for official performance comparisons.

## Suite

`benchmarks/suites/nightly.toml` contains the targets, ordered case IDs, and
suite-wide measurement settings. Commands and filesystem paths are derived from
the source and build layouts, so the manifest stays short and portable.

The nightly suite has 16 cases per target:

| Provider | Cases | Coverage |
|---|---:|---|
| Triton | 13 | Five generic memory/elementwise cases, six softmax, normalization, GEMM, and attention cases, and two GPT-OSS-20B model shapes |
| hipBLASLt/TensileLite | 3 | FP16 GEMM, batched BF16 GEMM, and scaled FP8 GEMM |

The five generic Triton cases cover a 32 MiB contiguous FP32 copy, a
boundary-sized FP32 vector addition, a 2048x2048 FP16 transpose, an irregular
FP32 gather, and a contended FP32 atomic addition. The six existing Triton
cases cover aligned and boundary FP16 softmax, BF16 RMSNorm, aligned and ragged
BF16 GEMM, and FP16 attention. The two GPT-OSS cases use GPT-OSS-20B dimensions
for BF16 RMSNorm (hidden size 2880) and grouped-query sliding attention (64
query heads, 8 key/value heads, head dimension 64, sequence/window 128).

This produces 32 case/target cells in a full run. Inputs and launch
configurations are fixed, and measured runs do not autotune. The cells execute
sequentially, while each simulation uses eight worker partitions--one for each
XCD in both target configurations. Changing that count creates a different
benchmark configuration and its results must not be compared directly with
the eight-thread series.

MXFP4 MoE projections are not included. Although the pinned hipBLASLt API
defines `HIP_R_4F_E2M1` and VEC32 UE8M0 scale metadata, its installed gfx950
solution library requires MXFP4 for both GEMM inputs and does not provide the
BF16-activation by MXFP4-weight operation used by GPT-OSS. Adding an
MXFP4-shaped case would therefore claim unsupported math or require a separate
activation-quantization operation with different timing semantics.

## Measurement

Each cell runs in a fresh process. The workload performs allocation,
initialization, compilation or descriptor setup, and algorithm selection before
measurement. It then runs three untimed warmups followed by 21 timed samples.
Every warmup and sample launches the operation and synchronizes the device; a
sample is the host elapsed time around that launch-and-synchronize pair.

Only the 21 samples contribute to the reported minimum, median, and maximum.
Process startup, setup, warmups, JSON output, and teardown are excluded. For a
Triton case, one sample contains one kernel launch. For a hipBLASLt
case, one sample contains one `hipblasLtMatmul` call, which may launch more than
one internal kernel.

This first version does not establish numerical correctness for the benchmark
adapters. It validates execution and timing integrity only: the runner rejects
failed processes, timeouts, malformed workload JSON, identity or target
mismatches, and timing vectors that do not contain the requested number of
positive samples. Separate correctness coverage can be added later without
putting reference computation or output copies inside the timing boundary.

## Install and build

Use Python 3.12 and install the pinned dependencies:

```bash
src=/path/to/rocm-systems/emulation/rocjitsu
python=/path/to/python3.12
"$python" -m venv /path/to/rocjitsu-benchmark-env
python=/path/to/rocjitsu-benchmark-env/bin/python
"$python" -m pip install -r "$src/benchmarks/requirements.txt"
```

The requirements use AMD's multi-architecture wheel index and provide the ROCm
SDK, PyTorch, Triton, hipBLASLt, and target-specific gfx950/gfx1250 device
libraries. The benchmark build consumes those installed packages; it does not
fetch or build hipBLASLt, TensileLite, or other native dependencies from
source. The native packages are binary-only. The index publishes the 27 KiB
pure-Python `rocm` metadata package only as a source archive, so it is the sole
exception; pip builds no ROCm library or benchmark component from source.

Configure a Release build against the SDK installed in the environment:

```bash
build=/path/to/rocjitsu-build-release
rocm=$("$(dirname "$python")/rocm-sdk" path --root)
export LD_LIBRARY_PATH="$rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S "$src" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DRJ_BUILD_BENCHMARKS=ON \
  -DLTO=OFF \
  -DRJ_ENABLE_ASAN=OFF \
  -DRJ_ENABLE_MSAN=OFF \
  -DRJ_ENABLE_TSAN=OFF \
  -DRJ_ENABLE_UBSAN=OFF \
  -DROCM_PATH="$rocm" \
  -DPython3_EXECUTABLE="$python"

cmake --build "$build" --target rocjitsu_benchmark_workloads
```

`ROCM_PATH` is authoritative: CMake clears cached package locations and finds
HIP, hipBLAS-common, and hipBLASLt package configurations only beneath that
SDK. The build produces the rocjitsu executable, the host hipBLASLt adapter,
and the logging, race, and throughput plugin modules; the Triton cases use the
checked-in Python workload module. Normal rocjitsu builds are unchanged because
`RJ_BUILD_BENCHMARKS` defaults to `OFF`. Keeping the SDK library directory on
`LD_LIBRARY_PATH` also lets PyTorch resolve the shared libraries shipped in the
same environment.

Build the selected checkout immediately before every official run. Before
creating an output directory, the runner requires `CMAKE_BUILD_TYPE=Release`,
`LTO=OFF`, all four sanitizer options above set to `OFF`, and
`CMAKE_HOME_DIRECTORY` equal to the current rocjitsu checkout. It also requires
the cached `ROCM_PATH` to exactly equal the `_rocm_sdk_devel` package imported
by the Python environment. These checks prevent accidental Debug, LTO,
sanitizer, mixed-checkout, or mixed-SDK measurements, but they do not infer
whether binaries became stale after a source change.

## Run

Run the module from the rocjitsu source directory:

```bash
cd "$src"

"$python" -m benchmarks.runner \
  --build-dir "$build" \
  --output /path/to/results/run-001
```

The checked-in nightly manifest is the default. `--manifest` selects another
manifest; repeated `--target` and `--case` flags select a development subset.
`--warmups` and `--samples` override the manifest for short experiments, and
`--list` prints the selected matrix without running it. The runner refuses to
overwrite an existing output directory. Sample counts must be odd so the
reported median is always one of the observed integer-nanosecond samples.

Each cell has a 300-second default timeout. A cell failure is recorded and does
not discard results from other cells. The command returns failure when any
selected cell fails. The runner flushes a progress line when the suite and each
cell start and finish. Cell messages include matrix position, case, target,
status, elapsed wall time, and the median for a successful measurement, so a
long sequential run remains observable in local terminals and GitHub Actions.

## CI publication proof of concept

`.github/workflows/rocjitsu-benchmark-publish-poc.yml` runs the complete
`nightly.toml` matrix in two isolated lanes:

- a push to `users/IanWood1/rocjitsu-benchmark-ci-poc` exercises the workflow
  before merge and writes only to
  `users/IanWood1/rocjitsu-benchmark-results-staging`;
- a push to `develop`, or a manual dispatch whose selected ref is exactly
  `develop`, writes to `users/IanWood1/rocjitsu-benchmark-results`.

Manual dispatches of other refs are skipped. The benchmark job has a
210-minute timeout. It has read-only repository access and uploads a 14-day
diagnostic artifact containing any available `run.json` and `cases/`, plus
workflow metadata. Diagnostic staging and upload use `always()`, so setup
failures, benchmark cell timeouts, and malformed or missing raw results still
leave an Actions artifact when the job reaches those steps. The Triton cache is
deliberately excluded.

A separate publisher job is the only job with `contents: write`. It runs only
for a finalized schema-v1 nightly result tied to the workflow commit; a
finalized partial result remains publishable even though the benchmark job is
failed when every failed case already has a catalog definition or another
target completed that case. On a new data branch, a case that fails on every
target before emitting its problem metadata remains only in the diagnostic
artifact because no valid dashboard definition can be created for it. The
publisher validates the raw result again, then invokes
`python -m benchmarks.dashboard_publish` to update dashboard-ready resources
under `public/data/`. The generated run identity is
`github-<run-id>-attempt-<attempt>`, automatic and manual triggers remain
distinct, and the hosted catalog is explicitly marked as demo data. Both
results branches use dashboard branch identity `develop`; the staging branch
provides repository-level isolation for feature-run data, while the dashboard
provenance and diagnostic `workflow.txt` retain the actual triggering ref.

Publication uses the workflow's `GITHUB_TOKEN` and a signed-off
`github-actions[bot]` commit. For each non-force push attempt it discards the
previous generated commit, fetches the latest remote results branch, and reruns
the publisher against that state. It makes up to 20 attempts with bounded,
randomized backoff. This reconstructs `public/data/index.json` and any
content-addressed catalog from all current runs rather than rebasing a stale
generated index through a conflict. The results branch is created as an orphan
if it does not exist. Repository policy must allow Actions to request
`contents: write`, and branch rules must allow the bot to create and update both
POC results branches.

This workflow proves the same-repository publication and dashboard-data shape;
it does not deploy GitHub Pages. Its autoscaled `azure-linux-scale-rocm` runner
is not the reserved, tuned benchmark machine, and all hosted data is marked as
demo data, so these timings must not be mixed into official performance
history. The workflow reports the runner's CPU and cgroup allocation to make
the available host parallelism visible in each job log.

## Plugin overhead

`benchmarks/suites/plugin-overhead.toml` is a compact plugin comparison suite.
It selects the contiguous copy, boundary softmax, aligned BF16 GEMM, and
GPT-OSS grouped-query attention cases on both targets, producing eight cells
per run. Run it once for each profile, using a different output directory:

```bash
for profile in none logging race throughput; do
  "$python" -m benchmarks.runner \
    --build-dir "$build" \
    --manifest benchmarks/suites/plugin-overhead.toml \
    --plugin-profile "$profile" \
    --output "/path/to/results/plugin-$profile"
done
```

Profiles are deliberately individual rather than combinations: `none` enables
no plugin, while `logging`, `race`, and `throughput` each enable only the named
plugin. Their run configuration IDs are `plugins-<profile>-v1`. Every cell
receives a generated `config.json` derived from the
checked-in target config; the base config is never modified. Enabled profiles
route plugin output to
`cases/<case>/<target>/plugins/<plugin>.log`. A run is not successful unless
each expected report exists. Each invocation remains an independent schema-v1
run so profile results can be compared without changing benchmark identity.

## Artifacts

The output root contains schema-v1 `run.json`. It records:

- overall status, timestamps, and wall time;
- configuration identity, plugin profile/list, and the SHA-256 digest of each
  checked-in target configuration;
- provenance including the rocjitsu commit and commit timestamp, dirty state,
  validated build type and ROCm SDK path/version, Python, PyTorch, Triton, and
  installed package versions;
- host and operating-system information;
- the selected matrix and effective warmup/sample policy;
- each cell's provider, parameters, raw timings, minimum, median, maximum,
  status, artifact paths, and failure message.

Per-cell files are retained under `cases/<case>/<target>/`:

```text
workload.json
stdout.txt
stderr.txt
config.json
```

`workload.json` uses `rocjitsu.benchmark.workload.v1` and contains the case,
target, parameters, and raw synchronized timings emitted by the workload. The
generated config makes the exact target and plugin setup reproducible, and its
path is recorded in the cell's artifact map. The stdout and stderr files make
failures diagnosable without expanding the root artifact. Enabled plugin
profiles also record their report paths under `artifacts.pluginReports`. If a
workload fails before emitting JSON, `workload.json` may be absent; the logs,
config, and finalized partial `run.json` remain available.

Plugin reports are whole-process diagnostics: they include deterministic input
initialization, the compile/descriptor setup launch, warmups, and timed samples.
They are retained for investigation and are not treated as sample-scoped
metrics. Plugin overhead comparisons use the synchronized timing samples in
`workload.json`, whose boundary is unchanged across profiles.

The runner intentionally has no historical commit selector, backfill engine,
built-in comparison, or production scheduler. Historical sampling (including
commits around known performance changes), production post-submit automation,
retention, and Pages deployment remain follow-up work. The separate
`benchmarks.dashboard_publish` module converts one finalized raw run into the
versioned static resources consumed by the dashboard.

## Adding a case

Implement the fixed case in the corresponding provider and add its ID to the
suite manifest. A workload must use the public runtime path, check the reported
target, emit the workload schema, return exactly the requested positive timing
samples, and avoid autotuning or network access during a run.
