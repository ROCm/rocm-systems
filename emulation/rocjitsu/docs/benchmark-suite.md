# Automated benchmark suite

Benchmark workloads, upstream kernel sources, suite manifests, the runner, and
result publisher live in
[rocjitsu-test-corpus](https://github.com/ROCm/rocjitsu-test-corpus).
See its `benchmarks/README.md` for workload definitions, source revisions,
measurement details, plugin profiles, and result formats.

The nightly suite runs 28 cases on each of `gfx950` and `gfx1250` (56 cells),
with eight simulator threads. Sixteen cases cover FP16/BF16 GEMMs across square,
tall, wide, long-reduction, and ragged shapes. The benchmark step has a 30-minute
timeout; dependency installation, building, and publication are separate.
Compilation, allocation, and warmup happen outside measured samples. Follow
[Benchmarking rocjitsu](benchmarking.md) for official performance comparisons.

## Local setup and execution

From a rocjitsu-test-corpus checkout containing the `benchmarks` package, use
Python 3.12 and a separate Release build of rocjitsu:

```bash
corpus="$HOME/work/rocjitsu-test-corpus"
src="$HOME/work/rocm-systems/emulation/rocjitsu"
build="$HOME/work/rocjitsu-benchmark-build"
python3.12 -m venv "$corpus/.venv"
python="$corpus/.venv/bin/python"
"$python" -m pip install -r "$corpus/benchmarks/requirements.txt"
export ROCM_PATH="$("$corpus/.venv/bin/rocm-sdk" path --root)"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"

cmake -S "$src" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DLTO=OFF \
  -DRJ_ENABLE_ASAN=OFF -DRJ_ENABLE_MSAN=OFF \
  -DRJ_ENABLE_TSAN=OFF -DRJ_ENABLE_UBSAN=OFF \
  -DROCM_PATH="$ROCM_PATH" -DPython3_EXECUTABLE="$python"
cmake --build "$build" --target rocjitsu_bin rocjitsu_shared \
  rocjitsu_plugin_logging_so rocjitsu_plugin_race_so rocjitsu_plugin_throughput_so

cd "$corpus"
"$python" -m benchmarks.runner \
  --rocjitsu-source-dir "$src" --build-dir "$build" \
  --manifest benchmarks/suites/smoke.toml --output /tmp/rocjitsu-benchmark-results
```

Suite TOML files define each Triton case and its input parameters.
Use `--list` to inspect cases without building or installing GPU dependencies.

## CI publication

`.github/workflows/rocjitsu-benchmarks.yml` runs the nightly suite
against an immutable corpus revision and records that revision in results.
The publisher checks out the same corpus commit and validates clean, matching
rocjitsu and corpus provenance before publication.

Pushes and manual dispatches on `develop` publish to
`shared/rocjitsu-benchmark-results`. Manual dispatches on other branches run the
benchmarks and upload diagnostics without publishing dashboard data.

Once the workflow is present on the repository's default branch, use the Actions
Run workflow branch selector or `gh workflow run rocjitsu-benchmarks.yml
--ref <branch>` to test a branch without another code change. A workflow introduced
only by a PR cannot be manually dispatched until it exists on the default branch.

Finalized partial runs remain publishable while the benchmark job reports
failure. Diagnostics are uploaded for failed runs. The publisher writes
`data/metadata.json`, `data/index.json`, immutable catalogs under
`data/test-catalogs/`, and target-grouped executions under `data/runs/`, matching
the [dashboard contract](https://github.com/ROCm/rocm-systems/blob/64c135a1314a94d7156ccb352c9ae48b65ec59a5/emulation/rocjitsu/website/docs/website-data-contract.md).
The benchmark job targets the `rocjitsu-benchmark` runner label; publication
runs separately on `ubuntu-24.04`. The site metadata enables the Beta label,
and results record the benchmark runner's machine ID.

Publication starts a fresh dataset; the old dashboard format is not migrated.
Website build and deployment remain separate from data publication. A fresh
dataset needs a completed Vanilla run before the dashboard can display it.
