# HRR tests

Layout, what each suite covers, and how to run it. CI runs everything here from
[`.github/workflows/hrr-ci.yml`](../../../.github/workflows/hrr-ci.yml); the commands below
are the same ones it uses.

## Layout

```text
tests/
  unit/          C++ (Catch2). hrr-unit-tests: CPU only, no GPU, no capture runtime.
  integration/   C++ (Catch2). hrr-integration-tests: real capture, then replay, on a GPU.
  config/        One YAML file per suite: every case, and where it runs.
  include/       Headers both suites include (hrr_test_common.hh, hrr_test_process.hh).
  scripts/       CI helpers that run a suite and report on it, plus the unittest
                 suites for those helpers and for ../tools. Not built.
```

`config/unit.yaml` and `config/integration.yaml` list every case the corresponding
binary must contain, and the rules for skipping one on an OS or a GPU architecture
(see [Configuring where a case runs](#configuring-where-a-case-runs)).
`integration/hrr_api_matrix_expectations.h` is generated; see
[`../tools/api-matrix`](../tools/api-matrix/README.md).

## Unit vs integration

The difference is what each suite needs and how much of HRR it exercises, not how small
the test is.

| | Unit (`hrr-unit-tests`) | Integration (`hrr-integration-tests`) |
|---|---|---|
| Exercises | One piece of HRR in-process: the archive wire format, the reader's recovery from torn or truncated archives, playback invariants such as pointer translation. | The whole chain: capture inside a real HIP runtime, write an archive, replay it with `hrr-playback`, compare the results. |
| Needs | HIP headers to build. No GPU, no capture-enabled runtime. | An AMD GPU and an `amdhip64` built with capture enabled, from the same commit as `hrr-playback`. |
| How it works | Calls the reader and playback code directly. Two recovery cases spawn `hrr-playback` for `--repair` / `--info`. | A driver case re-runs this same binary as a subprocess with `HIP_HRR_CAPTURE_OUTPUT` set, so a hidden workload runs under capture. It then runs `hrr-playback` and checks the replayed buffers byte for byte. |
| Platforms in CI | Linux and Windows. | Linux, one job per GPU family. |

Two things that are easy to trip over:

- Every case is named `Unit_HRR_*`, including the integration ones (the hip-tests naming
  convention). The prefix does not tell you which suite a case is in; the directory does.
- Most integration cases are not tests you run yourself. They are workloads the drivers
  spawn under capture, tagged `[.][hrr-direct]`. Running one by hand just executes the
  workload with no capture.

### Tags in the integration binary

| Tag | Meaning |
|---|---|
| `[hrr]` | A driver case. This is what you run. |
| `[hrr-direct]`, `[direct]` | A workload spawned by a driver as a subprocess. Do not run by hand. |
| `[.]` | Hidden. Catch2 skips it unless a tag or name selects it. |
| `[api-matrix]` | A tier of the per-API replay matrix (`T0` to `T5`), plus its host-side checks. `T0` to `T4` are also `[hrr]`, so `[hrr]~[direct]` selects them. |
| `[T5]` | The deprioritised-families tier. It has no `[hrr]` tag, so `[hrr]~[direct]` does not select it; run it explicitly by name or by this tag. |
| `[cpu]` | Needs no GPU. CI runs these even on a runner that has no device. |
| `[hrr-repro]` | A disabled reproducer. Never selected by CI. |

## Running the unit tests

```bash
cmake -S projects/hrr -B build/hrr -GNinja \
  -DROCM_PATH="${ROCM_PATH:-/opt/rocm}" \
  -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/opt/rocm}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DHRR_BUILD_PLAYBACK=ON \
  -DHRR_BUILD_TESTS=ON \
  -DHRR_BUILD_INTEGRATION_TESTS=OFF
cmake --build build/hrr
./build/hrr/tests/unit/hrr-unit-tests
```

`HRR_BUILD_PLAYBACK=ON` builds the in-tree `hrr-playback`, which two recovery cases
spawn. With it off, those cases need an `hrr-playback` on `PATH`.

If the SDK's HIP headers are older than this tree, pass
`-DHRR_HIP_INCLUDE_DIRS="<repo>/projects/hip/include;<repo>/projects/clr/hipamd/include"`
as CI does.

## Running the integration tests

1. Build `amdhip64` with capture compiled in. The tree has no CMake option for it, so
   pass the define:

   ```bash
   cmake -S projects/clr -B build/clr -GNinja \
     -DCMAKE_CXX_FLAGS=-DHIP_HRR_CAPTURE_ENABLED \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/opt/rocm}" \
     -DROCM_PATH="${ROCM_PATH:-/opt/rocm}" \
     -DHIP_COMMON_DIR="$PWD/projects/hip" \
     -DHIP_PLATFORM=amd -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF
   cmake --build build/clr --target amdhip64
   ```

2. Configure the tests as above with `-DHRR_BUILD_INTEGRATION_TESTS=ON`. Add
   `-DGPU_TARGETS=<arch>` and `-DCMAKE_HIP_ARCHITECTURES=<arch>` for the GPU you will
   run on; the workloads carry device kernels. Build `build/hrr` from the same commit
   as `build/clr`, or the generated payload layouts may disagree.

3. Put the capture build ahead of the SDK's runtime, then run the drivers the way CI
   does. The wrapper reads `config/integration.yaml`, detects the GPU, and skips the
   cases a rule names. The binary on its own ignores the config:

   ```bash
   export LD_LIBRARY_PATH="$PWD/build/clr/hipamd/lib:${ROCM_PATH:-/opt/rocm}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
   python3 projects/hrr/tests/scripts/run_catch2.py --xml results.xml --timeout 1200 \
     --config projects/hrr/tests/config/integration.yaml --arch auto \
     -- ./build/hrr/tests/integration/hrr-integration-tests
   ```

   CMake bakes the absolute paths of the integration binary and `hrr-playback` into
   the build, and the drivers re-run both by those paths, so do not move the build
   tree after building. To run one driver, name it:
   `./hrr-integration-tests Unit_HRR_GraphRoundtrip`. With no GPU, run only the
   CPU-only checks: `./hrr-integration-tests "[cpu]"`.

`HRR_TEST_IGNORE_CONFIG=1`, or `--ignore-config`, runs the cases the config would
skip. `--arch gfx1100` (or several, comma-separated) stands in for detection.

## Testing the Python helpers

`scripts/` holds both the helpers and their tests. The tests need only Python and
PyYAML (used by the api-matrix tools), not ROCm:

```bash
python3 -m unittest discover -s projects/hrr/tests/scripts -p 'test_*.py' -v
```

They cover `run_catch2.py`, `summarize_junit.py`, and `hrr_test_config.py`, plus
the tools in `../tools`: the HRR API ID compatibility check, the playback-handler default
in the generator, and the [api-matrix](../tools/api-matrix/README.md) scripts.

## Configuring where a case runs

The build is one fat binary for every architecture in `HRR_ARCHS`. Nothing about a
GPU is decided at build time. CI decides at run time, from
[`config/integration.yaml`](config/integration.yaml) (and
[`config/unit.yaml`](config/unit.yaml) for the CPU suite).

An empty entry runs on every OS and every GPU. A rule skips that case, or with a
suite-level `skip` every case, when every key under `when` matches. Keys are ANDed,
a list is ORed, a missing key matches anything, and a value may be a glob
(`gfx11*` is a family). `kind: disabled` is a temporary regression and requires a
public `https://` issue URL. `kind: unsupported` is permanent and needs only a
`reason`. On a machine with several GPUs a rule matches when it matches any of them.
`--arch none` is a run with no GPU, so an architecture rule does not match it.

One case, on one OS and one architecture:

```yaml
Unit_HRR_Regions_MemcpyFirstTouchMaterialization:
  skip:
    - when: {os: linux, arch: gfx1151}
      kind: disabled
      reason: HSA allow_access rejects the NPU agent
      issue: https://github.com/ROCm/rocm-systems/issues/1
```

Every case on one architecture, until a runner exists:

```yaml
skip:
  - when: {arch: gfx1250}
    kind: unsupported
    reason: no healthy MI455 runner yet
```

Direct workloads (`[hrr-direct]`, `[direct]`) are spawned by a driver and must not
carry rules. `targets.arch` is the same list as `HRR_ARCHS` in
`.github/workflows/hrr-ci.yml`; `hrr_test_config.py lint --workflow` checks that.

CI compares each binary's registered cases against its YAML, in both directions,
because a case behind an `#if` or a missing define disappears without any failure.
When you add or remove a case, regenerate the names from a built binary and say so
in the commit message. Existing rules are kept:

```bash
python3 projects/hrr/tests/scripts/hrr_test_config.py check \
  --binary build/hrr/tests/unit/hrr-unit-tests \
  --config projects/hrr/tests/config/unit.yaml --update
```

The same applies to the integration binary and `config/integration.yaml`. Listing
needs no GPU. A manual run can ignore the rules with
`workflow_dispatch` input `ignore_test_config`, which sets
`HRR_TEST_IGNORE_CONFIG`.
