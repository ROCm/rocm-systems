# rocDecode Test & Validation Scripts

Developer-facing helpers for building the samples and running a full local validation
sweep before opening a pull request. These wrap the CTest suite and the Python
conformance runners in [`testScripts/`](testScripts/README.md).

All commands are shown relative to the rocDecode project root (the directory containing
`build/`, `samples/`, and `test/`).

## Prerequisites

* `ROCM_PATH` pointing at your ROCm install — the build derives its compiler, install
  prefix, and CTest data path from it (default `/opt/rocm`). If ROCm is elsewhere (e.g. a
  TheRock or custom build), set it. Because Claude Code skills run cmake in a
  **non-interactive** shell that does not source `~/.bashrc`, set it in `~/.profile`:

  ```shell
  export ROCM_PATH="$HOME/TheRock_XXXX"
  ```

  `ROCM_PATH` must also be **writable** — `make install` installs into it, so a system
  `/opt/rocm` would need sudo (which the skill does not use); prefer a user-writable
  install such as a TheRock build in `$HOME`. `test/validate.sh --check-rocm` verifies both
  (compiler present + prefix writable) and fails fast with guidance.
* rocDecode built and installed (`make install`) — the CTest cases and conformance
  runners test against the installed library and data.
* [FFmpeg](https://ffmpeg.org/about.html) dev libraries — required for the samples and
  for the extended CTest cases:

  ```shell
  sudo apt install libavcodec-dev libavformat-dev libavutil-dev
  ```

* Python 3 with `pandas` and `tabulate` (used by the conformance runners):

  ```shell
  python3 -m pip install pandas tabulate
  ```

## `build_samples.sh` — build all sample apps

Configures and cleanly rebuilds every sample app under `samples/` (a `clean` target build
followed by a fresh build) into its own `build/` subdirectory. Uses `cmake --build`, so it
works regardless of the configured CMake generator (Make, Ninja, etc.).

```shell
test/build_samples.sh [JOBS]
```

* `JOBS` — parallel build jobs (optional, default: `nproc`).

## `validate.sh` — full validation sweep

Runs the CTest suite followed by the per-codec conformance tests, then prints a combined
summary box and exits `0` if everything passed, `1` otherwise. Per-phase logs are written
to `$HOME/rocDecode_validation_results/<timestamp>/` (outside the repo, to keep `git
status` clean; override the base with `ROCDECODE_VALIDATION_RESULTS_DIR`).

```shell
test/validate.sh [--build-dir DIR] [--skip-ctest] [--skip-conformance]
```

| Option | Description |
| --- | --- |
| `--build-dir DIR` | Path to the CMake build directory (default: `build`). |
| `--skip-ctest` | Skip the CTest phase. |
| `--skip-conformance` | Skip all conformance phases. |
| `-h`, `--help` | Show usage and exit. |

### Conformance stream location

Conformance streams are located via the `ROCDECODE_CONFORMANCE_DIR` environment variable
(default: `$HOME/rocDecodeConformance`). That directory must contain one subdirectory per
codec:

```
$ROCDECODE_CONFORMANCE_DIR/
├── AvcConformance/
├── Av1Conformance/
├── HevcConformance/
└── Vp9Conformance/
```

Point the variable at your local stream collection, e.g. add to your shell profile:

```shell
export ROCDECODE_CONFORMANCE_DIR="$HOME/Movies/rocDecodeConformance"
```

Codecs whose directory is missing are reported as `WARNING` and skipped, so the sweep
still runs (and CTest still passes) without the full data set.

## `perf_regression.py` — performance regression check

Measures decode FPS (running the `videoDecodePerf` sample on each stream) and compares it
against the GPU-specific baseline, flagging streams whose Avg FPS has dropped beyond the
tolerance. Prints a summary box, writes a per-stream comparison CSV to
`$HOME/rocDecode_perf_results/<timestamp>/` (outside the repo; override the base with
`ROCDECODE_PERF_RESULTS_DIR`), and exits `0` if there are no regressions, `1` otherwise.

```shell
test/perf_regression.py [--perf-dir DIR] [--baseline FILE] [--tolerance PCT] [--runs N] [--device ID] [--quick] [--check-rocm]
```

| Option | Description |
| --- | --- |
| `--perf-dir DIR` | Streams + baseline dir (default: `$ROCDECODE_PERF_DIR`). |
| `--baseline FILE` | Baseline HTML (default: `<perf-dir>/rocDecode_perf_baseline.html`). |
| `--tolerance PCT` | Regression threshold, % Avg FPS drop (default: `$ROCDECODE_PERF_TOLERANCE` or 5). |
| `--runs N` | Runs to average when confirming a flagged stream (default: 3). |
| `--device ID` | GPU device id (default: 0). |
| `--quick` | Fast check: one baseline stream per leaf subfolder, capped at ≤4K (8K skipped). |
| `--check-rocm` | Only verify the ROCm toolchain (`ROCM_PATH`) and exit. |

### GPU detection

The local GPU is detected (via `amd-smi` market name, then `rocm-smi`, then KFD
`gfx_target_version`) and mapped to the matching baseline column: `MI250X`, `MI300X`,
`MI300A`, `MI350`, `MI355`, `Navi31`, or `Navi48`. Because MI300X/MI300A share `gfx942` and
MI350/MI355 share `gfx950`, detection prefers the market name. Override with
`ROCDECODE_PERF_GPU=<column>` if needed.

### Perf stream + baseline location

Streams and the baseline file are located via the `ROCDECODE_PERF_DIR` environment variable
(default: `$HOME/rocDecodePerformance`), containing one subdirectory per codec plus the
baseline HTML:

```
$ROCDECODE_PERF_DIR/
├── AvcPerformance/
├── Av1Performance/
├── HevcPerformance/
├── Vp9Performance/
└── rocDecode_perf_baseline.html   # download from SharePoint
```

Point the variable at your local collection, e.g. add to your shell profile:

```shell
export ROCDECODE_PERF_DIR="$HOME/Movies/rocDecodePerformance"
```

The baseline is not distributed with the repo (it lives on an internal SharePoint site);
download `rocDecode_perf_baseline.html` into `$ROCDECODE_PERF_DIR` before running.

## Claude Code Skills

For developers using [Claude Code](https://claude.com/claude-code), the skills under
`.claude/skills/` wrap common rocDecode workflows so they can be invoked by name (e.g.
`/validate`). Each skill is defined by a `SKILL.md` file and typically drives the shell
scripts documented above.

The list of available skills will grow over time (for example, a future performance-test
skill). Add each new skill as its own subsection below.

### Common setup

These notes apply to every skill in this project:

* **Run from the project root** — start the Claude Code session in the rocDecode project
  root so the skill and its relative script paths resolve correctly.
* **Restart to pick up changes** — skills are discovered at session startup, so a freshly
  added or edited skill is only available after restarting the session.
* **Environment variables** — any variable a skill relies on (such as
  `ROCDECODE_CONFORMANCE_DIR`) must be visible to non-interactive shells. Set it in
  `~/.profile` or export it before launching Claude Code; otherwise the skill falls back to
  its default.

### `/validate` — full build and test sweep

Defined in `.claude/skills/validate/SKILL.md`. Runs the entire pipeline end to end:

1. `cmake -B build -DENABLE_EXTENDED_TESTS=ON` — configure the build (creates `build/` on a
   fresh checkout; enables the FFmpeg-based extended CTest cases).
2. `make clean -C build`, `make -j -C build`, `make install -C build` — rebuild and install.
3. `test/build_samples.sh` — build all sample apps.
4. `test/validate.sh` — run CTest + conformance and report the summary.

Invoke it by typing `/validate` in a Claude Code session started from the project root.

### `/perf-check` — performance regression check

Defined in `.claude/skills/perf-check/SKILL.md`. Measures decode FPS and compares it to the
GPU-specific baseline:

1. `test/perf_regression.py --check-rocm` — verify the ROCm toolchain (`ROCM_PATH`).
2. `cmake -S samples/videoDecodePerf -B samples/videoDecodePerf/build` then
   `cmake --build samples/videoDecodePerf/build` — build the performance sample.
3. `test/perf_regression.py` — detect the GPU, run the perf test, compare to baseline, and
   report the summary.

Invoke it by typing `/perf-check` in a Claude Code session started from the project root,
or `/perf-check quick` for the fast one-stream-per-subfolder variant (`--quick`, ≤4K only).
Requires
`ROCDECODE_PERF_DIR` (streams + baseline) as described under
[`perf_regression.py`](#perf_regressionpy--performance-regression-check) above.

<!-- Add future skills here as new "### /<skill-name> — <summary>" subsections. -->

