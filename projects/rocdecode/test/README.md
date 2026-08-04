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
to `validation_results/<timestamp>/`.

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

<!-- Add future skills here as new "### /<skill-name> — <summary>" subsections. -->

