# benchmark-suite

A/B benchmark of command overhead (wall time, CPU time, peak RSS): run a
workload with and without a profiler such as `rocprof-sys-run` and compare.

## Quick start

```bash
./setup.sh                       # venv, `bench*` CLIs, workloads into ./build
source .venv/bin/activate
bench-scenario resource/scenarios/fixed_cost/noop_fixed_cost.json
bench-snapshot resource/snapshots/snapshot-suite.json     # run a whole suite
bench-snapshot compare <snapshot A> <snapshot B>          # exit 1 on a real regression
pytest                           # Python tests
```

## Layout

| Path | Contents |
| --- | --- |
| `src/bench/` | The Python package: CLI, runner, analysis, reports, scenario and snapshot handling, JSON schema |
| `workloads/` | C++ synthetic test programs (CMake, gtest in `workloads/tests/`); built into `build/` |
| `tests/` | pytest suite for `src/bench`, including a check that every shipped JSON file is valid |
| `resource/scenarios/<group>/` | Scenario definitions (inputs) |
| `resource/configs/` | rocprof-sys config files (`<tool>.cfg`) shared by the scenarios |
| `resource/results/<group>/` | Result JSON per scenario (outputs), mirrors `scenarios/` |
| `resource/snapshots/` | Suite files and `results/snapshot-<date-time>/` runs of a whole suite |

## Conventions

- A scenario's `workload` is relative to the scenario file and points into `build/`
  (`../../../build/<name>` from `resource/scenarios/<group>/`).
- A scenario's `run.output` goes to `resource/results/<group>/` (`../../results/<group>/<name>.json`).
- A tool's settings live in a shared `resource/configs/<tool>.cfg`. The scenario points at
  it with a `tool_env` entry `ROCPROFSYS_CONFIG_FILE` (relative to the scenario file, `:`-separated
  list allowed) and lists only what differs, such as `ROCPROFSYS_OUTPUT_PATH`. An env var
  beats the same key in the file, so `bench-scenario` and `bench-snapshot` refuse to start
  when a variable exported in your shell is also a key in a config file the scenario loads
  and the scenario does not set it in `tool_env`. The profiler aborts on a key in a config file that its build
  does not know, so a setting that exists only in some builds (for example
  `ROCPROFSYS_USE_MPI`) stays in `tool_env`, where an unknown variable is ignored. Put
  comments on their own line: an inline `#` is not stripped from text config files.
- `rocprofsys-*-output/` directories are profiler output written to the current
  directory; they are git-ignored. Run from a scratch directory to keep the tree clean.

The LEGEND section of every report explains the measured values; see `workloads/README.md` for the programs.
