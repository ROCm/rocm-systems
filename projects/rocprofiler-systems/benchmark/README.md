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
| `docs/` | Design and feature description (`feature_description.md`), runnable `examples/` |
| `resource/scenarios/<group>/` | Scenario definitions (inputs) |
| `resource/results/<group>/` | Result JSON per scenario (outputs), mirrors `scenarios/` |
| `resource/snapshots/` | Suite files and `results/snapshot-<date-time>/` runs of a whole suite |

## Conventions

- A scenario's `workload` is relative to the scenario file and points into `build/`
  (`../../../build/<name>` from `resource/scenarios/<group>/`).
- A scenario's `run.output` goes to `resource/results/<group>/` (`../../results/<group>/<name>.json`).
- `rocprofsys-*-output/` directories are profiler output written to the current
  directory; they are git-ignored. Run from a scratch directory to keep the tree clean.

See `docs/feature_description.md` for the measurement model and `workloads/README.md` for the programs.
