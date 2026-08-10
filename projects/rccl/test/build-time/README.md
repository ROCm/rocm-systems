# RCCL build-time regression harness

Clean-builds RCCL and fails when a change makes the build meaningfully slower.
`build_time.py` holds the measurement and a standalone CLI; `tests/` is a thin
pytest harness over the same functions so CI gets one reported result per GPU
target with proper skip/fail semantics.

## Two gates

**Comparison (what CI runs).** Builds the PR's base commit and its head on the
same runner and fails if head is more than `--max-regression-pct` slower. A
ratio cancels out machine speed, so it needs no per-machine tuning. Rounds
alternate base/head and the minimum of each wins, because build-time noise is
one-sided: interference can only ever make a build slower.

**Absolute.** Builds head only and fails if it exceeds `--threshold-sec`. Wall
clock is only meaningful relative to a known machine — a clean single-arch build
is around a minute on a 256-core host but will blow past five minutes on a small
runner — so this is for a runner whose budget someone has calibrated.

## Running it

Standalone, no pytest required:

```bash
./build_time.py --local-gpu --compare-base --max-regression-pct 10
./build_time.py --targets gfx942 --threshold-sec 600 --jobs 32
./build_time.py --help
```

Through pytest, which is how CI invokes it:

```bash
python -m venv venv && ./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python -m pytest -m compare --local-gpu --compare-base \
    --max-regression-pct 10 --repeat 2 --jobs 64
```

Each option also reads a default from the matching `RCCL_BUILD_TIME_*`
environment variable (`RCCL_BUILD_TIME_JOBS`, `RCCL_BUILD_TIME_REPEAT`,
`RCCL_BUILD_TIME_MAX_REGRESSION_PCT`, `RCCL_BUILD_TIME_TARGETS`,
`RCCL_BUILD_TIME_UPSTREAM`, `RCCL_BUILD_TIME_STRICT`, ...), so a runner config
can drive it through either `test_filter` arguments or `env_variables`.

Note that pytest already defines `--strict`, so the option that turns
unbuildable archs into failures is spelled `--strict-targets` here. The
standalone CLI still calls it `--strict`.

## What skips and what fails

| Situation | Result |
|---|---|
| Arch this ROCm toolchain cannot build | skip (`--strict-targets` makes it fail) |
| Base commit unreachable, e.g. shallow CI checkout | skip |
| Base and head are the same commit | skip |
| `--base-rev` names a revision that does not resolve | fail |
| hipcc missing or unable to compile HIP | fail |
| A clean build fails (after one retry) | fail |
| Head slower than the tolerance | fail |

The skips are deliberate: the RCCL default arch list runs ahead of released
ROCm, and a shallow checkout has no base history to compare against. Reporting
either as a build-time regression would be a lie. Because the runner maps an
all-skipped harness to SKIPPED rather than PASSED, a runner that silently stops
comparing is visible instead of showing green.

## Cost

Each measured target is a from-scratch configure plus build, and the comparison
gate runs `2 * --repeat` of them. Budget minutes per target, and note that
`ccache` is force-disabled (`CCACHE_DISABLE=1`) — a warm cache would make the
second half of an A/B comparison finish instantly and silently invalidate the
result.

## CI wiring

`tools/scripts/test_runner/configs/ci-precheckin.json` runs the comparison gate
as a pytest suite:

```json
"build_time": {
  "is_pytest": true,
  "setup_venv": true,
  "test_dir": "test/build-time",
  "num_gpus": 1,
  "timeout": 2400,
  "tests": [
    {
      "name": "BuildTime.LocalGpuArch",
      "test_filter": "-m compare --local-gpu --compare-base --max-regression-pct 10 --repeat 2 --jobs 64"
    }
  ]
}
```

`test_filter` starting with `-` is passed through to pytest as raw arguments
rather than being turned into a `-k` expression, which is what lets the gate's
options travel in the config.

CMake also registers an opt-in CTest (`-DENABLE_BUILD_TIME_TEST=ON`, target
`rccl-build-time`) that invokes `build_time.py` directly, so that path needs no
pytest.
