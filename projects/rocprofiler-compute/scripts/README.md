# Metric health reports

The metric-health utilities originate from
`users/feizheng10/aiprofcomp-865-health-utils`. This branch includes the
reporting utilities, not the source branch's single-pass allocator.
`single-pass` is the historical default run label; it does not enable that
allocator.

On a gfx1201 host, build or install this checkout's `rocprof-compute` as
described in the [project README](../README.md), then run:

```bash
bash scripts/run_metric_health_test_report.sh \
  --arch gfx1201 --therock /path/to/rocm --tag gfx1201-test \
  --iterations 500 --log /path/to/metric-health.log
```

Run this from the rocprofiler-compute project directory. The worker prefers
`install/bin/rocprof-compute` under the checkout, then other installed
executables, before falling back to the checkout's source. Check the logged
`rocprof-compute:` line to confirm which executable was used; an unrelated
installed version may not contain the gfx1201 changes. The worker builds
sample binaries in the checkout, runs three workloads, and writes under
`workloads/`. It uses `--overwrite` for the named workloads and truncates the
specified log, so choose a new tag and log path to preserve previous runs.
The default 500 iterations can take substantial GPU time. `--iterations 1`
is useful for a smoke test, not a median-stable health report.

The script reports `ALL_DONE` only if all required profiling and analysis
steps succeeded. Inspect the log for warnings, including missing counters.
gfx1201 WMMA collection may require an aqlprofile with event 233 enabled.
The report uses medians where available, and scalar values for Value-only
panels. gfx1201 Avg-only metrics without a Median or Min formula are shown as
N/A rather than mislabeled as medians; consult the analyze logs for their Avg.

For remote execution, copy [.metric_health_sites.env.example](.metric_health_sites.env.example)
to `.metric_health_sites.env`, fill in the `GFX1201_*` SSH settings, and run:

```bash
bash scripts/run_metric_health_test_report_local_driver.sh \
  --arch gfx1201 --tag gfx1201-test
bash scripts/run_metric_health_test_report_local_driver.sh \
  --arch gfx1201 --tag gfx1201-test --wait --fetch --report
```

The site file is ignored by Git and excluded from synchronization. The
driver defaults to SSH key authentication. A report and its fetched artifacts
are written under `validation-artifacts/metric-health-gfx1201-test/`.

For a run performed directly on the GPU host, arrange its files in the same
artifact layout and generate the report:

```bash
tag=gfx1201-test
artifacts="validation-artifacts/metric-health-${tag}"
mkdir -p "$artifacts/logs/gfx1201" "$artifacts/gfx1201"
cp -a "workloads/logs_gfx1201_${tag}/." "$artifacts/logs/gfx1201/"
cp -a "workloads/vcopy_${tag}_single-pass" \
  "workloads/nbody_${tag}_single-pass" \
  "workloads/mega_kernel_${tag}_single-pass" "$artifacts/gfx1201/"
python3 tools/generate_metric_health_report.py --arch gfx1201 \
  -p "$artifacts" --baseline-run single-pass --baseline-label "this branch" \
  --iterations 500 --out "$artifacts/gfx1201_health.html"
```

For a comparison, first collect and package a second run with `--run develop`
and `--profiler-root` pointing to an installed develop checkout. Pass the
baseline artifact directory with `--baseline` to the remote driver, or pass
`-p <baseline> -p <candidate>` to the report generator. A baseline without
gfx1201 support cannot provide a meaningful gfx1201 comparison.
