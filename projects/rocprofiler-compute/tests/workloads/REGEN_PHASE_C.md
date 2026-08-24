# Golden workload regeneration for Phase C

Golden workloads in this directory still carry `results_*.csv.gz` and `pmc_perf.csv.gz`.
Re-profile each arch on matching hardware, then commit the new `out/{pass}/` trees.

Delete this file once regeneration is complete.

## Checklist per arch

1. Re-profile on matching hardware.
2. Delete `results_*.csv.gz` and `pmc_perf.csv.gz` from each workload directory.
3. Confirm `out/` artifacts stage correctly:

```bash
git check-ignore -v tests/workloads/<name>/<arch>/out/<pass>/<pid>/<pid>.db
# should match the negation rule in .gitignore, not '**/out'
```

4. Verify:

```bash
python3 -m pytest tests/integration/test_analyze_workloads.py -k <ARCH> -q
```

## Expected layout

```
tests/workloads/<name>/<arch>/
├── out/
│   └── <pass>/
│       ├── <pid>_native_counter_collection.csv.gz
│       └── <pid>/
│           └── <pid>.db
├── profiling_config.yaml
├── sysinfo.csv
└── timestamps.csv
```

## Hardware

MI100, MI200, MI300A, MI300A_A1, MI300X_A1, MI350, and RDNA35_HALO.
