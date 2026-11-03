# rocSHMEM fixed-input link-width sweep

`sweep.py` is a Linux CI experiment, not a product-default change. Prepare the pinned rocSHMEM source and both native targets once outside timing, with all 31 architectures, `HIP_CLANG_NUM_PARALLEL_JOBS=1`, and compile-command export enabled. Keep the SDK immutable.

```sh
python3 benchmark/sweep.py \
  --source /absolute/source/projects/rocshmem \
  --build /absolute/build \
  --output /absolute/results/fresh-trials \
  --order 1,2,4,8,16,31 --targets both --repeat-best \
  --slot-lock /absolute/results/benchmark.lock \
  --expected-affinity-cpus 96 \
  --timeout-seconds 7200
```

Each width times info and functional links **separately and sequentially**, with Ninja `-j1`; only inner architecture concurrency changes. Every case requires the exact original 31 architecture IDs. Width 31 is the useful architecture-count ceiling. A wider value is rejected. The fastest passing width is selected by the sum of both link times, then rerun in `best,2,best,2` order. If width 2 wins, two additional reference trials replace the redundant four-case order. Repeat medians are reported separately from screening results; these are link-only totals, not project/CI savings or GPU runtime acceptance.

## Input and output guards

- Both product defaults remain one. Each configure changes only `ROCSHMEM_NATIVE_LINK_JOBS` in the isolated build cache.
- Compilation records must remain byte-identical. The known upstream embedded-header timestamp rewrite is suppressed only after exact content equality, with its original mtime and hash recorded.
- Link commands must differ solely by the requested `--offload-jobs=N`; width one contains no such option. Response files and compound shell actions are rejected rather than incompletely audited.
- Explicit compiler/object/archive/shared-library paths are hashed before and after every link. This does not claim complete dynamic provider closure; the supplied SDK must remain immutable.
- The dry run must contain exactly one final executable link, never compilation or other build work.
- Native executable hashes must match width one. Any mismatch is retained, excluded from selection, and gives a nonzero final exit status. Exact native parity also preserves embedded image bytes; architecture coverage is checked independently in argv.
- Profiling/diagnostic flags and environment overrides that can serialize or change driver work are rejected. `CC_PRINT_*` variables must be **unset**, not exported empty. The full environment is held as one in-memory snapshot and never dumped; rejected variable values are not printed.

## Memory and process accounting

The default one-second sampler reads Linux `/proc` for the owned process group and observed descendants, tracking PID start times against reuse. JSONL retains summed RSS, resident process/LLD counts, host available memory, cgroup remaining memory and permission gaps. CPU totals use child resource usage; observer CPU is reported separately and is not subtracted from elapsed time.

Summed RSS double-counts shared pages, samples are non-atomic, and short-lived processes or peaks can be missed. It is **not a guaranteed upper bound, unique memory, or proof that OOM cannot occur**. Resident LLD counts are not runnable-worker counts. Cgroup counters may include other work; their path and scope are recorded.

Available memory budget is the minimum of current host headroom, cgroup headroom, and optional `--max-summed-rss-gib`, after the `--min-available-gib` reserve (default 4 GiB). A 1.25x linear projection of observed RSS per width skips predicted-unsafe configurations explicitly. During execution, sampled RSS, host memory floor, dynamic cgroup memory floor and optional timeout can stop only the owned process group. Higher widths for that target are skipped after a resource/link failure. Architecture coverage is never reduced to fit memory. The projection is a conservative estimate, not a reliable hard bound.

The lock spans the complete sweep and repeats. SIGTERM and exceptions terminate the owned timed process group. Evidence output must be a fresh directory outside the build; existing run evidence is not overwritten.

## Artifacts and tests

`context.json`, `results.json`, `summary.json` and per-case configure/build logs, exact argv, input hashes, header checks, dry-run output, `memory.jsonl`, result JSON and raw binaries are retained. `failure.json` records a fatal failure only inside an output directory created by this invocation. Check skipped/rejected cases before interpreting the selected width. No automatic product adoption occurs.

```sh
python -m unittest discover -s ci-rocshmem-sweep/benchmark -p test_sweep.py -v
```

Local tests cover profiling controls, exact architecture/option checks, unexpected build work, process-tree/PID-reuse accounting, cgroup/RSS stopping, existing-output preservation, and a fully mocked 20-link both-target screening/repeat experiment. They do not constitute actual Linux compiler performance or GPU validation. Root owns CI bootstrap, workflow and publication; this directory contains benchmark code only.
