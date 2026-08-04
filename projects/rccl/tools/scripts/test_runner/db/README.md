# Test Runner -> results dashboard: results emission & scrape contract

This documents how `test_runner.py` emits results and how the results
dashboard consumes them.

## Two writers, one schema

Both paths write the same schema (`schema.sql`) and are idempotent on
`runs.run_id`, so a run may be delivered by either or both without duplication:

1. `test_runner.py --db-push` — best-effort direct push from the test host.
2. Dashboard-side ingest job — reads scraped `*.tar.gz` and upserts.

`run_id` is generated once per invocation as `YYYYMMDDThhmmssZ-<8hex>`.

Runs may carry **tags** (`--tag`/`--tags`) for dashboard filtering. Tags are set
at emit time; on re-ingest they are left untouched, so once a run is in the DB
its tags are mutable only via the dashboard (admin-gated).

## Enabling emission

```bash
# Local files only (durable; feeds the hourly scrape):
python3 test_runner.py -c configs/rccl_perf_tests.json --emit-results

# Local files + best-effort direct DB push:
export RCCL_RESULTS_DSN='postgresql://rccl_writer:***@dash-host:5432/rccl'
python3 test_runner.py -c configs/rccl_perf_tests.json --db-push
```

- `--db-push` implies `--emit-results`.
- If the DB push fails/times out, the run still succeeds and the local tarball is
  retained for the scrape. The DSN is read only from `RCCL_RESULTS_DSN`; it is
  never hardcoded or written into the emitted files.
- Enabling emission also turns on per-test **log capture** (via `tee`), which is
  what lets the emitter parse `busbw`/`algbw` from rccl-tests output. Default
  behaviour (no capture) is unchanged when emission is off.

## Output layout (`--results-dir`, default `<workspace>/results`)

```
results/
  <run_id>/
    run.json         # run manifest (sha, host, config, env, summary, coverage, tags)
    tests.jsonl      # one line per test result
    perf.jsonl       # one line per (size, place) perf row, with collective/avg
    coverage.json    # llvm-cov TOTAL percentages (if --coverage-report)
    logs/            # captured per-test logs (+ coverage rawfiles)
    report/          # llvm-cov HTML + function_coverage_report.txt (if generated)
  <run_id>.tar.gz    # self-contained snapshot of <run_id>/
  latest.tar.gz      # copy of the most recent snapshot (stable scrape target)
  latest.json        # most recent run manifest
  index.jsonl        # append-only: one summary line per run
```

## Scrape contract (hourly cron + scp)

The dashboard host pulls new snapshots on a schedule and ingests them:

```cron
# On the dashboard host. Pulls any new run tarballs, then ingests.
0 * * * * rsync -az --ignore-existing \
    testhost:/path/to/results/*.tar.gz /var/lib/rccl-dashboard/incoming/ \
    && /var/lib/rccl-dashboard/ingest.py /var/lib/rccl-dashboard/incoming
```

Notes:
- `rsync --ignore-existing` (or `scp` of only-new files) makes the pull
  idempotent; the ingest job is also idempotent on `run_id`.
- Scrapers that only want the newest run can grab `latest.tar.gz` / `latest.json`.
- `index.jsonl` lets a scraper enumerate historical runs without untarring.

## Applying the schema

```bash
psql "$RCCL_RESULTS_DSN" -f schema.sql
```
