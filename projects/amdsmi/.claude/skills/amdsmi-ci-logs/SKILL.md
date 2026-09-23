---
name: amdsmi-ci-logs
description: "Use when a GitHub Actions run or job for an amd-smi / rocm-systems PR failed and the web UI shows truncated output, when you need the full failing log, or when you need CI artifacts (test reports, packages). Triggers: an actions/runs/<id>/job/<id> URL, 'where are the logs', 'output is truncated', 'where are the artifacts', a red check on a PR."
---

# CI Logs — amd-smi (GitHub Actions)

Get the **full** log and artifacts for a rocm-systems Actions run when the web UI
truncates output. The failing assertion and the uploaded test reports live in the
raw log / artifacts, not the truncated job view.

**Not** the internal Gerrit/Zuul CI (`gerrit.core.linux.amd.com`) — that's a
separate system.

## When to Use

- A pasted `.../actions/runs/<RUN_ID>/job/<JOB_ID>` URL
- "The unit-test output is truncated / cut off"
- "Where do I see if there are artifacts?"
- A red check on a PR and you need the failing step's full text

## From a URL or run id

A job URL encodes both ids: `.../actions/runs/<RUN_ID>/job/<JOB_ID>?pr=<PR>`.

```bash
REPO=ROCm/rocm-systems
gh run view <RUN_ID> --repo $REPO                 # summary: jobs, conclusion, artifacts
gh run view <RUN_ID> --repo $REPO --log-failed    # only failed steps (fastest triage)
gh run view --repo $REPO --job <JOB_ID> --log     # one job, full log
```

`--log-failed` only filters at the **run** level. Combined with `--job` it is
silently ignored and you get the whole job log.

A rocm-systems run is big (a failed-only run log can exceed 80k lines), so
redirect to a file and grep. Never `head -n` a run log for triage: the amdsmi
failure is usually far below the first screenful of some other project's output.

```bash
gh run view <RUN_ID> --repo $REPO --log > "${TMPDIR:-/tmp}/ci-<RUN_ID>.log"
grep -nE 'FAILED|Error|assert' "${TMPDIR:-/tmp}/ci-<RUN_ID>.log"
```

## Artifacts (test reports, packages)

Artifacts are **not** in the job log view — download them:

```bash
gh run view <RUN_ID> --repo $REPO                          # lists artifact names
gh run download <RUN_ID> --repo $REPO -D "${TMPDIR:-/tmp}/ci-<RUN_ID>"
gh run download <RUN_ID> --repo $REPO -n <name> -D <dir>   # one named artifact
```

## From a PR number (no URL yet)

```bash
gh pr checks <PR#> --repo $REPO                   # checks + run links
BR=$(gh pr view <PR#> --repo $REPO --json headRefName -q .headRefName)
gh run list --repo $REPO --branch "$BR" --limit 10
```

## Helper

`gh-ci-logs.sh <run-id|run-url|job-url> [--failed] [-r owner/repo] [-o outdir]`
parses the id(s), writes the summary, full log, and artifacts under
`$TMPDIR/ci-<run-id>/`, and prints that directory. Repo defaults to
`ROCm/rocm-systems` (override with `-r` or `GH_CI_REPO`). It exits non-zero if
the run can't be read, so a 404 never lands in the log file as if it were output.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Reading the truncated web/job view | `--log` / `--log-failed` gives the full text |
| `head -200` on a run log to triage | The amdsmi failure is often tens of thousands of lines in — `grep` instead |
| Expecting `--log-failed` to filter a single job | It only filters at run level; job logs come back whole |
| Looking for test reports in the log | They're **artifacts** — `gh run download` |
| Dumping a huge log inline | Redirect to `$TMPDIR`, then `grep` |
| Guessing which run belongs to a PR | `gh pr checks <PR#>` links the exact run |
| Using this for internal Gerrit/Zuul CI | Wrong CI — different system, different tooling |
