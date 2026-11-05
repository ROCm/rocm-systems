#!/usr/bin/env bash
# Fetch full GitHub Actions logs + artifacts for a run or job.
# The web UI truncates output and hides artifacts; this pulls the raw log,
# the failing steps, and any uploaded artifacts into one directory.
#
# Usage:
#   gh-ci-logs.sh <run-id | run-url | job-url> [-r owner/repo] [-o outdir] [--failed]
#
# Examples:
#   gh-ci-logs.sh 31407615392
#   gh-ci-logs.sh https://github.com/ROCm/rocm-systems/actions/runs/31407615392/job/93517652084 --failed
#
# Repo defaults to ROCm/rocm-systems (override with -r or GH_CI_REPO).
set -uo pipefail

REPO="${GH_CI_REPO:-ROCm/rocm-systems}"
OUTDIR=""
LOG_FLAG="--log"
ARG=""
need_val() { [[ $# -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 2; }; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--repo)  need_val "$@"; REPO="$2"; shift 2;;
    -o|--out)   need_val "$@"; OUTDIR="$2"; shift 2;;
    --failed)   LOG_FLAG="--log-failed"; shift;;
    -h|--help)  tail -n +2 "$0" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0;;
    -*) echo "unknown flag: $1" >&2; exit 2;;
    *)  ARG="$1"; shift;;
  esac
done

[[ -n "$ARG" ]] || { echo "error: need a run id, run URL, or job URL" >&2; exit 2; }
command -v gh >/dev/null || { echo "error: gh CLI not found" >&2; exit 2; }

# Parse run id (and optional job id) from a URL or bare id.
RUN_ID=""; JOB_ID=""
if [[ "$ARG" =~ /actions/runs/([0-9]+)(/job/([0-9]+))? ]]; then
  RUN_ID="${BASH_REMATCH[1]}"; JOB_ID="${BASH_REMATCH[3]:-}"
elif [[ "$ARG" =~ ^[0-9]+$ ]]; then
  RUN_ID="$ARG"
else
  echo "error: could not parse a run id from '$ARG'" >&2; exit 2
fi

OUTDIR="${OUTDIR:-${TMPDIR:-/tmp}/ci-${RUN_ID}}"
mkdir -p "$OUTDIR"
echo "repo=$REPO run=$RUN_ID job=${JOB_ID:-<all>} out=$OUTDIR" >&2

# Run summary: jobs, conclusion, artifact names.
# Keep stderr out of the output files so a failed call never looks like a log.
if ! gh run view "$RUN_ID" --repo "$REPO" > "$OUTDIR/summary.txt" 2> "$OUTDIR/gh.err"; then
  echo "error: cannot read run $RUN_ID in $REPO" >&2
  cat "$OUTDIR/gh.err" >&2
  rm -f "$OUTDIR/summary.txt" "$OUTDIR/gh.err"
  exit 1
fi

# gh ignores --log-failed when a single job is selected, so don't claim otherwise.
if [[ -n "$JOB_ID" && "$LOG_FLAG" == "--log-failed" ]]; then
  echo "note: --failed does not apply to a single job; writing the full job log" >&2
fi

# Full log — a single job if one was given, else the whole run.
if [[ -n "$JOB_ID" ]]; then
  LOG_FILE="$OUTDIR/job-${JOB_ID}.log"
  gh run view --repo "$REPO" --job "$JOB_ID" --log > "$LOG_FILE" 2> "$OUTDIR/gh.err"
else
  LOG_FILE="$OUTDIR/run.log"
  gh run view "$RUN_ID" --repo "$REPO" "$LOG_FLAG" > "$LOG_FILE" 2> "$OUTDIR/gh.err"
fi

if [[ ! -s "$LOG_FILE" ]]; then
  echo "warning: no log retrieved (expired or still running)" >&2
  cat "$OUTDIR/gh.err" >&2
  rm -f "$LOG_FILE"
fi
rm -f "$OUTDIR/gh.err"

# Artifacts (test reports, packages) — hidden behind the run summary page.
if gh run download "$RUN_ID" --repo "$REPO" -D "$OUTDIR/artifacts" 2>/dev/null; then
  echo "artifacts -> $OUTDIR/artifacts" >&2
else
  echo "no downloadable artifacts (or none retained)" >&2
fi

echo "$OUTDIR"
