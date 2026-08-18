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

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--repo)  REPO="$2"; shift 2;;
    -o|--out)   OUTDIR="$2"; shift 2;;
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
gh run view "$RUN_ID" --repo "$REPO" > "$OUTDIR/summary.txt" 2>&1 || true

# Full log — a single job if one was given, else the whole run.
if [[ -n "$JOB_ID" ]]; then
  gh run view --repo "$REPO" --job "$JOB_ID" "$LOG_FLAG" > "$OUTDIR/job-${JOB_ID}.log" 2>&1 || true
else
  gh run view "$RUN_ID" --repo "$REPO" "$LOG_FLAG" > "$OUTDIR/run.log" 2>&1 || true
fi

# Artifacts (test reports, packages) — hidden behind the run summary page.
if gh run download "$RUN_ID" --repo "$REPO" -D "$OUTDIR/artifacts" 2>/dev/null; then
  echo "artifacts -> $OUTDIR/artifacts" >&2
else
  echo "no downloadable artifacts (or none retained)" >&2
fi

echo "$OUTDIR"
