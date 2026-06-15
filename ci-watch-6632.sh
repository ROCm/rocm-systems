#!/usr/bin/env bash
#
# CI watch for a PR: rerun flaky CI failures once a workflow run is terminal.
#
# For each watched workflow run it: checks whether the run is completed, and if
# so, counts the failed jobs (ignoring "(xfail)" jobs for TheRock CI) and reruns
# all failed jobs in parallel via `gh run rerun --failed`.
#
# The xfail guard prevents an infinite loop: if the ONLY failures are expected
# "(xfail)" jobs, nothing is rerun.
#
# Usage:
#   ./ci-watch-6632.sh                 # one cycle
#   ./ci-watch-6632.sh --watch         # loop every 30 min until Ctrl-C
#   ./ci-watch-6632.sh --watch -i 600  # loop every 10 min
#   PR=1234 REPO=owner/repo ./ci-watch-6632.sh
set -euo pipefail

PR="${PR:-6632}"
REPO="${REPO:-ROCm/rocm-systems}"
INTERVAL="${INTERVAL:-1800}"
WATCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch) WATCH=1 ;;
    -i | --interval)
      INTERVAL="$2"
      shift
      ;;
    -h | --help)
      grep '^#' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
  shift
done

# Extract the most recent run id for a given workflow name from the PR rollup.
run_id_for() {
  local workflow="$1"
  gh pr view "$PR" --repo "$REPO" --json statusCheckRollup --jq \
    "[.statusCheckRollup[] | select(.workflowName==\"${workflow}\") | .detailsUrl]
     | map(capture(\"runs/(?<id>[0-9]+)\").id) | last // empty"
}

# Number of failed jobs in a run. With guard=1, "(xfail)" jobs are excluded.
failed_count() {
  local run_id="$1" guard="$2" filter='.conclusion=="failure"'
  [[ "$guard" == "1" ]] && filter+=' and (.name|contains("(xfail)")|not)'
  gh run view "$run_id" --repo "$REPO" --json jobs --jq \
    "[.jobs[] | select(${filter})] | length"
}

# Check one run; rerun its failures if terminal and there is a real failure.
process_run() {
  local label="$1" workflow="$2" guard="$3" run_id status nfail

  run_id="$(run_id_for "$workflow")"
  if [[ -z "$run_id" ]]; then
    echo "  ${label}: no run found"
    return
  fi

  status="$(gh run view "$run_id" --repo "$REPO" --json status --jq .status)"
  if [[ "$status" != "completed" ]]; then
    echo "  ${label} (${run_id}): ${status} — no action"
    return
  fi

  nfail="$(failed_count "$run_id" "$guard")"
  if [[ "$nfail" -eq 0 ]]; then
    echo "  ${label} (${run_id}): terminal, no real failures — no action"
    return
  fi

  echo "  ${label} (${run_id}): terminal, ${nfail} failure(s) — rerunning --failed"
  gh run rerun "$run_id" --repo "$REPO" --failed
}

cycle() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] CI watch — PR #${PR} (${REPO})"
  process_run "TheRock CI" "TheRock CI" 1
  process_run "rocprofiler-sdk Code Coverage" "rocprofiler-sdk Code Coverage" 0
  process_run "rocprofiler-sdk CI (Core)" "rocprofiler-sdk Continuous Integration" 0
}

if [[ "$WATCH" -eq 1 ]]; then
  while true; do
    cycle
    echo "  sleeping ${INTERVAL}s…"
    sleep "$INTERVAL"
  done
else
  cycle
fi
