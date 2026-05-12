#!/usr/bin/env bash
#
# Compare per-job wall time and ccache stats between two CI runs.
#
# Usage:
#     ./measure-ci-ccache.sh <cold_run_id> <warm_run_id> [repo]
#
# Defaults to the repository the current `gh` context resolves to.
# Emits a markdown table on stdout: job name, cold time, warm time,
# speedup, ccache hit% lifted from the "ccache stats" step log.
#
# Requires: gh CLI authenticated, jq.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <cold_run_id> <warm_run_id> [repo]" >&2
    exit 2
fi

cold_id="$1"
warm_id="$2"
repo_arg=()
if [[ $# -ge 3 ]]; then
    repo_arg=(--repo "$3")
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

fetch_jobs() {
    local run_id="$1"
    local out="$2"
    gh run view "${run_id}" "${repo_arg[@]}" \
        --json jobs \
        --jq '[.jobs[] | {
                name,
                duration: ((.completedAt | fromdate) - (.startedAt | fromdate)),
                logs_url: .url
              }]' >"${out}"
}

fetch_jobs "${cold_id}" "${tmpdir}/cold.json"
fetch_jobs "${warm_id}" "${tmpdir}/warm.json"

# Pull ccache stats step logs for hit-rate extraction.
fetch_log_dir() {
    local run_id="$1"
    local dir="$2"
    mkdir -p "${dir}"
    gh run view "${run_id}" "${repo_arg[@]}" --log >"${dir}/all.log" 2>/dev/null || true
}

fetch_log_dir "${cold_id}" "${tmpdir}/cold-logs"
fetch_log_dir "${warm_id}" "${tmpdir}/warm-logs"

# Extract per-job hit rate from "ccache stats" sections in the merged log.
# `gh run view --log` prefixes each line with "<job>\t<step>\t<line>".
hit_rate_for_job() {
    local log="$1"
    local job="$2"
    awk -F'\t' -v j="${job}" '
        $1 == j && $2 == "ccache stats" {
            if (match($0, /Hits:[[:space:]]+[0-9]+[[:space:]]*\/[[:space:]]*[0-9]+[[:space:]]*\([0-9.]+ ?%\)/)) {
                print substr($0, RSTART, RLENGTH)
                exit
            }
            if (match($0, /[Cc]ache hit rate[^0-9]+[0-9.]+ ?%/)) {
                print substr($0, RSTART, RLENGTH)
                exit
            }
        }
    ' "${log}" || true
}

printf '| Job | Cold (s) | Warm (s) | Speedup | Warm ccache hit |\n'
printf '|---|---:|---:|---:|---|\n'

mapfile -t jobs < <(jq -r '.[].name' "${tmpdir}/cold.json")

for job in "${jobs[@]}"; do
    cold_dur="$(jq -r --arg n "${job}" '.[] | select(.name == $n) | .duration' "${tmpdir}/cold.json")"
    warm_dur="$(jq -r --arg n "${job}" '.[] | select(.name == $n) | .duration' "${tmpdir}/warm.json")"
    if [[ -z "${warm_dur}" || "${warm_dur}" == "null" ]]; then
        warm_dur="-"
        speedup="-"
    else
        speedup="$(awk -v c="${cold_dur}" -v w="${warm_dur}" 'BEGIN { if (w > 0) printf "%.2fx", c / w; else print "-" }')"
    fi
    hit="$(hit_rate_for_job "${tmpdir}/warm-logs/all.log" "${job}")"
    if [[ -z "${hit}" ]]; then
        hit="(no stats)"
    fi
    printf '| %s | %s | %s | %s | %s |\n' \
        "${job}" "${cold_dur}" "${warm_dur}" "${speedup}" "${hit}"
done
