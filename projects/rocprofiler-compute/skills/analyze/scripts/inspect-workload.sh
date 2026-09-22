#!/usr/bin/env bash

set -u

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s <workload-directory>\n' "$0" >&2
    exit 2
fi

workload_directory=$1
search_root=${workload_directory}

if [[ ! -d "${workload_directory}" ]]; then
    printf 'FAIL directory: %s\n' "${workload_directory}" >&2
    exit 1
fi

if [[ "${search_root}" == -* ]]; then
    search_root="./${search_root}"
fi

find_first() {
    local pattern=$1

    find "${search_root}" -maxdepth 2 -type f -name "${pattern}" \
        -print -quit 2>/dev/null
}

status=0
config_path=$(find_first profiling_config.yaml)
counter_path=$(find_first 'pmc_perf.csv*')
result_path=$(find_first 'results_*.csv*')
sampling_path=$(find_first '*_ps_file_results.json')
database_path=$(find_first '*.db')

if [[ -n "${config_path}" ]]; then
    printf 'PASS profile config: %s\n' "${config_path}"
else
    printf 'FAIL profile config: profiling_config.yaml not found\n' >&2
    status=1
fi

if [[ -n "${counter_path}" || -n "${result_path}" ]]; then
    printf 'PASS counter data: %s\n' "${counter_path:-${result_path}}"
elif [[ -n "${sampling_path}" ]]; then
    printf 'PASS PC sampling data: %s\n' "${sampling_path}"
else
    printf 'FAIL profile data: counters or PC sampling results not found\n' >&2
    status=1
fi

if [[ -n "${database_path}" ]]; then
    printf 'INFO retained rocpd database: %s\n' "${database_path}"
else
    printf 'INFO retained rocpd database: none\n'
fi

exit "${status}"
