#!/usr/bin/env bash

set -u

status=0
rocm_root=${ROCM_PATH:-/opt/rocm}

if command -v amd-smi >/dev/null 2>&1; then
    if amd-smi >/dev/null 2>&1; then
        printf 'PASS GPU visible: amd-smi\n'
    else
        printf 'FAIL GPU visible: amd-smi returned non-zero\n' >&2
        status=1
    fi
else
    printf 'FAIL command: amd-smi\n' >&2
    status=1
fi

if [[ -r "${rocm_root}/.info/version" ]]; then
    printf 'PASS ROCm version: %s\n' "$(< "${rocm_root}/.info/version")"
else
    printf 'FAIL ROCm version: %s/.info/version is unreadable\n' \
        "${rocm_root}" >&2
    status=1
fi

if command -v rocprof-compute >/dev/null 2>&1; then
    rocprof-compute --version || status=1
else
    printf 'FAIL command: rocprof-compute\n' >&2
    status=1
fi

exit "${status}"
