#!/usr/bin/env bash
# Run every subdirectory's test-examples.sh and aggregate pass/fail.
#
# Usage: test-all-examples.sh [--build-root DIR] [--test-dir DIR] [--gpu N] [--compat-off]
#
# --build-root  Root of a build tree mirroring this layout (e.g. build/examples).
#               Each suite is invoked with "$build_root/<suite>" as its BUILD_DIR.
#               If omitted, each suite defaults to its own source directory
#               (matches an in-tree build where binaries sit next to sources).
# Other flags are forwarded verbatim to each suite's test-examples.sh.

set -u
set -o pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

build_root=""
declare -a forward=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-root) build_root="$2"; shift 2 ;;
        --test-dir|--gpu) forward+=("$1" "$2"); shift 2 ;;
        --compat-off)     forward+=("$1");      shift ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "Unexpected argument: $1" >&2; exit 2 ;;
    esac
done

suites=()
while IFS= read -r -d '' f; do suites+=("$f"); done < <(
    find "$script_dir" -mindepth 2 -maxdepth 2 -name test-examples.sh -print0 | sort -z
)

if (( ${#suites[@]} == 0 )); then
    echo "No test-examples.sh found under $script_dir" >&2
    exit 2
fi

overall_rc=0
declare -a results=()

for suite_script in "${suites[@]}"; do
    suite_dir=$(dirname "$suite_script")
    suite_name=$(basename "$suite_dir")
    declare -a args=()
    if [[ -n "$build_root" ]]; then
        args+=("$build_root/$suite_name")
    fi
    args+=("${forward[@]}")

    echo
    echo "############################################################"
    echo "# $suite_name"
    echo "############################################################"
    if bash "$suite_script" "${args[@]}"; then
        results+=("$suite_name: PASS")
    else
        rc=$?
        results+=("$suite_name: FAIL (exit $rc)")
        overall_rc=1
    fi
done

echo
echo "============================================================"
echo "Aggregate results:"
printf '  %s\n' "${results[@]}"
exit $overall_rc
