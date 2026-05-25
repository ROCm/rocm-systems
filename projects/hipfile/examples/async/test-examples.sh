#!/usr/bin/env bash
# Run the hipFile async/ examples end-to-end and verify their outputs.
#
# Usage: test-examples.sh [BUILD_DIR] [--test-dir DIR] [--gpu N] [--compat-off]
#
# BUILD_DIR defaults to the directory containing this script. All four async
# examples are READ_FILE -> WRITE_FILE round-trip copies; this script feeds
# each one a random 1 MiB input and asserts the output is byte-for-byte equal.

set -u
set -o pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

build_dir=""
test_dir=""
gpu=""
compat_off=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-dir)   test_dir="$2"; shift 2 ;;
        --gpu)        gpu="$2";      shift 2 ;;
        --compat-off) compat_off=1;  shift ;;
        -h|--help)
            sed -n '2,10p' "$0"
            exit 0
            ;;
        *)
            if [[ -z "$build_dir" ]]; then
                build_dir="$1"; shift
            else
                echo "Unexpected argument: $1" >&2; exit 2
            fi
            ;;
    esac
done

[[ -z "$build_dir" ]] && build_dir="$script_dir"
if [[ ! -d "$build_dir" ]]; then
    echo "BUILD_DIR does not exist: $build_dir" >&2
    exit 2
fi

if [[ -z "$test_dir" ]]; then
    test_dir="/tmp/hipfile-async-tests.$$"
    mkdir -p "$test_dir"
    trap 'rm -rf "$test_dir"' EXIT
else
    mkdir -p "$test_dir"
fi

if (( compat_off )); then
    export HIPFILE_ALLOW_COMPAT_MODE=false
fi

declare -a gpu_arg=()
[[ -n "$gpu" ]] && gpu_arg=("$gpu")

pass=0
fail=0
declare -a failed=()

report() {
    local name="$1" status="$2" msg="${3:-}"
    if [[ "$status" == "PASS" ]]; then
        printf '  [PASS] %-50s %s\n' "$name" "$msg"
        pass=$((pass + 1))
    else
        printf '  [FAIL] %-50s %s\n' "$name" "$msg" >&2
        fail=$((fail + 1))
        failed+=("$name")
    fi
}

run() {
    local name="$1"; shift
    [[ "$1" == "--" ]] && shift
    local log="$test_dir/${name//\//_}.log"
    if "$@" >"$log" 2>&1; then
        return 0
    else
        local rc=$?
        printf '    \xe2\x86\xb3 exit=%d, log: %s\n' "$rc" "$log" >&2
        tail -n 20 "$log" | sed 's/^/      /' >&2
        return $rc
    fi
}

echo "==> hipFile async/ test suite"
echo "    build_dir = $build_dir"
echo "    test_dir  = $test_dir"
[[ -n "$gpu" ]] && echo "    gpu       = $gpu"
(( compat_off )) && echo "    HIPFILE_ALLOW_COMPAT_MODE=false"
echo

input="$test_dir/input.1m.bin"
dd if=/dev/urandom of="$input" bs=1 count=$((1024 * 1024)) status=none

ASYNC_EXAMPLES=(
    roundtrip-async
    roundtrip-async-nonblocking-stream
    roundtrip-async-multi-stream
    roundtrip-async-multi-stream-registered
)

for name in "${ASYNC_EXAMPLES[@]}"; do
    exe="$build_dir/$name"
    out="$test_dir/$name.out"
    if [[ ! -x "$exe" ]]; then
        report "$name" FAIL "binary not found ($exe)"
        continue
    fi
    if run "$name" -- "$exe" "$input" "$out" "${gpu_arg[@]}"; then
        if cmp -s "$input" "$out"; then
            report "$name" PASS "(byte-equal round-trip of 1 MiB input)"
        else
            in_sz=$(stat -c%s "$input")
            out_sz=$(stat -c%s "$out" 2>/dev/null || echo "missing")
            report "$name" FAIL "output differs (in=${in_sz} out=${out_sz})"
        fi
    else
        report "$name" FAIL "non-zero exit"
    fi
done

echo
echo "==> Summary: $pass passed, $fail failed"
if (( fail > 0 )); then
    printf '    failed: %s\n' "${failed[@]}"
    exit 1
fi
exit 0
