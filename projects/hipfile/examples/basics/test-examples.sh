#!/usr/bin/env bash
# Run the hipFile basics/ examples end-to-end and verify their outputs.
#
# Usage: test-examples.sh [BUILD_DIR] [--test-dir DIR] [--gpu N] [--compat-off]
#
# BUILD_DIR defaults to the directory containing this script (i.e. assumes the
# binaries sit alongside the .cpp sources, as happens for an in-tree build that
# follows the basics/CMakeLists.txt layout). Override if your build tree puts
# them elsewhere, e.g. build/examples/basics.
#
# Each example self-verifies internally via an FNV-1a hash; this script adds
# external cmp / size checks where they are meaningful, and reports a per-test
# pass/fail summary. Exits non-zero if any test fails.

set -u
set -o pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

build_dir=""
test_dir=""
gpu=""
compat_off=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-dir)  test_dir="$2"; shift 2 ;;
        --gpu)       gpu="$2";      shift 2 ;;
        --compat-off) compat_off=1; shift ;;
        -h|--help)
            sed -n '2,12p' "$0"
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
    test_dir="/tmp/hipfile-basics-tests.$$"
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
ignored=0
declare -a failed=()
declare -a ignored_results=()

bin() {
    local name="$1"
    local path="$build_dir/$name"
    if [[ ! -x "$path" ]]; then
        echo "MISSING: $path" >&2
        return 1
    fi
    printf '%s' "$path"
}

filesize() { stat -c%s "$1"; }

report() {
    local name="$1" status="$2" msg="${3:-}"
    case "$status" in
        PASS)
            printf '  [PASS] %-50s %s\n' "$name" "$msg"
            pass=$((pass + 1))
            ;;
        IGNORED-PASS)
            printf '  [IGN-OK]  %-47s %s\n' "$name" "$msg"
            ignored=$((ignored + 1))
            ignored_results+=("$name: PASS - $msg")
            ;;
        IGNORED-FAIL)
            printf '  [IGN-FAIL] %-46s %s\n' "$name" "$msg"
            ignored=$((ignored + 1))
            ignored_results+=("$name: FAIL - $msg")
            ;;
        *)
            printf '  [FAIL] %-50s %s\n' "$name" "$msg" >&2
            fail=$((fail + 1))
            failed+=("$name")
            ;;
    esac
}

run() {
    # run NAME -- CMD ARGS...   captures output, returns rc
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

# Make an INPUT file of N bytes, block-aligned for O_DIRECT (4 KiB safe default).
make_input() {
    local path="$1" bytes="$2"
    dd if=/dev/urandom of="$path" bs=1 count="$bytes" status=none
}

echo "==> hipFile basics/ test suite"
echo "    build_dir = $build_dir"
echo "    test_dir  = $test_dir"
[[ -n "$gpu" ]] && echo "    gpu       = $gpu"
(( compat_off )) && echo "    HIPFILE_ALLOW_COMPAT_MODE=false"
echo

# Common inputs. 1 MiB is a comfortable size for the iterative/copy tests and
# safely exceeds subregion-write's 8192-byte skip.
input_1m="$test_dir/input.1m.bin"
make_input "$input_1m" $((1024 * 1024))

# -----------------------------------------------------------------------------
# 1. bufregister-write OUTPUT [GPUID]   (128 KiB pattern, internal hash check)
# -----------------------------------------------------------------------------
name="bufregister-write"
out="$test_dir/$name.out"
if exe=$(bin "$name"); then
    if run "$name" -- "$exe" "$out" "${gpu_arg[@]}"; then
        sz=$(filesize "$out")
        if [[ "$sz" == "$((128 * 1024))" ]]; then
            report "$name" PASS "(${sz} bytes)"
        else
            report "$name" FAIL "expected 131072 bytes, got $sz"
        fi
    else
        report "$name" FAIL "non-zero exit"
    fi
else
    report "$name" FAIL "binary not found"
fi
bufreg_out="$out"

# -----------------------------------------------------------------------------
# 2. no-bufregister-write OUTPUT [GPUID]   (1 MiB pattern)
# -----------------------------------------------------------------------------
name="no-bufregister-write"
out="$test_dir/$name.out"
if exe=$(bin "$name"); then
    if run "$name" -- "$exe" "$out" "${gpu_arg[@]}"; then
        sz=$(filesize "$out")
        if [[ "$sz" == "$((1024 * 1024))" ]]; then
            report "$name" PASS "(${sz} bytes)"
        else
            report "$name" FAIL "expected 1048576 bytes, got $sz"
        fi
    else
        report "$name" FAIL "non-zero exit"
    fi
else
    report "$name" FAIL "binary not found"
fi

# -----------------------------------------------------------------------------
# 3. no-odirect-write OUTPUT [GPUID]   (128 KiB pattern, same fill as #1)
# -----------------------------------------------------------------------------
name="no-odirect-write"
out="$test_dir/$name.out"
if exe=$(bin "$name"); then
    if run "$name" -- "$exe" "$out" "${gpu_arg[@]}"; then
        sz=$(filesize "$out")
        if [[ "$sz" != "$((128 * 1024))" ]]; then
            report "$name" FAIL "expected 131072 bytes, got $sz"
        elif [[ -f "$bufreg_out" ]] && cmp -s "$bufreg_out" "$out"; then
            report "$name" PASS "(matches bufregister-write byte-for-byte)"
        elif [[ -f "$bufreg_out" ]]; then
            report "$name" FAIL "128 KiB but differs from bufregister-write output"
        else
            report "$name" PASS "(${sz} bytes; could not cross-check)"
        fi
    else
        report "$name" FAIL "non-zero exit"
    fi
else
    report "$name" FAIL "binary not found"
fi

# -----------------------------------------------------------------------------
# 4. iterative-read INPUT OUTPUT [GPUID]   (full copy, expect cmp equal)
# -----------------------------------------------------------------------------
name="iterative-read"
out="$test_dir/$name.out"
if exe=$(bin "$name"); then
    if run "$name" -- "$exe" "$input_1m" "$out" "${gpu_arg[@]}"; then
        if cmp -s "$input_1m" "$out"; then
            report "$name" PASS "(byte-equal copy of input.1m.bin)"
        else
            report "$name" FAIL "output differs from input"
        fi
    else
        report "$name" FAIL "non-zero exit"
    fi
else
    report "$name" FAIL "binary not found"
fi

# -----------------------------------------------------------------------------
# 5. iterative-devmem-offset-read INPUT OUTPUT [GPUID]   (full copy, equal)
# -----------------------------------------------------------------------------
name="iterative-devmem-offset-read"
out="$test_dir/$name.out"
if exe=$(bin "$name"); then
    if run "$name" -- "$exe" "$input_1m" "$out" "${gpu_arg[@]}"; then
        if cmp -s "$input_1m" "$out"; then
            report "$name" PASS "(byte-equal copy of input.1m.bin)"
        else
            report "$name" FAIL "output differs from input"
        fi
    else
        report "$name" FAIL "non-zero exit"
    fi
else
    report "$name" FAIL "binary not found"
fi

# -----------------------------------------------------------------------------
# 6. subregion-write INPUT OUTPUT [GPUID]
#    Writes INPUT bytes [8192 .. end). Expect size = size(IN) - 8192 and a
#    byte-for-byte match against INPUT skipping the first 8192 bytes.
# -----------------------------------------------------------------------------
name="subregion-write"
out="$test_dir/$name.out"
SUB_OFFSET=8192
if exe=$(bin "$name"); then
    if run "$name" -- "$exe" "$input_1m" "$out" "${gpu_arg[@]}"; then
        in_sz=$(filesize "$input_1m")
        out_sz=$(filesize "$out")
        expect=$((in_sz - SUB_OFFSET))
        if [[ "$out_sz" != "$expect" ]]; then
            report "$name" FAIL "expected $expect bytes, got $out_sz"
        elif cmp -s -i "${SUB_OFFSET}:0" "$input_1m" "$out"; then
            report "$name" PASS "(input[${SUB_OFFSET}:] equals output)"
        else
            report "$name" FAIL "size matched but content diverges"
        fi
    else
        report "$name" FAIL "non-zero exit"
    fi
else
    report "$name" FAIL "binary not found"
fi

# -----------------------------------------------------------------------------
# 7. various-mem-rw INPUT OUTPUT MODE [GPUID]
#    MODE: 1=device, 2=managed, 3=pinned-host. All three should reproduce
#    INPUT byte-for-byte.
# -----------------------------------------------------------------------------
#   Modes 2 and 3 are known to be rejected by hipFile on some setups with
#   error -5013 ("Memory type backing pointer is incompatible with hipFile").
#   They are still executed for visibility but do not count toward the
#   pass/fail tally or the script's exit status.
for mode in 1 2 3; do
    case "$mode" in
        1) label="device";      ignore=0 ;;
        2) label="managed";     ignore=1 ;;
        3) label="pinned-host"; ignore=1 ;;
    esac
    name="various-mem-rw[$label]"
    out="$test_dir/various-mem-rw.$label.out"
    pass_status="PASS"; fail_status="FAIL"
    if (( ignore )); then
        pass_status="IGNORED-PASS"; fail_status="IGNORED-FAIL"
    fi
    if exe=$(bin "various-mem-rw"); then
        if run "$name" -- "$exe" "$input_1m" "$out" "$mode" "${gpu_arg[@]}"; then
            if cmp -s "$input_1m" "$out"; then
                report "$name" "$pass_status" "(byte-equal copy)"
            else
                report "$name" "$fail_status" "output differs from input"
            fi
        else
            report "$name" "$fail_status" "non-zero exit"
        fi
    else
        report "$name" "$fail_status" "binary not found"
    fi
done

# -----------------------------------------------------------------------------
# 8. roundtrip-verify CREATED COPIED [GPUID]   (fully self-contained)
# -----------------------------------------------------------------------------
name="roundtrip-verify"
created="$test_dir/$name.created"
copied="$test_dir/$name.copied"
if exe=$(bin "$name"); then
    if run "$name" -- "$exe" "$created" "$copied" "${gpu_arg[@]}"; then
        if cmp -s "$created" "$copied"; then
            report "$name" PASS "(self-test passed, files match externally)"
        else
            report "$name" FAIL "self-test exited 0 but files differ externally"
        fi
    else
        report "$name" FAIL "non-zero exit"
    fi
else
    report "$name" FAIL "binary not found"
fi

echo
echo "==> Summary: $pass passed, $fail failed, $ignored ignored"
if (( ignored > 0 )); then
    echo "    ignored results (not counted toward exit status):"
    printf '      - %s\n' "${ignored_results[@]}"
fi
if (( fail > 0 )); then
    printf '    failed: %s\n' "${failed[@]}"
    exit 1
fi
exit 0
