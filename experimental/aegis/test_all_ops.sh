#!/bin/bash
# Test each llama.cpp operation individually under AegisBit instrumentation.
# Writes output to a temp file so process crashes don't truncate results.
AEGIS_DIR=/home/djavady/aegis_two
TEST_BIN=/home/djavady/llama_cpp/build-hip/bin/test-backend-ops
LIB=$AEGIS_DIR/build/src/libaegisbit.so
TMPFILE=$(mktemp)
STRIP_ANSI='s/\x1b\[[0-9;]*m//g'

PASS_OPS=()
FAIL_OPS=()
SKIP_OPS=()
TIMEOUT_OPS=()
TEARDOWN_OPS=()

OPS=$(LD_PRELOAD=$LIB $TEST_BIN --list-ops 2>/dev/null | grep '^ ' | sed 's/^ *//')

total=0
for op in $OPS; do
    total=$((total + 1))
    echo -n "[$total] $op ... "

    # Write to file so crash doesn't lose output
    timeout 120 bash -c "LD_PRELOAD=$LIB $TEST_BIN -o '$op' > '$TMPFILE' 2>&1"
    exit_code=$?
    output=$(sed "$STRIP_ANSI" "$TMPFILE")

    if [ $exit_code -eq 124 ]; then
        echo "TIMEOUT"
        TIMEOUT_OPS+=("$op")
        continue
    fi

    # Look for "X/Y tests passed" 
    rocm_passed=$(echo "$output" | grep -oP '\d+/\d+ tests passed' | head -1)
    fail_lines=$(echo "$output" | grep -P '\bFAIL\b' | grep -v 'Backend\|backend' | head -5)
    fail_count=$(echo "$output" | grep -cP '\bFAIL\b')
    has_illegal=$(echo "$output" | grep -c 'HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION')
    has_abort=$(echo "$output" | grep -c 'Aborted\|SIGABRT')

    passed_num=""
    total_num=""
    if [ -n "$rocm_passed" ]; then
        passed_num=$(echo "$rocm_passed" | grep -oP '^\d+')
        total_num=$(echo "$rocm_passed" | grep -oP '(?<=/)\d+')
    fi

    if [ "$has_illegal" -gt 0 ]; then
        echo "ILLEGAL_INSTRUCTION $rocm_passed"
        FAIL_OPS+=("$op (ILLEGAL_INSTRUCTION)")
    elif [ -n "$rocm_passed" ] && [ "$total_num" = "0" ]; then
        echo "SKIP (0/0 tests)"
        SKIP_OPS+=("$op")
    elif [ -n "$rocm_passed" ] && [ "$passed_num" != "$total_num" ]; then
        actual_fails=$((total_num - passed_num))
        echo "FAIL $rocm_passed"
        FAIL_OPS+=("$op ($rocm_passed)")
    elif [ -n "$rocm_passed" ] && [ "$passed_num" = "$total_num" ]; then
        if [ $exit_code -ne 0 ]; then
            echo "PASS $rocm_passed (teardown exit=$exit_code)"
            TEARDOWN_OPS+=("$op")
        else
            echo "PASS $rocm_passed"
        fi
        PASS_OPS+=("$op")
    elif [ $exit_code -ne 0 ]; then
        # Check if there were individual test OK lines
        ok_count=$(echo "$output" | grep -cP ':\s*OK$')
        if [ "$ok_count" -gt 0 ]; then
            echo "PARTIAL (${ok_count} OKs, crash exit=$exit_code before summary)"
            FAIL_OPS+=("$op (partial ${ok_count} OKs, crash exit=$exit_code)")
        else
            echo "CRASH (exit=$exit_code)"
            FAIL_OPS+=("$op (crash exit=$exit_code)")
        fi
    else
        echo "SKIP/UNKNOWN"
        SKIP_OPS+=("$op")
    fi
done

rm -f "$TMPFILE"

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "PASS:             ${#PASS_OPS[@]} (${#TEARDOWN_OPS[@]} with teardown crash)"
echo "FAIL:             ${#FAIL_OPS[@]}"
echo "SKIP (no GPU):    ${#SKIP_OPS[@]}"
echo "TIMEOUT:          ${#TIMEOUT_OPS[@]}"
echo ""

if [ ${#FAIL_OPS[@]} -gt 0 ]; then
    echo "FAILED OPERATIONS:"
    for f in "${FAIL_OPS[@]}"; do
        echo "  - $f"
    done
    echo ""
fi

if [ ${#TIMEOUT_OPS[@]} -gt 0 ]; then
    echo "TIMED OUT OPERATIONS:"
    for t in "${TIMEOUT_OPS[@]}"; do
        echo "  - $t"
    done
    echo ""
fi

echo "PASSED OPERATIONS:"
for p in "${PASS_OPS[@]}"; do
    echo "  - $p"
done
