#!/bin/bash
# Run rccl-UnitTests individually with a timeout.
# Usage: ./run-tests.sh [timeout_seconds] [test_binary]

TIMEOUT=${1:-60}
BINARY=${2:-./build/release/test/rccl-UnitTests}

if [ ! -x "$BINARY" ]; then
  echo "ERROR: $BINARY not found or not executable"
  exit 1
fi

TESTS=$("$BINARY" --gtest_list_tests 2>/dev/null)
if [ $? -ne 0 ]; then
  echo "ERROR: failed to list tests"
  exit 1
fi

PASS=0
FAIL=0
TIMEOUT_COUNT=0
TOTAL=0

CHILD_PID=""

cleanup() {
  echo ""
  echo "Interrupted."
  if [ -n "$CHILD_PID" ]; then
    kill -9 -"$CHILD_PID" 2>/dev/null  # kill the whole process group
  fi
  echo "========================================"
  echo "Total: $TOTAL  Pass: $PASS  Fail: $FAIL  Timeout: $TIMEOUT_COUNT"
  echo "========================================"
  exit 1
}

trap cleanup INT TERM

CURRENT_SUITE=""
IN_TESTS=0
while IFS= read -r line; do
  # Skip the environment variable banner — test suites end with "."
  # and test cases are indented with spaces. Skip everything else.
  if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*/?\. ]]; then
    IN_TESTS=1
    CURRENT_SUITE="$line"
    continue
  fi
  [[ $IN_TESTS -eq 0 ]] && continue
  if [[ -z "$line" ]]; then
    continue
  fi
  if [[ "$line" =~ ^[[:space:]] ]]; then
    TEST_NAME=$(echo "$line" | sed 's/^[[:space:]]*//' | awk '{print $1}')
    FULL_TEST="${CURRENT_SUITE}${TEST_NAME}"
    ((TOTAL++))
    printf "[%4d] %-80s " "$TOTAL" "$FULL_TEST"
    setsid timeout -s 9 "$TIMEOUT" "$BINARY" --gtest_filter="$FULL_TEST" > /dev/null 2>&1 &
    CHILD_PID=$!
    wait $CHILD_PID
    rc=$?
    kill -9 -"$CHILD_PID" 2>/dev/null  # mop up any survivors; no-op if already gone
    CHILD_PID=""
    if [ $rc -eq 0 ]; then
      echo "PASS"
      ((PASS++))
    elif [ $rc -eq 137 ]; then
      echo "TIMEOUT"
      ((TIMEOUT_COUNT++))
    else
      echo "FAIL (exit $rc)"
      ((FAIL++))
    fi
  fi
done <<< "$TESTS"

echo ""
echo "========================================"
echo "Total: $TOTAL  Pass: $PASS  Fail: $FAIL  Timeout: $TIMEOUT_COUNT"
echo "========================================"
