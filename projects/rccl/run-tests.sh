#!/bin/bash
# Run rccl-UnitTests individually with a timeout.
# Usage: ./run-tests.sh [-t timeout_seconds] [-b test_binary] [-f testfile]

TIMEOUT=60
BINARY=./build/release/test/rccl-UnitTests
FILE=""

while getopts "t:b:f:" opt; do
  case $opt in
    t) TIMEOUT=$OPTARG ;;
    b) BINARY=$OPTARG ;;
    f) FILE=$OPTARG ;;
    *) echo "Usage: $0 [-t timeout] [-b binary] [-f testfile]" >&2; exit 1 ;;
  esac
done

TMPFILE=""

cleanup() {
  echo ""
  echo "Interrupted."
  if [ -n "$CHILD_PID" ]; then
    kill -9 -"$CHILD_PID" 2>/dev/null  # kill the whole process group
  fi
  echo "========================================"
  echo "Total: $TOTAL  Pass: $PASS  Fail: $FAIL  Timeout: $TIMEOUT_COUNT"
  echo "========================================"
  [ -n "$TMPFILE" ] && rm -f "$TMPFILE"
  exit 1
}

trap cleanup INT TERM

if [ ! -x "$BINARY" ]; then
  echo "ERROR: $BINARY not found or not executable"
  exit 1
fi

if [ -n "$FILE" ]; then
  if [ ! -r "$FILE" ]; then
    echo "ERROR: $FILE not found or not readable"
    exit 1
  fi
else
  TMPFILE=$(mktemp)
  "$BINARY" --gtest_list_tests 2>/dev/null | awk '
    /^[A-Za-z_][A-Za-z0-9_]*\/?\./ { suite=$0; next }
    /^[^[:space:]]/ { suite="" }
    /^[[:space:]]/ { if (suite) { sub(/^[[:space:]]+/, ""); sub(/ .*/, ""); print suite $0 } }
  ' > "$TMPFILE"
  if [ $? -ne 0 ] || [ ! -s "$TMPFILE" ]; then
    echo "ERROR: failed to list tests"
    rm -f "$TMPFILE"
    exit 1
  fi
  FILE=$TMPFILE
fi

PASS=0
FAIL=0
TIMEOUT_COUNT=0
TOTAL=0

CHILD_PID=""

while IFS= read -r line; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  ((TOTAL++))
  printf "[%4d] %-50s " "$TOTAL" "$line"
  setsid timeout -s 9 "$TIMEOUT" "$BINARY" --gtest_filter="$line" > /dev/null 2>&1 &
  CHILD_PID=$!
  wait $CHILD_PID >/dev/null 2>&1
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
done < "$FILE"

[ -n "$TMPFILE" ] && rm -f "$TMPFILE"

echo ""
echo "========================================"
echo "Total: $TOTAL  Pass: $PASS  Fail: $FAIL  Timeout: $TIMEOUT_COUNT"
echo "========================================"
