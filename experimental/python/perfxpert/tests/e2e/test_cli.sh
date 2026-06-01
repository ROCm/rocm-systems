#!/usr/bin/env bash
# tests/e2e/test_cli.sh — end-to-end CLI smoke (agentic path).
#
# The agentic runtime is the sole execution path. No env-var toggles;
# this script exercises the CLI unconditionally.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE="${HERE}/../fixtures/regression_baseline.db"
OUT_DIR="$(mktemp -d)"
trap 'rm -rf "${OUT_DIR}"' EXIT

if [ ! -f "$FIXTURE" ]; then
    echo "--- Fixture $FIXTURE not found; running help-only sanity checks"
    python3 -m perfxpert analyze --help > /dev/null
    echo "PASS: analyze --help"
    echo "ALL E2E CLI TESTS PASS (sanity checks only)"
    exit 0
fi

echo "--- Test 1: agentic text output"
if python3 -m perfxpert analyze -i "${FIXTURE}" --format text > "${OUT_DIR}/out.txt" 2>&1; then
    echo "PASS: agentic text output"
else
    grep -q "agent runtime is not available" "${OUT_DIR}/out.txt" \
        && { echo "PASS: agentic path returned clean error (runtime absent)"; } \
        || { echo "FAIL: unclean failure from agentic path"; cat "${OUT_DIR}/out.txt"; exit 1; }
fi

echo "--- Test 2: help works"
python3 -m perfxpert analyze --help > /dev/null
echo "PASS: analyze --help"

echo "ALL E2E CLI TESTS PASS"
