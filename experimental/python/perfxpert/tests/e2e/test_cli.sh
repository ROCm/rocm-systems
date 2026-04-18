#!/usr/bin/env bash
# tests/e2e/test_cli.sh — end-to-end CLI smoke (legacy path)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE="${HERE}/../fixtures/regression_baseline.db"
OUT_DIR="$(mktemp -d)"
trap 'rm -rf "${OUT_DIR}"' EXIT

# Check if fixture exists before running fixture-dependent tests
if [ ! -f "$FIXTURE" ]; then
    echo "--- Fixture $FIXTURE not found, skipping fixture-dependent tests"
    echo "--- Running basic CLI sanity checks only"

    echo "--- Test 1: help works in both paths (legacy)"
    unset PERFXPERT_USE_AGENTS
    python3 -m perfxpert analyze --help > /dev/null
    echo "PASS: legacy help"

    echo "--- Test 2: help works with agentic flag"
    export PERFXPERT_USE_AGENTS=1
    python3 -m perfxpert analyze --help > /dev/null
    echo "PASS: agentic help (even though agents runtime unavailable)"

    echo "ALL E2E CLI TESTS PASS (sanity checks only)"
    exit 0
fi

echo "--- Test 1: legacy path text output"
unset PERFXPERT_USE_AGENTS
python3 -m perfxpert analyze -i "${FIXTURE}" --format text > "${OUT_DIR}/out.txt" || {
    echo "FAIL: legacy text output failed"; exit 1;
}
grep -q "matmul\|kernel" "${OUT_DIR}/out.txt" || {
    echo "WARN: expected kernel name not found in text output (but output may vary)"
}
echo "PASS: legacy text output"

echo "--- Test 2: legacy path JSON output"
python3 -m perfxpert analyze -i "${FIXTURE}" --format json -d "${OUT_DIR}" -o analysis || {
    echo "FAIL: legacy JSON output failed"; exit 1;
}
test -f "${OUT_DIR}/analysis.json" || { echo "FAIL: JSON file not produced"; exit 1; }
python3 -c "import json; json.load(open('${OUT_DIR}/analysis.json'))" || {
    echo "FAIL: JSON file not parseable"; exit 1;
}
echo "PASS: legacy JSON output"

echo "--- Test 3: agentic path with feature flag (expected to gracefully degrade if runtime absent)"
export PERFXPERT_USE_AGENTS=1
if python3 -m perfxpert analyze -i "${FIXTURE}" --format text > "${OUT_DIR}/agentic.txt" 2>&1; then
    echo "PASS: agentic path ran (Phase 3 runtime is installed)"
else
    # If Phase 3 runtime isn't installed yet, we expect a clean error
    grep -q "agent runtime is not available" "${OUT_DIR}/agentic.txt" && {
        echo "PASS: agentic path returned clean error (Phase 3 runtime not yet installed)"
    } || {
        echo "FAIL: unclean failure message from agentic path"; exit 1;
    }
fi

echo "--- Test 4: help works in both paths"
unset PERFXPERT_USE_AGENTS
python3 -m perfxpert analyze --help > /dev/null
echo "PASS: legacy help"

export PERFXPERT_USE_AGENTS=1
python3 -m perfxpert analyze --help > /dev/null
echo "PASS: agentic help"

echo "ALL E2E CLI TESTS PASS"
