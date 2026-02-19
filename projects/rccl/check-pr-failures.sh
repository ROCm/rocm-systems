#!/bin/bash

# Script to check all open RCCL PRs for failed checks
# Usage: ./check-pr-failures.sh [--verbose]

set -e

VERBOSE=false
if [[ "$1" == "--verbose" ]]; then
    VERBOSE=true
fi

echo "=========================================="
echo "Checking Open RCCL PRs for Failed Checks"
echo "=========================================="
echo ""

# Get list of open PRs with "project: rccl" label
echo "Fetching open PRs with 'project: rccl' label..."
PR_LIST=$(gh pr list --label "project: rccl" --state open --json number --limit 100 | python3 -c "import sys, json; data=json.load(sys.stdin); print(' '.join([str(pr['number']) for pr in data]))")

if [ -z "$PR_LIST" ]; then
    echo "No open PRs found with 'project: rccl' label."
    exit 0
fi

TOTAL_PRS=$(echo $PR_LIST | wc -w)
echo "Found $TOTAL_PRS open PRs"
echo ""

# Arrays to track results
declare -a PRS_WITH_FAILURES
declare -a PRS_ALL_PASSING
declare -a PRS_WITH_PENDING

# Associative array to track failure counts per check
declare -A CHECK_FAILURE_COUNTS

# Temporary file for detailed output
DETAIL_FILE=$(mktemp)

# Check each PR
for PR_NUM in $PR_LIST; do
    echo -n "Checking PR #$PR_NUM... "

    # Get check status
    CHECKS=$(gh pr checks $PR_NUM 2>/dev/null || echo "")

    if [ -z "$CHECKS" ]; then
        echo "no checks"
        continue
    fi

    # Count failures, pending, and passes
    FAILURES=$(echo "$CHECKS" | grep -E "\s(fail|FAIL)\s" | wc -l || true)
    PENDING=$(echo "$CHECKS" | grep -E "\spending\s" | wc -l || true)
    PASSES=$(echo "$CHECKS" | grep -E "\spass\s" | wc -l || true)

    if [ "$FAILURES" -gt 0 ]; then
        echo "FAILURES FOUND ($FAILURES failed, $PENDING pending, $PASSES passed)"
        PRS_WITH_FAILURES+=($PR_NUM)

        # Store detailed failure info
        echo "=== PR #$PR_NUM ===" >> "$DETAIL_FILE"
        gh pr view $PR_NUM --json title,url | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"Title: {data['title']}\nURL: {data['url']}\")" >> "$DETAIL_FILE"
        echo "" >> "$DETAIL_FILE"
        echo "Failed Checks:" >> "$DETAIL_FILE"
        echo "$CHECKS" | grep -E "\s(fail|FAIL)\s" >> "$DETAIL_FILE" || true
        echo "" >> "$DETAIL_FILE"

        # Count failures per check
        while IFS= read -r line; do
            # Extract check name (first tab-delimited field)
            CHECK_NAME=$(echo "$line" | cut -f1)
            if [ -n "$CHECK_NAME" ]; then
                if [ -z "${CHECK_FAILURE_COUNTS[$CHECK_NAME]}" ]; then
                    CHECK_FAILURE_COUNTS[$CHECK_NAME]=0
                fi
                CHECK_FAILURE_COUNTS[$CHECK_NAME]=$((CHECK_FAILURE_COUNTS[$CHECK_NAME] + 1))
            fi
        done < <(echo "$CHECKS" | grep -E "\s(fail|FAIL)\s" || true)

    elif [ "$PENDING" -gt 0 ]; then
        echo "pending checks ($PENDING pending, $PASSES passed)"
        PRS_WITH_PENDING+=($PR_NUM)
    else
        echo "all passing ($PASSES passed)"
        PRS_ALL_PASSING+=($PR_NUM)
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total PRs checked: $TOTAL_PRS"
echo "PRs with failures: ${#PRS_WITH_FAILURES[@]}"
echo "PRs with pending checks: ${#PRS_WITH_PENDING[@]}"
echo "PRs all passing: ${#PRS_ALL_PASSING[@]}"
echo ""

if [ ${#PRS_WITH_FAILURES[@]} -gt 0 ]; then
    echo "=========================================="
    echo "PRs with Failed Checks"
    echo "=========================================="
    for PR_NUM in "${PRS_WITH_FAILURES[@]}"; do
        echo "- PR #$PR_NUM: https://github.com/ROCm/rocm-systems/pull/$PR_NUM"
    done
    echo ""

    if [ "$VERBOSE" = true ]; then
        echo "=========================================="
        echo "Detailed Failure Information"
        echo "=========================================="
        cat "$DETAIL_FILE"
    else
        echo "Run with --verbose flag to see detailed failure information"
        echo ""
    fi
fi

if [ ${#PRS_WITH_PENDING[@]} -gt 0 ]; then
    echo "=========================================="
    echo "PRs with Pending Checks"
    echo "=========================================="
    for PR_NUM in "${PRS_WITH_PENDING[@]}"; do
        echo "- PR #$PR_NUM: https://github.com/ROCm/rocm-systems/pull/$PR_NUM"
    done
    echo ""
fi

if [ ${#PRS_ALL_PASSING[@]} -gt 0 ]; then
    echo "=========================================="
    echo "PRs with All Checks Passing"
    echo "=========================================="
    for PR_NUM in "${PRS_ALL_PASSING[@]}"; do
        echo "- PR #$PR_NUM: https://github.com/ROCm/rocm-systems/pull/$PR_NUM"
    done
    echo ""
fi

# Display failure counts per check
if [ ${#CHECK_FAILURE_COUNTS[@]} -gt 0 ]; then
    echo "=========================================="
    echo "Failure Count by Check"
    echo "=========================================="
    # Sort by count (descending) then by name
    for check in "${!CHECK_FAILURE_COUNTS[@]}"; do
        echo "${CHECK_FAILURE_COUNTS[$check]} $check"
    done | sort -rn -k1,1 -k2 | while read -r count name; do
        echo "$name: $count PR(s) failed"
    done
    echo ""
fi

# Clean up
rm -f "$DETAIL_FILE"

echo "Done!"
