#!/bin/bash

# Script to generate a detailed report of all open RCCL PRs and their check status
# Usage: ./check-pr-failures-detailed.sh [output_file.md]

set -e

OUTPUT_FILE="${1:-pr-check-report.md}"

echo "Generating detailed PR check report..."
echo "Output file: $OUTPUT_FILE"
echo ""

# Start the markdown report
{
    echo "# RCCL Pull Requests - Check Status Report"
    echo ""
    echo "Generated on: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "## Summary"
    echo ""
} > "$OUTPUT_FILE"

# Get list of open PRs
echo "Fetching PR list..."
PR_DATA=$(gh pr list --label "project: rccl" --state open --json number,title,url,author --limit 100)
TOTAL_PRS=$(echo "$PR_DATA" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))")

echo "Total PRs: $TOTAL_PRS" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Process each PR
echo "Processing $TOTAL_PRS PRs..."
declare -A PR_STATUS
declare -A PR_TITLES
declare -A PR_URLS
declare -A PR_AUTHORS
declare -A PR_FAILURES

FAIL_COUNT=0
PENDING_COUNT=0
PASS_COUNT=0
NO_CHECK_COUNT=0

# Get PR numbers
PR_NUMBERS=$(echo "$PR_DATA" | python3 -c "import sys, json; data=json.load(sys.stdin); print(' '.join([str(pr['number']) for pr in data]))")

# Create temp file for check data
TEMP_CHECKS=$(mktemp)

for PR_NUM in $PR_NUMBERS; do
    echo -n "Processing PR #$PR_NUM... "

    # Extract PR info (do this once)
    PR_INFO=$(echo "$PR_DATA" | python3 -c "import sys, json
data=json.load(sys.stdin)
pr=[p for p in data if p['number']==$PR_NUM][0]
# Escape special chars for bash
title=pr['title'].replace('\"', '\\\"').replace('\$', '\\\$')
url=pr['url']
author=pr['author']['login']
print(f'{title}|||{url}|||{author}')")

    IFS='|||' read -r TITLE URL AUTHOR <<< "$PR_INFO"
    PR_TITLES[$PR_NUM]="$TITLE"
    PR_URLS[$PR_NUM]="$URL"
    PR_AUTHORS[$PR_NUM]="$AUTHOR"

    # Get checks
    CHECKS=$(gh pr checks $PR_NUM 2>/dev/null || echo "")

    if [ -z "$CHECKS" ]; then
        PR_STATUS[$PR_NUM]="no-checks"
        ((NO_CHECK_COUNT++))
        echo "no checks"
        continue
    fi

    # Count check statuses
    FAIL_CHECKS=$(echo "$CHECKS" | grep -cE "\s(fail|FAIL)\s" || true)
    PENDING_CHECKS=$(echo "$CHECKS" | grep -cE "\spending\s" || true)
    PASS_CHECKS=$(echo "$CHECKS" | grep -cE "\spass\s" || true)

    if [ "$FAIL_CHECKS" -gt 0 ]; then
        PR_STATUS[$PR_NUM]="failed"
        # Save failed checks
        echo "$CHECKS" | grep -E "\s(fail|FAIL)\s" > "${TEMP_CHECKS}.${PR_NUM}" || true
        ((FAIL_COUNT++))
        echo "FAILED ($FAIL_CHECKS failures)"
    elif [ "$PENDING_CHECKS" -gt 0 ]; then
        PR_STATUS[$PR_NUM]="pending"
        ((PENDING_COUNT++))
        echo "pending ($PENDING_CHECKS pending)"
    else
        PR_STATUS[$PR_NUM]="passing"
        ((PASS_COUNT++))
        echo "passing ($PASS_CHECKS passed)"
    fi
done

echo ""
echo "Generating report..."

# Write summary statistics
{
    echo "| Status | Count | Percentage |"
    echo "|--------|-------|------------|"
    echo "| ❌ Failed | $FAIL_COUNT | $(awk "BEGIN {printf \"%.1f\", ($FAIL_COUNT/$TOTAL_PRS)*100}")% |"
    echo "| ⏳ Pending | $PENDING_COUNT | $(awk "BEGIN {printf \"%.1f\", ($PENDING_COUNT/$TOTAL_PRS)*100}")% |"
    echo "| ✅ Passing | $PASS_COUNT | $(awk "BEGIN {printf \"%.1f\", ($PASS_COUNT/$TOTAL_PRS)*100}")% |"
    if [ "$NO_CHECK_COUNT" -gt 0 ]; then
        echo "| ⚪ No Checks | $NO_CHECK_COUNT | $(awk "BEGIN {printf \"%.1f\", ($NO_CHECK_COUNT/$TOTAL_PRS)*100}")% |"
    fi
    echo ""
    echo "---"
    echo ""
    echo "## PRs with Failed Checks"
    echo ""
} >> "$OUTPUT_FILE"

# Process failed PRs
if [ "$FAIL_COUNT" -gt 0 ]; then
    for PR_NUM in $PR_NUMBERS; do
        if [ "${PR_STATUS[$PR_NUM]}" != "failed" ]; then
            continue
        fi

        {
            echo "### PR #$PR_NUM - ${PR_TITLES[$PR_NUM]}"
            echo ""
            echo "- **Author:** ${PR_AUTHORS[$PR_NUM]}"
            echo "- **URL:** ${PR_URLS[$PR_NUM]}"
            echo ""
            echo "**Failed Checks:**"
            echo ""
            echo '```'
            if [ -f "${TEMP_CHECKS}.${PR_NUM}" ]; then
                cat "${TEMP_CHECKS}.${PR_NUM}"
            fi
            echo '```'
            echo ""
        } >> "$OUTPUT_FILE"
    done
else
    echo "No PRs with failed checks! 🎉" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

# Add pending PRs section
{
    echo ""
    echo "---"
    echo ""
    echo "## PRs with Pending Checks"
    echo ""
} >> "$OUTPUT_FILE"

if [ "$PENDING_COUNT" -gt 0 ]; then
    for PR_NUM in $PR_NUMBERS; do
        if [ "${PR_STATUS[$PR_NUM]}" != "pending" ]; then
            continue
        fi

        {
            echo "### PR #$PR_NUM - ${PR_TITLES[$PR_NUM]}"
            echo ""
            echo "- **Author:** ${PR_AUTHORS[$PR_NUM]}"
            echo "- **URL:** ${PR_URLS[$PR_NUM]}"
            echo ""
        } >> "$OUTPUT_FILE"
    done
else
    echo "No PRs with pending checks." >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

# Add passing PRs section
{
    echo ""
    echo "---"
    echo ""
    echo "## PRs with All Checks Passing"
    echo ""
} >> "$OUTPUT_FILE"

if [ "$PASS_COUNT" -gt 0 ]; then
    for PR_NUM in $PR_NUMBERS; do
        if [ "${PR_STATUS[$PR_NUM]}" != "passing" ]; then
            continue
        fi

        echo "- **PR #$PR_NUM** - ${PR_TITLES[$PR_NUM]} (by ${PR_AUTHORS[$PR_NUM]}) - [${PR_URLS[$PR_NUM]}](${PR_URLS[$PR_NUM]})" >> "$OUTPUT_FILE"
    done
    echo "" >> "$OUTPUT_FILE"
else
    echo "No PRs with all checks passing." >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

# Clean up temp files
rm -f "${TEMP_CHECKS}"*

echo ""
echo "Report generated successfully: $OUTPUT_FILE"
echo ""
echo "Summary:"
echo "  Total PRs: $TOTAL_PRS"
echo "  Failed: $FAIL_COUNT"
echo "  Pending: $PENDING_COUNT"
echo "  Passing: $PASS_COUNT"
if [ "$NO_CHECK_COUNT" -gt 0 ]; then
    echo "  No checks: $NO_CHECK_COUNT"
fi
