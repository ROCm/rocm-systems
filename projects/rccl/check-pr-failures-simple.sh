#!/bin/bash

# Simple and fast script to generate a PR check report
# Usage: ./check-pr-failures-simple.sh [output_file.md]

OUTPUT_FILE="${1:-pr-check-report.md}"

echo "Generating PR check report (simplified version)..."
echo "Output file: $OUTPUT_FILE"
echo ""

# Get list of open PRs
echo "Fetching open PRs..."
PR_LIST=$(gh pr list --label "project: rccl" --state open --json number,title,url,author --limit 100)

if [ -z "$PR_LIST" ]; then
    echo "No open PRs found."
    exit 0
fi

TOTAL=$(echo "$PR_LIST" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))")

# Generate report
cat > "$OUTPUT_FILE" << EOF
# RCCL Pull Requests - Check Status Report

Generated on: $(date '+%Y-%m-%d %H:%M:%S')

Total Open PRs: **$TOTAL**

---

## Open Pull Requests

| PR # | Title | Author | Link | Status |
|------|-------|--------|------|--------|
EOF

echo "Processing $TOTAL PRs (this may take a few minutes)..."

echo "$PR_LIST" | python3 - << 'PYTHON_SCRIPT' >> "$OUTPUT_FILE"
import sys
import json
import subprocess

data = json.load(sys.stdin)

for i, pr in enumerate(data, 1):
    pr_num = pr['number']
    title = pr['title'].replace('|', '\\|')  # Escape pipes for markdown table
    author = pr['author']['login']
    url = pr['url']

    # Get check status (suppress errors)
    try:
        result = subprocess.run(
            ['gh', 'pr', 'checks', str(pr_num)],
            capture_output=True,
            text=True,
            timeout=10
        )
        checks_output = result.stdout

        # Count statuses
        fail_count = checks_output.count(' fail ')
        pending_count = checks_output.count(' pending ')

        if fail_count > 0:
            status = f"❌ {fail_count} failed"
        elif pending_count > 0:
            status = f"⏳ {pending_count} pending"
        elif checks_output.strip():
            status = "✅ passing"
        else:
            status = "⚪ no checks"
    except:
        status = "❓ unknown"

    print(f"| #{pr_num} | {title} | {author} | [View]({url}) | {status} |")
    print(f"PR #{pr_num}... done ({i}/{len(data)})", file=sys.stderr)

PYTHON_SCRIPT

echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "**Legend:**" >> "$OUTPUT_FILE"
echo "- ❌ Failed: PR has failing checks" >> "$OUTPUT_FILE"
echo "- ⏳ Pending: PR has checks in progress" >> "$OUTPUT_FILE"
echo "- ✅ Passing: All checks passed" >> "$OUTPUT_FILE"
echo "- ⚪ No checks: No checks configured" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo ""
echo "Report generated successfully: $OUTPUT_FILE"
echo ""
echo "To get detailed failure information for a specific PR, run:"
echo "  gh pr checks <PR_NUMBER>"
