#!/bin/bash

# List all open RCCL PRs in a clean table format
# Usage: ./list-open-prs.sh [--markdown]

MARKDOWN=false
if [[ "$1" == "--markdown" ]]; then
    MARKDOWN=true
fi

echo "Fetching open RCCL PRs..."
PR_DATA=$(gh pr list --label "project: rccl" --state open --json number,title,author,url,updatedAt --limit 100)

TOTAL=$(echo "$PR_DATA" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))")

if [ "$MARKDOWN" = true ]; then
    # Markdown format
    echo "# RCCL Open Pull Requests"
    echo ""
    echo "Total: **$TOTAL** PRs"
    echo ""
    echo "| PR # | Title | Author | Last Updated |"
    echo "|------|-------|--------|--------------|"
    echo "$PR_DATA" | python3 -c "
import sys, json
from datetime import datetime
data = json.load(sys.stdin)
for pr in data:
    title = pr['title'].replace('|', '\\\\|')
    updated = pr['updatedAt'][:10]  # Just the date
    print(f\"| [#{pr['number']}]({pr['url']}) | {title} | {pr['author']['login']} | {updated} |\")
"
    echo ""
    echo "---"
    echo ""
    echo "**To check status of a specific PR:**"
    echo "\`\`\`bash"
    echo "gh pr checks <PR_NUMBER>"
    echo "\`\`\`"
else
    # Terminal format
    printf "\n%-8s %-60s %-20s %-12s\n" "PR #" "Title" "Author" "Updated"
    printf "%-8s %-60s %-20s %-12s\n" "----" "-----" "------" "-------"
    echo "$PR_DATA" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for pr in data:
    title = pr['title'][:57] + '...' if len(pr['title']) > 60 else pr['title']
    author = pr['author']['login'][:17] + '...' if len(pr['author']['login']) > 20 else pr['author']['login']
    updated = pr['updatedAt'][:10]
    print(f\"{pr['number']:<8} {title:<60} {author:<20} {updated:<12}\")
"
    echo ""
    echo "Total: $TOTAL PRs"
    echo ""
    echo "To check failures, use: ./check-pr-failures.sh"
    echo "To check a specific PR: gh pr checks <PR_NUMBER>"
fi
