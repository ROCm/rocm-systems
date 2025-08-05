# GitHub Workflows Documentation

## PR Review Labels Management

This repository uses an automated workflow to manage PR labels based on review status.

### Workflow: `pr-review-label-manager.yml`

**Triggers:**
- When a review is submitted or dismissed on a PR
- When new commits are pushed to a PR (synchronize)
- When a PR is opened or marked ready for review
- When review requests are added or removed

**Purpose:** Automatically manages review status labels based on the current state of PR reviews

**Label Logic:**
- **"ready for peer review"**: Applied when:
  - PR has no approvals
  - PR has changes requested
  - New commits are pushed (resets review state)
- **"ready for 2nd review"**: Applied when:
  - PR has exactly 1 approval
  - No changes are requested
- **"ready for merge"**: Applied when:
  - PR has 2 or more approvals
  - No changes are requested

**Special Behaviors:**
- Draft PRs are ignored
- Reviews from the PR author are not counted
- Only the most recent review from each reviewer is considered
- Comment-only reviews are ignored
- Labels are automatically created if they don't exist

### Required Labels

Make sure these labels exist in your repository:
- `ready for peer review` (suggested color: #d4c5f9)
- `ready for 2nd review` (suggested color: #fbca04)
- `ready for merge` (suggested color: #0e8a16)

### Creating Labels

To create these labels, go to your repository's Issues → Labels page and create them manually, or use the GitHub CLI:

```bash
gh label create "ready for peer review" --color d4c5f9 --description "PR needs initial review"
gh label create "ready for 2nd review" --color fbca04 --description "PR has one approval, needs second review"
gh label create "ready for merge" --color 0e8a16 --description "PR has sufficient approvals and is ready to merge"
```

### Notes

- The workflows only count reviews from users other than the PR author
- Only the most recent review from each reviewer is considered
- Reviews with "COMMENTED" state are ignored (only APPROVED and CHANGES_REQUESTED matter)
- The workflows require the `GITHUB_TOKEN` with `pull-requests: write` permission