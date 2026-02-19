# RCCL PR Check Scripts

Quick scripts to monitor open RCCL pull requests and their CI status.

## Available Scripts

### 1. `list-open-prs.sh` - Quick PR List (⚡ FASTEST)
Lists all open PRs in a clean table format.

**Usage:**
```bash
# Terminal table
./list-open-prs.sh

# Markdown format
./list-open-prs.sh --markdown > prs.md
```

**Speed:** Instant (~2 seconds)

---

### 2. `check-pr-failures.sh` - Full Check Status (⏱️ SLOW)
Checks CI status for all open PRs. **Warning: This is slow** (~5-10 minutes for 40+ PRs).

**Usage:**
```bash
# Basic summary
./check-pr-failures.sh

# Detailed with all failures
./check-pr-failures.sh --verbose
```

**Speed:** ~5-10 minutes for 40 PRs (GitHub API is slow)

**What it does:**
- Fetches all open PRs
- Runs `gh pr checks <number>` for each PR (this is the slow part)
- Categorizes PRs by status (failed/pending/passing)
- Shows failure details in verbose mode

---

### 3. Manual Check (Recommended for specific PRs)

To check a specific PR quickly:

```bash
# Check PR status
gh pr checks 3258

# Check PR status and view in browser
gh pr view 3258 --web
```

## Recommended Workflow

### Daily Workflow
```bash
# 1. Quick list of all PRs
./list-open-prs.sh

# 2. Check specific PRs you care about
gh pr checks 3258
gh pr checks 3254

# 3. Optional: Full check (go get coffee ☕)
./check-pr-failures.sh
```

### Weekly Report
```bash
# Generate full status (takes ~10 min)
./check-pr-failures.sh --verbose > weekly-report-$(date +%Y-%m-%d).txt

# Or just list PRs
./list-open-prs.sh --markdown > weekly-prs.md
```

## Why Is It Slow?

The `gh pr checks` command makes API calls to GitHub for each PR individually:
- 44 PRs × ~10 seconds each = ~7 minutes
- GitHub API rate limiting
- No batch API available for check status

**Workaround:** Check specific PRs manually instead of checking all PRs.

## Understanding Check Status

Common check names:

| Check Name | What It Does |
|------------|--------------|
| `trigger-rocm-ci` | Triggers the ROCm CI pipeline |
| `mci/rocm-libraries/precheckin(rccl)` | Math CI precheckin tests |
| `mci/rocm-libraries/extra(rccl)` | Math CI additional tests |
| `PSDB` | PSDB CI system |
| `Linux (hip-tests, ...)` | Linux HIP tests |
| `Windows (hip-tests, ...)` | Windows HIP tests |
| `TheRock CI Summary` | TheRock CI summary check |

## Quick Reference

```bash
# List all open PRs (fast)
./list-open-prs.sh

# Check one PR
gh pr checks 3258

# Check all PRs (slow)
./check-pr-failures.sh

# Detailed check (very slow)
./check-pr-failures.sh --verbose

# View PR in browser
gh pr view 3258 --web

# Get PR info
gh pr view 3258

# List only failing PRs (after running full check)
./check-pr-failures.sh | grep "FAILURES FOUND"
```

## Tips

1. **Don't run full checks unnecessarily** - It's slow and hits GitHub API rate limits
2. **Check specific PRs** - Use `gh pr checks <NUMBER>` for PRs you're working on
3. **Use list-open-prs.sh first** - Get an overview before diving into specifics
4. **Run full checks weekly** - Not daily
5. **Bookmark important PRs** - Check those regularly, ignore the rest

## Example Session

```bash
$ ./list-open-prs.sh
Fetching open RCCL PRs...

PR #     Title                                            Author               Updated
----     -----                                            ------               -------
3258     [RCCL] fix DMABUF support check fail             paklui               2026-02-16
3254     [RCCL] Fix p2p_batching...                       isaki001             2026-02-16
...

Total: 44 PRs

$ gh pr checks 3258
trigger-rocm-ci           fail    4s      https://...
mci/.../precheckin(rccl)  pass    0       https://...
...

$ # That PR has failures, let me check another one
$ gh pr checks 3261
labeler                   pass    6s      https://...
```

## Troubleshooting

### Script hangs forever
- `Ctrl+C` to cancel
- Use `list-open-prs.sh` instead
- Check specific PRs with `gh pr checks <NUM>`

### "gh: command not found"
```bash
# Install GitHub CLI
sudo apt install gh
gh auth login
```

### Rate limited
- Wait a few minutes
- GitHub has API rate limits
- Checking too many PRs too quickly triggers this

## Files

- `list-open-prs.sh` - Fast PR listing ⚡
- `check-pr-failures.sh` - Full status check (slow) ⏱️
- `README-PR-SCRIPTS.md` - This file

---

**TL;DR:** Use `list-open-prs.sh` for quick lists, `gh pr checks <NUM>` for specific PRs, and only run full checks when you have time to spare.
