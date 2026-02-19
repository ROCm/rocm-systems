# RCCL PR Check Scripts

These scripts help you monitor the status of all open RCCL pull requests and identify which ones have failing checks.

## Prerequisites

- GitHub CLI (`gh`) must be installed and authenticated
- Python 3 (for JSON parsing)
- Bash shell

## Scripts

### 1. `check-pr-failures.sh` - Quick Summary

A fast terminal-based script that provides a quick overview of PR check statuses.

**Usage:**
```bash
# Basic usage - shows summary only
./check-pr-failures.sh

# Verbose mode - shows detailed failure information
./check-pr-failures.sh --verbose
```

**Output:**
- Lists total PRs checked
- Counts PRs with failures, pending checks, and passing checks
- Shows PR numbers and links
- In verbose mode, displays all failed checks for each PR

**Example:**
```bash
$ ./check-pr-failures.sh
==========================================
Checking Open RCCL PRs for Failed Checks
==========================================

Found 44 open PRs

Checking PR #3261... all passing (1 passed)
Checking PR #3258... FAILURES FOUND (1 failed, 1 pending, 11 passed)
...

==========================================
Summary
==========================================
Total PRs checked: 44
PRs with failures: 38
PRs with pending checks: 3
PRs all passing: 3
```

### 2. `check-pr-failures-detailed.sh` - Markdown Report

Generates a comprehensive Markdown report file with detailed information about all PRs.

**Usage:**
```bash
# Generate report with default filename (pr-check-report.md)
./check-pr-failures-detailed.sh

# Generate report with custom filename
./check-pr-failures-detailed.sh my-report.md
```

**Output:**
- Creates a Markdown file with:
  - Summary table with counts and percentages
  - Section for PRs with failed checks (including list of failed checks)
  - Section for PRs with pending checks
  - Section for PRs with all checks passing
- Report includes PR title, author, URL, and check details

**Example:**
```bash
$ ./check-pr-failures-detailed.sh weekly-report.md
Generating detailed PR check report...
Output file: weekly-report.md

Processing PRs...
Processing PR #3261... passing
Processing PR #3258... FAILED
...

Report generated successfully: weekly-report.md

Summary:
  Total PRs: 44
  Failed: 38
  Pending: 3
  Passing: 3
```

## Check Status Categories

- **Failed** (❌): PR has one or more checks that failed
- **Pending** (⏳): PR has checks still running (no failures)
- **Passing** (✅): All checks have completed successfully

## Common Workflows

### Daily Check
Quick check to see if any new failures appeared:
```bash
./check-pr-failures.sh
```

### Weekly Report
Generate a detailed report for team review:
```bash
./check-pr-failures-detailed.sh weekly-report-$(date +%Y-%m-%d).md
```

### Find Specific Failures
Use verbose mode and grep to find specific types of failures:
```bash
./check-pr-failures.sh --verbose | grep -A 5 "mci/rocm-libraries"
```

### Monitor Specific PR
To check a specific PR:
```bash
gh pr checks <PR_NUMBER>
```

## Understanding Check Names

Common check names you'll see:

- `trigger-rocm-ci` - Triggers the ROCm CI pipeline
- `mci/rocm-libraries/precheckin(rccl)` - Math CI precheckin tests
- `mci/rocm-libraries/extra(rccl)` - Math CI extra tests
- `mci/rocm-libraries/static-analysis(rccl)` - Static analysis checks
- `PSDB` - PSDB CI system
- `Linux (hip-tests, rocprofiler-tests) / Test / Test hip-tests` - Linux HIP tests
- `Windows (hip-tests, rocprofiler-tests) / Test / Test hip-tests` - Windows HIP tests
- `TheRock CI Summary` - TheRock CI summary check
- `Trigger Azure CI` - Azure CI trigger

## Troubleshooting

### "gh: command not found"
Install GitHub CLI:
```bash
# Ubuntu/Debian
sudo apt install gh

# Or download from https://cli.github.com/
```

### "You are not logged into any GitHub hosts"
Authenticate with GitHub:
```bash
gh auth login
```

### No PRs found
Make sure you're in the correct repository directory:
```bash
cd /path/to/rocm-systems
```

## Tips

1. **Run regularly**: Set up a cron job to generate reports automatically
2. **Compare reports**: Save reports over time to track progress
3. **Filter by failure type**: Use grep to find specific types of failures
4. **Share reports**: The Markdown reports are perfect for sharing with the team

## Examples

### Create weekly report and open in browser
```bash
REPORT="pr-report-$(date +%Y-%m-%d).md"
./check-pr-failures-detailed.sh "$REPORT"
# Convert to HTML and open (requires pandoc)
pandoc "$REPORT" -o "${REPORT%.md}.html" && open "${REPORT%.md}.html"
```

### Find all PRs failing Math CI
```bash
./check-pr-failures.sh --verbose | grep -B 3 "mci/"
```

### Count PRs by failure type
```bash
./check-pr-failures.sh --verbose | grep "fail" | awk '{print $1}' | sort | uniq -c
```

## Maintenance

These scripts use the GitHub CLI to query PR status. They should continue to work as long as:
- GitHub CLI is installed and authenticated
- The repository structure remains the same
- PRs continue to use the "project: rccl" label

Last updated: 2026-02-16
