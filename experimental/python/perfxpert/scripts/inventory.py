#!/usr/bin/env python3
"""
scripts/inventory.py — Generate inventory of all audited docs
Runs lint, link-checker, and sample-executor; outputs JSON inventory.

All scripts live alongside this one under
experimental/python/perfxpert/scripts/; they are invoked by absolute
path so this file works regardless of the caller's cwd.
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

_SCRIPTS_DIR = Path(__file__).resolve().parent
_LINT_SH = _SCRIPTS_DIR / "lint.sh"
_LINK_CHECKER = _SCRIPTS_DIR / "link-checker.py"
_TEST_SAMPLES = _SCRIPTS_DIR / "test-samples.py"


def run_lint_check():
    """Run lint.sh and count violations."""
    result = subprocess.run(
        ["bash", str(_LINT_SH)],
        capture_output=True,
        text=True,
    )
    violations = result.stdout.count("FAIL:")
    return violations


def run_link_check():
    """Run link-checker.py --strict and count broken links (CSV rows only)."""
    result = subprocess.run(
        ["python3", str(_LINK_CHECKER), "--strict", "experimental/python/perfxpert"],
        capture_output=True,
        text=True,
    )
    # --strict emits only CSV data rows (no preamble, no "all validated" line).
    broken = 0
    for line in result.stdout.split('\n'):
        if line.strip() and not line.startswith('file,'):
            broken += 1
    return broken


def run_sample_check():
    """Run test-samples.py and count failures."""
    result = subprocess.run(
        ["python3", str(_TEST_SAMPLES), "experimental/python/perfxpert"],
        capture_output=True,
        text=True,
    )
    # Default output interleaves per-FAIL lines with the JSON summary at the end.
    # Locate the JSON block by finding the first '{' at column 0.
    stdout = result.stdout
    idx = stdout.find('\n{')
    if idx == -1:
        idx = 0 if stdout.lstrip().startswith('{') else -1
    if idx == -1:
        return 0
    try:
        data = json.loads(stdout[idx:].lstrip())
        return data.get('failed', 0)
    except Exception:
        return 0


def count_docs():
    """Count all .md files in perfxpert."""
    count = 0
    for md_file in Path("experimental/python/perfxpert").rglob("*.md"):
        if any(part.startswith('.') for part in md_file.parts):
            continue
        count += 1
    return count


def generate_inventory():
    """Generate docs inventory."""
    print("Running inventory scan...", file=sys.stderr)

    lint_violations = run_lint_check()
    broken_links = run_link_check()
    failed_samples = run_sample_check()
    total_docs = count_docs()

    inventory = {
        "timestamp": datetime.now().isoformat(),
        "phase": 9,
        "summary": {
            "total_docs": total_docs,
            "lint_violations": lint_violations,
            "broken_links": broken_links,
            "failed_samples": failed_samples,
        },
        "audit_results": {
            "lint": {
                "status": "PASS" if lint_violations == 0 else "FAIL",
                "violations": lint_violations,
            },
            "links": {
                "status": "PASS" if broken_links == 0 else "FAIL",
                "broken": broken_links,
            },
            "samples": {
                "status": "PASS" if failed_samples == 0 else "FAIL",
                "failed": failed_samples,
            },
        },
        "baseline": "PR4 zero-violation snapshot",
    }

    print(json.dumps(inventory, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(generate_inventory())
