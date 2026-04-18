#!/usr/bin/env python3
"""
docs/inventory.py — Generate inventory of all audited docs
Runs lint, link-checker, and sample-executor; outputs JSON inventory
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

def run_lint_check():
    """Run lint.sh and count violations."""
    result = subprocess.run(
        ["bash", "docs/lint.sh"],
        capture_output=True,
        text=True,
    )
    violations = result.stdout.count("FAIL:")
    return violations


def run_link_check():
    """Run link-checker.py and count broken links."""
    result = subprocess.run(
        ["python3", "docs/link-checker.py", "experimental/python/perfxpert"],
        capture_output=True,
        text=True,
    )
    # Parse CSV output
    broken = 0
    for line in result.stdout.split('\n'):
        if line and not line.startswith('Found') and not line.startswith('file,'):
            broken += 1
    return broken


def run_sample_check():
    """Run test-samples.py and count failures."""
    result = subprocess.run(
        ["python3", "docs/test-samples.py", "experimental/python/perfxpert"],
        capture_output=True,
        text=True,
    )
    try:
        data = json.loads(result.stdout)
        return data.get('failed', 0)
    except:
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
        "baseline": "PR1 initial scan",
    }

    print(json.dumps(inventory, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(generate_inventory())
