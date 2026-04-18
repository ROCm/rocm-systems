#!/usr/bin/env python3
"""Unit tests for docs/inventory.py."""

import subprocess
import json
from pathlib import Path


def test_inventory_exists():
    """Inventory script must exist."""
    inventory_script = Path("docs/inventory.py")
    assert inventory_script.exists(), "docs/inventory.py does not exist"


def test_inventory_output_valid_json():
    """Inventory output must be valid JSON."""
    result = subprocess.run(
        ["python3", "docs/inventory.py"],
        capture_output=True,
        text=True,
    )
    # Parse the JSON output (skip stderr messages)
    for line in result.stdout.split('\n'):
        if line.startswith('{'):
            # Found JSON start
            json_str = line + '\n' + '\n'.join(
                l for l in result.stdout.split('\n')[1:]
            )
            try:
                data = json.loads(json_str)
                assert "summary" in data
                assert "audit_results" in data
                return
            except json.JSONDecodeError:
                pass
    assert False, "Could not parse valid JSON from inventory output"


def test_inventory_has_required_fields():
    """Inventory must have required fields."""
    result = subprocess.run(
        ["python3", "docs/inventory.py"],
        capture_output=True,
        text=True,
    )

    # Extract JSON
    for line in result.stdout.split('\n'):
        if line.startswith('{'):
            json_str = '\n'.join(
                l for l in result.stdout.split('\n')[result.stdout.split('\n').index(line):]
            )
            data = json.loads(json_str)
            break

    required_fields = [
        "timestamp",
        "phase",
        "summary",
        "audit_results",
    ]

    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    # Check summary
    summary = data["summary"]
    assert "total_docs" in summary
    assert "lint_violations" in summary
    assert "broken_links" in summary
    assert "failed_samples" in summary


def test_inventory_summary_counts():
    """Inventory summary counts should be non-negative."""
    result = subprocess.run(
        ["python3", "docs/inventory.py"],
        capture_output=True,
        text=True,
    )

    for line in result.stdout.split('\n'):
        if line.startswith('{'):
            json_str = '\n'.join(
                l for l in result.stdout.split('\n')[result.stdout.split('\n').index(line):]
            )
            data = json.loads(json_str)
            break

    summary = data["summary"]
    assert summary["total_docs"] > 0, "Should have at least 1 doc"
    assert summary["lint_violations"] >= 0
    assert summary["broken_links"] >= 0
    assert summary["failed_samples"] >= 0


if __name__ == "__main__":
    test_inventory_exists()
    print("✓ test_inventory_exists passed")
