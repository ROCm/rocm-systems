# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit test for tools/counter_grouping_inspector.py

Single test that loops through all supported architectures and verifies:
- Return code of counter_grouping_inspector.py script is 0
- Buckets and counter assignments are non-zero for each architecture
"""

import subprocess
import sys
from pathlib import Path

# Path to counter_grouping_inspector.py script
COUNTER_GROUPING_INSPECTOR_SCRIPT = (
    Path(__file__).resolve().parent.parent / "tools" / "counter_grouping_inspector.py"
)

# Ensure src directory is in Python path for imports
_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


def test_counter_grouping_inspector_all_supported_archs():
    """Test counter_grouping_inspector.py works for all supported architectures.

    For each supported architecture:
    1. Run counter_grouping_inspector.py --arch <arch> and verify return code is 0
    2. Parse output to verify buckets > 0 and counter assignments > 0
    """
    from utils.mi_gpu_spec import mi_gpu_specs

    supported_archs = list(mi_gpu_specs.get_gpu_series_dict().keys())
    assert len(supported_archs) > 0, "Should have at least one supported architecture"

    failed_archs = []
    results = {}

    for arch in supported_archs:
        # Run the counter_grouping_inspector.py script for this architecture
        result = subprocess.run(
            [sys.executable, str(COUNTER_GROUPING_INSPECTOR_SCRIPT), "--arch", arch],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Check return code
        if result.returncode != 0:
            stderr_preview = result.stderr[:500] if result.stderr else "None"
            failed_archs.append(
                f"{arch}: Return code {result.returncode}\n  stderr: {stderr_preview}"
            )
            results[arch] = {
                "return_code": result.returncode,
                "buckets": 0,
                "counter_assignments": 0,
            }
            continue

        # Parse output to find bucket and counter assignment counts
        # Look for "Summary: X bucket(s), Y counter assignment(s)." line
        stdout = result.stdout
        buckets = 0
        counter_assignments = 0

        for line in stdout.split("\n"):
            if "Summary:" in line and "bucket" in line:
                # Parse line like "Summary: 5 bucket(s), 120 counter assignment(s)."
                parts = line.split()
                for i, part in enumerate(parts):
                    if "bucket" in part and i > 0:
                        try:
                            buckets = int(parts[i - 1])
                        except ValueError:
                            pass
                    # Check for counter assignment pattern
                    is_assignment = i + 1 < len(parts) and "assignment" in parts[i + 1]
                    if "counter" in part and is_assignment:
                        try:
                            counter_assignments = int(parts[i - 1])
                        except (ValueError, IndexError):
                            pass

        results[arch] = {
            "return_code": result.returncode,
            "buckets": buckets,
            "counter_assignments": counter_assignments,
        }

        # Verify non-zero values
        if buckets == 0:
            failed_archs.append(f"{arch}: buckets = 0 (expected > 0)")
        if counter_assignments == 0:
            failed_archs.append(f"{arch}: counter_assignments = 0 (expected > 0)")

    # Report results
    print("\n" + "=" * 60)
    print("Counter Grouping Inspector Test Results:")
    print("=" * 60)
    for arch, data in results.items():
        is_success = (
            data["return_code"] == 0
            and data["buckets"] > 0
            and data["counter_assignments"] > 0
        )
        status = "✓" if is_success else "✗"
        print(
            f"  {status} {arch}: return_code={data['return_code']}, "
            f"buckets={data['buckets']}, "
            f"counter_assignments={data['counter_assignments']}"
        )
    print("=" * 60)

    # Assert no failures
    assert len(failed_archs) == 0, (
        f"Counter grouping inspector failed for {len(failed_archs)} architecture(s):\n"
        + "\n".join(f"  - {err}" for err in failed_archs)
    )


def test_counter_grouping_inspector_invalid_arch():
    """Invalid --arch must exit non-zero (argparse rejects unknown choices)."""
    result = subprocess.run(
        [
            sys.executable,
            str(COUNTER_GROUPING_INSPECTOR_SCRIPT),
            "--arch",
            "__not_a_valid_arch__",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    # argparse exits with status 2 for argument errors (see SystemExit in argparse docs)
    # including when --arch is not in the script's allowed choices list.
    assert result.returncode == 2
    err = (result.stderr or "") + (result.stdout or "")
    assert "invalid choice" in err.lower()
