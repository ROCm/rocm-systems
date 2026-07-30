#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Manual self-check for tools/counter_grouping_inspector.py (developer tooling).

Not part of the automatic CI/pytest suite. Run from project root:

    python3 tools/test_counter_grouping_inspector_manual.py
"""

import subprocess
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TOOLS_DIR.parent
_COUNTER_GROUPING_INSPECTOR = _TOOLS_DIR / "counter_grouping_inspector.py"

_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.mi_gpu_spec import mi_gpu_specs  # noqa: E402


def check_all_supported_archs() -> None:
    supported_archs = list(mi_gpu_specs.get_gpu_series_dict().keys())
    assert supported_archs, "Should have at least one supported architecture"

    failed: list[str] = []
    results: dict[str, dict[str, int]] = {}

    for arch in supported_archs:
        result = subprocess.run(
            [sys.executable, str(_COUNTER_GROUPING_INSPECTOR), "--arch", arch],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=_PROJECT_ROOT,
        )

        if result.returncode != 0:
            stderr_preview = result.stderr[:500] if result.stderr else "None"
            failed.append(
                f"{arch}: Return code {result.returncode}\n  stderr: {stderr_preview}"
            )
            results[arch] = {
                "return_code": result.returncode,
                "buckets": 0,
                "counter_assignments": 0,
            }
            continue

        buckets = 0
        counter_assignments = 0
        for line in result.stdout.split("\n"):
            if "Summary:" in line and "bucket" in line:
                parts = line.split()
                for idx, part in enumerate(parts):
                    if "bucket" in part and idx > 0:
                        try:
                            buckets = int(parts[idx - 1])
                        except ValueError:
                            pass
                    is_assignment = (
                        idx + 1 < len(parts) and "assignment" in parts[idx + 1]
                    )
                    if "counter" in part and is_assignment:
                        try:
                            counter_assignments = int(parts[idx - 1])
                        except (ValueError, IndexError):
                            pass

        results[arch] = {
            "return_code": result.returncode,
            "buckets": buckets,
            "counter_assignments": counter_assignments,
        }

        if buckets == 0:
            failed.append(f"{arch}: buckets = 0 (expected > 0)")
        if counter_assignments == 0:
            failed.append(f"{arch}: counter_assignments = 0 (expected > 0)")

    print("\n" + "=" * 60)
    print("Counter Grouping Inspector (manual check) Results:")
    print("=" * 60)
    for arch, data in results.items():
        success = (
            data["return_code"] == 0
            and data["buckets"] > 0
            and data["counter_assignments"] > 0
        )
        mark = "✓" if success else "✗"
        print(
            f"  {mark} {arch}: return_code={data['return_code']}, "
            f"buckets={data['buckets']}, "
            f"counter_assignments={data['counter_assignments']}"
        )
    print("=" * 60)

    assert not failed, (
        "counter_grouping_inspector failed for "
        f"{len(failed)} architecture(s):\n" + "\n".join(f"  - {err}" for err in failed)
    )


def check_invalid_arch_exits_with_argparse_error() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_COUNTER_GROUPING_INSPECTOR),
            "--arch",
            "__not_a_valid_arch__",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 2
    combined = (result.stderr or "") + (result.stdout or "")
    assert "invalid choice" in combined.lower()


def main() -> None:
    check_all_supported_archs()
    check_invalid_arch_exits_with_argparse_error()
    print("Manual checks passed.")


if __name__ == "__main__":
    main()
