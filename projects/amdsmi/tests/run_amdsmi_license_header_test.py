#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
run_amdsmi_license_header_test.py
=================================

Exercises check_license_headers.py against synthetic files. A header rewrite can
strip the opening ``/*`` of an old block comment while leaving its ``*/`` tail
behind; the two SPDX lines are then present but the file no longer compiles, so
a presence-only check passes it through. These cases pin both the presence and
the well-formedness behavior.

Scratch files are created under projects/amdsmi so the checker can resolve them
to project-relative paths, and are removed when the test exits.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
CHECKER = TESTS_DIR / "check_license_headers.py"

SPDX = "// Copyright Advanced Micro Devices, Inc.\n// SPDX-License-Identifier: MIT\n"
ORPHANED_TAIL = "\n * See LICENSE file for full license text.\n */\n"

CASES = [
    ("well_formed.cc", SPDX + "\n#include <cstdio>\nint main() { return 0; }\n", 0),
    ("missing_header.cc", "#include <cstdio>\nint main() { return 0; }\n", 1),
    ("orphaned_tail.cc", SPDX + ORPHANED_TAIL + "\n#include <cstdio>\n", 1),
    ("orphaned_tail.h", SPDX + ORPHANED_TAIL + "\n#ifndef X_H_\n#define X_H_\n#endif\n", 1),
    ("intact_block.cc", SPDX + "\n/* a normal block comment */\n#include <cstdio>\n", 0),
    # .in build templates are out of scope even without a header.
    ("skipped_template.py.in", '__version__ = "1.0"\n', 0),
]


def run_checker(path: Path) -> int:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(path)], capture_output=True, text=True
    ).returncode


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp:
        for name, text, expected in CASES:
            path = Path(tmp) / name
            path.write_text(text)
            actual = run_checker(path)
            verdict = "PASS" if actual == expected else "FAIL"
            print(f"{verdict}: {name} exit={actual} expected={expected}")
            if actual != expected:
                failures.append(name)

    if failures:
        sys.exit("check_license_headers.py gave the wrong verdict for: " + ", ".join(failures))

    print(f"PASS: all {len(CASES)} license-header cases behaved as expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
