#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for check_hrr_api_compat.py."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import check_hrr_api_compat as compat  # noqa: E402


def header(version: int, entries: list[tuple[str, int]]) -> str:
    enum_entries = "\n".join(
        f"    HRR_API_{name} = {value}," for name, value in entries
    )
    return f"""\
#define HRR_VERSION ((uint16_t){version}u)
typedef enum hrr_api_id {{
{enum_entries}
    HRR_API_COUNT = {len(entries)}
}} hrr_api_id_t;
"""


class ScratchHeaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name) / "base.h"
        self.current = Path(self.tmp.name) / "current.h"

    def write(self, base: str, current: str) -> None:
        self.base.write_text(base, encoding="utf-8")
        self.current.write_text(current, encoding="utf-8")

    def run_main(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = compat.main(
                [
                    "--base-header",
                    str(self.base),
                    "--current-header",
                    str(self.current),
                ]
            )
        return rc, out.getvalue(), err.getvalue()


class CompatibilityTest(ScratchHeaders):
    def test_unchanged_map_does_not_require_a_bump(self):
        text = header(6, [("RUNTIME", 0), ("COMPILER", 1)])
        self.write(text, text)
        rc, out, _ = self.run_main()
        self.assertEqual(rc, 0)
        self.assertIn("version 6 -> 6", out)

    def test_compiler_tail_append_does_not_require_a_bump(self):
        self.write(
            header(6, [("RUNTIME", 0), ("COMPILER", 1)]),
            header(6, [("RUNTIME", 0), ("COMPILER", 1), ("COMPILER_NEW", 2)]),
        )
        rc, _, _ = self.run_main()
        self.assertEqual(rc, 0)

    def test_runtime_append_that_moves_compiler_id_requires_a_bump(self):
        self.write(
            header(6, [("RUNTIME", 0), ("COMPILER", 1)]),
            header(6, [("RUNTIME", 0), ("RUNTIME_NEW", 1), ("COMPILER", 2)]),
        )
        rc, _, err = self.run_main()
        self.assertEqual(rc, 1)
        self.assertIn("HRR_API_COMPILER: 1 -> 2", err)
        self.assertIn("increment HRR_VERSION", err)

    def test_one_bump_covers_multiple_moved_ids(self):
        self.write(
            header(6, [("RUNTIME", 0), ("COMPILER_A", 1), ("COMPILER_B", 2)]),
            header(
                7,
                [
                    ("RUNTIME", 0),
                    ("RUNTIME_NEW_A", 1),
                    ("RUNTIME_NEW_B", 2),
                    ("COMPILER_A", 3),
                    ("COMPILER_B", 4),
                ],
            ),
        )
        rc, _, _ = self.run_main()
        self.assertEqual(rc, 0)

    def test_version_must_not_decrease(self):
        self.write(header(7, [("API", 0)]), header(6, [("API", 0)]))
        rc, _, err = self.run_main()
        self.assertEqual(rc, 1)
        self.assertIn("decreased from 7 to 6", err)


class ParseFailureTest(ScratchHeaders):
    def assert_parse_error(self, current: str, expected: str):
        self.write(header(6, [("API", 0)]), current)
        rc, _, err = self.run_main()
        self.assertEqual(rc, 2)
        self.assertIn(expected, err)

    def test_rejects_duplicate_ids(self):
        self.assert_parse_error(
            header(6, [("API_A", 0), ("API_B", 0)]), "duplicate API ID 0"
        )

    def test_rejects_malformed_enum_entries(self):
        self.assert_parse_error(
            """\
#define HRR_VERSION ((uint16_t)6u)
typedef enum hrr_api_id {
    HRR_API_A,
    HRR_API_COUNT = 1
} hrr_api_id_t;
""",
            "malformed hrr_api_id_t entry",
        )

    def test_rejects_missing_version(self):
        self.assert_parse_error(
            """\
typedef enum hrr_api_id {
    HRR_API_A = 0,
    HRR_API_COUNT = 1
} hrr_api_id_t;
""",
            "HRR_VERSION definition not found",
        )


if __name__ == "__main__":
    unittest.main()
