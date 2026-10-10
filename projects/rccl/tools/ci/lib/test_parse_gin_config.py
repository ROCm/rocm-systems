#!/usr/bin/env python3
"""Unit tests for parse_gin_config.py."""

import json
import tempfile
import unittest
from pathlib import Path

from parse_gin_config import _FIELD_SEP, format_rows, parse_config


class ParseGinConfigTest(unittest.TestCase):
    def test_empty_env_preserves_gtest_filter_in_args(self) -> None:
        """Regression: adjacent tabs made bash assign --gtest_filter to env_flags."""
        config = parse_config(
            _write_json(
                {
                    "mca": "",
                    "debug_env": [],
                    "tests": [
                        {
                            "name": "fixtures-case",
                            "kind": "fixtures",
                            "bin": "rccl-UnitTestsFixtures",
                            "env": [],
                            "args": "--gtest_filter=GinRocshmemGdaTemplateTest.*",
                        }
                    ],
                }
            )
        )
        rows = format_rows(config)
        test_row = next(r for r in rows if r.startswith("test" + _FIELD_SEP))
        parts = test_row.split(_FIELD_SEP)
        self.assertEqual(len(parts), 6)
        self.assertEqual(parts[0], "test")
        self.assertEqual(parts[1], "fixtures-case")
        self.assertEqual(parts[2], "fixtures")
        self.assertEqual(parts[3], "rccl-UnitTestsFixtures")
        self.assertEqual(parts[4], "")
        self.assertEqual(parts[5], "--gtest_filter=GinRocshmemGdaTemplateTest.*")

    def test_pytest_kind_preserves_args(self) -> None:
        # Both pytest files run through one case. parse_config and format_rows
        # branch on neither bin nor kind -- format_rows joins positionally -- so
        # a second standalone copy of this test could not fail on its own. The
        # env lists differ in length so the join is pinned at both arities.
        cases = [
            (
                "bcast-pytest",
                "test_Broadcast.py",
                ["NCCL_CUMEM_ENABLE=1", "HSA_NO_SCRATCH_RECLAIM=1"],
                "-x NCCL_CUMEM_ENABLE=1 -x HSA_NO_SCRATCH_RECLAIM=1",
            ),
            (
                "rs-pytest",
                "test_ReduceScatterGinSdma.py",
                ["NCCL_CUMEM_ENABLE=1"],
                "-x NCCL_CUMEM_ENABLE=1",
            ),
        ]
        for name, binary, env, expected_env in cases:
            with self.subTest(bin=binary):
                config = parse_config(
                    _write_json(
                        {
                            "mca": "",
                            "debug_env": [],
                            "tests": [
                                {
                                    "name": name,
                                    "kind": "pytest",
                                    "bin": binary,
                                    "env": env,
                                    "args": "-k GinSdma -v",
                                }
                            ],
                        }
                    )
                )
                rows = format_rows(config)
                test_row = next(r for r in rows if r.startswith("test" + _FIELD_SEP))
                parts = test_row.split(_FIELD_SEP)
                self.assertEqual(parts[1], name)
                self.assertEqual(parts[2], "pytest")
                self.assertEqual(parts[3], binary)
                self.assertEqual(parts[4], expected_env)
                self.assertEqual(parts[5], "-k GinSdma -v")


def _write_json(data: dict) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(data, handle)
    handle.close()
    return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
