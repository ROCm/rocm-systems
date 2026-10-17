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
        config = parse_config(
            _write_json(
                {
                    "mca": "",
                    "debug_env": [],
                    "tests": [
                        {
                            "name": "bcast-pytest",
                            "kind": "pytest",
                            "bin": "test_Broadcast.py",
                            "env": ["NCCL_CUMEM_ENABLE=1", "HSA_NO_SCRATCH_RECLAIM=1"],
                            "args": "-k GinSdma -v",
                        }
                    ],
                }
            )
        )
        rows = format_rows(config)
        test_row = next(r for r in rows if r.startswith("test" + _FIELD_SEP))
        parts = test_row.split(_FIELD_SEP)
        self.assertEqual(parts[2], "pytest")
        self.assertEqual(parts[3], "test_Broadcast.py")
        self.assertEqual(
            parts[4], "-x NCCL_CUMEM_ENABLE=1 -x HSA_NO_SCRATCH_RECLAIM=1"
        )
        self.assertEqual(parts[5], "-k GinSdma -v")

    def test_gating_row_lists_gating_tests(self) -> None:
        config = parse_config(
            _write_json(
                {
                    "tests": [
                        {"name": "sdma-d3", "kind": "rccl-tests", "bin": "alltoall_perf"},
                        {"name": "gda-ut", "kind": "fixtures", "bin": "rccl-UnitTestsFixtures"},
                    ],
                    "gating_tests": ["sdma-d3"],
                }
            )
        )
        rows = format_rows(config)
        self.assertIn("gating" + _FIELD_SEP + "sdma-d3", rows)

    def test_gating_row_is_empty_without_gating_tests(self) -> None:
        config = parse_config(
            _write_json(
                {"tests": [{"name": "sdma-d3", "kind": "rccl-tests", "bin": "alltoall_perf"}]}
            )
        )
        self.assertIn("gating" + _FIELD_SEP, format_rows(config))

    def test_unknown_gating_test_is_rejected(self) -> None:
        """A renamed test must not silently drop out of the nightly gate."""
        path = _write_json(
            {
                "tests": [{"name": "sdma-d3", "kind": "rccl-tests", "bin": "alltoall_perf"}],
                "gating_tests": ["sdma-d3", "sdma-ag-d3"],
            }
        )
        with self.assertRaisesRegex(ValueError, "sdma-ag-d3"):
            parse_config(path)


def _write_json(data: dict) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(data, handle)
    handle.close()
    return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
