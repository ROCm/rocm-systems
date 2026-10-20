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
        # Nothing in parse_config or format_rows branches on bin or kind; the two
        # rows only pin the positional join at two env-list arities.
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

    def test_shipped_gin_tests_json_pytest_bins_are_supported(self) -> None:
        """Catch typos in shipped pytest entries before an 8-GPU node burns steps 1-6."""
        shipped = Path(__file__).resolve().parent / "gin-tests.json"
        config = parse_config(shipped)
        supported = {"test_Broadcast.py", "test_ReduceScatterGinSdma.py"}
        pytest_bins = {t.bin for t in config.tests if t.kind == "pytest"}
        self.assertTrue(pytest_bins, "shipped gin-tests.json has no pytest entries")
        self.assertTrue(
            pytest_bins <= supported,
            "unsupported pytest bin(s) in gin-tests.json: {}".format(
                sorted(pytest_bins - supported)
            ),
        )


def _write_json(data: dict) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(data, handle)
    handle.close()
    return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
