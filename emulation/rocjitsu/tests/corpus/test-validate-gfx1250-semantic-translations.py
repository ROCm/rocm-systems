#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

VALIDATOR_PATH = Path(__file__).with_name("validate-gfx1250-semantic-translations.py")
SPEC = importlib.util.spec_from_file_location(
    "gfx1250_semantic_validator", VALIDATOR_PATH
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)

SOURCE_ID = "fnv1a64:0123456789abcdef"
OTHER_SOURCE_ID = "fnv1a64:fedcba9876543210"


def runtime_record(
    *,
    source_id: str,
    changed: int,
    outcome: str = "translated",
    translation_status: int = 0,
    status: int = 0,
) -> str:
    return (
        "[hsa-hotswap-rj] eager translation "
        f"source_id={source_id} "
        "input_revision=b0 output_revision=a0 "
        f"outcome={outcome} changed={changed} "
        "input_bytes=64 output_bytes=96 "
        f"translation_status={translation_status} status={status}\n"
    )


class LoadExpectationsTest(unittest.TestCase):
    def load_expectations(self, payload: object) -> dict[str, int]:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "expectations.json"
            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            return VALIDATOR.load_expectations(path)

    def load_entry(self, entry: dict[str, object]) -> dict[str, int]:
        return self.load_expectations({"schema": 1, "tests": {"semantic_test": entry}})

    def assert_invalid(self, entry: dict[str, object]) -> None:
        with self.assertRaises(ValueError):
            self.load_entry(entry)

    def test_accepts_valid_rewrite_counts(self) -> None:
        for value in (0, 3):
            with self.subTest(value=value):
                self.assertEqual(
                    self.load_entry({"instructions_requiring_rewrite": value}),
                    {"semantic_test": value},
                )

    def test_rejects_invalid_rewrite_counts(self) -> None:
        entries = (
            {"instructions_requiring_rewrite": -1},
            {"instructions_requiring_rewrite": False},
            {},
            {"instructions_requiring_rewrite": 0.0},
            {"instructions_requiring_rewrite": "0"},
            {"instructions_requiring_rewrite": None},
        )
        for entry in entries:
            with self.subTest(entry=entry):
                self.assert_invalid(entry)

    def test_rejects_invalid_payloads(self) -> None:
        payloads = (
            {"tests": {"semantic_test": {"instructions_requiring_rewrite": 0}}},
            {
                "schema": 2,
                "tests": {"semantic_test": {"instructions_requiring_rewrite": 0}},
            },
            {"schema": 1},
            {"schema": 1, "tests": {}},
            {"schema": 1, "tests": []},
            {
                "schema": 1,
                "tests": {"": {"instructions_requiring_rewrite": 0}},
            },
            {"schema": 1, "tests": {"semantic_test": 0}},
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    self.load_expectations(payload)


class RuntimeEvidenceTest(unittest.TestCase):
    def test_accepts_expected_successes_with_one_matching_fixture(self) -> None:
        runtime_text = runtime_record(
            source_id=SOURCE_ID,
            changed=3,
        ) + runtime_record(
            source_id=OTHER_SOURCE_ID,
            changed=0,
        )

        self.assertEqual(
            VALIDATOR.validate_runtime_evidence(runtime_text, SOURCE_ID, 3, 2),
            (1, 2),
        )

    def test_rejects_matching_success_plus_failed_record(self) -> None:
        runtime_text = runtime_record(
            source_id=SOURCE_ID,
            changed=3,
        ) + runtime_record(
            source_id=OTHER_SOURCE_ID,
            changed=0,
            outcome="translation_failed",
            translation_status=1,
            status=1,
        )

        with self.assertRaisesRegex(ValueError, "1 successful records"):
            VALIDATOR.validate_runtime_evidence(runtime_text, SOURCE_ID, 3, 2)

    def test_rejects_anonymous_legacy_records(self) -> None:
        runtime_text = (
            "[hsa-hotswap-rj] eager translated "
            "input_bytes=64 output_bytes=96 status=0\n"
        )

        with self.assertRaisesRegex(
            ValueError, "no successful runtime record matching"
        ):
            VALIDATOR.validate_runtime_evidence(runtime_text, SOURCE_ID, 3, 1)


if __name__ == "__main__":
    unittest.main()
