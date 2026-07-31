#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path
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
