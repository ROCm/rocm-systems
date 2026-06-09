#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
import sys
import unittest


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools" / "hip_api_manifest"
sys.path.insert(0, str(TOOLS_DIR))

from api_diff import diff_manifests  # noqa: E402


def base_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "functions": {
            "hipInit": {
                "return_type": "hipError_t",
                "params": [{"type": "unsigned int", "name": "flags"}],
            },
        },
        "records": {
            "hipExample_t": {
                "fields": [{"type": "int", "name": "value"}],
            },
        },
        "enums": {
            "hipExampleEnum": {
                "constants": [
                    {"name": "hipExampleZero", "value": "0"},
                ],
            },
        },
    }


class ManifestDiffTests(unittest.TestCase):
    def test_unchanged_manifest(self) -> None:
        diff = diff_manifests(base_manifest(), base_manifest())
        self.assertEqual(diff.status, "UNCHANGED")
        self.assertEqual(diff.changes, ())

    def test_added_function_is_minor(self) -> None:
        old = base_manifest()
        new = base_manifest()
        functions = dict(new["functions"])
        functions["hipNewApi"] = {"return_type": "hipError_t", "params": []}
        new["functions"] = functions
        diff = diff_manifests(old, new)
        self.assertEqual(diff.status, "MINOR")
        self.assertEqual(diff.changes[0].kind, "function_added")

    def test_changed_function_signature_is_major(self) -> None:
        old = base_manifest()
        new = base_manifest()
        functions = dict(new["functions"])
        functions["hipInit"] = {
            "return_type": "hipError_t",
            "params": [{"type": "unsigned long", "name": "flags"}],
        }
        new["functions"] = functions
        diff = diff_manifests(old, new)
        self.assertEqual(diff.status, "MAJOR")
        self.assertEqual(diff.changes[0].kind, "function_signature_changed")

    def test_record_layout_change_is_major(self) -> None:
        old = base_manifest()
        new = base_manifest()
        records = dict(new["records"])
        records["hipExample_t"] = {
            "fields": [
                {"type": "int", "name": "value"},
                {"type": "int", "name": "added"},
            ],
        }
        new["records"] = records
        diff = diff_manifests(old, new)
        self.assertEqual(diff.status, "MAJOR")
        self.assertEqual(diff.changes[0].kind, "record_layout_changed")

    def test_added_enum_constant_is_minor(self) -> None:
        old = base_manifest()
        new = base_manifest()
        enums = dict(new["enums"])
        enums["hipExampleEnum"] = {
            "constants": [
                {"name": "hipExampleZero", "value": "0"},
                {"name": "hipExampleOne", "value": "1"},
            ],
        }
        new["enums"] = enums
        diff = diff_manifests(old, new)
        self.assertEqual(diff.status, "MINOR")
        self.assertEqual(diff.changes[0].kind, "enum_constants_added")

    def test_changed_enum_value_is_major(self) -> None:
        old = base_manifest()
        new = base_manifest()
        enums = dict(new["enums"])
        enums["hipExampleEnum"] = {
            "constants": [
                {"name": "hipExampleZero", "value": "7"},
            ],
        }
        new["enums"] = enums
        diff = diff_manifests(old, new)
        self.assertEqual(diff.status, "MAJOR")
        self.assertEqual(diff.changes[0].kind, "enum_changed")


if __name__ == "__main__":
    unittest.main()
