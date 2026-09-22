#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

TOOL_PATH = Path(__file__).with_name("list-test-selectors.py")
SPEC = importlib.util.spec_from_file_location("list_test_selectors", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class SelectorConfigTest(unittest.TestCase):
    def test_loads_target_specific_selectors(self) -> None:
        config = self._load(
            {
                "gfx1100": {
                    "cts": [
                        "cts.gfx1100.fpsan.full_id",
                        "unqualified_name",
                    ]
                },
                "gfx950": {"cts": ["cts.gfx950.fpsan.full_id"]},
            }
        )

        self.assertEqual(
            TOOL.selectors_for(config, "gfx1100", "cts"),
            ("cts.gfx1100.fpsan.full_id", "unqualified_name"),
        )
        self.assertEqual(TOOL.selectors_for(config, "gfx1201", "cts"), ())

    def test_rejects_invalid_configurations(self) -> None:
        invalid_payloads = (
            None,
            [],
            {},
            {"not-a-target": {"cts": ["test"]}},
            {"gfx1100": {}},
            {"gfx1100": {"cts": []}},
            {"gfx1100": {"cts": [""]}},
            {"gfx1100": {"cts": [" leading-space"]}},
            {"gfx1100": {"cts": ["contains,comma"]}},
            {"gfx1100": {"cts": ["duplicate", "duplicate"]}},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    self._load(payload)

    def test_reports_invalid_json_as_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "selectors.json"
            path.write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                TOOL.load_selector_config(path)

    def _load(self, payload):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "selectors.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return TOOL.load_selector_config(path)


if __name__ == "__main__":
    unittest.main()
