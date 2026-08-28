#!/usr/bin/env python3
"""Focused schema checks for code.json SQTT funcmap metadata."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import rocprof_trace_decoder.codegen as codegen


class CodegenFuncmapTest(unittest.TestCase):
    def generate(self, funcmaps: list[codegen._SqttFuncmap]):
        code_objects = [
            codegen.CodeObject("packed.co", 10),
            codegen.CodeObject("legacy.co", 11),
        ][: len(funcmaps)]
        with (
            patch.object(codegen, "_normalize_code_objects", return_value=code_objects),
            patch.object(codegen, "build_address_ranges", return_value=[]),
            patch.object(codegen, "parse_disassembly", return_value=[]),
            patch.object(codegen, "read_symbol_labels", return_value=({}, {})),
            patch.object(codegen, "_read_sqtt_funcmap", side_effect=funcmaps),
        ):
            return codegen.generate_code_artifacts(code_objects).code_index.document

    def test_adds_packed_metadata_without_changing_funcmap_rows(self):
        doc = self.generate(
            [
                codegen._parse_sqtt_funcmap(
                    b"M:shader_clock_bits=12;shader_clock_shift=4\n"
                    b"F:7:scope@source.cpp:1\n"
                    b"U:8:user\n"
                    b"R:8:extra_payload_count=1\n"
                )
            ]
        )

        self.assertEqual(
            doc["sqtt_funcmap"],
            [[10, 7, "F", "scope", "source.cpp:1", 0], [10, 8, "U", "user", "", 0]],
        )
        self.assertTrue(all(len(row) == 6 for row in doc["sqtt_funcmap"]))
        self.assertEqual(doc["sqtt_funcmap_layout"], [[10, 12, 4]])
        self.assertEqual(doc["sqtt_funcmap_payloads"], [[10, 8, 1]])

    def test_legacy_funcmap_exports_rows_without_a_packed_layout(self):
        doc = self.generate(
            [codegen._parse_sqtt_funcmap(b"F:7:scope\nR:7:extra_payload_count=1\n")]
        )

        self.assertEqual(doc["sqtt_funcmap"], [[10, 7, "F", "scope", "", 0]])
        self.assertNotIn("sqtt_funcmap_layout", doc)
        self.assertEqual(doc["sqtt_funcmap_payloads"], [[10, 7, 1]])

    def test_incomplete_packed_layout_is_ignored(self):
        doc = self.generate([codegen._parse_sqtt_funcmap(b"M:shader_clock_bits=12\nF:7:scope\n")])

        self.assertEqual(doc["sqtt_funcmap"], [[10, 7, "F", "scope", "", 0]])
        self.assertNotIn("sqtt_funcmap_layout", doc)

    def test_packed_layout_does_not_require_named_entries(self):
        doc = self.generate([codegen._parse_sqtt_funcmap(b"M:shader_clock_bits=12;shader_clock_shift=4\n")])

        self.assertNotIn("sqtt_funcmap", doc)
        self.assertEqual(doc["sqtt_funcmap_layout"], [[10, 12, 4]])


if __name__ == "__main__":
    unittest.main()
