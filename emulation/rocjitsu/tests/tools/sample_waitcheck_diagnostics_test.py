#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Protect a lossless diagnostic population from aliased sample outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(sys.argv.pop(1))


class SampleWaitcheckDiagnosticsTest(unittest.TestCase):
    def test_rejects_input_and_output_aliases_before_writing(self) -> None:
        record = {
            "schema": "rj-waitcheck-diagnostic-v1",
            "input": "kernel.hsaco",
            "target": "gfx1201",
            "code_object_index": 0,
            "kernel_name": "kernel",
            "kernel_entry": 0,
            "kernel_entry_hex": "0x0",
            "counter": "km",
            "access": "use",
            "register": {"class": "vgpr", "index": 0, "width": 1},
            "section": ".text",
            "section_offset": 8,
            "section_offset_hex": "0x8",
            "file_offset": 264,
            "producer_section_offset": 0,
            "producer_section_offset_hex": "0x0",
            "producer_file_offset": 256,
            "required_count": 0,
            "instruction": "v_mov_b32 v1, v0",
            "producer_instruction": "s_load_b32 s0, s[0:1], 0",
            "message": "missing wait",
            "repro_command": "rj_waitcheck kernel.hsaco",
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            population = root / "population.jsonl"
            population.write_text((json.dumps(record) + "\n") * 2)
            original_hash = hashlib.sha256(population.read_bytes()).hexdigest()
            sample = root / "sample.jsonl"
            hardlink = root / "hardlink.jsonl"
            hardlink.hardlink_to(population)
            for output, extra in [
                (population, []),
                (sample, ["--unique-output", str(hardlink)]),
                (sample, ["--metadata", str(population)]),
                (sample, ["--triage-markdown", str(population)]),
            ]:
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), str(population), str(output),
                     "--count", "1", *extra],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("aliases", result.stderr)
                self.assertEqual(hashlib.sha256(population.read_bytes()).hexdigest(),
                                 original_hash)


if __name__ == "__main__":
    unittest.main()
