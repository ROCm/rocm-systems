#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

SCRIPT = Path(__file__).parents[2] / "tools/vfio_guest_firmware.py"
SPEC = importlib.util.spec_from_file_location("vfio_guest_firmware", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
firmware = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(firmware)


class FirmwareFixtureTest(unittest.TestCase):
    def test_generates_only_the_named_synthetic_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "amdgpu"
            firmware.generate(output)

            self.assertEqual(
                {path.name for path in output.iterdir()}, set(firmware.FIXTURES)
            )
            for path in output.iterdir():
                blob = path.read_bytes()
                size, header_size = struct.unpack_from("<II", blob)
                self.assertEqual(size, len(blob))
                self.assertGreaterEqual(header_size, 32)
                self.assertEqual(len(blob[: firmware.FIXED_HEADER_BYTES]), 0x100)
                self.assertIn(struct.pack("<I", firmware.SENTINEL), blob)

    def test_output_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            firmware.generate(first)
            firmware.generate(second)

            for name in firmware.FIXTURES:
                self.assertEqual(
                    (first / name).read_bytes(), (second / name).read_bytes()
                )

    def test_refuses_to_replace_a_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "gc_12_1_0_mec.bin").write_bytes(b"preserve")

            with self.assertRaisesRegex(firmware.FixtureError, "refusing to replace"):
                firmware.generate(output)

            self.assertEqual((output / "gc_12_1_0_mec.bin").read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
