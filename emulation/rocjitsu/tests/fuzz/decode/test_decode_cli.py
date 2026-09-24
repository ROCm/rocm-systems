#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


class DecodeCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        decoder = os.environ.get("RJ_DECODE_FUZZ")
        if not decoder:
            raise unittest.SkipTest("RJ_DECODE_FUZZ is not set")
        cls.decoder = Path(decoder)

    def run_decoder(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.decoder), *arguments],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_replays_exact_window_as_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample = Path(temporary) / "s_nop.bin"
            sample.write_bytes(struct.pack("<4I", 0xBF800000, 0, 0, 0))
            process = self.run_decoder("--input", str(sample), "--json")
            self.assertEqual(process.returncode, 0, process.stderr)
            record = json.loads(process.stdout)
            self.assertEqual(record["status"], "valid")
            self.assertEqual(record["mnemonic"], "s_nop")

    def test_rejects_wrong_sized_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample = Path(temporary) / "short.bin"
            sample.write_bytes(bytes(15))
            process = self.run_decoder("--input", str(sample), "--json")
            self.assertEqual(process.returncode, 2)
            self.assertIn("exactly 16 bytes", process.stderr)

    def test_replay_preserves_concrete_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample = Path(temporary) / "gfx1251_only.bin"
            sample.write_bytes(struct.pack("<4I", 0x7E083AFA, 0xFF015002, 0, 0))

            gfx1251 = self.run_decoder(
                "--input", str(sample), "--json", "--target", "gfx1251"
            )
            self.assertEqual(gfx1251.returncode, 0, gfx1251.stderr)
            self.assertEqual(json.loads(gfx1251.stdout)["mnemonic"], "v_mov_b64_e32")

            gfx1250 = self.run_decoder(
                "--input", str(sample), "--json", "--target", "gfx1250"
            )
            self.assertEqual(gfx1250.returncode, 0, gfx1250.stderr)
            self.assertEqual(json.loads(gfx1250.stdout)["status"], "invalid")

    def test_rejects_non_amdgpu_target(self):
        process = self.run_decoder("--emit-seeds", "unused", "--target", "risc-v")
        self.assertEqual(process.returncode, 2)
        self.assertIn("unsupported decoder target", process.stderr)

    def test_rejects_json_outside_replay_mode(self):
        for arguments in (("--afl", "--json"), ("--emit-seeds", "unused", "--json")):
            with self.subTest(arguments=arguments):
                process = self.run_decoder(*arguments)
                self.assertEqual(process.returncode, 2)
                self.assertIn("Usage:", process.stderr)

    def test_emits_and_rejects_reusing_seed_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            seeds = Path(temporary) / "seeds"
            first = self.run_decoder("--emit-seeds", str(seeds), "--target", "gfx950")
            self.assertEqual(first.returncode, 0, first.stderr)
            files = sorted(seeds.iterdir())
            self.assertGreater(len(files), 100)
            self.assertTrue(all(path.stat().st_size == 16 for path in files))

            second = self.run_decoder("--emit-seeds", str(seeds), "--target", "gfx950")
            self.assertEqual(second.returncode, 2)
            self.assertIn("not empty", second.stderr)

    def test_emits_gfx1251_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            seeds = Path(temporary) / "seeds"
            process = self.run_decoder(
                "--emit-seeds", str(seeds), "--target", "gfx1251"
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            files = list(seeds.iterdir())
            self.assertGreater(len(files), 100)
            self.assertTrue(all(path.stat().st_size == 16 for path in files))


if __name__ == "__main__":
    unittest.main()
