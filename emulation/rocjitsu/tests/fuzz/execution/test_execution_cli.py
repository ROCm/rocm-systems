#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class ExecutionFuzzCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        binary = os.environ.get("RJ_EXECUTION_FUZZ")
        if not binary:
            raise unittest.SkipTest("RJ_EXECUTION_FUZZ is not set")
        cls.binary = Path(binary)

    def run_execution_fuzz(
        self, *arguments: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.binary), *(str(argument) for argument in arguments)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_help_lists_execution_modes(self) -> None:
        result = self.run_execution_fuzz("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--target auto|NAME", result.stdout)
        self.assertIn("--threads auto|N", result.stdout)
        self.assertIn("--async auto|on|off", result.stdout)
        self.assertIn("--emit-seeds DIRECTORY", result.stdout)

    def test_raw_replay_reports_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "input.bin"
            input_path.write_bytes(b"\x05\x00\x00\x00\x80\xbf")
            result = self.run_execution_fuzz(
                "--input", input_path, "--target", "gfx1250"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            stats = json.loads(result.stdout)
            self.assertEqual(stats["target"], "gfx1250")
            self.assertEqual(stats["inputs"], 1)
            self.assertEqual(stats["inputs_executed"], 1)
            self.assertEqual(stats["executed"], 2)

    def test_target_override_ignores_input_selector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "input.bin"
            input_path.write_bytes(b"\x00\x00\x00\x00\xc0\xbf")
            result = self.run_execution_fuzz(
                "--input", input_path, "--target", "gfx1250"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            stats = json.loads(result.stdout)
            self.assertEqual(stats["target"], "gfx1250")
            self.assertEqual(stats["inputs_executed"], 1)
            self.assertEqual(stats["executed"], 2)
            automatic = self.run_execution_fuzz("--input", input_path)
            self.assertEqual(automatic.returncode, 0, automatic.stderr)
            self.assertEqual(json.loads(automatic.stdout)["executed"], 0)

    def test_emit_raw_seeds_creates_afl_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            seed_directory = Path(temporary) / "seeds"
            result = self.run_execution_fuzz(
                "--emit-seeds", seed_directory, "--target", "gfx1250"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            seeds = sorted(seed_directory.glob("*.bin"))
            self.assertEqual(len(seeds), 128)
            self.assertEqual(
                {path.stat().st_size for path in seeds}, {8, 32, 256, 1024}
            )
            self.assertTrue(all(path.read_bytes()[0] == 10 for path in seeds))
            self.assertEqual({path.read_bytes()[1] for path in seeds}, {0, 1, 6, 7})

    def test_invalid_async_mode_is_rejected(self) -> None:
        result = self.run_execution_fuzz("--input", __file__, "--async", "sometimes")
        self.assertEqual(result.returncode, 2)
        self.assertIn("async mode must be auto, on, or off", result.stderr)

    def test_newly_advertised_targets_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "input.bin"
            input_path.write_bytes(b"\x00")
            for target in ("gfx1100", "gfx1151"):
                result = self.run_execution_fuzz(
                    "--input", input_path, "--target", target
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["target"], target)


if __name__ == "__main__":
    unittest.main()
