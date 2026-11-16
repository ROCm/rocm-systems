# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the actual benchmark executable without a GPU or assembler."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cli_fixtures import write_code_object

benchmark = Path(sys.argv[1]).resolve()


class DecodeBenchmarkCliProcessTest(unittest.TestCase):
    def invoke(self, *args, status=0):
        result = subprocess.run(
            [str(benchmark), *map(str, args)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(
            result.returncode, status, (args, result.returncode, result.stderr)
        )
        return result.stdout if status == 0 else result.stderr

    def test_process_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.co"
            write_code_object(source)
            existing = self.invoke(
                source,
                "--target",
                "gfx950",
                "--iterations",
                "2",
                "--invalid-limit",
                "1",
            )
            self.assertIn("2 valid instructions", existing)
            self.assertIn("valid: 4 attempts", existing)
            self.assertIn("invalid: 2 attempts", existing)
            self.assertEqual(existing.count("0 unexpected"), 2)
            extended = self.invoke(
                "-t=gfx950", "-n=2", "--invalid-limit=0", "--", source
            )
            self.assertIn("valid: 4 attempts", extended)
            help_text = self.invoke(source, "--help")
            self.assertIn("--iterations, -n", help_text)
            self.assertIn("Decoder settings:", help_text)
            self.assertIn("exactly one", self.invoke(status=1))
            self.assertIn("exactly one", self.invoke("", status=1))
            self.assertIn(
                "unsupported decoder target",
                self.invoke(source, "--target=bad", status=1),
            )
            self.assertIn(
                "expected an integer", self.invoke(source, "--iterations=0", status=1)
            )
            self.assertIn("requires a value", self.invoke(source, "--target", status=1))
            self.assertIn(
                "valid: 4 attempts",
                self.invoke(source, "-n=1", "--iterations=2", "--invalid-limit=0"),
            )
            self.assertIn("exactly one", self.invoke(source, source, status=1))
            self.assertIn("unknown flag", self.invoke("--help", "--unknown", status=1))
            self.assertIn(
                "does not match", self.invoke(source, "--target=gfx942", status=2)
            )


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
