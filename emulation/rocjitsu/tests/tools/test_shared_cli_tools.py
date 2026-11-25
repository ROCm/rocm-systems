# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Integration checks for every native CLI consumer, without a GPU or SDK."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cli_fixtures import write_code_object

translate, pretranslate, discovery, fuzz, launcher = (
    Path(arg).resolve() for arg in sys.argv[1:]
)


class SharedCliToolsProcessTest(unittest.TestCase):
    def invoke(self, tool, *args, status=0, cwd=None):
        result = subprocess.run(
            [str(tool), *map(str, args)], cwd=cwd, capture_output=True, timeout=20
        )
        self.assertEqual(
            result.returncode,
            status,
            (tool.name, args, result.returncode, result.stdout, result.stderr),
        )
        return result.stdout + result.stderr

    def test_shared_cli_process_behavior(self):
        for tool, failure in (
            (translate, 1),
            (pretranslate, 2),
            (discovery, 1),
            (fuzz, 2),
            (launcher, 1),
        ):
            self.assertIn(b"Usage:", self.invoke(tool, "--help"))
            self.assertIn(b"--help, -h", self.invoke(tool, "-h"))
            self.assertIn(
                b"unknown flag",
                self.invoke(tool, "--help", "--unknown", status=failure),
            )

        launcher_help = self.invoke(launcher, "--help")
        self.assertIn(b"Launch overrides apply only to this invocation", launcher_help)
        self.assertIn(b"do not rewrite the source config", launcher_help)
        self.assertIn(b"off disables checks", launcher_help)
        self.assertIn(b"on enables ordinary checks", launcher_help)
        self.assertIn(b"all includes gfx1250 XCNT checks", launcher_help)
        self.assertIn(b"ROCJITSU_ENABLE_VFIO=ON", launcher_help)
        self.assertIn(b"memory-backend-memfd", launcher_help)
        self.assertIn(b"memory-backend=mem", launcher_help)

        self.assertIn(
            b"application arguments must follow --",
            self.invoke(launcher, "bare-application", status=1),
        )
        self.assertIn(
            b"application arguments must follow --",
            self.invoke(launcher, "bare-application", "--", "app", status=1),
        )
        self.assertIn(
            b"application arguments must follow --",
            self.invoke(
                launcher,
                "bare-application",
                "--config",
                "missing.json",
                "--",
                "app",
                status=1,
            ),
        )
        self.assertIn(b"requires a value", self.invoke(launcher, "--config", status=1))
        self.assertIn(
            b"nonempty path", self.invoke(launcher, "--vfio-socket=", status=1)
        )
        self.assertIn(
            b"requires a value", self.invoke(pretranslate, "--store-root", status=2)
        )
        self.assertIn(
            b"unsigned integer",
            self.invoke(launcher, "--cpu-thread-budget=4294967296", status=1),
        )
        self.assertIn(
            b"unsigned integer",
            self.invoke(launcher, "--vfio-ready-fd=2147483648", status=1),
        )
        self.assertIn(
            b"expected a decimal or hexadecimal",
            self.invoke(translate, "--code-object-index=0x", status=1),
        )
        self.assertIn(
            b"at most 65535",
            self.invoke(translate, "--debug-conservative-liveness=65536", status=1),
        )
        self.assertIn(
            b"unsupported value",
            self.invoke(translate, "--input-target=unknown", status=1),
        )
        self.assertIn(
            b"unsupported value",
            self.invoke(translate, "--output-mode=unknown", status=1),
        )
        self.assertIn(
            b"requires a value", self.invoke(translate, "--input-revision", status=1)
        )
        self.assertIn(b"Usage:", self.invoke(fuzz, "unexpected-positional", status=2))

        with tempfile.TemporaryDirectory(prefix="rj-cli-tools-") as directory:
            root = Path(directory).resolve()
            source = root / "-input.co"
            write_code_object(source)
            self.assertIn(
                b"no generation named",
                self.invoke(discovery, "missing", root / "output.bin", status=1),
            )
            self.invoke(discovery, "--", "gfx1250", "-discovery.bin", cwd=root)
            self.assertGreater((root / "-discovery.bin").stat().st_size, 0)
            # After a generation operand, --help is a legacy output filename.
            self.invoke(discovery, "gfx1250", "--help", cwd=root)
            self.assertGreater((root / "--help").stat().st_size, 0)
            # The pretranslator should distinguish expected target rejections from I/O failures.
            skipped = self.invoke(
                pretranslate,
                f"--store-root={root / 'store'}",
                "--portable",
                "--",
                source,
            )
            self.assertIn(b"skipped=1 failed=0", skipped)
            skipped = self.invoke(
                pretranslate,
                "--store-root",
                root / "store",
                "--fail-on-skipped",
                source,
                status=1,
            )
            self.assertIn(b"skipped=1 failed=0", skipped)
            failed = self.invoke(
                pretranslate,
                "--store-root",
                root / "store",
                root / "absent.co",
                status=1,
            )
            self.assertIn(b"cannot read", failed)
            # Repeated options exercise the full decoder path and prove the last value wins.
            sample = root / "sample.bin"
            sample.write_bytes(bytes.fromhex("000080bf") + bytes(12))
            replay = self.invoke(
                fuzz, f"--input={sample}", "--json", "--target=bogus", "--target=gfx950"
            )
            self.assertIn(b'"mnemonic":"s_nop"', replay.replace(b" ", b""))
            self.assertIn(
                b"unsupported decoder target: bogus",
                self.invoke(
                    fuzz,
                    f"--input={sample}",
                    "--json",
                    "--target=gfx950",
                    "--target=bogus",
                    status=2,
                ),
            )
            # A leading-dash input reaches domain validation. This minimal fixture
            # intentionally has no kernel descriptor; full translation is covered by the smoke suite.
            translated = self.invoke(
                translate,
                "--input-target=gfx950",
                "--output-target=gfx950",
                "--output-mode=code-object",
                "--",
                "-input.co",
                cwd=root,
                status=3,
            )
            self.assertIn(b"kernel descriptors are required", translated)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
