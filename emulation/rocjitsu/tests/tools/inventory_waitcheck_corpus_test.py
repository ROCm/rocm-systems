#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify that a HIP host object yields standalone AMDGPU ELF artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest


SCRIPT, WAITCHECK, INPUT = map(Path, sys.argv[1:4])
del sys.argv[1:4]


class InventoryWaitcheckCorpusTest(unittest.TestCase):
    def test_extracts_hip_fatbin(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            extracted = root / "objects"
            manifest = root / "manifest.jsonl"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(INPUT), "--waitcheck", str(WAITCHECK),
                 "--project", "pytorch", "--source", "test-fixture", "--provenance",
                 "llvm", "--extract-dir", str(extracted), "--target", "gfx942",
                 "--output", str(manifest)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(rows[0]["artifact_count"], 1)
            artifact = next(row for row in rows if row["kind"] == "artifact")
            path = Path(artifact["artifact"])
            self.assertNotEqual(path, INPUT)
            self.assertTrue(path.is_relative_to(extracted))
            data = path.read_bytes()
            self.assertEqual(data[:4], b"\x7fELF")
            self.assertEqual(struct.unpack_from("<H", data, 18)[0], 224)
            self.assertEqual(artifact["container"], str(INPUT))
            self.assertEqual(artifact["container_target"], "gfx942")

    def test_parses_container_once_for_bulk_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            extracted = root / "objects"
            manifest = root / "manifest.jsonl"
            log = root / "calls.log"
            wrapper = root / "waitcheck-wrapper.sh"
            wrapper.write_text(
                "#!/bin/sh\n"
                f"printf '%s\\n' \"$*\" >> {shlex.quote(str(log))}\n"
                f"exec {shlex.quote(str(WAITCHECK))} \"$@\"\n"
            )
            wrapper.chmod(0o700)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(INPUT), "--waitcheck", str(wrapper),
                 "--project", "pytorch", "--source", "test-fixture", "--provenance",
                 "llvm", "--extract-dir", str(extracted), "--target", "gfx942",
                 "--jobs", "16", "--output", str(manifest)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = log.read_text().splitlines()
            self.assertEqual(sum("--extract-code-objects" in call for call in calls), 1)
            self.assertFalse(any("--extract-code-object " in call for call in calls))
            self.assertFalse(any("--list-code-objects" in call for call in calls))

    def test_target_filter_keeps_artifacts_from_earlier_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            container = root / "raw-container.bin"
            container.write_bytes(
                b"prefix" + INPUT.read_bytes() + b"middle"
                + INPUT.with_name("vector_add_gfx1100.o").read_bytes()
            )
            extracted = root / "objects"
            artifacts = []
            for target in ("gfx942", "gfx1100"):
                manifest = root / f"{target}.jsonl"
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), str(container), "--waitcheck",
                     str(WAITCHECK), "--project", "pytorch", "--source",
                     "test-fixture", "--provenance", "llvm", "--extract-dir",
                     str(extracted), "--target", target, "--output", str(manifest)],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                rows = [json.loads(line) for line in manifest.read_text().splitlines()]
                self.assertEqual(rows[0]["artifact_count"], 1)
                artifacts.append(next(row for row in rows if row["kind"] == "artifact"))

            for artifact in artifacts:
                path = Path(artifact["artifact"])
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 artifact["artifact_sha256"])

    def test_recursive_rerun_ignores_its_output_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            corpus = Path(temp) / "corpus"
            corpus.mkdir()
            (corpus / "vector_add.o").write_bytes(INPUT.read_bytes())
            extracted = corpus / "objects"
            manifest = corpus / "manifest.jsonl"
            artifacts = []
            for _ in range(2):
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), str(corpus), "--recursive", "--waitcheck",
                     str(WAITCHECK), "--project", "pytorch", "--source",
                     "test-fixture", "--provenance", "llvm", "--extract-dir",
                     str(extracted), "--target", "gfx942", "--output", str(manifest)],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                rows = [json.loads(line) for line in manifest.read_text().splitlines()]
                self.assertEqual(rows[0]["artifact_count"], 1)
                artifacts.append(next(row for row in rows if row["kind"] == "artifact"))
            self.assertEqual(artifacts[0]["artifact"], artifacts[1]["artifact"])
            self.assertEqual(artifacts[0]["artifact_sha256"], artifacts[1]["artifact_sha256"])


if __name__ == "__main__":
    unittest.main()
