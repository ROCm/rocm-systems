#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stdlib-only unit tests for ``dynolog_integration.build``."""

import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Allow running directly as well as via unittest discovery from
# ``projects/rdc/tests``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynolog_integration import build


def _version_code_quiet(rocm_dir: Path) -> int:
    """Call rocm_version_code, swallowing its ``::warning::`` stdout.

    The fallback paths emit a GitHub workflow ``::warning::`` command; without
    this, running the tests inside the workflow would surface those as
    misleading run-level annotations.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return build.rocm_version_code(rocm_dir)


class RocmVersionCodeTest(unittest.TestCase):
    def _rocm_dir_with_version(self, contents: str) -> Path:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        info = tmp / ".info"
        info.mkdir()
        (info / "version").write_text(contents)
        return tmp

    def test_parses_major_minor_patch(self):
        rocm_dir = self._rocm_dir_with_version("7.2.0-12345\n")
        self.assertEqual(build.rocm_version_code(rocm_dir), 70200)

    def test_parses_two_digit_minor(self):
        rocm_dir = self._rocm_dir_with_version("10.1.3\n")
        self.assertEqual(build.rocm_version_code(rocm_dir), 100103)

    def test_missing_version_file_falls_back(self):
        empty = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        self.assertEqual(_version_code_quiet(empty), build._DEFAULT_ROCM_VERSION_CODE)

    def test_unparseable_version_falls_back(self):
        rocm_dir = self._rocm_dir_with_version("not-a-version\n")
        self.assertEqual(_version_code_quiet(rocm_dir), build._DEFAULT_ROCM_VERSION_CODE)


if __name__ == "__main__":
    unittest.main()
