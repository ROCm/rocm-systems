#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The version stamp must not turn off git's ownership check."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

STAMP = Path(__file__).resolve().parents[2] / "playback" / "HrrVersionStamp.cmake"
PLAYBACK = STAMP.parent


class VersionStampTest(unittest.TestCase):
    def test_no_wildcard_safe_directory_and_owned_repo_still_stamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "hrr_build_info.h"
            env = os.environ.copy()
            env.pop("GIT_CONFIG_GLOBAL", None)
            env.pop("GIT_CONFIG_SYSTEM", None)
            proc = subprocess.run(
                ["cmake", "-DSRC_DIR=" + str(PLAYBACK), "-DOUT_FILE=" + str(out),
                 "-P", str(STAMP)],
                check=False, capture_output=True, text=True, env=env)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertFalse((Path(tmp) / "hrr_version_stamp.gitconfig").exists())
            text = out.read_text()
            self.assertNotIn("directory = *", text)
            self.assertIn('#define HRR_BUILD_REV "', text)
            self.assertNotIn('HRR_BUILD_REV "unknown"', text)


if __name__ == "__main__":
    unittest.main()
