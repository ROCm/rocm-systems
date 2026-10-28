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
    def stamp(self, extra_env):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "hrr_build_info.h"
            env = os.environ.copy()
            env.pop("GIT_CONFIG_GLOBAL", None)
            env.pop("GIT_CONFIG_SYSTEM", None)
            env.update(extra_env)
            proc = subprocess.run(
                ["cmake", "-DSRC_DIR=" + str(PLAYBACK), "-DOUT_FILE=" + str(out),
                 "-P", str(STAMP)],
                check=False, capture_output=True, text=True, env=env)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return out.read_text()

    def test_owned_repo_still_stamps(self):
        text = self.stamp({})
        self.assertIn('#define HRR_BUILD_REV "', text)
        self.assertNotIn('HRR_BUILD_REV "unknown"', text)

    def test_checkout_owned_by_someone_else_reports_unknown(self):
        # Git treats the checkout as owned by another user. A stamp that
        # bypasses the ownership check, by whatever route, still reads it.
        text = self.stamp({"GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"})
        self.assertIn('HRR_BUILD_REV "unknown"', text)


if __name__ == "__main__":
    unittest.main()
