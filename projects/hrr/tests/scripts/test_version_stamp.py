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


ASSUME_OTHER_OWNER = {"GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"}


def git_env(extra_env):
    env = os.environ.copy()
    env.pop("GIT_CONFIG_GLOBAL", None)
    env.pop("GIT_CONFIG_SYSTEM", None)
    env.update(extra_env)
    return env


def git_enforces_ownership():
    """True when git refuses a repository it believes someone else owns.

    Not every git does: one older than the ownership check ignores the
    variable, and one whose system or global config says safe.directory = *
    (as on GitHub's hosted runners) trusts every directory. Where git trusts
    everything, the stamp reading the repository is correct and the test has
    nothing to prove.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env = git_env(ASSUME_OTHER_OWNER)
        subprocess.run(["git", "init", "-q", tmp], check=True, env=env)
        proc = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=tmp,
                              check=False, capture_output=True, env=env)
        return proc.returncode != 0


class VersionStampTest(unittest.TestCase):
    def stamp(self, extra_env):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "hrr_build_info.h"
            env = git_env(extra_env)
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

    @unittest.skipUnless(git_enforces_ownership(),
                         "this git does not enforce the ownership check")
    def test_checkout_owned_by_someone_else_reports_unknown(self):
        # Git treats the checkout as owned by another user. A stamp that
        # bypasses the ownership check, by whatever route, still reads it.
        text = self.stamp(ASSUME_OTHER_OWNER)
        self.assertIn('HRR_BUILD_REV "unknown"', text)


if __name__ == "__main__":
    unittest.main()
