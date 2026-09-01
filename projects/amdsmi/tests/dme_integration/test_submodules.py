#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stdlib-only unit tests for ``dme_integration.submodules``."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Allow running directly as well as via unittest discovery from
# ``projects/amdsmi/tests``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dme_integration import submodules


class RewriteGitmodulesSshToHttpsTest(unittest.TestCase):
    def _write(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".gitmodules", delete=False, encoding="utf-8"
        )
        tmp.write(text)
        tmp.close()
        path = Path(tmp.name)
        self.addCleanup(path.unlink)
        return path

    def test_rewrites_scp_style_url(self):
        path = self._write('[submodule "proto"]\n\turl = git@github.com:org/repo.git\n')
        self.assertTrue(submodules._rewrite_gitmodules_ssh_to_https(path))
        self.assertEqual(
            path.read_text(), '[submodule "proto"]\n\turl = https://github.com/org/repo.git\n'
        )

    def test_rewrites_ssh_scheme_url(self):
        path = self._write('[submodule "proto"]\n\turl = ssh://git@github.com/org/repo.git\n')
        self.assertTrue(submodules._rewrite_gitmodules_ssh_to_https(path))
        self.assertEqual(
            path.read_text(), '[submodule "proto"]\n\turl = https://github.com/org/repo.git\n'
        )

    def test_already_https_returns_false(self):
        content = '[submodule "proto"]\n\turl = https://github.com/org/repo.git\n'
        path = self._write(content)
        self.assertFalse(submodules._rewrite_gitmodules_ssh_to_https(path))
        self.assertEqual(path.read_text(), content)

    def test_nonexistent_file_returns_false(self):
        self.assertFalse(
            submodules._rewrite_gitmodules_ssh_to_https(Path("/nonexistent/.gitmodules"))
        )

    def test_idempotent(self):
        path = self._write('[submodule "proto"]\n\turl = git@github.com:org/repo.git\n')
        self.assertTrue(submodules._rewrite_gitmodules_ssh_to_https(path))
        rewritten = path.read_text()
        # Second call is a no-op: nothing left to rewrite.
        self.assertFalse(submodules._rewrite_gitmodules_ssh_to_https(path))
        self.assertEqual(path.read_text(), rewritten)


class CloneAtRefTest(unittest.TestCase):
    _SHA = "779265ed4f4423af9f0c52de11c5e92d51d8cd00"

    def test_tag_uses_clone_dash_b(self):
        with mock.patch.object(submodules, "run") as run:
            submodules._clone_at_ref("repo", Path("/dst"), "v1.4.2")
        # A tag/branch clones directly with -b and never checks out a SHA.
        (cmd,), _ = run.call_args
        self.assertIn("-b", cmd)
        self.assertIn("v1.4.2", cmd)
        self.assertEqual(run.call_count, 1)

    def test_sha_clones_then_checks_out(self):
        with mock.patch.object(submodules, "run") as run:
            submodules._clone_at_ref("repo", Path("/dst"), self._SHA)
        cmds = [c.args[0] for c in run.call_args_list]
        # SHA path: plain clone (no -b) then a detached checkout of the SHA.
        self.assertNotIn("-b", cmds[0])
        self.assertEqual(cmds[1][:3], ["git", "checkout", "--detach"])
        self.assertIn(self._SHA, cmds[1])

    def test_sha_recurse_updates_submodules(self):
        with mock.patch.object(submodules, "run") as run:
            submodules._clone_at_ref("repo", Path("/dst"), self._SHA, recurse=True)
        cmds = [c.args[0] for c in run.call_args_list]
        self.assertIn("--recurse-submodules", cmds[0])
        self.assertTrue(any("submodule" in c and "update" in c for c in cmds))


if __name__ == "__main__":
    unittest.main()
