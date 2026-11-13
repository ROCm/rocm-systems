"""Tests for how test_jax_collective.py checks out the JAX test sources.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import test_jax_collective as tjc

# jax 0.11.2 on PyPI: jax.version._git_hash, the commit of tag jax-v0.11.2.
JAX_0_11_2 = "32544801e26115ac1794926d027148abf3baf009"

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
       "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main"]


def make_checkout(path: Path, files=("tests/pmap_test.py",)) -> str:
    """Create a git repo at *path* with *files* committed; return HEAD."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(GIT + ["init", "-q"], cwd=path, check=True)
    for name in files:
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text("")
    subprocess.run(GIT + ["add", "-A"], cwd=path, check=True)
    subprocess.run(GIT + ["commit", "-q", "-m", "x"], cwd=path, check=True)
    return subprocess.run(GIT + ["rev-parse", "HEAD"], cwd=path, check=True,
                          capture_output=True, text=True).stdout.strip()


class CloneJaxTestSourcesTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.jax_src = Path(tmp.name) / "jax-src"
        self.set_installed_hash(JAX_0_11_2)

    def set_installed_hash(self, git_hash, has_attr=True):
        fake_jax = types.ModuleType("jax")
        fake_jax.__version__ = "0.11.2"
        fake_jax.version = types.SimpleNamespace(**({"_git_hash": git_hash} if has_attr else {}))
        patcher = mock.patch.dict(sys.modules, {"jax": fake_jax})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_offline(self):
        """Run clone_jax_test_sources with real git, failing on any fetch."""
        real_run = subprocess.run

        def run(cmd, **kwargs):
            if "fetch" in cmd:
                raise AssertionError(f"unexpected fetch: {cmd}")
            return real_run(cmd, **kwargs)

        with mock.patch.object(tjc.subprocess, "run", side_effect=run):
            tjc.clone_jax_test_sources(self.jax_src)

    def test_fetches_tests_at_installed_commit(self):
        calls = []

        def run(cmd, **kwargs):
            calls.append((cmd, kwargs.get("cwd")))
            if cmd[:2] == ["git", "checkout"]:
                (self.jax_src / "tests").mkdir()
                (self.jax_src / "tests" / "pmap_test.py").touch()
            return subprocess.CompletedProcess(cmd, 0)

        with mock.patch.object(tjc.subprocess, "run", side_effect=run):
            tjc.clone_jax_test_sources(self.jax_src)
        self.assertEqual([cmd for cmd, _ in calls], [
            ["git", "init", "-q"],
            ["git", "remote", "add", "origin", "https://github.com/jax-ml/jax.git"],
            ["git", "sparse-checkout", "set", "tests/"],
            ["git", "fetch", "--depth=1", "--filter=blob:none", "origin", JAX_0_11_2],
            ["git", "checkout", "-q", "FETCH_HEAD"],
        ])
        self.assertTrue(all(cwd == self.jax_src for _, cwd in calls))

    def test_fetches_into_existing_empty_directory(self):
        self.jax_src.mkdir()
        with mock.patch.object(tjc.subprocess, "run") as run:
            with self.assertRaises(SystemExit):  # the mock fetches no tests/
                tjc.clone_jax_test_sources(self.jax_src)
        self.assertEqual(run.call_args_list[0].args[0], ["git", "init", "-q"])

    def test_rejects_hash_that_is_not_a_full_commit_sha(self):
        for bad in ("", "Unknown", None, JAX_0_11_2[:7], JAX_0_11_2.upper()):
            with self.subTest(git_hash=bad):
                self.set_installed_hash(bad)
                with mock.patch.object(tjc.subprocess, "run") as run:
                    with self.assertRaises(SystemExit):
                        tjc.clone_jax_test_sources(self.jax_src)
                run.assert_not_called()

    def test_rejects_jax_without_git_hash(self):
        self.set_installed_hash(None, has_attr=False)
        with self.assertRaises(SystemExit):
            tjc.clone_jax_test_sources(self.jax_src)

    def test_fails_if_fetch_has_no_tests(self):
        with mock.patch.object(tjc.subprocess, "run"):
            with self.assertRaises(SystemExit):
                tjc.clone_jax_test_sources(self.jax_src)

    def test_reuses_checkout_at_installed_commit(self):
        self.set_installed_hash(make_checkout(self.jax_src))
        self.run_offline()

    def test_rejects_checkout_at_another_commit(self):
        make_checkout(self.jax_src)
        with self.assertRaises(SystemExit):
            self.run_offline()

    def test_rejects_checkout_without_a_commit(self):
        self.jax_src.mkdir()
        subprocess.run(GIT + ["init", "-q"], cwd=self.jax_src, check=True)
        with self.assertRaises(SystemExit):
            self.run_offline()

    def test_rejects_checkout_at_installed_commit_without_tests(self):
        self.set_installed_hash(make_checkout(self.jax_src, files=("README",)))
        with self.assertRaises(SystemExit):
            self.run_offline()

    def test_rejects_directory_that_is_not_a_checkout(self):
        (self.jax_src / "tests").mkdir(parents=True)
        (self.jax_src / "tests" / "pmap_test.py").touch()
        with mock.patch.object(tjc.subprocess, "run") as run:
            with self.assertRaises(SystemExit):
                tjc.clone_jax_test_sources(self.jax_src)
        run.assert_not_called()

    def test_rejects_checked_out_jax_package(self):
        make_checkout(self.jax_src, files=("tests/pmap_test.py", "jax/__init__.py"))
        with mock.patch.object(tjc.subprocess, "run") as run:
            with self.assertRaises(SystemExit):
                tjc.clone_jax_test_sources(self.jax_src)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
