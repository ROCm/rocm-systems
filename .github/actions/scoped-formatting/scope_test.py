"""Exercise formatting selection against real Git histories and workflow scopes."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from scope import matches_scope

SCRIPT = Path(__file__).with_name("scope.py")
WORKFLOWS = SCRIPT.parents[2] / "workflows"


def workflow_scope(name: str) -> dict[str, str]:
    workflow = yaml.load(
        (WORKFLOWS / f"{name}-formatting.yml").read_text(), Loader=yaml.BaseLoader
    )
    return next(
        step["with"]
        for step in workflow["jobs"]["pre-commit"]["steps"]
        if step.get("uses") == "./.github/actions/scoped-formatting"
    )


class GitScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self.environment = {
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }
        self.git("-c", "init.templateDir=", "init", "--initial-branch=base")
        self.git("config", "user.name", "Formatting Test")
        self.git("config", "user.email", "formatting@example.com")
        self.git("commit", "--allow-empty", "-m", "base")
        self.git("checkout", "-b", "feature")

    def git(self, *arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=self.repository,
            env=self.environment,
            stderr=subprocess.PIPE,
            text=True,
        ).strip()

    def write(self, path: str, text: str = "content\n") -> None:
        target = self.repository / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self) -> None:
        self.git("add", "--all")
        self.git("commit", "-m", "changes")

    def run_scope(
        self,
        *,
        include: str = r"^runtimes/",
        exclude: str = "",
        base: str = "base",
        head: str = "feature",
        success: bool = True,
    ) -> dict[str, str]:
        output = self.root / "output"
        output.write_text("")
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(SCRIPT),
                "--base",
                base,
                "--head",
                head,
                "--include",
                include,
                "--exclude",
                exclude,
                "--output",
                str(output),
            ],
            cwd=self.repository,
            env=self.environment,
            capture_output=True,
            text=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(), "")
        return dict(line.split("=", 1) for line in output.read_text().splitlines())

    def test_match_after_three_thousand_unrelated_paths(self) -> None:
        for index in range(3100):
            self.write(f"docs/file-{index:04d}.txt")
        self.write("runtimes/last.rs")
        self.commit()
        self.assertEqual(self.run_scope()["affected"], "true")

    def test_unaffected_and_empty_changes_succeed(self) -> None:
        self.assertEqual(self.run_scope()["affected"], "false")
        self.write("docs/readme.md")
        self.commit()
        self.assertEqual(self.run_scope()["affected"], "false")

    def test_base_only_changes_do_not_activate_scope(self) -> None:
        ancestor = self.git("rev-parse", "base")
        self.git("checkout", "base")
        self.write("runtimes/base-only.rs")
        self.commit()
        self.git("checkout", "feature")
        self.write("docs/pr.md")
        self.commit()
        result = self.run_scope()
        self.assertEqual(result["affected"], "false")
        self.assertEqual(result["base"], ancestor)
        self.assertEqual(result["head"], self.git("rev-parse", "feature"))

    def test_sparse_checkout_keeps_out_of_worktree_changes(self) -> None:
        self.write(".github/actions/scoped-formatting/placeholder")
        self.commit()
        self.git("branch", "-f", "base", "feature")
        self.write("runtimes/hidden.rs")
        self.commit()
        self.git("sparse-checkout", "set", ".github/actions/scoped-formatting")
        self.assertFalse((self.repository / "runtimes/hidden.rs").exists())
        self.assertEqual(self.run_scope()["affected"], "true")

    def test_blobless_checkout_needs_no_source_blobs_or_network(self) -> None:
        self.write(".github/actions/scoped-formatting/placeholder")
        self.commit()
        self.git("branch", "-f", "base", "feature")
        self.write("runtimes/unfetched.rs", "source blob must stay on the server\n")
        self.commit()
        self.git("config", "uploadpack.allowFilter", "true")
        clone = self.root / "clone"
        self.git(
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            self.repository.as_uri(),
            str(clone),
        )
        self.repository = clone
        self.git("sparse-checkout", "set", ".github/actions/scoped-formatting")
        self.git("checkout", "feature")
        self.assertIn("?", self.git("rev-list", "--objects", "--missing=print", "HEAD"))
        self.git("remote", "set-url", "origin", (self.root / "unavailable").as_uri())
        self.assertEqual(self.run_scope(base="origin/base")["affected"], "true")

    def test_rename_out_of_scope_and_deletion_activate_scope(self) -> None:
        self.write("runtimes/original.rs")
        self.commit()
        self.git("branch", "-f", "base", "feature")
        self.git("mv", "runtimes/original.rs", "moved.rs")
        self.commit()
        self.assertEqual(self.run_scope()["affected"], "true")
        self.git("checkout", "base", "--", "runtimes/original.rs")
        self.git("rm", "moved.rs")
        self.commit()
        self.git("rm", "runtimes/original.rs")
        self.commit()
        self.assertEqual(self.run_scope()["affected"], "true")

    def test_filename_bytes_and_delimiters_are_preserved(self) -> None:
        for path in ("runtimes/a b.rs", "runtimes/a\nb.rs", "runtimes/\udcff.rs"):
            self.write(path)
        self.commit()
        self.assertEqual(
            self.run_scope(include=r"^runtimes/a\nb\.rs$")["affected"], "true"
        )
        self.assertEqual(self.run_scope(include="\udcff")["affected"], "true")

    def test_missing_revision_and_invalid_pattern_fail_without_outputs(self) -> None:
        self.run_scope(base="missing", success=False)
        self.run_scope(include="[", success=False)

    def test_multiple_merge_bases_fail_without_outputs(self) -> None:
        parent = self.git("rev-parse", "base")
        tree = self.git("rev-parse", "base^{tree}")
        left = self.git("commit-tree", tree, "-p", parent, "-m", "left")
        right = self.git("commit-tree", tree, "-p", parent, "-m", "right")
        base = self.git(
            "commit-tree", tree, "-p", left, "-p", right, "-m", "base merge"
        )
        head = self.git(
            "commit-tree", tree, "-p", right, "-p", left, "-m", "head merge"
        )
        self.run_scope(base=base, head=head, success=False)


class WorkflowScopeTests(unittest.TestCase):
    def affected(self, workflow: str, path: str) -> bool:
        scope = workflow_scope(workflow)
        return matches_scope([path], scope["include"], scope.get("exclude", ""))

    def test_root_config_and_workflow_inputs(self) -> None:
        for name in ("runtimes", "rocjitsu"):
            with self.subTest(name=name):
                self.assertTrue(self.affected(name, ".pre-commit-config.yaml"))
                self.assertTrue(
                    self.affected(name, f".github/workflows/{name}-formatting.yml")
                )
                self.assertFalse(self.affected(name, "docs/readme.md"))
        self.assertTrue(self.affected("runtimes", "runtimes/src/lib.rs"))
        self.assertTrue(self.affected("rocjitsu", "emulation/mirage/src/lib.rs"))
        self.assertTrue(self.affected("rocjitsu", "emulation/rocjitsu/test.py"))

    def test_project_scopes_and_documentation_exclusions(self) -> None:
        for name in ("cuid", "rocprofiler-compute"):
            with self.subTest(name=name):
                self.assertTrue(self.affected(name, f"projects/{name}/src/runtime.cpp"))
                self.assertTrue(
                    self.affected(name, f".github/workflows/{name}-formatting.yml")
                )
                for path in ("README.md", "docs/conf.py", ".readthedocs.yaml"):
                    self.assertFalse(self.affected(name, f"projects/{name}/{path}"))
                self.assertFalse(self.affected(name, "runtimes/src/lib.rs"))
                self.assertTrue(self.affected(name, f"projects/{name}/README.md\n"))

    def test_shared_action_changes_activate_every_scope(self) -> None:
        for name in ("runtimes", "rocjitsu", "cuid", "rocprofiler-compute"):
            with self.subTest(name=name):
                self.assertTrue(
                    self.affected(name, ".github/actions/scoped-formatting/scope.py")
                )


if __name__ == "__main__":
    unittest.main()
