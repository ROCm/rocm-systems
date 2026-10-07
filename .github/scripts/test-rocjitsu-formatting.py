#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the formatting workflow's file selection in temporary Git repos."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows/rocjitsu-formatting.yml"
CONFIG_PATH = ".pre-commit-config.yaml"
WORKFLOW_PATH = ".github/workflows/rocjitsu-formatting.yml"
MIRAGE_FILE = "emulation/mirage/core/src/metric.rs"
ROCJITSU_FILE = "emulation/rocjitsu/tests/example.cpp"


class FormattingSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow = yaml.safe_load(WORKFLOW.read_text())
        cls.script = next(
            step["run"]
            for step in workflow["jobs"]["pre-commit"]["steps"]
            if step.get("name", "").startswith("Run pre-commit")
        )

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.record = self.root / "pre-commit.json"
        self.git("init", "--quiet", "--initial-branch=main")
        self.git("config", "user.name", "Formatting Test")
        self.git("config", "user.email", "formatting-test@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.write(CONFIG_PATH, "repos: []\n")
        self.write(WORKFLOW_PATH, "name: formatting\n")
        self.write(MIRAGE_FILE, "//! Metrics module.\n")
        self.write(ROCJITSU_FILE, "// Test fixture.\n")
        self.write(".github/scripts/unrelated.py", "pass\n")
        self.write("README.md", "Repository\n")
        self.base = self.commit()
        hook = self.bin_dir / "pre-commit"
        hook.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "pathlib.Path(os.environ['FORMATTING_TEST_RECORD']).write_text(\n"
            "    json.dumps(sys.argv[1:]))\n"
            "sys.exit(int(os.environ.get('FORMATTING_TEST_EXIT', '0')))\n"
        )
        hook.chmod(0o755)

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-c", f"core.hooksPath={self.root / 'hooks'}", *args],
            cwd=self.repo,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()

    def write(self, path, contents):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)

    def commit(self):
        self.git("add", "--all")
        self.git("commit", "--quiet", "-m", "Update formatting test fixtures.")
        return self.git("rev-parse", "HEAD")

    def run_workflow(self, *, base=None, head=None, hook_exit=0):
        base = self.base if base is None else base
        head = self.git("rev-parse", "HEAD") if head is None else head
        script = self.script.replace(
            "${{ github.event.pull_request.base.sha }}", base
        ).replace("${{ github.event.pull_request.head.sha }}", head)
        environment = {
            **os.environ,
            "BASE_SHA": base,
            "HEAD_SHA": head,
            "PATH": f"{self.bin_dir}{os.pathsep}{os.environ['PATH']}",
            "FORMATTING_TEST_RECORD": str(self.record),
            "FORMATTING_TEST_EXIT": str(hook_exit),
        }
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
            cwd=self.repo,
            env=environment,
            capture_output=True,
            text=True,
        )

    def selected_files(self):
        arguments = json.loads(self.record.read_text())
        self.assertEqual(arguments[:2], ["run", "--files"])
        return arguments[2:]

    def test_source_change_checks_only_changed_file(self):
        self.write(MIRAGE_FILE, "//! Updated metrics module.\n")
        self.commit()
        result = self.run_workflow()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.selected_files(), [MIRAGE_FILE])

    def test_hook_config_change_checks_both_complete_subtrees(self):
        self.write(CONFIG_PATH, "repos: []\n# Update a hook version.\n")
        self.commit()
        result = self.run_workflow()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertCountEqual(
            self.selected_files(), [CONFIG_PATH, MIRAGE_FILE, ROCJITSU_FILE]
        )

    def test_workflow_change_checks_both_complete_subtrees(self):
        self.write(WORKFLOW_PATH, "name: updated-formatting\n")
        self.commit()
        result = self.run_workflow()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertCountEqual(
            self.selected_files(), [WORKFLOW_PATH, MIRAGE_FILE, ROCJITSU_FILE]
        )

    def test_full_scan_filters_untracked_and_missing_files(self):
        self.write(CONFIG_PATH, "repos: []\n# Update formatting.\n")
        self.commit()
        self.write("emulation/mirage/untracked.rs", "// Untracked.\n")
        (self.repo / ROCJITSU_FILE).unlink()
        result = self.run_workflow()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertCountEqual(self.selected_files(), [CONFIG_PATH, MIRAGE_FILE])

    def test_full_scan_deduplicates_changed_source_files(self):
        self.write(CONFIG_PATH, "repos: []\n# Update formatting.\n")
        self.write(MIRAGE_FILE, "//! Updated metrics module.\n")
        self.commit()
        result = self.run_workflow()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertCountEqual(
            self.selected_files(), [CONFIG_PATH, MIRAGE_FILE, ROCJITSU_FILE]
        )

    def test_deleted_files_do_not_reach_pre_commit(self):
        (self.repo / MIRAGE_FILE).unlink()
        self.commit()
        result = self.run_workflow()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.record.exists())

    def test_sparse_checkout_can_leave_no_files_to_check(self):
        self.write(MIRAGE_FILE, "//! Updated metrics module.\n")
        self.commit()
        (self.repo / MIRAGE_FILE).unlink()
        result = self.run_workflow()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.record.exists())

    def test_filenames_with_whitespace_reach_pre_commit_intact(self):
        path = "emulation/mirage/file with space\nand newline.rs"
        self.write(path, "// Fixture.\n")
        self.commit()
        result = self.run_workflow()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.selected_files(), [path])

    def test_merge_checkout_does_not_add_base_only_changes(self):
        self.git("checkout", "--quiet", "-b", "pr")
        self.write(MIRAGE_FILE, "//! Updated metrics module.\n")
        head = self.commit()
        self.git("checkout", "--quiet", "main")
        self.write(ROCJITSU_FILE, "// Changed on the base branch.\n")
        base = self.commit()
        self.git("merge", "--quiet", "-m", "Merge formatting test fixtures.", "pr")
        result = self.run_workflow(base=base, head=head)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.selected_files(), [MIRAGE_FILE])

    def test_pre_commit_failure_fails_the_step(self):
        self.write(MIRAGE_FILE, "//! Updated metrics module.\n")
        self.commit()
        result = self.run_workflow(hook_exit=23)
        self.assertEqual(result.returncode, 23, result.stderr)

    def test_invalid_git_revision_fails_without_invoking_pre_commit(self):
        result = self.run_workflow(head="missing-revision")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.record.exists())


if __name__ == "__main__":
    unittest.main()
