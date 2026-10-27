# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rccl_change_sets.py."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parent))
import rccl_change_sets as sets
from workflow_filters import DEFAULT_TYPES, STATUS_ONLY_IFS, pull_request_trigger

REPO_ROOT = Path(__file__).resolve().parents[3]
OTHER_BRANCHES = ["", "main", "rccl-develop", "rocprofiler-compute-develop"]

# Per COVERAGE_RULES name: (paths it covers, look-alikes it must not cover).
EXAMPLES = {
    "host-only unit tests": (
        [
            "projects/rccl/test/host/CMakeLists.txt",
            "projects/rccl/test/host/run_host_tests.sh",
            "projects/rccl/test/host/fakes/os_fakes.cc",
        ],
        [
            "projects/rccl/test/host",
            "projects/rccl/test/hostile.cc",
            "projects/rccl/test/hostfoo/x.cc",
            "projects/rccl/test/CMakeLists.txt",
            "projects/rccl/src/test/host/x.cc",
            "projects/rccl-tests/test/host/x.cc",
        ],
    ),
}


class CoverageRulesTest(unittest.TestCase):
    def test_every_rule_has_examples(self):
        self.assertEqual(set(EXAMPLES), {r.name for r in sets.COVERAGE_RULES})
        for name, (covered, lookalikes) in EXAMPLES.items():
            with self.subTest(rule=name):
                self.assertTrue(covered)
                self.assertTrue(lookalikes)

    def test_covered_examples_on_rule_branches(self):
        for rule in sets.COVERAGE_RULES:
            for path in EXAMPLES[rule.name][0]:
                for branch in rule.branches:
                    with self.subTest(rule=rule.name, path=path, branch=branch):
                        self.assertTrue(sets.covered_elsewhere(path, branch))

    def test_covered_examples_not_on_other_branches(self):
        for rule in sets.COVERAGE_RULES:
            for path in EXAMPLES[rule.name][0]:
                for branch in OTHER_BRANCHES:
                    if branch in rule.branches:
                        continue
                    with self.subTest(rule=rule.name, path=path, branch=branch):
                        self.assertFalse(sets.covered_elsewhere(path, branch))

    def test_lookalikes_never_covered(self):
        for rule in sets.COVERAGE_RULES:
            for path in EXAMPLES[rule.name][1]:
                for branch in rule.branches:
                    with self.subTest(rule=rule.name, path=path, branch=branch):
                        self.assertFalse(sets.covered_elsewhere(path, branch))


class CoverageWorkflowTest(unittest.TestCase):
    """A rule may only claim coverage where its workflow actually runs."""

    def test_workflow_runs_on_rule_branches(self):
        for rule in sets.COVERAGE_RULES:
            with self.subTest(rule=rule.name):
                trigger = pull_request_trigger(REPO_ROOT / rule.workflow)
                self.assertTrue(trigger.branches, f"{rule.workflow}: no branches")
                self.assertLessEqual(set(rule.branches), trigger.branches)

    def test_workflow_runs_on_every_push(self):
        for rule in sets.COVERAGE_RULES:
            with self.subTest(rule=rule.name):
                trigger = pull_request_trigger(REPO_ROOT / rule.workflow)
                self.assertLessEqual(DEFAULT_TYPES, trigger.types)
                self.assertLessEqual(
                    set(trigger.ifs), STATUS_ONLY_IFS, "an if: can skip the tests"
                )

    def test_workflow_paths_include_rule_patterns(self):
        for rule in sets.COVERAGE_RULES:
            paths = pull_request_trigger(REPO_ROOT / rule.workflow).paths
            prefixes = [p.removesuffix("**") for p in paths if p.endswith("/**")]
            for pattern in rule.patterns:
                with self.subTest(rule=rule.name, pattern=pattern):
                    self.assertTrue(
                        any(pattern.startswith(prefix) for prefix in prefixes),
                        f"{rule.workflow} paths {paths} do not include {pattern}",
                    )


class WorkflowFiltersTest(unittest.TestCase):
    def _parse(self, body, jobs=""):
        with tempfile.TemporaryDirectory() as tmp:
            wf = Path(tmp) / "wf.yml"
            wf.write_text(
                "on:\n  pull_request:\n"
                + body
                + "  workflow_dispatch:\n\njobs:\n  a:\n    runs-on: x\n"
                + jobs
            )
            return pull_request_trigger(wf)

    def test_reads_both_branch_forms(self):
        trigger = self._parse(
            "    branches: [main, develop]  # trailing comment\n"
            "    # full-line comment\n"
            "  # shallow comment\n"
            "    paths:\n      - 'projects/rccl/**'\n"
        )
        self.assertEqual(trigger.branches, {"main", "develop"})
        self.assertEqual(trigger.paths, ["projects/rccl/**"])
        self.assertEqual(trigger.types, DEFAULT_TYPES)
        self.assertEqual(trigger.ifs, [])
        self.assertEqual(
            self._parse("    branches:\n      - develop\n").branches, {"develop"}
        )

    def test_reads_types_and_ifs(self):
        trigger = self._parse(
            "    types: [labeled]\n",
            jobs="    if: contains(labels, 'x')\n"
            "    steps:\n"
            "      - if: ${{ always() }}\n"
            "        run: a\n"
            "      - name: b\n"
            "        if: github.event_name == 'push'\n",
        )
        self.assertEqual(trigger.types, {"labeled"})
        self.assertEqual(
            trigger.ifs,
            ["contains(labels, 'x')", "always()", "github.event_name == 'push'"],
        )

    def test_rejects_what_it_cannot_vouch_for(self):
        for body in [
            "    branches: [develop]\n  # note\n    paths-ignore:\n      - 'x/**'\n",
            "    branches-ignore: [main]\n",
            "    paths:\n      - 'projects/rccl/**'\n"
            "      - '!projects/rccl/test/**'\n",
            "    paths: ['projects/rccl/**', '!projects/rccl/test/host/**']\n",
        ]:
            with self.subTest(body=body), self.assertRaises(AssertionError):
                self._parse(body)


class IsDocTest(unittest.TestCase):
    def test_docs(self):
        for path in [
            "README.md",
            "projects/rccl/README.md",
            "projects/rccl/docs/api.rst",
            "projects/rccl/docs/conf.py",
        ]:
            with self.subTest(path=path):
                self.assertTrue(sets.is_doc(path))

    def test_not_docs(self):
        for path in ["projects/rccl/src/init.cc", "projects/rccl/test/host/x.cc"]:
            with self.subTest(path=path):
                self.assertFalse(sets.is_doc(path))


if __name__ == "__main__":
    unittest.main()
