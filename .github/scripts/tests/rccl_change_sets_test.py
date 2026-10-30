# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rccl_change_sets.py."""

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parent))
import rccl_change_sets as sets
import rccl_detect_host_test_changes as host_detect
from workflow_filters import (
    DEFAULT_TYPES,
    STATUS_ONLY_IFS,
    job_lines,
    pull_request_trigger,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OTHER_BRANCHES = ["", "main", "rccl-develop", "rocprofiler-compute-develop"]

# Per COVERAGE_RULES name: (paths it covers, look-alikes it must not cover).
EXAMPLES = {
    "host-only unit tests": (
        [
            "projects/rccl/test/host/CMakeLists.txt",
            "projects/rccl/test/host/run_host_tests.sh",
            "projects/rccl/test/host/fakes/os_fakes.cc",
            "projects/rccl/test/test_categories_micro.yaml",
            "projects/rccl/test/test_categories_micro_net.yaml",
        ],
        [
            "projects/rccl/test/host",
            "projects/rccl/test/hostile.cc",
            "projects/rccl/test/hostfoo/x.cc",
            "projects/rccl/test/CMakeLists.txt",
            "projects/rccl/src/test/host/x.cc",
            "projects/rccl-tests/test/host/x.cc",
            "projects/rccl/test/test_categories.yaml",
            "projects/rccl/test/test_categories_fixtures.yaml",
            "projects/rccl/test/test_categories_mpi.yaml",
            "projects/rccl/test/test_categories_micro.yml",
            "projects/rccl/test/test_categories_microbench.yaml",
        ],
    ),
    "test runner": (
        [
            "projects/rccl/tools/scripts/test_runner/configs/ci-precheckin.json",
            "projects/rccl/tools/scripts/test_runner/lib/test_executor.py",
            "projects/rccl/tools/scripts/test_runner/test_runner.py",
        ],
        [
            "projects/rccl/tools/scripts/test_runner",
            "projects/rccl/tools/scripts/test_runner_x/a.py",
            "projects/rccl/tools/scripts/topo_expl/x.py",
            "projects/rccl/tools/test_runner/a.py",
        ],
    ),
}


# rccl-host-unit-tests.yml runs on every path. Its `changes` job runs the
# detector, and the test jobs skip unless it says the tests are needed.
HOST_TESTS_WORKFLOW = ".github/workflows/rccl-host-unit-tests.yml"
HOST_TESTS_RUN_IF = "needs.changes.outputs.rccl == 'true'"
HOST_TESTS_CHANGES_LINES = [
    "rccl: ${{ steps.filter.outputs.rccl }}",
    "id: filter",
    "run: python3 .github/scripts/rccl_detect_host_test_changes.py --base-ref HEAD^",
]


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
                allowed = set(STATUS_ONLY_IFS)
                if rule.workflow == HOST_TESTS_WORKFLOW:
                    allowed.add(HOST_TESTS_RUN_IF)
                self.assertLessEqual(
                    set(trigger.ifs), allowed, "an if: can skip the tests"
                )

    def test_change_detection_covers_rule(self):
        for rule in sets.COVERAGE_RULES:
            with self.subTest(rule=rule.name):
                if rule.workflow != HOST_TESTS_WORKFLOW:
                    trigger = pull_request_trigger(REPO_ROOT / rule.workflow)
                    self.assertTrue(trigger.paths, "no paths: and no detector")
                    continue
                for path in [*rule.patterns, *EXAMPLES[rule.name][0]]:
                    self.assertTrue(host_detect.touches_host_tests([path]), path)

    def test_host_tests_wiring(self):
        workflow = REPO_ROOT / HOST_TESTS_WORKFLOW
        # Either would turn a failed detection or test into a pass.
        text = workflow.read_text(encoding="utf-8")
        for forbidden in ("continue-on-error", "|| true"):
            self.assertNotIn(forbidden, text)

        jobs = job_lines(workflow)
        for line in HOST_TESTS_CHANGES_LINES:
            self.assertIn(line, jobs["changes"])

        # The gate must always run, wait on every other job, and fail when
        # detection fails or did not say "false".
        gate = jobs["host-gate"]
        self.assertEqual([l for l in gate if l.startswith("if:")], ["if: always()"])
        needs = next((l for l in gate if l.startswith("needs: [")), None)
        self.assertIsNotNone(needs, "gate needs: must be a flow list")
        self.assertEqual(
            set(re.findall(r"[\w-]+", needs.removeprefix("needs:"))),
            set(jobs) - {"host-gate"},
        )
        for line in [
            'if [[ "${CHANGES_RESULT}" != "success" ]]; then',
            'if [[ "${RCCL_CHANGED}" == "false" ]]; then',
            "RESULTS: ${{ join(needs.*.result, ' ') }}",
        ]:
            self.assertIn(line, gate)

    def test_workflow_paths_include_rule_patterns(self):
        for rule in sets.COVERAGE_RULES:
            paths = pull_request_trigger(REPO_ROOT / rule.workflow).paths
            if not paths:
                continue  # Covered by test_change_detection_covers_rule.
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

    def test_job_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf = Path(tmp) / "wf.yml"
            wf.write_text(
                "on:\n  pull_request:\n\n"
                "jobs:\n"
                "  a:\n    runs-on: x  # note\n    # comment\n\n"
                "  b-c:\n    needs: [a]\n    steps:\n      - run: y\n"
                "env:\n  Z: 1\n"
            )
            self.assertEqual(
                job_lines(wf),
                {"a": ["runs-on: x"], "b-c": ["needs: [a]", "steps:", "- run: y"]},
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
