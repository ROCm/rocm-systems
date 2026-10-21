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
import rccl_change_sets as sets

REPO_ROOT = Path(__file__).resolve().parents[3]
OTHER_BRANCHES = ["", "main", "rccl-develop", "rocprofiler-compute-develop"]


def pull_request_filters(workflow: Path) -> tuple[set[str], list[str]]:
    """(branches, paths) of a workflow's pull_request trigger.

    Strict on purpose: anything it cannot vouch for (negations, *-ignore keys,
    unknown keys) fails the test instead of being skipped.
    """
    lines = [
        re.sub(r"\s+#.*$", "", l)
        for l in workflow.read_text(encoding="utf-8").splitlines()
    ]
    start = next(i for i, l in enumerate(lines) if re.match(r"^  pull_request:\s*$", l))
    filters: dict[str, list[str]] = {}
    key = None
    for line in lines[start + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith("    "):
            break
        if m := re.match(r"^    ([\w-]+):\s*(.*)$", line):
            key, inline = m.groups()
            if key not in ("branches", "paths", "types"):
                raise AssertionError(f"{workflow}: unsupported pull_request key {key}")
            filters[key] = re.findall(r"[^\[\],'\"\s]+", inline)
        elif key and (item := re.match(r"^\s+-\s*['\"]?([^'\"\s]+)", line)):
            filters[key].append(item.group(1))
        else:
            raise AssertionError(f"{workflow}: cannot parse {line!r}")
    paths = filters.get("paths", [])
    if any(p.startswith("!") for p in paths):
        raise AssertionError(f"{workflow}: negated paths {paths}")
    return set(filters.get("branches", [])), paths


class CoverageRulesTest(unittest.TestCase):
    def test_every_rule_has_examples(self):
        for rule in sets.COVERAGE_RULES:
            with self.subTest(rule=rule.name):
                self.assertTrue(rule.branches)
                self.assertTrue(rule.covered_examples)
                self.assertTrue(rule.lookalike_examples)

    def test_covered_examples_on_rule_branches(self):
        for rule in sets.COVERAGE_RULES:
            for path in rule.covered_examples:
                for branch in rule.branches:
                    with self.subTest(rule=rule.name, path=path, branch=branch):
                        self.assertTrue(sets.covered_elsewhere(path, branch))

    def test_covered_examples_not_on_other_branches(self):
        for rule in sets.COVERAGE_RULES:
            for path in rule.covered_examples:
                for branch in OTHER_BRANCHES:
                    if branch in rule.branches:
                        continue
                    with self.subTest(rule=rule.name, path=path, branch=branch):
                        self.assertFalse(sets.covered_elsewhere(path, branch))

    def test_lookalikes_never_covered(self):
        for rule in sets.COVERAGE_RULES:
            for path in rule.lookalike_examples:
                for branch in rule.branches:
                    with self.subTest(rule=rule.name, path=path, branch=branch):
                        self.assertFalse(sets.covered_elsewhere(path, branch))


class CoverageWorkflowTest(unittest.TestCase):
    """A rule may only claim coverage where its workflow actually runs."""

    def test_workflow_runs_on_rule_branches(self):
        for rule in sets.COVERAGE_RULES:
            with self.subTest(rule=rule.name):
                branches, _ = pull_request_filters(REPO_ROOT / rule.workflow)
                self.assertTrue(branches, f"{rule.workflow}: no pull_request branches")
                self.assertLessEqual(set(rule.branches), branches)

    def test_workflow_paths_include_rule_patterns(self):
        for rule in sets.COVERAGE_RULES:
            _, paths = pull_request_filters(REPO_ROOT / rule.workflow)
            prefixes = [p.removesuffix("**") for p in paths if p.endswith("/**")]
            for pattern in rule.patterns:
                with self.subTest(rule=rule.name, pattern=pattern):
                    self.assertTrue(
                        any(pattern.startswith(prefix) for prefix in prefixes),
                        f"{rule.workflow} paths {paths} do not include {pattern}",
                    )

    def _parse(self, body):
        with tempfile.TemporaryDirectory() as tmp:
            wf = Path(tmp) / "wf.yml"
            wf.write_text("on:\n  pull_request:\n" + body + "  workflow_dispatch:\n")
            return pull_request_filters(wf)

    def test_parser_reads_both_branch_forms(self):
        self.assertEqual(
            self._parse(
                "    branches: [main, develop]  # trailing comment\n"
                "    # full-line comment\n"
                "  # shallow comment\n"
                "    paths:\n      - 'projects/rccl/**'\n"
            ),
            ({"main", "develop"}, ["projects/rccl/**"]),
        )
        self.assertEqual(
            self._parse("    branches:\n      - develop\n"), ({"develop"}, [])
        )

    def test_parser_rejects_what_it_cannot_vouch_for(self):
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
