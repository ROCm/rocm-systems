# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check formatting scope selection and workflow wiring."""

from pathlib import Path
import sys
import unittest

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS.parent / "actions/changed-files"))

from scope import matches_scope

sys.path.insert(0, str(WORKFLOWS.parents[1] / "tools/systems_pr_bot"))
import policy_check

POLICY = policy_check.load_policy(
    WORKFLOWS.parents[1] / "tools/systems_pr_bot/policy.yml"
)

WORKFLOW = yaml.load(
    (WORKFLOWS / "scoped-formatting.yml").read_text(), Loader=yaml.BaseLoader
)
STEPS = WORKFLOW["jobs"]["pre-commit"]["steps"]
ALWAYS_INCLUDE = STEPS[1]["with"]["always-include"]


def workflow_scope(name: str) -> dict[str, str]:
    workflow = yaml.load(
        (WORKFLOWS / "scoped-formatting.yml").read_text(), Loader=yaml.BaseLoader
    )
    return next(
        scope
        for scope in workflow["jobs"]["pre-commit"]["strategy"]["matrix"]["include"]
        if scope["scope"] == name
    )


class WorkflowScopeTests(unittest.TestCase):
    def test_every_formatting_check_is_required_by_policy(self) -> None:
        scopes = WORKFLOW["jobs"]["pre-commit"]["strategy"]["matrix"]["include"]
        names = [scope["check-name"] for scope in scopes]
        self.assertEqual(len(names), len(set(names)))
        self.assertLessEqual(set(names), set(POLICY.required_checks))
        self.assertEqual(
            WORKFLOW["jobs"]["pre-commit"]["strategy"]["fail-fast"], "false"
        )
        for scope in scopes:
            self.assertTrue(scope["sparse-checkout"].strip(), scope["scope"])

    def test_each_scope_failure_is_enforced_by_the_actual_bot(self) -> None:
        for name in POLICY.required_checks:
            runs = [
                {"name": check, "conclusion": "failure" if check == name else "success"}
                for check in POLICY.required_checks
            ]
            with self.subTest(check=name):
                missing, failing, _ = policy_check.summarize_required_checks(
                    POLICY, runs
                )
                self.assertEqual(missing, [])
                self.assertEqual(failing, [f"{name}=failure"])
                rows = {
                    row.name: row
                    for row in policy_check.build_check_results(POLICY, runs)
                }
                self.assertFalse(rows[name].passed)

    def test_missing_and_running_checks_remain_pending(self) -> None:
        for name in POLICY.required_checks:
            for conclusion in ("missing", None):
                runs = [
                    {
                        "name": check,
                        "conclusion": conclusion if check == name else "success",
                    }
                    for check in POLICY.required_checks
                    if check != name or conclusion != "missing"
                ]
                with self.subTest(check=name, conclusion=conclusion):
                    missing, failing, _ = policy_check.summarize_required_checks(
                        POLICY, runs
                    )
                    self.assertEqual(missing, [name] if conclusion == "missing" else [])
                    self.assertEqual(failing, [])
                    rows = {
                        row.name: row
                        for row in policy_check.build_check_results(POLICY, runs)
                    }
                    self.assertTrue(rows[name].pending)
                    self.assertFalse(rows[name].passed)

    def test_metadata_results_do_not_override_failed_or_running_checks(self) -> None:
        for conclusion in ("failure", None):
            for name in POLICY.required_checks:
                runs = [
                    {
                        "name": check,
                        "conclusion": conclusion if check == name else "success",
                    }
                    for check in POLICY.required_checks
                ]
                runs.append(
                    {"name": "metadata-only / pre-commit", "conclusion": "skipped"}
                )
                with self.subTest(check=name, conclusion=conclusion):
                    rows = {
                        row.name: row
                        for row in policy_check.build_check_results(POLICY, runs)
                    }
                    self.assertFalse(rows[name].passed)
                    self.assertEqual(rows[name].pending, conclusion is None)

    def test_formatter_uses_the_inspected_comparison(self) -> None:
        _, _, checkout, setup, formatting = STEPS
        for step in (checkout, setup, formatting):
            self.assertEqual(step["if"], "steps.scope.outputs.affected == 'true'")
        self.assertEqual(checkout["with"]["fetch-depth"], "0")
        self.assertEqual(
            checkout["with"]["sparse-checkout"], "${{ matrix.sparse-checkout }}"
        )
        self.assertEqual(
            checkout["with"]["submodules"], "${{ matrix.submodules || 'false' }}"
        )
        self.assertEqual(
            formatting["uses"],
            "pre-commit/action@2c7b3805fd2a0fd8c1884dcaebf91fc102a13ecd",
        )
        self.assertIn(
            "--config ${{ matrix.config || '.pre-commit-config.yaml' }}",
            formatting["with"]["extra_args"],
        )
        self.assertIn(
            "--from-ref ${{ steps.scope.outputs.base }}",
            formatting["with"]["extra_args"],
        )
        self.assertIn(
            "--to-ref ${{ steps.scope.outputs.head }}", formatting["with"]["extra_args"]
        )

    def affected(self, workflow: str, path: str, event: str = "pull_request") -> bool:
        scope = workflow_scope(workflow)
        exclude = scope.get("exclude", "") if event == "pull_request" else ""
        return matches_scope([path], scope["include"], exclude, ALWAYS_INCLUDE)

    def test_root_config_and_workflow_inputs(self) -> None:
        for name in ("runtimes", "rocjitsu"):
            with self.subTest(name=name):
                self.assertTrue(self.affected(name, ".pre-commit-config.yaml"))
                self.assertTrue(
                    self.affected(name, ".github/workflows/scoped-formatting.yml")
                )
                self.assertFalse(self.affected(name, "docs/readme.md"))
        self.assertTrue(self.affected("runtimes", "runtimes/src/lib.rs"))
        self.assertTrue(self.affected("rocjitsu", "emulation/mirage/src/lib.rs"))
        self.assertTrue(self.affected("rocjitsu", "emulation/rocjitsu/test.py"))

    def test_rocjitsu_fuzzing_scripts_activate_formatting(self) -> None:
        for filename in (
            "collect_rocjitsu_afl_results.py",
            "rocjitsu_fuzz_report.js",
            "run_rocjitsu_afl_campaign.py",
        ):
            for relative in (filename, f"tests/test_{filename}"):
                with self.subTest(path=relative):
                    self.assertTrue(
                        self.affected("rocjitsu", f".github/scripts/{relative}")
                    )
        self.assertFalse(self.affected("rocjitsu", ".github/scripts/other.py"))
        self.assertIn(
            ".github", workflow_scope("rocjitsu")["sparse-checkout"].splitlines()
        )

    def test_cuid_uses_shared_directory_for_configuration_and_pushes(self) -> None:
        self.assertEqual(
            workflow_scope("cuid")["config"], "shared/cuid/.pre-commit-config.yaml"
        )
        self.assertIn("shared/cuid/**", WORKFLOW["on"]["push"]["paths"])
        self.assertFalse(self.affected("cuid", "projects/cuid/src/runtime.cpp"))

    def test_project_scopes_and_documentation_exclusions(self) -> None:
        for name, directory in (
            ("cuid", "shared/cuid"),
            ("rocprofiler-compute", "projects/rocprofiler-compute"),
        ):
            with self.subTest(name=name):
                self.assertTrue(self.affected(name, f"{directory}/src/runtime.cpp"))
                self.assertTrue(
                    self.affected(name, ".github/workflows/scoped-formatting.yml")
                )
                for path in ("README.md", "docs/conf.py", ".readthedocs.yaml"):
                    self.assertFalse(self.affected(name, f"{directory}/{path}"))
                self.assertFalse(self.affected(name, "runtimes/src/lib.rs"))
                self.assertTrue(self.affected(name, f"{directory}/README.md\n"))

    def test_shared_infrastructure_changes_activate_every_scope(self) -> None:
        for name in ("runtimes", "rocjitsu", "cuid", "rocprofiler-compute"):
            for path in (
                ".github/actions/changed-files/scope.py",
                ".github/workflows/scoped-formatting.yml",
            ):
                with self.subTest(name=name, path=path):
                    self.assertTrue(self.affected(name, path))

    def test_documentation_exclusions_apply_only_to_pull_requests(self) -> None:
        for name, directory in (
            ("cuid", "shared/cuid"),
            ("rocprofiler-compute", "projects/rocprofiler-compute"),
        ):
            for relative in (
                "README.md",
                "guide.rst",
                "docs/conf.py",
                ".readthedocs.yaml",
            ):
                path = f"{directory}/{relative}"
                with self.subTest(scope=name, path=path):
                    self.assertFalse(self.affected(name, path, "pull_request"))
                    self.assertTrue(self.affected(name, path, "push"))
            with self.subTest(scope=name):
                self.assertFalse(self.affected(name, "docs/conf.py", "push"))
                self.assertFalse(self.affected(name, "runtimes/src/lib.rs", "push"))


if __name__ == "__main__":
    unittest.main()
