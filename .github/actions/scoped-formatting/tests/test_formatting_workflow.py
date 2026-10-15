"""Check formatting scope selection and workflow wiring."""

from pathlib import Path
import sys
import unittest

import yaml

ACTION_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACTION_DIRECTORY.parent / "changed-files"))

from scope import matches_scope

WORKFLOWS = ACTION_DIRECTORY.parents[1] / "workflows"
ACTION = yaml.load(
    (ACTION_DIRECTORY / "action.yml").read_text(), Loader=yaml.BaseLoader
)
ALWAYS_INCLUDE = ACTION["runs"]["steps"][0]["with"]["always-include"]


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
    def test_every_scope_runs_on_every_pr_target(self) -> None:
        workflow = yaml.load(
            (WORKFLOWS / "scoped-formatting.yml").read_text(), Loader=yaml.BaseLoader
        )
        self.assertEqual(
            workflow["on"]["pull_request"],
            {
                "types": [
                    "opened",
                    "synchronize",
                    "reopened",
                    "ready_for_review",
                    "edited",
                ]
            },
        )
        job = workflow["jobs"]["pre-commit"]
        self.assertEqual(job["name"], "${{ matrix.check-name }}")
        for key in ("if", "needs", "continue-on-error"):
            self.assertNotIn(key, job)
        self.assertEqual(job["strategy"]["fail-fast"], "false")
        scopes = job["strategy"]["matrix"]["include"]
        self.assertEqual(
            {scope["check-name"] for scope in scopes},
            {
                "pre-commit / runtimes",
                "pre-commit",
                "pre-commit / cuid",
                "pre-commit / rocprofiler-compute",
            },
        )
        self.assertEqual(len(scopes), 4)
        for scope in scopes:
            self.assertNotIn("enabled", scope)
        for step in job["steps"]:
            self.assertNotIn("if", step)
            self.assertNotIn("continue-on-error", step)
        checkout, formatting = job["steps"]
        self.assertEqual(checkout["with"]["fetch-depth"], "0")
        self.assertIn(
            ".github/actions/changed-files",
            checkout["with"]["sparse-checkout"].splitlines(),
        )
        self.assertEqual(formatting["uses"], "./.github/actions/scoped-formatting")
        self.assertEqual(
            formatting["with"],
            {
                "include": "${{ matrix.include }}",
                "exclude": "${{ github.event_name == 'pull_request' && matrix.exclude || '' }}",
                "config": "${{ matrix.config || '.pre-commit-config.yaml' }}",
                "sparse-checkout": "${{ matrix.sparse-checkout }}",
                "submodules": "${{ matrix.submodules || 'false' }}",
            },
        )

    def test_formatter_uses_the_inspected_comparison(self) -> None:
        scope, checkout, setup, formatting = ACTION["runs"]["steps"]
        self.assertEqual(scope["uses"], "./.github/actions/changed-files")
        self.assertEqual(scope["with"]["include"], "${{ inputs.include }}")
        self.assertEqual(scope["with"]["exclude"], "${{ inputs.exclude }}")
        for step in (checkout, setup, formatting):
            self.assertEqual(step["if"], "steps.scope.outputs.affected == 'true'")
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

    def test_project_scopes_and_documentation_exclusions(self) -> None:
        for name in ("cuid", "rocprofiler-compute"):
            with self.subTest(name=name):
                self.assertTrue(self.affected(name, f"projects/{name}/src/runtime.cpp"))
                self.assertTrue(
                    self.affected(name, ".github/workflows/scoped-formatting.yml")
                )
                for path in ("README.md", "docs/conf.py", ".readthedocs.yaml"):
                    self.assertFalse(self.affected(name, f"projects/{name}/{path}"))
                self.assertFalse(self.affected(name, "runtimes/src/lib.rs"))
                self.assertTrue(self.affected(name, f"projects/{name}/README.md\n"))

    def test_shared_infrastructure_changes_activate_every_scope(self) -> None:
        for name in ("runtimes", "rocjitsu", "cuid", "rocprofiler-compute"):
            for path in (
                ".github/actions/scoped-formatting/action.yml",
                ".github/actions/changed-files/scope.py",
                ".github/workflows/scoped-formatting.yml",
            ):
                with self.subTest(name=name, path=path):
                    self.assertTrue(self.affected(name, path))

    def test_documentation_exclusions_apply_only_to_pull_requests(self) -> None:
        for name in ("cuid", "rocprofiler-compute"):
            for relative in (
                "README.md",
                "guide.rst",
                "docs/conf.py",
                ".readthedocs.yaml",
            ):
                path = f"projects/{name}/{relative}"
                with self.subTest(scope=name, path=path):
                    self.assertFalse(self.affected(name, path, "pull_request"))
                    self.assertTrue(self.affected(name, path, "push"))
            with self.subTest(scope=name):
                self.assertFalse(self.affected(name, "docs/conf.py", "push"))
                self.assertFalse(self.affected(name, "runtimes/src/lib.rs", "push"))


if __name__ == "__main__":
    unittest.main()
