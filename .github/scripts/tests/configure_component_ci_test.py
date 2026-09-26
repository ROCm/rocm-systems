import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_component_ci


class WorkflowDispatchTest(unittest.TestCase):
    def _run(self, dispatch_jobs):
        env = {
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "WORKFLOW_DISPATCH_JOBS": dispatch_jobs,
        }
        with patch.dict(os.environ, env), patch(
            "configure_component_ci.set_github_output"
        ) as mock_output:
            configure_component_ci.main()
            return mock_output.call_args[0][0]

    def test_all_runs_everything(self):
        out = self._run("all")
        self.assertEqual(out["nvidia"], "true")
        self.assertEqual(out["wsl"], "true")
        self.assertEqual(out["hip-contract-tests"], "true")

    def test_single_job(self):
        out = self._run("nvidia")
        self.assertEqual(out["nvidia"], "true")
        self.assertEqual(out["wsl"], "false")
        self.assertEqual(out["hip-contract-tests"], "false")

    def test_subset_of_jobs(self):
        out = self._run("wsl hip-contract-tests")
        self.assertEqual(out["nvidia"], "false")
        self.assertEqual(out["wsl"], "true")
        self.assertEqual(out["hip-contract-tests"], "true")

    def test_empty_input_runs_nothing(self):
        out = self._run("")
        self.assertEqual(out["nvidia"], "false")
        self.assertEqual(out["wsl"], "false")
        self.assertEqual(out["hip-contract-tests"], "false")


class NvidiaPatternTest(unittest.TestCase):
    def _run(self, changed_files):
        with patch(
            "configure_component_ci.get_changed_files", return_value=changed_files
        ), patch("configure_component_ci.set_github_output") as mock_output, patch.dict(
            os.environ, {"GITHUB_EVENT_NAME": "pull_request"}
        ):
            configure_component_ci.main()
            return mock_output.call_args[0][0]

    def test_clr_change_triggers_nvidia(self):
        self.assertEqual(self._run(["projects/clr/src/main.cpp"])["nvidia"], "true")

    def test_hip_change_triggers_nvidia(self):
        self.assertEqual(
            self._run(["projects/hip/src/hip_runtime.cpp"])["nvidia"], "true"
        )

    def test_hip_tests_change_triggers_nvidia(self):
        self.assertEqual(
            self._run(["projects/hip-tests/catch/unit/test_hip.cpp"])["nvidia"], "true"
        )

    def test_hipother_change_triggers_nvidia(self):
        self.assertEqual(
            self._run(["projects/hipother/src/hipother.cpp"])["nvidia"], "true"
        )

    def test_hip_nvidia_ci_workflow_triggers_nvidia(self):
        self.assertEqual(
            self._run([".github/workflows/hip-nvidia-ci.yml"])["nvidia"], "true"
        )

    def test_unrelated_change_does_not_trigger_nvidia(self):
        self.assertEqual(
            self._run(["projects/rocminfo/src/main.cpp"])["nvidia"], "false"
        )


class WslPatternTest(unittest.TestCase):
    def _run(self, changed_files):
        with patch(
            "configure_component_ci.get_changed_files", return_value=changed_files
        ), patch("configure_component_ci.set_github_output") as mock_output, patch.dict(
            os.environ, {"GITHUB_EVENT_NAME": "pull_request"}
        ):
            configure_component_ci.main()
            return mock_output.call_args[0][0]

    def test_rocr_runtime_change_triggers_wsl(self):
        self.assertEqual(
            self._run(["projects/rocr-runtime/src/runtime.cpp"])["wsl"], "true"
        )

    def test_wkmi_change_triggers_wsl(self):
        self.assertEqual(
            self._run(["shared/amdgpu-windows-interop/wkmi/src/wkmi.cpp"])["wsl"],
            "true",
        )

    def test_rocr_runtime_wsl_workflow_triggers_wsl(self):
        self.assertEqual(
            self._run([".github/workflows/rocr-runtime-wsl.yml"])["wsl"], "true"
        )

    def test_unrelated_change_does_not_trigger_wsl(self):
        self.assertEqual(
            self._run(["projects/rocminfo/src/main.cpp"])["wsl"], "false"
        )


class HipContractTestsPatternTest(unittest.TestCase):
    def _run(self, changed_files):
        with patch(
            "configure_component_ci.get_changed_files", return_value=changed_files
        ), patch("configure_component_ci.set_github_output") as mock_output, patch.dict(
            os.environ, {"GITHUB_EVENT_NAME": "pull_request"}
        ):
            configure_component_ci.main()
            return mock_output.call_args[0][0]

    def test_hip_include_change_triggers_contract_tests(self):
        self.assertEqual(
            self._run(["projects/hip/include/hip/hip_runtime.h"])[
                "hip-contract-tests"
            ],
            "true",
        )

    def test_catch_contract_change_triggers_contract_tests(self):
        self.assertEqual(
            self._run(["projects/hip-tests/catch/contract/test_contract.cpp"])[
                "hip-contract-tests"
            ],
            "true",
        )

    def test_catch_tools_change_triggers_contract_tests(self):
        self.assertEqual(
            self._run(["projects/hip-tests/catch/tools/tool.py"])[
                "hip-contract-tests"
            ],
            "true",
        )

    def test_contract_yaml_triggers_contract_tests(self):
        self.assertEqual(
            self._run(["projects/hip-tests/catch/config/configs/contract.yaml"])[
                "hip-contract-tests"
            ],
            "true",
        )

    def test_test_plan_md_triggers_contract_tests(self):
        self.assertEqual(
            self._run(["projects/hip-tests/catch/TEST_PLAN.md"])[
                "hip-contract-tests"
            ],
            "true",
        )

    def test_hip_contract_coverage_workflow_triggers_contract_tests(self):
        self.assertEqual(
            self._run([".github/workflows/hip-contract-coverage.yml"])[
                "hip-contract-tests"
            ],
            "true",
        )

    def test_unrelated_change_does_not_trigger_contract_tests(self):
        self.assertEqual(
            self._run(["projects/rocminfo/src/main.cpp"])["hip-contract-tests"],
            "false",
        )


class OverlapTest(unittest.TestCase):
    """Files that appear in multiple pattern lists should trigger multiple jobs."""

    def _run(self, changed_files):
        with patch(
            "configure_component_ci.get_changed_files", return_value=changed_files
        ), patch("configure_component_ci.set_github_output") as mock_output, patch.dict(
            os.environ, {"GITHUB_EVENT_NAME": "pull_request"}
        ):
            configure_component_ci.main()
            return mock_output.call_args[0][0]

    def test_clr_triggers_both_nvidia_and_wsl(self):
        out = self._run(["projects/clr/src/main.cpp"])
        self.assertEqual(out["nvidia"], "true")
        self.assertEqual(out["wsl"], "true")

    def test_component_ci_workflow_triggers_all_jobs(self):
        out = self._run([".github/workflows/component-ci.yml"])
        self.assertEqual(out["nvidia"], "true")
        self.assertEqual(out["wsl"], "true")
        self.assertEqual(out["hip-contract-tests"], "true")

    def test_empty_diff_triggers_nothing(self):
        out = self._run([])
        self.assertEqual(out["nvidia"], "false")
        self.assertEqual(out["wsl"], "false")
        self.assertEqual(out["hip-contract-tests"], "false")


if __name__ == "__main__":
    unittest.main()
