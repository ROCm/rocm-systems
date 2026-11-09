import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lib.test_config import TestConfigProcessor
from lib.test_executor import TestExecutor


_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_RCCL = Path(__file__).resolve().parents[4]


class RcclEpArgumentConfigTest(unittest.TestCase):
    def test_build_disables_ep_with_mpi_and_preserves_normal_build_flags(self):
        processor = TestConfigProcessor(str(_CONFIG_DIR / "mi355x_thor2_roce_func.json"))
        for skip_mpi in (False, True):
            for cached_options in ("", "-DENABLE_MPI_TESTS=ON -DENABLE_RCCL_EP_TESTS=ON",
                                   {"ENABLE_MPI_TESTS": "ON", "ENABLE_RCCL_EP_TESTS": "ON"}):
                with self.subTest(skip_mpi=skip_mpi, cached_options=cached_options):
                    executor = TestExecutor.__new__(TestExecutor)
                    executor.using_custom_lib = False
                    executor.args = SimpleNamespace(
                        no_build=False, verbose=False, skip_mpi_check=skip_mpi,
                        coverage_report=False,
                    )
                    executor.paths = {"workdir": str(_RCCL), "rocm_path": "/opt/rocm"}
                    executor.build_config = dict(processor.config["build_configuration"])
                    executor.build_config["cmake_options"] = cached_options
                    original_flags = list(executor.build_config["install_flags"])
                    with patch.object(executor, "_rocm_root", return_value="/opt/rocm"), \
                         patch("lib.test_executor.subprocess.run",
                               return_value=SimpleNamespace(returncode=0)) as run:
                        self.assertTrue(executor.build_rccl())
                    command = run.call_args.args[0]
                    for flag in ("--enable-mpi-tests", "--enable-rccl-ep-tests"):
                        self.assertEqual(flag in command, not skip_mpi)
                    options = command[command.index("--cmake-options") + 1].split()
                    if skip_mpi:
                        for name in ("ENABLE_MPI_TESTS", "ENABLE_RCCL_EP_TESTS"):
                            values = [option for option in options if option.startswith(f"-D{name}=")]
                            self.assertEqual(values[-1], f"-D{name}=OFF")
                    self.assertEqual(executor.build_config["install_flags"], original_flags)

    def test_argument_checks_are_built_and_enabled_on_eight_local_gpus(self):
        processor = TestConfigProcessor(str(_CONFIG_DIR / "mi355x_thor2_roce_func.json"))
        flags = processor.config["build_configuration"]["install_flags"]
        self.assertIn("-t", flags)
        self.assertIn("--enable-mpi-tests", flags)
        self.assertIn("--enable-rccl-ep-tests", flags)

        suite = processor.combine_configs("rccl_ep_argument_checks")
        self.assertEqual(suite["binary"], "rccl-UnitTestsMPI")
        self.assertTrue(suite["is_gtest"])
        self.assertEqual(suite["num_ranks"], 8)
        self.assertEqual(suite["num_nodes"], 1)
        self.assertEqual(suite["num_gpus"], 8)
        self.assertEqual(suite["env_variables"]["NCCL_CUMEM_ENABLE"], "1")
        self.assertEqual(suite["timeout"], 180)
        self.assertEqual(len(suite["tests"]), 1)
        self.assertEqual(
            suite["tests"][0]["test_filter"],
            "RcclEpDispatchCombineTest.CreateRejectsOutOfRangeArguments",
        )
        enabled = [
            entry
            for entry in processor.config["test_suites"]
            if entry["config"] == "rccl_ep_argument_checks" and entry.get("enabled", True)
        ]
        self.assertEqual(len(enabled), 1)
        source = (_RCCL / "test/RcclEpDispatchCombineMPITests.cpp").read_text(encoding="utf-8")
        self.assertIn("TEST_F(RcclEpDispatchCombineTest, CreateRejectsOutOfRangeArguments)", source)


if __name__ == "__main__":
    unittest.main()
