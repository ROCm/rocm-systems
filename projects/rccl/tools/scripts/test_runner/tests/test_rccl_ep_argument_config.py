import unittest
from pathlib import Path

from lib.test_config import TestConfigProcessor


_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_RCCL = Path(__file__).resolve().parents[4]


class RcclEpArgumentConfigTest(unittest.TestCase):
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
