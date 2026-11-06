import re
import unittest
from pathlib import Path

from lib.test_config import TestConfigProcessor


_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_RCCL = Path(__file__).resolve().parents[4]


class RecorderLogLevelConfigTest(unittest.TestCase):
    def test_end_to_end_cases_are_enabled_with_bounded_runs(self):
        source = (_RCCL / "test/_RecorderTests.cpp").read_text(encoding="utf-8")
        cases = {
            f"Recorder.{name}"
            for name in re.findall(r"TEST\(Recorder,\s*(LogLevel\w+)\)", source)
        }
        self.assertEqual(len(cases), 3)

        for filename in ("mi300x_mellanox_ib.json", "mi355x_thor2_roce.json"):
            with self.subTest(config=filename):
                processor = TestConfigProcessor(str(_CONFIG_DIR / filename))
                suite = processor.combine_configs("unit_tests_standard")
                self.assertEqual(suite["binary"], "rccl-UnitTests")
                self.assertTrue(suite["is_gtest"])
                enabled = [
                    entry
                    for entry in processor.config["test_suites"]
                    if entry["config"] == "unit_tests_standard" and entry.get("enabled", True)
                ]
                self.assertEqual(len(enabled), 1)
                selected = [
                    entry for entry in suite["tests"] if entry.get("test_filter") in cases
                ]
                self.assertEqual(len(selected), len(cases))
                self.assertEqual({entry["test_filter"] for entry in selected}, cases)
                for entry in selected:
                    self.assertEqual(entry["timeout"], 60)


if __name__ == "__main__":
    unittest.main()
