import json
import os
import tempfile
import unittest

from check_test_filters import config_tests, parse_gtest_list, positive_clauses, unmatched_clauses

LISTING = """Running main() from gtest_main.cc
RmaMultiSegmentPostMPITest.
  ChainsAroundWrBatchSizeLandIntact
  IPutSignalSendQueueOversubscribe
Sizes/ParamTest.
  Runs/0  # GetParam() = 4
"""


class CheckTestFiltersTest(unittest.TestCase):
    def setUp(self):
        self.names = parse_gtest_list(LISTING)

    def test_parses_plain_and_parameterized_names(self):
        self.assertEqual(self.names, [
            "RmaMultiSegmentPostMPITest.ChainsAroundWrBatchSizeLandIntact",
            "RmaMultiSegmentPostMPITest.IPutSignalSendQueueOversubscribe",
            "Sizes/ParamTest.Runs/0",
        ])

    def test_negative_clauses_are_ignored(self):
        self.assertEqual(positive_clauses("A.*:B.x-C.y:D.*"), ["A.*", "B.x"])
        self.assertEqual(positive_clauses(""), ["*"])
        self.assertEqual(positive_clauses("-A.x"), ["*"])
        self.assertEqual(positive_clauses("ALL"), ["*"])

    def test_reports_the_stale_clause_only(self):
        stale = "RmaMultiSegmentMPITest.IPutSignalSendQueueOversubscribe"
        self.assertEqual(unmatched_clauses(f"RmaMultiSegmentPostMPITest.*:{stale}", self.names), [stale])
        self.assertEqual(unmatched_clauses("*/ParamTest.Runs/*", self.names), [])

    def test_config_tests_resolves_extends_and_binary(self):
        config = {
            "test_configurations": {
                "parent": {"is_gtest": True, "binary": "build/rccl-UnitTestsMPI", "tests": []},
                "child": {"extends": "parent", "tests": [
                    {"name": "a", "test_filter": "A.*"},
                    {"name": "b", "test_filter": "B.*", "binary": "rccl-UnitTestsFixtures"},
                    {"name": "c", "is_gtest": False},
                ]},
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.json")
            with open(path, "w") as f:
                json.dump(config, f)
            got = [(cname, test["name"], binary) for cname, test, binary in config_tests(path)]
        self.assertEqual(got, [("child", "a", "rccl-UnitTestsMPI"), ("child", "b", "rccl-UnitTestsFixtures")])


if __name__ == "__main__":
    unittest.main()
