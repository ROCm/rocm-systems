import unittest

from check_test_filters import parse_gtest_list, positive_clauses, unmatched_clauses

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

    def test_reports_the_stale_clause_only(self):
        stale = "RmaMultiSegmentMPITest.IPutSignalSendQueueOversubscribe"
        self.assertEqual(unmatched_clauses(f"RmaMultiSegmentPostMPITest.*:{stale}", self.names), [stale])
        self.assertEqual(unmatched_clauses("*/ParamTest.Runs/*", self.names), [])


if __name__ == "__main__":
    unittest.main()
