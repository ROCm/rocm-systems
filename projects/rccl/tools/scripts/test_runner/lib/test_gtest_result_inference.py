#!/usr/bin/env python3
"""Unit tests for test_executor's gtest result inference.

The regression these guard: a --gtest_filter matching no test makes Google Test
print "Running 0 tests from 0 test suites" and exit 0, with no per-test [ OK ] /
[ SKIPPED ] / [ FAILED ] line. Both inference paths fell through to PASSED, so a
test_filter with a typo (e.g. "UBR_AlltoAll.X" for a suite really named
"UBR_AllToAll") reported green while executing nothing at all.

Zero selected tests is now SKIPPED, matching what the pytest path already does
for "no tests collected".
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_executor import (
    infer_gtest_result_from_json_file,
    infer_gtest_result_from_output,
)

FILTER_MATCHED_NOTHING = (
    "Note: Google Test filter = UBR_AlltoAll.OutOfPlace_MultiNode\n"
    "[==========] Running 0 tests from 0 test suites.\n"
    "[==========] 0 tests from 0 test suites ran. (0 ms total)\n"
    "[  PASSED  ] 0 tests.\n"
)

ONE_TEST_PASSED = (
    "[==========] Running 1 test from 1 test suite.\n"
    "[ RUN      ] Suite.Case\n"
    "[       OK ] Suite.Case (1 ms)\n"
    "[  PASSED  ] 1 test.\n"
)

ONE_TEST_SKIPPED = (
    "[==========] Running 1 test from 1 test suite.\n"
    "[ RUN      ] Suite.Case\n"
    "[  SKIPPED ] Suite.Case (0 ms)\n"
)

ONE_TEST_FAILED = (
    "[==========] Running 1 test from 1 test suite.\n"
    "[ RUN      ] Suite.Case\n"
    "[  FAILED  ] Suite.Case (0 ms)\n"
)


def _write_json(payload):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    return path


class TestInferFromOutput(unittest.TestCase):
    def test_filter_matched_nothing_is_skipped_not_passed(self):
        self.assertEqual(infer_gtest_result_from_output(FILTER_MATCHED_NOTHING, 0), "SKIPPED")

    def test_legacy_test_cases_wording_also_detected(self):
        out = "[==========] Running 0 tests from 0 test cases.\n"
        self.assertEqual(infer_gtest_result_from_output(out, 0), "SKIPPED")

    def test_passing_run_still_passes(self):
        self.assertEqual(infer_gtest_result_from_output(ONE_TEST_PASSED, 0), "PASSED")

    def test_skipped_run_still_skips(self):
        self.assertEqual(infer_gtest_result_from_output(ONE_TEST_SKIPPED, 0), "SKIPPED")

    def test_failed_run_still_fails(self):
        self.assertEqual(infer_gtest_result_from_output(ONE_TEST_FAILED, 0), "FAILED")

    def test_nonzero_exit_fails(self):
        self.assertEqual(infer_gtest_result_from_output(ONE_TEST_PASSED, 1), "FAILED")

    def test_timeout_exit_reports_timeout(self):
        self.assertEqual(infer_gtest_result_from_output("", 124), "TIMEOUT")

    def test_empty_output_is_unchanged(self):
        """The missing-JSON fallback calls this with "" -- must stay PASSED on exit 0."""
        self.assertEqual(infer_gtest_result_from_output("", 0), "PASSED")


class TestInferFromJsonFile(unittest.TestCase):
    def test_report_with_no_tests_is_skipped_not_passed(self):
        path = _write_json({"tests": 0, "failures": 0, "testsuites": []})
        try:
            self.assertEqual(infer_gtest_result_from_json_file(path, 0), "SKIPPED")
        finally:
            os.unlink(path)

    def test_report_with_passing_test_passes(self):
        path = _write_json({
            "tests": 1,
            "testsuites": [
                {"name": "Suite", "testsuite": [{"name": "Case", "result": "COMPLETED"}]}
            ],
        })
        try:
            self.assertEqual(infer_gtest_result_from_json_file(path, 0), "PASSED")
        finally:
            os.unlink(path)

    def test_report_with_skipped_test_skips(self):
        path = _write_json({
            "tests": 1,
            "testsuites": [
                {"name": "Suite", "testsuite": [{"name": "Case", "result": "SKIPPED"}]}
            ],
        })
        try:
            self.assertEqual(infer_gtest_result_from_json_file(path, 0), "SKIPPED")
        finally:
            os.unlink(path)

    def test_report_with_failure_fails(self):
        path = _write_json({
            "tests": 1,
            "testsuites": [
                {
                    "name": "Suite",
                    "testsuite": [
                        {"name": "Case", "result": "COMPLETED",
                         "failures": [{"failure": "boom"}]}
                    ],
                }
            ],
        })
        try:
            self.assertEqual(infer_gtest_result_from_json_file(path, 0), "FAILED")
        finally:
            os.unlink(path)

    def test_missing_json_falls_back_to_exit_code(self):
        missing = os.path.join(tempfile.gettempdir(), "rccl-no-such-gtest-report.json")
        self.assertFalse(os.path.exists(missing))
        self.assertEqual(infer_gtest_result_from_json_file(missing, 0), "PASSED")
        self.assertEqual(infer_gtest_result_from_json_file(missing, 1), "FAILED")


if __name__ == "__main__":
    unittest.main()
