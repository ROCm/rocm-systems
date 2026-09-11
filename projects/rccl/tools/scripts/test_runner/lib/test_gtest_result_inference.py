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
    collect_gtest_case_details,
    count_gtest_cases_from_json,
    count_gtest_cases_from_json_file,
    count_pytest_cases_from_junit,
    format_case_counts,
    format_duplicate_tree,
    format_issue_tree,
    format_unique_tree,
    infer_gtest_result_from_json_file,
    infer_gtest_result_from_output,
    make_run_identity_key,
    merge_timeout_details,
    stamp_run_identity,
    summarize_case_uniqueness,
    synthetic_case_detail,
    wrap_mpi_program,
    _sum_case_counts,
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


WILDCARD_SUITE_JSON = {
    "tests": 4,
    "failures": 1,
    "testsuites": [
        {
            "name": "DdaIpcEligibilityTest",
            "tests": 4,
            "testsuite": [
                {"name": "Eligible_Ipc", "result": "COMPLETED"},
                {"name": "Ineligible_NoIpc", "result": "COMPLETED"},
                {"name": "SkipOnArch", "result": "SKIPPED"},
                {"name": "Broken", "result": "COMPLETED",
                 "failures": [{"failure": "boom"}]},
            ],
        }
    ],
}


class TestCountGtestCases(unittest.TestCase):
    def test_wildcard_suite_expands_to_leaf_cases(self):
        counts = count_gtest_cases_from_json(WILDCARD_SUITE_JSON)
        self.assertEqual(counts, {
            "cases": 4, "passed": 2, "failed": 1, "skipped": 1, "timeout": 0,
        })

    def test_empty_report_is_zero_cases(self):
        counts = count_gtest_cases_from_json({"tests": 0, "testsuites": []})
        self.assertEqual(counts["cases"], 0)

    def test_file_helper_reads_the_same_counts(self):
        path = _write_json(WILDCARD_SUITE_JSON)
        try:
            self.assertEqual(
                count_gtest_cases_from_json_file(path),
                {"cases": 4, "passed": 2, "failed": 1, "skipped": 1, "timeout": 0},
            )
        finally:
            os.unlink(path)

    def test_missing_file_is_unknown(self):
        missing = os.path.join(tempfile.gettempdir(), "rccl-no-such-gtest-report.json")
        self.assertIsNone(count_gtest_cases_from_json_file(missing))

    def test_format_and_sum(self):
        self.assertEqual(
            format_case_counts({"cases": 4, "passed": 2, "failed": 1, "skipped": 1}),
            "4 cases (2 passed, 1 failed, 1 skipped)",
        )
        self.assertEqual(
            format_case_counts({"cases": 1, "passed": 0, "failed": 0, "skipped": 0, "timeout": 1}),
            "1 cases (1 timed out)",
        )
        total = _sum_case_counts([
            {"cases": 4, "passed": 2, "failed": 1, "skipped": 1},
            None,
            {"cases": 1, "passed": 1, "failed": 0, "skipped": 0},
        ])
        self.assertEqual(total, {
            "cases": 5, "passed": 3, "failed": 1, "skipped": 1, "timeout": 0,
        })

    def test_collect_names_and_statuses(self):
        details = collect_gtest_case_details(WILDCARD_SUITE_JSON)
        self.assertEqual(
            [(d["full_name"], d["status"]) for d in details],
            [
                ("DdaIpcEligibilityTest.Eligible_Ipc", "PASSED"),
                ("DdaIpcEligibilityTest.Ineligible_NoIpc", "PASSED"),
                ("DdaIpcEligibilityTest.SkipOnArch", "SKIPPED"),
                ("DdaIpcEligibilityTest.Broken", "FAILED"),
            ],
        )

    def test_issue_tree_groups_failed_and_skipped(self):
        tree = format_issue_tree([
            {
                "config_suite": "Unit Tests - Fixtures (Debug)",
                "config_entry": "DdaIpcEligibilityTest",
                "details": collect_gtest_case_details(WILDCARD_SUITE_JSON),
            },
            {
                "config_suite": "Unit Tests - Fixtures (Debug)",
                "config_entry": "ArgCheck",
                "details": [
                    {"suite": "ArgCheckTest", "case": "Ok",
                     "full_name": "ArgCheckTest.Ok", "status": "PASSED"},
                ],
            },
            {
                "config_suite": "CE Tests - 2-Rank",
                "config_entry": "CE_AlltoAll_2Ranks",
                "details": [
                    {"suite": "CeMPI_AlltoAll", "case": "TwoRanks",
                     "full_name": "CeMPI_AlltoAll.TwoRanks", "status": "FAILED"},
                ],
            },
            {
                "config_suite": "NET Transport - Ethernet (Multi-Node)",
                "config_entry": "NET_AllTests_2Nodes_ETH",
                "details": merge_timeout_details(
                    None, "NET_AllTests_2Nodes_ETH",
                    "NetMPITest.AllTests", 600,
                ),
            },
        ])
        expected = "\n".join([
            "Failed/skipped/timeout cases:",
            "  +- Unit Tests - Fixtures (Debug)",
            "  |  `- DdaIpcEligibilityTest",
            "  |     +- SKIPPED DdaIpcEligibilityTest.SkipOnArch",
            "  |     `- FAILED  DdaIpcEligibilityTest.Broken",
            "  +- CE Tests - 2-Rank",
            "  |  `- CE_AlltoAll_2Ranks",
            "  |     `- FAILED  CeMPI_AlltoAll.TwoRanks",
            "  `- NET Transport - Ethernet (Multi-Node)",
            "     `- NET_AllTests_2Nodes_ETH",
            "        `- TIMEOUT NetMPITest.AllTests (timed out after 600s)",
        ])
        self.assertEqual(tree, expected)

    def test_timeout_keeps_finished_leaves_and_adds_placeholder(self):
        merged = merge_timeout_details(
            collect_gtest_case_details(WILDCARD_SUITE_JSON),
            "DdaIpcEligibilityTest", "DdaIpcEligibilityTest.*", 30,
        )
        statuses = [(d["full_name"], d["status"]) for d in merged]
        self.assertIn(("DdaIpcEligibilityTest.Broken", "FAILED"), statuses)
        self.assertIn(("DdaIpcEligibilityTest.* (timed out after 30s)", "TIMEOUT"), statuses)

    def test_issue_tree_includes_failed_entry_without_json(self):
        tree = format_issue_tree([
            {
                "config_suite": "Symmetric Abort + CheckMode Tests - Single Node",
                "config_entry": "SymCheckMode_Local_DebugLocal",
                "details": None,
                "executed": True,
                "config_result": "FAILED",
                "run_identity": "id",
            },
            {
                "config_suite": "Symmetric Abort + CheckMode Tests - Single Node",
                "config_entry": "SymCheckMode_Local_DebugGlobal",
                "details": [
                    synthetic_case_detail(
                        "SymCheckMode_Local_DebugGlobal",
                        "SymCheckMode_Local.DebugGlobal_HostPointer_Rejected",
                        "FAILED",
                    )
                ],
                "executed": True,
                "config_result": "FAILED",
            },
        ])
        self.assertIn("FAILED  SymCheckMode_Local_DebugLocal", tree)
        self.assertIn("FAILED  SymCheckMode_Local.DebugGlobal_HostPointer_Rejected", tree)


class TestWrapMpiProgram(unittest.TestCase):
    def test_rank0_only_sets_gtest_output(self):
        wrapped = wrap_mpi_program("/bin/rccl-UnitTestsMPI --gtest_filter=Foo.Bar", "/tmp/rccl.json")
        self.assertIn("OMPI_COMM_WORLD_RANK", wrapped)
        self.assertIn("GTEST_OUTPUT=json:/tmp/rccl.json", wrapped)
        self.assertIn('if [ "$rank" = "0" ]', wrapped)
        self.assertNotIn("--gtest_output", wrapped)

    def test_non_gtest_has_no_json_env(self):
        wrapped = wrap_mpi_program("/bin/rccl-UnitTestsMPI")
        self.assertNotIn("GTEST_OUTPUT", wrapped)
        self.assertIn("exec /bin/rccl-UnitTestsMPI", wrapped)


def _leaf(full_name, status, identity, suite=None, case=None):
    if suite is None or case is None:
        suite, _, case = full_name.partition(".")
        if not case:
            case = full_name
            suite = full_name
    return {
        "suite": suite,
        "case": case,
        "full_name": full_name,
        "status": status,
        "run_identity": identity,
    }


class TestRunIdentity(unittest.TestCase):
    def test_ignores_runner_rewritten_env(self):
        a = make_run_identity_key(
            binary="rccl-UnitTestsMPI",
            num_ranks=8,
            env_vars={"NCCL_NET": "IBVerbs", "LD_LIBRARY_PATH": "/tmp/a"},
        )
        b = make_run_identity_key(
            binary="/build/test/rccl-UnitTestsMPI",
            num_ranks=8,
            env_vars={"NCCL_NET": "IBVerbs", "LD_LIBRARY_PATH": "/tmp/b"},
        )
        self.assertEqual(a, b)

    def test_different_env_is_different_identity(self):
        a = make_run_identity_key(binary="rccl-UnitTestsMPI", env_vars={"NCCL_ALGO": "Ring"})
        b = make_run_identity_key(binary="rccl-UnitTestsMPI", env_vars={"NCCL_ALGO": "Tree"})
        self.assertNotEqual(a, b)


class TestUniqueAndDuplicateCases(unittest.TestCase):
    def test_same_leaf_same_env_is_duplicate_not_removed(self):
        identity = make_run_identity_key(
            binary="rccl-UnitTestsMPI", num_ranks=8, env_vars={"NCCL_NET": "IBVerbs"}
        )
        entries = [
            {
                "config_suite": "P2P Tests - Complete Suite",
                "config_entry": "P2P_SendRecvRegistration",
                "details": [
                    _leaf("P2pMPITest.P2pSendRecvRegistrationTest", "SKIPPED", identity),
                ],
            },
            {
                "config_suite": "P2P Tests - Complete Suite",
                "config_entry": "P2P_AllTests",
                "details": [
                    _leaf("P2pMPITest.P2pSendRecvRegistrationTest", "SKIPPED", identity),
                    _leaf("P2pMPITest.IpcGraphRegisterBufferTest", "SKIPPED", identity),
                ],
            },
        ]
        summary = summarize_case_uniqueness(entries)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["unique"], 2)
        self.assertEqual(summary["duplicate_cases"], 1)
        self.assertEqual(summary["duplicate_extra"], 1)
        self.assertEqual(summary["unique_skipped"], 2)

        unique_tree = format_unique_tree(summary["unique_entries"])
        expected_unique = "\n".join([
            "Unique test cases:",
            "  `- P2P Tests - Complete Suite",
            "     +- P2P_SendRecvRegistration",
            "     |  `- SKIPPED P2pMPITest.P2pSendRecvRegistrationTest",
            "     `- P2P_AllTests",
            "        `- SKIPPED P2pMPITest.IpcGraphRegisterBufferTest",
        ])
        self.assertEqual(unique_tree, expected_unique)

        dup_tree = format_duplicate_tree(summary["duplicate_entries"])
        expected_dup = "\n".join([
            "Duplicate cases:",
            "  `- P2P Tests - Complete Suite",
            "     +- P2P_SendRecvRegistration",
            "     |  `- DUPLICATE P2pMPITest.P2pSendRecvRegistrationTest (2 runs)",
            "     `- P2P_AllTests",
            "        `- DUPLICATE P2pMPITest.P2pSendRecvRegistrationTest (2 runs)",
        ])
        self.assertEqual(dup_tree, expected_dup)

    def test_same_leaf_different_env_is_unique(self):
        default_id = make_run_identity_key(
            binary="rccl-UnitTests", env_vars={"RCCL_ENABLE_HOST_GRAPH": "1"}
        )
        sym_id = make_run_identity_key(
            binary="rccl-UnitTests", env_vars={"NCCL_SYM_KERNEL": "ALL"}
        )
        leaf = "RcclHostApi.CommSplit"
        entries = [
            {
                "config_suite": "Host API Tests",
                "config_entry": "host_api_default",
                "details": [_leaf(leaf, "PASSED", default_id)],
            },
            {
                "config_suite": "Host API Tests",
                "config_entry": "host_api_symmem",
                "details": [_leaf(leaf, "PASSED", sym_id)],
            },
        ]
        summary = summarize_case_uniqueness(entries)
        self.assertEqual(summary["unique"], 2)
        self.assertEqual(summary["duplicate_cases"], 0)
        self.assertEqual(format_duplicate_tree(summary["duplicate_entries"]), "")

    def test_mixed_status_duplicate_keeps_counts(self):
        identity = make_run_identity_key(binary="rccl-UnitTests", env_vars={})
        entries = [
            {
                "config_suite": "Unit Tests - Fixtures (Debug)",
                "config_entry": "DdaAlltoAllThresholdTest",
                "details": [_leaf(
                    "DdaAlltoAllThresholdTest.SymmetricSupport_Disabled",
                    "FAILED", identity,
                )],
            },
            {
                "config_suite": "Unit Tests - Standard Collectives",
                "config_entry": "DdaAlltoAllThresholdTest_Again",
                "details": [_leaf(
                    "DdaAlltoAllThresholdTest.SymmetricSupport_Disabled",
                    "PASSED", identity,
                )],
            },
        ]
        summary = summarize_case_uniqueness(entries)
        self.assertEqual(summary["unique"], 1)
        self.assertEqual(summary["unique_failed"], 1)
        self.assertEqual(summary["duplicate_extra"], 1)
        dup_tree = format_duplicate_tree(summary["duplicate_entries"])
        self.assertIn("(2 runs: FAILED, PASSED)", dup_tree)

    def test_stamp_does_not_mutate_original(self):
        original = [{"suite": "S", "case": "C", "full_name": "S.C", "status": "PASSED"}]
        stamped = stamp_run_identity(original, "id-1")
        self.assertEqual(stamped[0]["run_identity"], "id-1")
        self.assertNotIn("run_identity", original[0])

    def test_suppressed_disabled_leaves_are_not_executed(self):
        payload = {
            "testsuites": [{
                "name": "MemManagerRealMem",
                "testsuite": [
                    {"name": "Track_Something", "result": "COMPLETED", "failures": []},
                    {
                        "name": "DISABLED_Track_RealHipMalloc_Scratch",
                        "result": "SUPPRESSED",
                        "failures": [],
                    },
                ],
            }]
        }
        details = collect_gtest_case_details(payload)
        self.assertEqual(
            [(d["full_name"], d["status"]) for d in details],
            [("MemManagerRealMem.Track_Something", "PASSED")],
        )
        counts = count_gtest_cases_from_json(payload)
        self.assertEqual(counts, {
            "cases": 1, "passed": 1, "failed": 0, "skipped": 0, "timeout": 0,
        })

    def test_missing_report_counts_as_one_executed_and_unique_case(self):
        # JSON missing after MPI abort / write race: Unique used to exceed Total
        # because uniqueness synthesized a leaf that case_counts never saw.
        entries = [
            {
                "config_suite": "Grow MPI Tests (2-rank)",
                "config_entry": "Grow_ConfigInheritance",
                "details": None,
                "executed": True,
                "config_result": "PASSED",
                "run_identity": make_run_identity_key(binary="rccl-UnitTestsMPI"),
            },
            {
                "config_suite": "Symmetric Abort + CheckMode Tests - Single Node",
                "config_entry": "SymCheckMode_Local_DebugLocal",
                "details": None,
                "executed": True,
                "config_result": "FAILED",
                "run_identity": make_run_identity_key(
                    binary="rccl-UnitTestsMPI", env_vars={"NCCL_CHECK_MODE": "DEBUG_LOCAL"}
                ),
            },
            {
                "config_suite": "Symmetric Abort + CheckMode Tests - Single Node",
                "config_entry": "SymCheckMode_Local_DebugGlobal",
                "details": None,
                "executed": True,
                "config_result": "FAILED",
                "run_identity": make_run_identity_key(
                    binary="rccl-UnitTestsMPI", env_vars={"NCCL_CHECK_MODE": "DEBUG_GLOBAL"}
                ),
            },
        ]
        summary = summarize_case_uniqueness(entries)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["unique"], 3)
        self.assertEqual(summary["duplicate_cases"], 0)
        self.assertEqual(summary["executed_passed"], 1)
        self.assertEqual(summary["executed_failed"], 2)
        self.assertEqual(summary["unique_passed"], 1)
        self.assertEqual(summary["unique_failed"], 2)
        self._assert_count_invariants(summary)

    def test_synthetic_detail_uses_gtest_filter_name(self):
        leaf = synthetic_case_detail(
            "SymCheckMode_Local_DebugLocal",
            "SymCheckMode_Local.DebugLocal_HostPointer_Rejected",
            "FAILED",
        )
        self.assertEqual(leaf["full_name"], "SymCheckMode_Local.DebugLocal_HostPointer_Rejected")
        self.assertEqual(leaf["status"], "FAILED")

    def test_no_duplicate_executed_matches_unique(self):
        identity = make_run_identity_key(binary="rccl-UnitTests", env_vars={})
        entries = [
            {
                "config_suite": "S",
                "config_entry": "A",
                "details": [_leaf("Suite.Pass", "PASSED", identity)],
            },
            {
                "config_suite": "S",
                "config_entry": "B",
                "details": [_leaf("Suite.Fail", "FAILED", identity)],
            },
        ]
        summary = summarize_case_uniqueness(entries)
        self.assertEqual(summary["executed_failed"], summary["unique_failed"])
        self.assertEqual(summary["executed_passed"], summary["unique_passed"])
        self._assert_count_invariants(summary)

    def _assert_count_invariants(self, summary):
        self.assertEqual(
            summary["unique"] + summary["duplicate_extra"],
            summary["total"],
        )
        self.assertEqual(
            summary["executed_passed"]
            + summary["executed_failed"]
            + summary["executed_skipped"]
            + summary["executed_timeout"]
            + summary["executed_other"],
            summary["total"],
        )
        self.assertEqual(
            summary["unique_passed"]
            + summary["unique_failed"]
            + summary["unique_skipped"]
            + summary["unique_timeout"]
            + summary["unique_other"],
            summary["unique"],
        )


class TestCountPytestCases(unittest.TestCase):
    def test_junit_counts_each_testcase(self):
        xml = """<?xml version="1.0"?>
<testsuites>
  <testsuite name="ir" tests="3" failures="1" skipped="1">
    <testcase classname="t" name="a"/>
    <testcase classname="t" name="b"><skipped message="x"/></testcase>
    <testcase classname="t" name="c"><failure message="boom"/></testcase>
  </testsuite>
</testsuites>
"""
        fd, path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w") as f:
            f.write(xml)
        try:
            self.assertEqual(
                count_pytest_cases_from_junit(path),
                {"cases": 3, "passed": 1, "failed": 1, "skipped": 1, "timeout": 0},
            )
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
