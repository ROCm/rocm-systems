#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""Unit tests for test_executor's gtest result inference.

The regression these guard: a --gtest_filter matching no test makes Google Test
print "Running 0 tests from 0 test suites" and exit 0, with no per-test [ OK ] /
[ SKIPPED ] / [ FAILED ] line. Both inference paths fell through to PASSED, so a
test_filter with a typo (e.g. "UBR_AlltoAll.X" for a suite really named
"UBR_AllToAll") reported green while executing nothing at all.

Zero selected tests is now SKIPPED, matching what the pytest path already does
for "no tests collected".
"""

import argparse
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from lib.test_executor import (
    TestExecutor,
    _counts_from_details,
    collect_gtest_case_details,
    collect_gtest_case_details_from_file,
    collect_pytest_case_details_from_junit,
    format_case_counts,
    format_duplicate_tree,
    format_issue_tree,
    infer_gtest_result_from_json_file,
    infer_gtest_result_from_output,
    make_run_identity_key,
    merge_process_failure_details,
    merge_timeout_details,
    relocate_rma_reload_counter,
    stamp_run_identity,
    suite_disposition,
    summarize_case_uniqueness,
    synthetic_case_detail,
    wrap_mpi_program,
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
        """Exit 0 with no report uses the stdout fallback.

        A non-zero exit returns FAILED before the file is opened, so it does
        not exercise this path. The patch fails the test if exit 0 stops
        calling the fallback.
        """
        missing = os.path.join(tempfile.gettempdir(), "rccl-no-such-gtest-report.json")
        self.assertFalse(os.path.exists(missing))
        with patch(
            "lib.test_executor.infer_gtest_result_from_output",
            return_value="PASSED",
        ) as fallback:
            self.assertEqual(infer_gtest_result_from_json_file(missing, 0), "PASSED")
        fallback.assert_called_once_with("", 0)

    def test_nonzero_exit_fails_even_when_json_passed(self):
        """A passing report must not override a non-zero process exit."""
        path = _write_json({
            "tests": 1,
            "testsuites": [
                {"name": "Suite", "testsuite": [{"name": "Case", "result": "COMPLETED"}]}
            ],
        })
        try:
            self.assertEqual(infer_gtest_result_from_json_file(path, 0), "PASSED")
            self.assertEqual(infer_gtest_result_from_json_file(path, 1), "FAILED")
        finally:
            os.unlink(path)

    def test_leaf_without_result_agrees_with_details(self):
        path = _write_json({
            "tests": 1,
            "testsuites": [
                {"name": "Suite", "testsuite": [{"name": "Case"}]}
            ],
        })
        try:
            details = collect_gtest_case_details_from_file(path)
            self.assertEqual(details[0]["status"], "FAILED")
            self.assertEqual(
                infer_gtest_result_from_json_file(path, 0, details=details),
                "FAILED",
            )
        finally:
            os.unlink(path)

    def test_started_leaf_without_result_is_failed(self):
        path = _write_json({
            "tests": 1,
            "testsuites": [
                {"name": "Suite", "testsuite": [{"name": "Case", "status": "RUN"}]}
            ],
        })
        try:
            details = collect_gtest_case_details_from_file(path)
            self.assertEqual(details[0]["status"], "FAILED")
            self.assertEqual(infer_gtest_result_from_json_file(path, 0, details=details), "FAILED")
        finally:
            os.unlink(path)

    def test_notrun_leaf_without_result_is_skipped(self):
        path = _write_json({
            "tests": 1,
            "testsuites": [
                {"name": "Suite", "testsuite": [{"name": "Case", "status": "NOTRUN"}]}
            ],
        })
        try:
            details = collect_gtest_case_details_from_file(path)
            self.assertEqual(details[0]["status"], "SKIPPED")
            self.assertEqual(infer_gtest_result_from_json_file(path, 0, details=details), "SKIPPED")
        finally:
            os.unlink(path)


class TestProcessFailureDetails(unittest.TestCase):
    def test_nonzero_mpi_exit_adds_failure_to_passing_rank0_report(self):
        details = [{
            "suite": "Suite",
            "case": "Case",
            "full_name": "Suite.Case",
            "status": "PASSED",
        }]
        merged = merge_process_failure_details(details, "MPI Config", "Suite.Case")
        self.assertEqual([item["status"] for item in merged], ["PASSED", "FAILED"])
        self.assertEqual(
            merged[-1]["full_name"],
            "MPI Config (process exited non-zero)",
        )
        self.assertEqual([item["status"] for item in details], ["PASSED"])

    def test_existing_failed_leaf_does_not_add_duplicate(self):
        details = [{
            "suite": "Suite",
            "case": "Case",
            "full_name": "Suite.Case",
            "status": "FAILED",
        }]
        self.assertIs(
            merge_process_failure_details(details, "MPI Config", "Suite.Case"),
            details,
        )


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
        counts = _counts_from_details(collect_gtest_case_details(WILDCARD_SUITE_JSON))
        self.assertEqual(counts, {
            "cases": 4, "passed": 2, "failed": 1, "skipped": 1, "timeout": 0,
            "disabled": 0,
        })

    def test_empty_report_is_zero_cases(self):
        counts = _counts_from_details(collect_gtest_case_details({"tests": 0, "testsuites": []}))
        self.assertEqual(counts["cases"], 0)

    def test_file_helper_reads_the_same_counts(self):
        path = _write_json(WILDCARD_SUITE_JSON)
        try:
            self.assertEqual(
                _counts_from_details(collect_gtest_case_details_from_file(path)),
                {"cases": 4, "passed": 2, "failed": 1, "skipped": 1, "timeout": 0,
                 "disabled": 0},
            )
        finally:
            os.unlink(path)

    def test_missing_file_is_unknown(self):
        missing = os.path.join(tempfile.gettempdir(), "rccl-no-such-gtest-report.json")
        self.assertIsNone(collect_gtest_case_details_from_file(missing))

    def test_format_and_sum(self):
        self.assertEqual(
            format_case_counts({"cases": 4, "passed": 2, "failed": 1, "skipped": 1}),
            "4 cases (2 passed, 1 failed, 1 skipped)",
        )
        self.assertEqual(
            format_case_counts({"cases": 1, "passed": 0, "failed": 0, "skipped": 0, "timeout": 1}),
            "1 cases (1 timed out)",
        )
        self.assertEqual(
            format_case_counts({
                "cases": 2, "passed": 0, "failed": 0, "skipped": 0,
                "timeout": 0, "disabled": 2,
            }),
            "2 cases (2 disabled)",
        )

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
        self.assertNotIn("IBVerbs", a)
        self.assertNotIn("LD_LIBRARY_PATH", a)

    def test_different_env_is_different_identity(self):
        a = make_run_identity_key(binary="rccl-UnitTestsMPI", env_vars={"NCCL_ALGO": "Ring"})
        b = make_run_identity_key(binary="rccl-UnitTestsMPI", env_vars={"NCCL_ALGO": "Tree"})
        self.assertNotEqual(a, b)
        self.assertNotIn("Ring", a)
        self.assertNotIn("Tree", b)

    def test_reload_counter_leaves_tmp(self):
        workspace = tempfile.mkdtemp(prefix="rccl-identity-")
        try:
            configured = "/tmp/rccl_rma_reload_put_signal_counter.txt"
            env = {"RCCL_RMA_RELOAD_COUNTER_FILE": configured}
            relocate_rma_reload_counter(env, workspace)
            path = env["RCCL_RMA_RELOAD_COUNTER_FILE"]
            parent = os.path.join(workspace, "rma_reload_counters")
            self.assertTrue(path.startswith(parent + os.sep))
            self.assertNotEqual(path, configured)
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(os.stat(parent).st_mode & 0o777, 0o700)
            self.assertEqual(os.stat(os.path.dirname(path)).st_mode & 0o777, 0o700)
        finally:
            shutil.rmtree(workspace)


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
        self.assertNotIn("unique_entries", summary)

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
        counts = _counts_from_details(details)
        self.assertEqual(counts, {
            "cases": 1, "passed": 1, "failed": 0, "skipped": 0, "timeout": 0,
            "disabled": 0,
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
        self.assertEqual(summary["total_passed"], 1)
        self.assertEqual(summary["total_failed"], 2)
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

    def test_synthetic_detail_keeps_config_name_for_colon_filter(self):
        leaf = synthetic_case_detail(
            "RcclAllReduceDdaDecision",
            "Rcclwrap.Gfx942_SymOff_MidMsg_TakesDda:Rcclwrap.Gfx1250_CeEligible_StillTakesDda",
            "FAILED",
        )
        self.assertEqual(leaf["full_name"], "RcclAllReduceDdaDecision")
        self.assertNotIn(":", leaf["full_name"])

    def test_no_duplicate_total_matches_unique(self):
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
        self.assertEqual(summary["total_failed"], summary["unique_failed"])
        self.assertEqual(summary["total_passed"], summary["unique_passed"])
        self._assert_count_invariants(summary)

    def _assert_count_invariants(self, summary):
        self.assertEqual(
            summary["unique"] + summary["duplicate_extra"],
            summary["total"],
        )
        self.assertEqual(
            summary["total_passed"]
            + summary["total_failed"]
            + summary["total_skipped"]
            + summary["total_timeout"]
            + summary["total_disabled"]
            + summary["total_other"],
            summary["total"],
        )
        self.assertEqual(
            summary["unique_passed"]
            + summary["unique_failed"]
            + summary["unique_skipped"]
            + summary["unique_timeout"]
            + summary["unique_disabled"]
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
                _counts_from_details(collect_pytest_case_details_from_junit(path)),
                {"cases": 3, "passed": 1, "failed": 1, "skipped": 1, "timeout": 0,
                 "disabled": 0},
            )
        finally:
            os.unlink(path)


class TestDisabledConfigAccounting(unittest.TestCase):
    def _executor(self, test_name=None):
        ex = TestExecutor.__new__(TestExecutor)
        ex.args = argparse.Namespace(test_name=test_name)
        ex.emit_enabled = False
        ex.test_results = []
        ex.test_names = []
        ex.test_durations = []
        ex.test_suites = []
        ex.test_case_counts = []
        ex.test_case_details = []
        ex.test_run_identities = []
        ex.test_executed = []
        ex.test_records = []
        ex.rerun_results = []
        return ex

    def test_disabled_entries_are_in_total_and_unique(self):
        identity = make_run_identity_key(extra="disabled:NET IB:InitA")
        entries = [
            {
                "config_suite": "Unit Tests",
                "config_entry": "ArgCheck",
                "details": [
                    _leaf("ArgCheckTest.Ok", "PASSED",
                          make_run_identity_key(binary="rccl-UnitTests")),
                ],
            },
            {
                "config_suite": "NET IB - Initialization Tests",
                "config_entry": "InitA",
                "details": [
                    _leaf("NetIbMPITest.InitA", "DISABLED", identity),
                ],
            },
            {
                "config_suite": "NET IB - Initialization Tests",
                "config_entry": "InitB",
                "details": [
                    _leaf("NetIbMPITest.InitB", "DISABLED",
                          make_run_identity_key(extra="disabled:NET IB:InitB")),
                ],
            },
        ]
        summary = summarize_case_uniqueness(entries)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["unique"], 3)
        self.assertEqual(summary["duplicate_cases"], 0)
        self.assertEqual(summary["total_passed"], 1)
        self.assertEqual(summary["total_disabled"], 2)
        self.assertEqual(summary["unique_disabled"], 2)
        self._assert_count_invariants(summary)

    def test_record_disabled_suite_counts_each_config_test(self):
        ex = self._executor()
        ex.record_disabled_suite({
            "suite_details": {"name": "NET IB - Initialization Tests"},
            "tests": [
                {"name": "InitA", "test_filter": "NetIbMPITest.InitA"},
                {"name": "InitB", "test_filter": "NetIbMPITest.InitB"},
            ],
        })
        self.assertEqual(ex.test_results, ["DISABLED", "DISABLED"])
        self.assertEqual(ex.test_names, ["InitA", "InitB"])
        self.assertEqual(
            [c["cases"] for c in ex.test_case_counts],
            [1, 1],
        )
        summary = summarize_case_uniqueness(ex._summary_case_entries())
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["unique"], 2)
        self.assertEqual(summary["total_disabled"], 2)
        self._assert_count_invariants(summary)

    def test_record_disabled_suite_honors_test_name_filter(self):
        ex = self._executor(test_name="InitA")
        ex.record_disabled_suite({
            "suite_details": {"name": "NET IB"},
            "tests": [
                {"name": "InitA", "test_filter": "NetIbMPITest.InitA"},
                {"name": "InitB", "test_filter": "NetIbMPITest.InitB"},
            ],
        })
        self.assertEqual(ex.test_names, ["InitA"])

    def test_record_disabled_suite_emit_duration_is_float(self):
        ex = self._executor()
        ex.emit_enabled = True
        ex.record_disabled_suite({
            "suite_details": {"name": "NET IB"},
            "tests": [{"name": "InitA", "test_filter": "NetIbMPITest.InitA"}],
        })
        self.assertEqual(ex.test_durations, [0.0])
        self.assertIsInstance(ex.test_durations[0], float)
        self.assertEqual(ex.test_records[0]["duration"], 0.0)
        self.assertIsInstance(ex.test_records[0]["duration"], float)

    def test_print_summary_adds_disabled_into_total_and_unique(self):
        ex = self._executor()
        # NET IB is first so a DISABLED leaf that leaked into the issue tree
        # would be a non-last suite and render as "+- NET IB". CE Tests is the
        # non-last real issue suite, so that glyph is reachable.
        ex.test_suites = ["NET IB", "CE Tests", "Socket Tests", "Unit Tests"]
        ex.test_names = ["InitA", "AlltoAll", "SendRecv", "ArgCheck"]
        ex.test_results = ["DISABLED", "FAILED", "SKIPPED", "PASSED"]
        ex.test_durations = [0.0, 0.4, 0.2, 1.5]
        ex.test_case_counts = [
            {"cases": 1, "passed": 0, "failed": 0, "skipped": 0,
             "timeout": 0, "disabled": 1},
            {"cases": 1, "passed": 0, "failed": 1, "skipped": 0,
             "timeout": 0, "disabled": 0},
            {"cases": 1, "passed": 0, "failed": 0, "skipped": 1,
             "timeout": 0, "disabled": 0},
            {"cases": 1, "passed": 1, "failed": 0, "skipped": 0,
             "timeout": 0, "disabled": 0},
        ]
        disabled_id = make_run_identity_key(extra="disabled:NET IB:InitA")
        failed_id = make_run_identity_key(binary="rccl-CE")
        skipped_id = make_run_identity_key(binary="rccl-Socket")
        passed_id = make_run_identity_key(binary="rccl-UnitTests")
        ex.test_case_details = [
            [_leaf("NetIbMPITest.InitA", "DISABLED", disabled_id)],
            [_leaf("CeMPI.AlltoAll", "FAILED", failed_id)],
            [_leaf("Socket.SendRecv", "SKIPPED", skipped_id)],
            [_leaf("ArgCheckTest.Ok", "PASSED", passed_id)],
        ]
        ex.test_run_identities = [disabled_id, failed_id, skipped_id, passed_id]
        ex.test_executed = [False, True, True, True]
        buf = io.StringIO()
        with redirect_stdout(buf):
            ex.print_summary()
        out = buf.getvalue()
        self.assertIn("DISABLED", out)
        self.assertRegex(out, r"Config entries:\s+4")
        self.assertRegex(out, r"Passed:\s+1")
        self.assertRegex(out, r"Failed:\s+1")
        self.assertRegex(out, r"Skipped:\s+1")
        self.assertRegex(out, r"Disabled:\s+1")
        self.assertRegex(out, r"Total:\s+4")
        self.assertRegex(out, r"Unique:\s+4")
        issue = out.split("Failed/skipped/timeout cases:", 1)[1].split("Duplicate cases:", 1)[0]
        self.assertIn("\n  +- CE Tests\n", issue)
        self.assertIn("\n  `- Socket Tests\n", issue)
        self.assertNotIn("NET IB", issue)

    def _assert_count_invariants(self, summary):
        TestUniqueAndDuplicateCases()._assert_count_invariants(summary)


class TestSuiteDisposition(unittest.TestCase):
    def _suite(self, name, enabled=True, smoke=False):
        return {
            "suite_details": {
                "name": name,
                "enabled": enabled,
                "smoke": smoke,
            }
        }

    def test_disabled_outside_smoke_is_skip_not_disabled(self):
        suite = self._suite("NET IB", enabled=False, smoke=False)
        self.assertEqual(suite_disposition(suite, smoke_only=True), "skip_scope")

    def test_disabled_smoke_suite_is_disabled(self):
        suite = self._suite("Smoke Off", enabled=False, smoke=True)
        self.assertEqual(suite_disposition(suite, smoke_only=True), "disabled")

    def test_disabled_off_suite_name_filter_is_skip(self):
        suite = self._suite("NET IB", enabled=False)
        self.assertEqual(
            suite_disposition(suite, smoke_only=False, suite_name_filter="Unit*"),
            "skip_name",
        )

    def test_disabled_matching_suite_name_is_disabled(self):
        suite = self._suite("NET IB Tests", enabled=False)
        self.assertEqual(
            suite_disposition(suite, smoke_only=False, suite_name_filter="NET*"),
            "disabled",
        )

    def test_enabled_all_scope_is_run(self):
        suite = self._suite("Unit Tests", enabled=True)
        self.assertEqual(suite_disposition(suite, smoke_only=False), "run")


if __name__ == "__main__":
    unittest.main()
