"""Tests for which PyTorch c10d tests run and how a run is judged.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.fspath(SCRIPTS))
import test_pytorch_c10d as tp  # noqa: E402
from rccl_ci_utils import parse_junit_xml  # noqa: E402

OPS = "test.distributed.test_c10d_ops_nccl.ProcessGroupNCCLOpTest"


def case(name, outcome="passed", cls=OPS, file="", message=""):
    return {"name": name, "classname": cls, "file": file,
            "outcome": outcome, "message": message}


def smoke_cases():
    return [case(n) for n in tp.SMOKE_TESTS]


def all_cases():
    """As many passing tests in each file as its floor."""
    out = []
    for f, n in tp.ALL_MIN_RAN.items():
        module = f.removesuffix(".py")
        out += [case(f"test_{i}", cls=f"test.distributed.{module}.T") for i in range(n)]
    return out


class Targets(unittest.TestCase):
    def test_every_file_in_the_all_scope_has_a_floor(self):
        self.assertEqual({tp.target_file(t) for t in tp.ALL_TARGETS}, set(tp.ALL_MIN_RAN))

    def test_smoke_targets_are_part_of_all(self):
        self.assertLessEqual(set(tp.SMOKE_TARGETS), set(tp.ALL_TARGETS))

    def test_smoke_tests_are_unique_and_not_known_failures(self):
        self.assertEqual(len(tp.SMOKE_TESTS), len(set(tp.SMOKE_TESTS)))
        with open(SCRIPTS / "known_failures.json") as f:
            known = {k.rsplit(".", 1)[-1] for k in json.load(f)["tests"]}
        self.assertFalse(set(tp.SMOKE_TESTS) & known)

    def test_target_file_drops_the_class(self):
        self.assertEqual(tp.target_file("test_c10d_functional_native.py::TestWithNCCL"),
                         "test_c10d_functional_native.py")


class SmokeVerdict(unittest.TestCase):
    def test_every_smoke_test_ran(self):
        v = tp.judge("smoke", smoke_cases(), set(), 0)
        self.assertEqual(v["exit_code"], 0, v["errors"])
        self.assertEqual(v["smoke_ran"], len(tp.SMOKE_TESTS))

    def test_a_missing_name_fails_even_when_extra_tests_matched(self):
        # -k substrings select extra tests; they must not stand in for a
        # smoke test that is gone.
        cases = smoke_cases()[1:] + [case("test_send_recv_complex"),
                                     case("test_send_recv_object_list")]
        v = tp.judge("smoke", cases, set(), 0)
        self.assertEqual(v["exit_code"], 1)
        self.assertIn(tp.SMOKE_TESTS[0], v["errors"][0])

    def test_skipped_after_an_earlier_failure_did_not_run(self):
        cases = smoke_cases()
        cases[-1] = case(tp.SMOKE_TESTS[-1], "skipped",
                         message="Previous test failed, skipping")
        v = tp.judge("smoke", cases, set(), 0)
        self.assertEqual(v["exit_code"], 1)
        self.assertEqual(v["smoke_ran"], len(tp.SMOKE_TESTS) - 1)

    def test_a_known_failure_does_not_fail_the_run(self):
        cases = smoke_cases() + [case("test_flaky", "failed")]
        v = tp.judge("smoke", cases, {"test_flaky"}, 1)
        self.assertEqual(v["exit_code"], 0, v["errors"])
        self.assertEqual(v["known_failed"], ["ProcessGroupNCCLOpTest.test_flaky"])

    def test_an_unexpected_failure_fails_the_run(self):
        cases = smoke_cases()
        cases[0] = case(tp.SMOKE_TESTS[0], "failed")
        self.assertEqual(tp.judge("smoke", cases, set(), 1)["exit_code"], 1)

    def test_no_results_fail_the_run(self):
        self.assertEqual(tp.judge("smoke", [], set(), 1)["exit_code"], 1)


class AllVerdict(unittest.TestCase):
    def test_every_file_at_its_floor_passes(self):
        v = tp.judge("all", all_cases(), set(), 0)
        self.assertEqual(v["exit_code"], 0, v["errors"])

    def test_a_file_upstream_emptied_fails(self):
        cases = [c for c in all_cases() if ".test_c10d_spawn_nccl." not in c["classname"]]
        v = tp.judge("all", cases, set(), 0)
        self.assertEqual(v["exit_code"], 1)
        self.assertTrue(any(e.startswith("test_c10d_spawn_nccl.py:") for e in v["errors"]))

    def test_skipped_tests_do_not_count_toward_a_floor(self):
        cases = all_cases()
        for c in cases:
            if ".test_c10d_ops_nccl." in c["classname"]:
                c["outcome"] = "skipped"
        v = tp.judge("all", cases, set(), 0)
        self.assertTrue(any(e.startswith("test_c10d_ops_nccl.py:") for e in v["errors"]))

    def test_known_failures_count_as_run(self):
        cases = all_cases()
        known = set()
        for c in cases:
            if ".test_c10d_nccl." in c["classname"]:
                c["outcome"] = "failed"
                known.add(c["name"])
        v = tp.judge("all", cases, known, 1)
        self.assertEqual(v["exit_code"], 0, v["errors"])

    def test_a_test_recorded_twice_counts_once(self):
        # A call failure and its teardown error are two testcases; the second
        # must not stand in for a test that did not run.
        floor = tp.ALL_MIN_RAN["test_c10d_ops_nccl.py"]
        cases = [c for c in all_cases() if ".test_c10d_ops_nccl." not in c["classname"]]
        cases += [case(f"test_{i}") for i in range(floor - 2)]
        cases += [case("test_flaky", "failed"), case("test_flaky", "failed")]
        v = tp.judge("all", cases, {"test_flaky"}, 1)
        self.assertEqual(v["exit_code"], 1)
        self.assertTrue(any(e.startswith("test_c10d_ops_nccl.py:") for e in v["errors"]))
        self.assertEqual(v["ran_per_file"]["test_c10d_ops_nccl.py"], floor - 1)

    def test_pytest_errors_fail_the_run(self):
        self.assertEqual(tp.judge("all", all_cases(), set(), 3)["exit_code"], 1)
        # Exit 1 with nothing recorded as failed: something went wrong outside a test.
        self.assertEqual(tp.judge("all", all_cases(), set(), 1)["exit_code"], 1)


class FileOf(unittest.TestCase):
    def test_file_attribute_is_preferred(self):
        c = case("t", cls="whatever", file="test/distributed/test_c10d_object_collectives.py")
        self.assertEqual(tp.file_of(c, tp.ALL_MIN_RAN), "test_c10d_object_collectives.py")

    def test_classname_fallback_matches_the_whole_module_name(self):
        self.assertIsNone(tp.file_of(case("t", cls="test.distributed.test_c10d_nccl2.T"),
                                     tp.ALL_MIN_RAN))
        self.assertEqual(tp.file_of(case("t", cls="test.distributed.test_c10d_nccl.T"),
                                    tp.ALL_MIN_RAN), "test_c10d_nccl.py")


class KnownFailures(unittest.TestCase):
    def test_bare_and_class_qualified_keys(self):
        ops = case("test_send_recv_object_list")
        obj = case("test_send_recv_object_list",
                   cls="test.distributed.test_c10d_object_collectives.TestObjectCollectives")
        qualified = {"TestObjectCollectives.test_send_recv_object_list"}
        self.assertFalse(tp.is_known(ops, qualified))
        self.assertTrue(tp.is_known(obj, qualified))
        self.assertTrue(tp.is_known(ops, {"test_send_recv_object_list"}))

    def test_call_failure_and_teardown_error_count_once(self):
        cases = smoke_cases() + [case("test_x", "failed"), case("test_x", "failed")]
        v = tp.judge("smoke", cases, set(), 1)
        self.assertEqual(v["unexpected_failed"], ["ProcessGroupNCCLOpTest.test_x"])

    def test_known_failures_that_now_pass_are_reported(self):
        v = tp.judge("smoke", smoke_cases(), {tp.SMOKE_TESTS[0], "test_gone"}, 0)
        self.assertEqual(v["now_passing"], [tp.SMOKE_TESTS[0]])


class JunitCases(unittest.TestCase):
    # Shapes pytest writes with -o junit_family=xunit1: a pass, a call failure
    # and its teardown error, a skip, and a file that fails to import.
    XML = """<?xml version="1.0"?>
<testsuites><testsuite tests="5" failures="1" errors="2" skipped="1">
  <testcase classname="test.distributed.test_c10d_ops_nccl.ProcessGroupNCCLOpTest"
            name="test_allreduce_ops" file="test/distributed/test_c10d_ops_nccl.py" time="2.0"/>
  <testcase classname="test.distributed.test_c10d_ops_nccl.ProcessGroupNCCLOpTest"
            name="test_broadcast_ops" file="test/distributed/test_c10d_ops_nccl.py" time="1.0">
    <failure message="AssertionError: Tensor-likes are not equal!">...</failure></testcase>
  <testcase classname="test.distributed.test_c10d_ops_nccl.ProcessGroupNCCLOpTest"
            name="test_broadcast_ops" file="test/distributed/test_c10d_ops_nccl.py" time="1.0">
    <error message="failed on teardown with &quot;TimeoutError&quot;">...</error></testcase>
  <testcase classname="test.distributed.test_c10d_ops_nccl.ProcessGroupNCCLOpTest"
            name="test_barrier" file="test/distributed/test_c10d_ops_nccl.py" time="0">
    <skipped message="Previous test failed, skipping"/></testcase>
  <testcase classname="" name="test.distributed.test_c10d_nccl"
            file="test/distributed/test_c10d_nccl.py">
    <error message="collection failure">AssertionError: missing tests</error></testcase>
</testsuite></testsuites>"""

    def parse(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.xml"
            p.write_text(self.XML)
            return parse_junit_xml(p)

    def test_cases_carry_classname_file_outcome_and_message(self):
        r = self.parse()
        self.assertEqual([c["outcome"] for c in r["cases"]],
                         ["passed", "failed", "failed", "skipped", "failed"])
        self.assertEqual(r["cases"][0]["file"], "test/distributed/test_c10d_ops_nccl.py")
        self.assertEqual(r["cases"][1]["message"], "AssertionError: Tensor-likes are not equal!")
        self.assertEqual(r["cases"][3]["message"], "Previous test failed, skipping")

    def test_a_call_failure_and_its_teardown_error_are_one_test(self):
        v = tp.judge("all", self.parse()["cases"], set(), 1)
        self.assertEqual(v["ran_per_file"]["test_c10d_ops_nccl.py"], 2)
        failed = v["unexpected_failed"]
        self.assertEqual(failed.count("ProcessGroupNCCLOpTest.test_broadcast_ops"), 1)

    def test_a_file_that_fails_to_import_fails_the_run(self):
        v = tp.judge("all", self.parse()["cases"], set(), 1)
        self.assertEqual(v["exit_code"], 1)
        self.assertIn("test.distributed.test_c10d_nccl", v["unexpected_failed"])
        self.assertTrue(any(e.startswith("test_c10d_nccl.py:") for e in v["errors"]))


if __name__ == "__main__":
    unittest.main()
