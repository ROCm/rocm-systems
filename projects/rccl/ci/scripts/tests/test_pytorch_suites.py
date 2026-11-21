"""Tests for the PyTorch distributed suite registry and verdict in test_pytorch_c10d.py.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.fspath(SCRIPTS))
import test_pytorch_c10d as tp  # noqa: E402
from rccl_ci_utils import parse_junit_xml  # noqa: E402

WORKFLOW = (SCRIPTS.parents[3] / ".github" / "workflows"
            / "therock-rccl-test-pytorch-distributed.yml")

FSDP = tp.SUITES["fsdp_dtensor"]
C10D = tp.SUITES["c10d_nccl"]


def case(cls, name, outcome="passed", file=""):
    return {"classname": cls, "name": name, "outcome": outcome, "file": file}


def all_files_pass(suite):
    """One passing testcase per file, classname as pytest writes it."""
    return [
        case("test.distributed." + f.removesuffix(".py").replace("/", ".") + ".T",
             f"test_{i}")
        for i, f in enumerate(suite.files)
    ]


class Registry(unittest.TestCase):
    def test_known_failure_files_exist_and_parse(self):
        for s in tp.SUITES.values():
            with open(SCRIPTS / s.known_failures_file) as f:
                self.assertIsInstance(json.load(f)["tests"], dict, s.name)

    def test_known_failure_entries_are_categorised(self):
        for s in tp.SUITES.values():
            with open(SCRIPTS / s.known_failures_file) as f:
                d = json.load(f)
            for test, cat in d["tests"].items():
                self.assertIn(cat, d.get("categories", {}), f"{s.name}: {test}")
                if s.qualified_ids:
                    self.assertRegex(test, r"^\w+\.test_\w+$", s.name)

    def test_fsdp_dtensor_timeout_override_fires_before_pytest_timeout(self):
        env = dict(FSDP.env)
        self.assertLess(int(env["DISTRIBUTED_TESTS_DEFAULT_TIMEOUT"]), FSDP.timeout_s)

    def test_artifact_stems_are_unique(self):
        stems = [s.artifact_stem for s in tp.SUITES.values()]
        self.assertEqual(len(stems), len(set(stems)))

    def test_c10d_nccl_keeps_its_names(self):
        # known_failures.json is keyed by bare names, and artifacts keep their
        # pre-suite names.
        self.assertFalse(C10D.qualified_ids)
        self.assertEqual(C10D.artifact_stem, "pytorch_c10d")
        self.assertEqual(C10D.known_failures_file, "known_failures.json")

    def test_workflow_all_scope_runs_every_suite(self):
        text = WORKFLOW.read_text()
        m = re.search(r"inputs\.test_scope == 'all' && '(\[[^']*\])'", text)
        self.assertIsNotNone(m, "matrix expression not found in workflow")
        self.assertEqual(sorted(json.loads(m.group(1))), sorted(tp.SUITES))

    def test_workflow_does_not_share_host_ipc(self):
        text = "\n".join(line for line in WORKFLOW.read_text().splitlines()
                         if not line.lstrip().startswith("#"))
        self.assertNotIn("--ipc", text)
        self.assertIn("--shm-size=10g", text)
        self.assertIn("--expect-shm-gib 10", text)


class FileOf(unittest.TestCase):
    def test_longest_module_wins(self):
        # "test_composability" is a file and a directory name.
        pp = case("test.distributed._composable.test_composability"
                  ".test_pp_composability.ComposabilityTest", "t")
        top = case("test.distributed.test_composability.ComposabilityTest", "t")
        self.assertEqual(tp.file_of(pp, FSDP),
                         "_composable/test_composability/test_pp_composability.py")
        self.assertEqual(tp.file_of(top, FSDP), "test_composability.py")

    def test_rootdir_prefix_does_not_matter(self):
        c = case("tensor.test_utils.UtilTest", "t")
        self.assertEqual(tp.file_of(c, FSDP), "tensor/test_utils.py")

    def test_file_attribute_preferred(self):
        c = case("whatever", "t", file="test/distributed/test_composability.py")
        self.assertEqual(tp.file_of(c, FSDP), "test_composability.py")
        c = case("whatever", "t",
                 file="distributed/_composable/test_composability/test_pp_composability.py")
        self.assertEqual(tp.file_of(c, FSDP),
                         "_composable/test_composability/test_pp_composability.py")

    def test_foreign_file_is_none(self):
        self.assertIsNone(tp.file_of(case("test.distributed.test_other.T", "t"), FSDP))


class Judge(unittest.TestCase):
    def test_full_suite_passes(self):
        cases = all_files_pass(FSDP)
        v = tp.judge(FSDP, "all", cases, FSDP.min_tests, set(), 0)
        self.assertEqual(v["exit_code"], 0, v["errors"])

    def test_a_file_with_only_skips_fails(self):
        cases = all_files_pass(FSDP)
        cases[3]["outcome"] = "skipped"
        v = tp.judge(FSDP, "all", cases, len(cases), set(), 0)
        self.assertEqual(v["exit_code"], 1)
        self.assertTrue(any(FSDP.files[3] in e for e in v["errors"]))

    def test_no_junit_is_a_failure(self):
        v = tp.judge(FSDP, "all", [], 0, set(), 0)
        self.assertEqual(v["exit_code"], 1)

    def test_known_failure_is_qualified_by_class(self):
        cases = all_files_pass(FSDP)
        cases.append(case("test.distributed.tensor.test_utils.UtilTest", "test_x", "failed"))
        cases.append(case("test.distributed.tensor.test_view_ops.ViewTest", "test_x", "failed"))
        v = tp.judge(FSDP, "all", cases, len(cases), {"UtilTest.test_x"}, 1)
        self.assertEqual(v["known_failed"], ["UtilTest.test_x"])
        self.assertEqual(v["unexpected_failed"], ["ViewTest.test_x"])
        self.assertEqual(v["exit_code"], 1)

    def test_only_known_failures_pass(self):
        cases = all_files_pass(FSDP)
        cases.append(case("test.distributed.tensor.test_utils.UtilTest", "test_x", "failed"))
        v = tp.judge(FSDP, "all", cases, FSDP.min_tests, {"UtilTest.test_x"}, 1)
        self.assertEqual(v["exit_code"], 0, v["errors"])

    def test_pytest_internal_error_fails_even_if_failures_known(self):
        cases = all_files_pass(FSDP)
        v = tp.judge(FSDP, "all", cases, len(cases), set(), 3)
        self.assertEqual(v["exit_code"], 1)

    def test_exit_1_without_recorded_failure_fails(self):
        cases = all_files_pass(FSDP)
        v = tp.judge(FSDP, "all", cases, len(cases), set(), 1)
        self.assertEqual(v["exit_code"], 1)

    def test_c10d_smoke_floor_and_bare_names(self):
        cases = [case("test.distributed.test_c10d_nccl.T", n) for n in C10D.smoke_tests]
        cases.append(case("test.distributed.test_c10d_nccl.T", "test_known", "failed"))
        v = tp.judge(C10D, "smoke", cases, len(cases), {"test_known"}, 1)
        self.assertEqual(v["exit_code"], 0, v["errors"])
        v = tp.judge(C10D, "smoke", cases[:5], 5, set(), 0)
        self.assertEqual(v["exit_code"], 1)

    def test_c10d_all_floor(self):
        cases = [case("test.distributed.test_c10d_nccl.T", f"t{i}") for i in range(10)]
        v = tp.judge(C10D, "all", cases, 10, set(), 0)
        self.assertEqual(v["exit_code"], 1)

    def test_suite_without_smoke_tier_runs_full_checks_under_smoke(self):
        cases = all_files_pass(FSDP)[:1]
        v = tp.judge(FSDP, "smoke", cases, 1, set(), 0)
        self.assertEqual(v["exit_code"], 1)

    def test_now_passing_reported(self):
        cases = all_files_pass(FSDP)
        cid = tp.case_id(cases[0], True)
        v = tp.judge(FSDP, "all", cases, len(cases), {cid}, 0)
        self.assertEqual(v["now_passing"], [cid])


class NotRunAndDuplicates(unittest.TestCase):
    CLS = "test.distributed.test_c10d_nccl2.ProcessGroupNCCL2Test"

    def test_call_failure_plus_teardown_error_counts_once(self):
        suite = tp.SUITES["c10d_nccl2"]
        cases = [case(self.CLS, "test_ok"),
                 case(self.CLS, "test_x", "failed"), case(self.CLS, "test_x", "failed")]
        v = tp.judge(suite, "all", cases, 2, set(), 1)
        self.assertEqual(v["unexpected_failed"], ["ProcessGroupNCCL2Test.test_x"])

    def test_previous_test_failed_skips_are_reported_as_not_run(self):
        suite = tp.SUITES["c10d_nccl2"]
        skip = dict(case(self.CLS, "test_after", "skipped"),
                    message="Previous test failed, skipping test_c10d_nccl2.ProcessGroupNCCL2Test.test_after")
        cap = dict(case(self.CLS, "test_cap", "skipped"), message="requires NVSHMEM")
        # Body skipped, teardown errored: a failure, not a not-run.
        both = dict(case(self.CLS, "test_both", "failed"), message="failed on teardown")
        both_skip = dict(case(self.CLS, "test_both", "skipped"), message="Previous test failed, skipping")
        cases = [case(self.CLS, "test_ok"), case(self.CLS, "test_x", "failed"),
                 skip, cap, both, both_skip]
        v = tp.judge(suite, "all", cases, suite.min_tests,
                     {"ProcessGroupNCCL2Test.test_x", "ProcessGroupNCCL2Test.test_both"}, 1)
        self.assertEqual(v["exit_code"], 0, v["errors"])
        self.assertEqual(v["not_run_after_failure"], ["ProcessGroupNCCL2Test.test_after"])


class JunitCases(unittest.TestCase):
    def test_cases_carry_classname_file_and_outcome(self):
        xml = """<?xml version="1.0"?>
<testsuites><testsuite tests="3" failures="1" errors="0">
  <testcase classname="test.distributed.tensor.test_utils.UtilTest" name="test_a"
            file="test/distributed/tensor/test_utils.py" time="1.0"/>
  <testcase classname="test.distributed.tensor.test_utils.UtilTest" name="test_b" time="1.0">
    <failure message="boom"/></testcase>
  <testcase classname="test.distributed.tensor.test_utils.UtilTest" name="test_c" time="0">
    <skipped/></testcase>
</testsuite></testsuites>"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.xml"
            p.write_text(xml)
            r = parse_junit_xml(p)
        self.assertEqual([c["outcome"] for c in r["cases"]], ["passed", "failed", "skipped"])
        self.assertEqual(r["cases"][0]["file"], "test/distributed/tensor/test_utils.py")
        self.assertEqual(r["failed"], ["test_b"])
        self.assertEqual(r["cases"][1]["message"], "boom")


if __name__ == "__main__":
    unittest.main()
