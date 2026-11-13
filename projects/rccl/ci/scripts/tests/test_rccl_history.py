"""Tests for the RCCL history sweep: its plan, its summary and its workflows.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import rccl_history as rh
import test_madengine as tm

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ENV = {"GITHUB_REPOSITORY": "ROCm/rocm-systems", "GITHUB_TOKEN": "t"}
WORKFLOWS = Path(__file__).resolve().parents[5] / ".github" / "workflows"


def resolved(spec, family):
    """fetch_rccl_build.resolve() of a build that exists: it still checks the spec."""
    kind, value = rh.fetch_rccl_build.parse_source(spec)
    version = value.rsplit("/", 1)[-1].split("-")[1] if kind == "url" else value
    return {"source": spec, "version": version, "archive": "wheel"}


class Plan(unittest.TestCase):
    def plan(self, candidate, baselines, in_bucket=lambda build: True):
        with mock.patch.object(tm, "rccl_artifact_exists",
                               side_effect=lambda b, g, r, **k: in_bucket(b)), \
             mock.patch.object(rh.fetch_rccl_build, "resolve", side_effect=resolved), \
             mock.patch.object(tm, "previous_scheduled_build", return_value="900") as latest:
            result = rh.plan(candidate, baselines, "gfx950-dcgpu", ENV)
        return result, latest

    def test_one_leg_per_baseline_in_order(self):
        (candidate, legs), _ = self.plan(
            "800", "release:10.1.0, nightly:10.1.0a20260822\n42346792638 run:7")
        self.assertEqual(candidate, "800")
        self.assertEqual([leg["label"] for leg in legs],
                         ["release-10.1.0", "nightly-10.1.0a20260822",
                          "run-42346792638", "run-7"])
        self.assertEqual(legs[0], {"baseline": "release:10.1.0", "label": "release-10.1.0"})
        self.assertEqual(legs[3], {"baseline": "7", "label": "run-7"})

    def test_a_url_label_is_unique_and_safe(self):
        url = "url:https://repo.amd.com/a/rocm_sdk_libraries-7.14.1-py3-none-linux_x86_64.whl"
        (_, legs), _ = self.plan("800", url)
        self.assertRegex(legs[0]["label"], r"^url-[A-Za-z0-9._-]+-[0-9a-f]{8}$")

    def test_no_candidate_takes_the_newest_scheduled_build(self):
        (candidate, _), latest = self.plan("", "release:10.1.0")
        self.assertEqual(candidate, "900")
        latest.assert_called_once_with("", "nightly", "gfx950", ENV)

    def test_every_bad_input_is_reported_at_once(self):
        with self.assertRaises(RuntimeError) as cm:
            self.plan("12x", "5, release:10.1.0 release:10.1.0, tag:1",
                      in_bucket=lambda build: build != "5")
        message = str(cm.exception)
        for part in ("must be a run ID, got '12x'", "5: its RCCL is not in the bucket",
                     "release:10.1.0: given twice", "unknown source 'tag'"):
            self.assertIn(part, message)

    def test_an_expired_candidate_fails(self):
        with self.assertRaisesRegex(RuntimeError, "candidate 800: its RCCL is not in the bucket"):
            self.plan("800", "release:10.1.0", in_bucket=lambda build: build != "800")

    def test_too_many_legs_fail(self):
        with self.assertRaisesRegex(RuntimeError, "at most 10 legs"):
            self.plan("800", ",".join(str(i) for i in range(1, 12)))

    def test_no_baselines_fail(self):
        with self.assertRaisesRegex(RuntimeError, "no baselines given"):
            self.plan("800", " , ")

    def test_stdout_carries_only_the_outputs(self):
        legs = [{"baseline": "release:10.1.0", "label": "release-10.1.0"}]
        out = io.StringIO()
        with mock.patch.object(rh, "plan", return_value=("800", legs)), \
             mock.patch.object(sys, "argv", ["rccl_history.py", "plan",
                                             "--amdgpu-family", "gfx950-dcgpu"]), \
             contextlib.redirect_stdout(out):
            rh.main()
        lines = out.getvalue().splitlines()
        self.assertEqual(lines[0], "candidate=800")
        self.assertEqual(json.loads(lines[1].removeprefix("matrix=")), {"include": legs})
        self.assertEqual(len(lines), 2)


def report(baseline_build="release:10.1.0") -> str:
    row = lambda v: {"metric_value": v, "tflops_avg": 700.0, "status": "pass"}
    comparisons = [{"precision": "BF16", tm.BASELINE: row(1743.0), tm.CANDIDATE: row(1743.7),
                    "ratio": 1743.7 / 1743.0}]
    return tm.generate_summary_report(
        "llama-3.1-70b-training", "2N/16GPU", 0, "[BF16] No regression", "c", "ruby",
        comparisons=comparisons,
        phase_rccl={tm.BASELINE: "2.30.7-HEAD:7da4302", tm.CANDIDATE: "2.32.3-develop:c7a66f1"},
        builds={tm.BASELINE: baseline_build, tm.CANDIDATE: "42768456967"}, report_only=True,
    )


class Summary(unittest.TestCase):
    def test_parses_the_report_test_madengine_writes(self):
        result = rh.parse_summary(report())
        self.assertEqual(result["status"], "PASSED")
        self.assertEqual(result["rccl"], {"baseline": "2.30.7-HEAD:7da4302",
                                          "candidate": "2.32.3-develop:c7a66f1"})
        self.assertEqual(result["build"], {"baseline": "release:10.1.0",
                                           "candidate": "CI build 42768456967"})
        self.assertEqual(result["precisions"], [{"precision": "BF16", "ratio": "1.0004",
                                                 "baseline": "1743.0", "candidate": "1743.7"}])

    def test_a_table_row_per_leg_and_precision_a_missing_leg_included(self):
        legs = [{"baseline": "release:10.1.0", "label": "release-10.1.0"},
                {"baseline": "release:7.13.0", "label": "release-7.13.0"}]
        prefix = "madengine-results-llama-3.1-70b-training-2N-gfx950-dcgpu"
        with tempfile.TemporaryDirectory() as tmp:
            leg = Path(tmp) / f"{prefix}-release-10.1.0"
            leg.mkdir()
            (leg / "madengine_summary.txt").write_text(report())
            table = rh.summary(Path(tmp), legs, prefix)
        self.assertIn("Candidate RCCL: 2.32.3-develop:c7a66f1", table)
        self.assertIn("| `release:10.1.0` | 2.30.7-HEAD:7da4302 | BF16 | 1743.0 | 1743.7 "
                      "| 1.0004 | PASSED |", table)
        self.assertIn("| `release:7.13.0` | | | | | | no results |", table)


@unittest.skipIf(yaml is None, "PyYAML not installed")
class Workflows(unittest.TestCase):
    def setUp(self):
        self.history = yaml.safe_load((WORKFLOWS / "therock-rccl-madengine-history.yml").read_text())
        self.reusable = yaml.safe_load((WORKFLOWS / "therock-rccl-test-madengine.yml").read_text())

    def test_the_sweep_is_dispatch_only_and_runs_its_legs_one_by_one(self):
        self.assertEqual(list(self.history[True]), ["workflow_dispatch"])
        ab = self.history["jobs"]["ab"]
        self.assertEqual(ab["uses"], "./.github/workflows/therock-rccl-test-madengine.yml")
        self.assertEqual(ab["strategy"]["max-parallel"], 1)
        self.assertFalse(ab["strategy"]["fail-fast"])
        self.assertIs(ab["with"]["report_only"], True)
        self.assertEqual(ab["with"]["run_label"], "${{ matrix.label }}")
        self.assertEqual(ab["with"]["baseline_artifact_run_id"], "${{ matrix.baseline }}")

    def test_no_dispatch_input_reaches_a_run_block(self):
        for job in self.history["jobs"].values():
            for step in job.get("steps", []):
                self.assertNotIn("${{", step.get("run", ""), step.get("name"))

    def test_the_new_inputs_are_call_only(self):
        call = self.reusable[True]["workflow_call"]["inputs"]
        dispatch = self.reusable[True]["workflow_dispatch"]["inputs"]
        for name in ("run_label", "report_only"):
            self.assertIn(name, call)
            self.assertNotIn(name, dispatch)
        self.assertLessEqual(len(dispatch), 10)

    def test_the_resolved_baseline_picks_how_it_is_fetched(self):
        steps = {s["name"]: s for s in self.reusable["jobs"]["test_madengine"]["steps"]}
        self.assertNotIn("if", steps["Resolve the A/B baseline build"])
        self.assertEqual(steps["Fetch baseline RCCL artifacts"]["if"],
                         "${{ steps.baseline.outputs.artifact_run_id != '' }}")
        fetch = steps["Fetch published baseline RCCL"]
        self.assertEqual(fetch["if"], "${{ steps.baseline.outputs.rccl_source != '' }}")
        self.assertEqual(fetch["env"]["BASELINE_RCCL_SOURCE"],
                         "${{ steps.baseline.outputs.rccl_source }}")
        self.assertIn('--output-dir "${BASELINE_ARTIFACTS_DIR}"', fetch["run"])
        run = steps["Run MADEngine workload"]
        self.assertEqual(run["env"]["BASELINE_RCCL_SOURCE"],
                         "${{ steps.baseline.outputs.rccl_source }}")
        self.assertIn('"${REPORT_ARGS[@]}"', run["run"])

    def test_the_run_label_keeps_legs_apart(self):
        job = self.reusable["jobs"]["test_madengine"]
        work_dir = job["env"]["WORK_DIR"]
        self.assertIn("inputs.run_label", work_dir)
        for name in ("XDG_CACHE_HOME", "CONDA_PKGS_DIRS"):
            self.assertTrue(job["env"][name].startswith(work_dir + "/"), name)
        steps = {s["name"]: s for s in job["steps"]}
        for name in ("Upload results", "Upload repro bundle"):
            self.assertIn("inputs.run_label", steps[name]["with"]["name"])

    def test_steps_touching_work_dir_wait_for_the_label_check(self):
        steps = {s["name"]: s for s in self.reusable["jobs"]["test_madengine"]["steps"]}
        self.assertIn("RUN_LABEL", steps["Validate arch"]["run"])
        for name in ("Upload results", "Upload repro bundle", "Cleanup work directory"):
            self.assertEqual(steps[name]["if"],
                             "always() && steps.validate.outcome == 'success'", name)


if __name__ == "__main__":
    unittest.main()
