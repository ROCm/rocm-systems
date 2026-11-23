"""Tests for the CoCo record test_madengine.py leaves, and publish_madengine_result.py.

CoCo refuses a record it cannot store as sent: a NaN, a key it does not know,
a point listed twice. These pin what the record carries, so a refusal is a
contract change rather than a surprise on a nightly run.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import argparse
import gzip
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import publish_madengine_result as publish
import test_madengine as tm

ENV = {
    "GITHUB_REPOSITORY": "ROCm/rocm-systems",
    "GITHUB_WORKFLOW_REF": "ROCm/rocm-systems/.github/workflows/therock-rccl-ci.yml@refs/heads/develop",
    "GITHUB_RUN_ID": "44031357229",
    "GITHUB_RUN_ATTEMPT": "2",
    "GITHUB_EVENT_NAME": "schedule",
    "GITHUB_SHA": "83e8199a5",
    "RUN_LABEL": "",
    "AMDGPU_FAMILIES": "gfx950-dcgpu",
    "ARTIFACT_RUN_ID": "44031357229",
    "BASELINE_ARTIFACT_RUN_ID": "",
    "BASELINE_RCCL_SOURCE": "release:7.0.1",
}


def args(**overrides):
    values = {"workload": "llama-3.1-70b-training", "nodes": 2, "cluster": "ruby",
              "report_only": False, "work_dir": None}
    values.update(overrides)
    return argparse.Namespace(**values)


def side(value, tflops=None, status="pass"):
    return {"metric_value": value, "tflops_avg": tflops, "status": status}


def super_row(model, metric, performance, precision=None, status="SUCCESS"):
    multi = {"precision": precision} if precision else {}
    return {"model": model, "metric": metric, "performance": performance,
            "status": status, "multi_results": multi}


class CocoComparisons(unittest.TestCase):
    def test_a_point_is_a_regression_below_the_workload_threshold(self):
        records = tm.coco_comparisons(
            [
                {"precision": "BF16", tm.BASELINE: side(100.0, 700.0),
                 tm.CANDIDATE: side(96.0, 690.0), "ratio": 0.96, "gate_values": (100.0, 96.0)},
                {"precision": "FP8", tm.BASELINE: side(100.0), tm.CANDIDATE: side(98.0),
                 "ratio": 0.98, "gate_values": (100.0, 98.0)},
            ],
            "tok_per_s_per_gpu_avg", "training",
        )
        self.assertEqual([r["regression"] for r in records], [True, False])
        self.assertEqual(records[0]["threshold"], tm.REGRESSION_THRESHOLD_TRAINING)
        self.assertEqual(records[0]["baseline_tflops"], 700.0)
        self.assertEqual(records[0]["metric"], "tok_per_s_per_gpu_avg")

    def test_a_point_missing_a_side_or_a_number_has_no_verdict(self):
        (record,) = tm.coco_comparisons(
            [{"precision": "isl1024_osl1024_con64", tm.BASELINE: None,
              tm.CANDIDATE: side(float("nan")), "ratio": None, "gate_values": None}],
            "throughput", "inference",
        )
        self.assertIsNone(record["regression"])
        self.assertIsNone(record["candidate_value"])
        self.assertIsNone(record["baseline_status"])
        self.assertEqual(record["threshold"], tm.REGRESSION_THRESHOLD_INFERENCE)

    def test_the_values_are_the_ones_the_gate_scored(self):
        stalled = {**side(94.7), "median": 100.2}
        (record,) = tm.coco_comparisons(
            [{"precision": "FP8", tm.BASELINE: {**side(100.0), "median": 100.0},
              tm.CANDIDATE: stalled, "ratio": 1.002, "metric": "median",
              "gate_values": (100.0, 100.2)}],
            "tok_per_s_per_gpu_avg", "training",
        )
        self.assertEqual((record["baseline_value"], record["candidate_value"]), (100.0, 100.2))
        self.assertAlmostEqual(record["ratio"], record["candidate_value"] / record["baseline_value"])
        self.assertFalse(record["regression"])


class CocoPhase(unittest.TestCase):
    def phase(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "perf_entry_super.json").write_text(json.dumps(rows))
            return tm.coco_phase(Path(tmp), 0, True)

    def test_rows_of_one_model_are_told_apart_by_their_precision(self):
        model = "primus_pyt_megatron_lm_train_llama-3.1-70b"
        phase = self.phase([
            super_row(model, "tok_per_s_per_gpu_avg", "1690.0", "BF16"),
            super_row(model, "tok_per_s_per_gpu_avg", "2704.0", "FP8"),
            super_row(model, "tok_per_s_per_gpu_avg", "2705.0", "FP8"),
        ])
        self.assertEqual([(r["point_key"], r["performance"]) for r in phase["rows"]],
                         [("BF16", 1690.0), ("FP8", 2705.0)])
        self.assertEqual((phase["exit_code"], phase["rccl_verified"]), (0, True))

    def test_a_sweep_point_comes_from_the_model_name_and_nan_becomes_null(self):
        row = super_row("pyt_sglang_deepseek-r1_1P1D_isl1024_osl1024_con64", "throughput", "nan")
        row["multi_results"] = {"ttft": float("nan"), "nested": [float("inf"), 1.0]}
        (record,) = self.phase([row])["rows"]
        self.assertEqual(record["point_key"], "isl1024_osl1024_con64")
        self.assertIsNone(record["performance"])
        self.assertEqual(record["multi_results"], {"ttft": None, "nested": [None, 1.0]})
        json.dumps(record, allow_nan=False)

    def test_a_phase_with_no_results_file_has_no_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(tm.coco_phase(Path(tmp), 1, None)["rows"], [])


class CocoRecord(unittest.TestCase):
    @mock.patch.dict(os.environ, ENV, clear=False)
    def test_the_run_context_comes_from_the_job_environment(self):
        record = tm.coco_record(
            args(report_only=True), verdict="fail",
            started_at=datetime(2026, 11, 1, 2, tzinfo=timezone.utc),
            builds={phase: tm.coco_build(phase) for phase in (tm.BASELINE, tm.CANDIDATE)},
            report="x" * (tm.COCO_REPORT_CHARS + 10), slurm_job_id="12345",
        )
        self.assertEqual(record["schema_version"], tm.COCO_SCHEMA_VERSION)
        self.assertEqual(record["workflow"], ".github/workflows/therock-rccl-ci.yml")
        self.assertEqual((record["run_id"], record["run_attempt"]), (44031357229, 2))
        self.assertEqual((record["workload_type"], record["gpus_per_node"]), ("training", 8))
        self.assertEqual(record["amdgpu_family"], "gfx950-dcgpu")
        self.assertTrue(record["report_only"])
        self.assertEqual(len(record["report"]), tm.COCO_REPORT_CHARS)
        self.assertEqual(record["builds"][tm.BASELINE]["source"], "release:7.0.1")
        self.assertIsNone(record["builds"][tm.BASELINE]["artifact_run_id"])
        self.assertEqual(record["phases"], {tm.BASELINE: None, tm.CANDIDATE: None})
        json.dumps(record, allow_nan=False)

    def test_a_run_stopped_before_its_phases_leaves_an_error_record_in_work_dir(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {**ENV, "WORK_DIR": f"{tmp}/leg"}), \
                mock.patch.object(tm, "write_github_summary"), \
                mock.patch.object(tm, "set_github_output"), \
                mock.patch.object(tm, "send_report"):
            with self.assertRaises(SystemExit):
                tm.fail_before_run(args(), "2N/16GPU", "No A/B baseline: none found")
            record = json.loads(Path(tmp, "leg", tm.COCO_RECORD_NAME).read_text())
        self.assertEqual(record["verdict"], "error")
        self.assertEqual(record["comparisons"], [])
        self.assertEqual(record["builds"][tm.CANDIDATE]["artifact_run_id"], "44031357229")
        self.assertIn("No A/B baseline", record["report"])

    def test_no_work_dir_means_no_record(self):
        with mock.patch.dict(os.environ, {"WORK_DIR": ""}):
            self.assertIsNone(tm.coco_record_path(args()))


class Publish(unittest.TestCase):
    RECORD = {"schema_version": 1, "verdict": "pass"}

    def test_the_request_is_gzipped_strict_json_under_the_record_version(self):
        request = publish.build_request("https://coco.example/", "coco_tok", self.RECORD)
        self.assertEqual(request.full_url, "https://coco.example/api/data/write/madengine-ab")
        self.assertEqual(request.get_header("Authorization"), "Bearer coco_tok")
        self.assertEqual(request.get_header("Content-encoding"), "gzip")
        self.assertEqual(request.get_header("X-coco-payload-version"), "1")
        self.assertEqual(json.loads(gzip.decompress(request.data)), self.RECORD)

    def test_only_a_server_side_failure_is_retried(self):
        self.assertTrue(publish.retryable(503))
        self.assertTrue(publish.retryable(429))
        self.assertFalse(publish.retryable(422))
        self.assertFalse(publish.retryable(401))

    def test_nothing_is_sent_until_both_url_and_token_are_set(self):
        with mock.patch.dict(os.environ, {"RCCL_COCO_URL": "https://coco.example",
                                          "RCCL_COCO_WEB_TOKEN": ""}), \
                mock.patch.object(publish, "publish") as send:
            self.assertEqual(publish.main(["publish", "/nonexistent.json"]), 0)
        send.assert_not_called()

    def test_a_leg_that_left_no_record_is_not_a_failure(self):
        with mock.patch.dict(os.environ, {"RCCL_COCO_URL": "https://coco.example",
                                          "RCCL_COCO_WEB_TOKEN": "coco_tok"}), \
                mock.patch.object(publish, "publish") as send:
            self.assertEqual(publish.main(["publish", "/nonexistent.json"]), 0)
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
