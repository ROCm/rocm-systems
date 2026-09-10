# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks import dashboard_publish as publisher


class DashboardPublishTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "public" / "data"

    @staticmethod
    def _test(
        logical_id: str = "triton.copy_fp32_32m",
        *,
        target: str = "gfx950",
        problem: dict[str, object] | None = None,
        status: str = "completed",
    ) -> dict[str, object]:
        timed_out = status == "timeout"
        return {
            "testId": f"{target}:{logical_id}",
            "logicalTestId": logical_id,
            "suite": "Triton",
            "name": "32 MiB contiguous FP32 copy",
            "target": target,
            "operation": "Copy",
            "dataType": "fp32",
            "problem": {"elements": 8_388_608} if problem is None else problem,
            "execMode": "functional",
            "numThreads": 1,
            "durationSeconds": 0.00125 if status == "completed" else None,
            "timing": {
                "unit": "ns",
                "samples": [1_200_000, 1_250_000, 1_300_000],
                "minimum": 1_200_000,
                "median": 1_250_000,
                "maximum": 1_300_000,
            },
            "status": status,
            "exitCode": 0 if status == "completed" else None,
            "timedOut": timed_out,
            "error": None if status == "completed" else f"{status} fixture",
            "artifacts": {},
        }

    def _raw(self, tests: list[dict[str, object]] | None = None) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "timestamp": "2026-09-01T20:00:00Z",
            "finishedAt": "2026-09-01T20:01:00+00:00",
            "status": "completed",
            "wallTimeSeconds": 60.0,
            "benchmarkSuite": "nightly",
            "targets": ["gfx950"],
            "measurement": {"warmups": 3, "samples": 3, "timeoutSeconds": 300},
            "configuration": {
                "id": "plugins-none-v1",
                "pluginProfile": "none",
                "plugins": [],
                "targetConfigSha256": {"gfx950": "b" * 64},
            },
            "provenance": {
                "rocjitsuCommitSha": "a" * 40,
                "rocjitsuCommitTimestamp": "2026-09-01T19:42:10-07:00",
                "dirty": False,
                "buildType": "Release",
                "rocmSdkPath": "/opt/rocm",
                "rocmSdkVersion": "7.2.0",
                "pythonVersion": "3.12.0",
                "torchVersion": "2.10.0",
                "tritonVersion": "3.6.0",
                "tritonCommitSha": None,
                "tensileLiteCommitSha": "c" * 12,
                "packages": {},
            },
            "environment": {"hostname": "benchmark-host"},
            "tests": tests or [self._test()],
        }

    def _publish(self, raw: dict[str, object] | None = None, **overrides):
        options = {
            "data_dir": self.data,
            "run_id": "github-123-attempt-1",
            "repository": "https://github.com/ROCm/rocm-systems",
            "environment_id": "rocm-7.2-sjc-01",
            "trigger": "auto",
            "branch": "develop",
            "machine_id": "sjc-rocjitsu-perf-01",
            "commit_order": 42,
            "commit_message": "Improve dispatch",
            "generated_at": "2026-09-01T20:02:00Z",
        }
        options.update(overrides)
        return publisher.publish(raw or self._raw(), **options)

    def test_normalizes_raw_run_and_omits_runner_only_fields(self) -> None:
        result = publisher.normalize_run(
            self._raw(),
            run_id="github-123-attempt-1",
            trigger="manual",
            branch="develop",
            environment_id="staging-rocm-7.2",
            machine_id="host-1",
            commit_order=7,
            commit_message="Candidate change",
            source_ref="users/example/candidate",
        )
        self.assertEqual(result["format"], publisher.RUN_FORMAT)
        self.assertEqual(result["schemaVersion"], 1)
        self.assertEqual(result["timestamp"], "2026-09-01T20:01:00Z")
        self.assertEqual(result["commitTimestamp"], "2026-09-02T02:42:10Z")
        self.assertEqual(result["configurationId"], "plugins-none-v1")
        self.assertEqual(result["commitOrder"], 7)
        self.assertEqual(result["machineId"], "host-1")
        self.assertNotIn("measurement", result)
        self.assertNotIn("timing", result["tests"][0])
        self.assertEqual(
            result["provenance"]["details"][0],
            {
                "key": "sourceRef",
                "label": "Source ref",
                "value": "users/example/candidate",
            },
        )
        self.assertEqual(result["provenance"]["commitMessage"], "Candidate change")

    def test_first_publish_writes_run_hashed_catalog_and_index(self) -> None:
        result = self._publish()
        self.assertTrue(result["changed"])
        index = publisher.load_json_document(self.data / "index.json")
        self.assertEqual(
            index,
            {
                "format": publisher.INDEX_FORMAT,
                "schemaVersion": 1,
                "generatedAt": "2026-09-01T20:02:00Z",
                "catalog": index["catalog"],
                "runs": [
                    {
                        "runId": "github-123-attempt-1",
                        "path": "runs/github-123-attempt-1.json",
                    }
                ],
            },
        )
        catalog_path = self.data / index["catalog"]
        self.assertEqual(
            index["catalog"],
            f"catalog-{hashlib.sha256(catalog_path.read_bytes()).hexdigest()[:12]}.json",
        )
        catalog = publisher.load_json_document(catalog_path)
        self.assertFalse(catalog["isDemo"])
        self.assertEqual(catalog["targets"], ["gfx950"])
        self.assertEqual(catalog["testCatalog"][0]["id"], "triton.copy_fp32_32m")
        run = publisher.load_json_document(
            self.data / "runs" / "github-123-attempt-1.json"
        )
        self.assertEqual(run["runId"], "github-123-attempt-1")
        self.assertEqual(run["tests"][0]["problem"], {"elements": 8_388_608})

    def test_repeat_is_idempotent_and_keeps_index_timestamp(self) -> None:
        self._publish()
        before = (self.data / "index.json").read_bytes()
        result = self._publish(generated_at="2026-09-03T00:00:00Z")
        self.assertFalse(result["changed"])
        self.assertEqual((self.data / "index.json").read_bytes(), before)
        self.assertEqual(len(publisher.load_json_document(self.data / "index.json")["runs"]), 1)

    def test_new_run_merges_catalog_and_preserves_history(self) -> None:
        self._publish()
        second = self._test(
            "triton.softmax_fp16_boundary", problem={"rows": 64, "columns": 4097}
        )
        second.update(
            {
                "testId": "gfx1250:triton.softmax_fp16_boundary",
                "target": "gfx1250",
                "name": "Boundary FP16 softmax",
                "operation": "Softmax",
                "dataType": "fp16",
            }
        )
        self._publish(
            self._raw([second]),
            run_id="github-124-attempt-1",
            generated_at="2026-09-02T20:02:00Z",
        )
        index = publisher.load_json_document(self.data / "index.json")
        self.assertEqual([item["runId"] for item in index["runs"]], [
            "github-123-attempt-1",
            "github-124-attempt-1",
        ])
        catalog = publisher.load_json_document(self.data / index["catalog"])
        self.assertEqual(catalog["targets"], ["gfx1250", "gfx950"])
        self.assertEqual(
            [item["id"] for item in catalog["testCatalog"]],
            ["triton.copy_fp32_32m", "triton.softmax_fp16_boundary"],
        )

    def test_conflicting_definition_fails_before_writing_resources(self) -> None:
        self._publish()
        before = (self.data / "index.json").read_bytes()
        conflicting = self._test(problem={"elements": 17})
        with self.assertRaisesRegex(publisher.PublishError, "definition conflicts"):
            self._publish(
                self._raw([conflicting]), run_id="github-124-attempt-1"
            )
        self.assertEqual((self.data / "index.json").read_bytes(), before)
        self.assertFalse(
            (self.data / "runs" / "github-124-attempt-1.json").exists()
        )

    def test_reusing_run_id_with_different_content_fails(self) -> None:
        self._publish()
        changed = self._raw()
        changed["tests"][0]["durationSeconds"] = 99.0
        with self.assertRaisesRegex(publisher.PublishError, "conflicting immutable"):
            self._publish(changed)

    def test_failed_result_reuses_catalog_problem(self) -> None:
        self._publish()
        failed = self._test(status="failed", problem=None)
        failed["problem"] = None
        self._publish(self._raw([failed]), run_id="github-124-attempt-1")
        run = publisher.load_json_document(
            self.data / "runs" / "github-124-attempt-1.json"
        )
        self.assertEqual(run["tests"][0]["problem"], {"elements": 8_388_608})
        self.assertIsNone(run["tests"][0]["durationSeconds"])

    def test_failed_result_without_catalog_definition_is_rejected(self) -> None:
        failed = self._test(status="failed")
        failed["problem"] = None
        with self.assertRaisesRegex(publisher.PublishError, "no existing catalog"):
            self._publish(self._raw([failed]))
        self.assertFalse(self.data.exists())

    def test_completed_result_cannot_borrow_a_catalog_problem(self) -> None:
        self._publish()
        completed = self._test()
        completed["problem"] = None
        with self.assertRaisesRegex(publisher.PublishError, "completed test.*no problem"):
            self._publish(self._raw([completed]), run_id="github-124-attempt-1")

    def test_requires_clean_matching_source_revision(self) -> None:
        dirty = self._raw()
        dirty["provenance"]["dirty"] = True
        with self.assertRaisesRegex(publisher.PublishError, "clean Rocjitsu checkout"):
            self._publish(dirty)
        with self.assertRaisesRegex(publisher.PublishError, "does not match expected SHA"):
            self._publish(self._raw(), expected_sha="d" * 40)
        self.assertFalse(self.data.exists())

    def test_missing_indexed_run_is_rejected_before_index_update(self) -> None:
        self._publish()
        index_path = self.data / "index.json"
        before = index_path.read_bytes()
        (self.data / "runs" / "github-123-attempt-1.json").unlink()
        with self.assertRaisesRegex(publisher.PublishError, "indexed dashboard run is missing"):
            self._publish(self._raw(), run_id="github-124-attempt-1")
        self.assertEqual(index_path.read_bytes(), before)
        self.assertFalse(
            (self.data / "runs" / "github-124-attempt-1.json").exists()
        )

    def test_mismatched_indexed_run_id_is_rejected_before_index_update(self) -> None:
        self._publish()
        run_path = self.data / "runs" / "github-123-attempt-1.json"
        run = publisher.load_json_document(run_path)
        run["runId"] = "github-different-attempt-1"
        run_path.write_text(json.dumps(run), encoding="utf-8")
        with self.assertRaisesRegex(publisher.PublishError, "mismatched runId"):
            self._publish(self._raw(), run_id="github-124-attempt-1")
        self.assertFalse(
            (self.data / "runs" / "github-124-attempt-1.json").exists()
        )

    def test_catalog_incompatible_indexed_run_is_rejected_before_update(self) -> None:
        self._publish()
        run_path = self.data / "runs" / "github-123-attempt-1.json"
        run = publisher.load_json_document(run_path)
        run["tests"][0]["problem"] = {"elements": 17}
        run_path.write_text(json.dumps(run), encoding="utf-8")
        with self.assertRaisesRegex(publisher.PublishError, "catalog field problem"):
            self._publish(self._raw(), run_id="github-124-attempt-1")
        self.assertFalse(
            (self.data / "runs" / "github-124-attempt-1.json").exists()
        )

    def test_repository_and_demo_mode_cannot_change_in_one_data_root(self) -> None:
        self._publish()
        with self.assertRaisesRegex(publisher.PublishError, "repository or isDemo"):
            self._publish(is_demo=True)
        with self.assertRaisesRegex(publisher.PublishError, "repository or isDemo"):
            self._publish(repository="https://github.com/example/fork")

    def test_strict_json_rejects_extra_document_duplicates_and_nan(self) -> None:
        malformed = {
            "extra": "{} {}",
            "duplicate": '{"value": 1, "value": 2}',
            "nan": '{"value": NaN}',
        }
        for name, text in malformed.items():
            with self.subTest(name=name):
                path = self.root / f"{name}.json"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(publisher.PublishError):
                    publisher.load_json_document(path)

    def test_cli_publishes_staging_metadata_and_source_ref(self) -> None:
        raw_path = self.root / "raw.json"
        raw_path.write_text(json.dumps(self._raw()), encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = publisher.main(
                [
                    "--raw-run",
                    str(raw_path),
                    "--data-dir",
                    str(self.data),
                    "--run-id",
                    "github-200-attempt-2",
                    "--repository",
                    "https://github.com/ROCm/rocm-systems",
                    "--environment-id",
                    "candidate-rocm-7.2",
                    "--trigger",
                    "manual",
                    "--branch",
                    "develop",
                    "--is-demo",
                    "--source-ref",
                    "users/example/candidate",
                    "--expected-sha",
                    "a" * 40,
                    "--generated-at",
                    "2026-09-04T00:00:00Z",
                ]
            )
        self.assertEqual(status, 0)
        self.assertTrue(json.loads(output.getvalue())["changed"])
        index = publisher.load_json_document(self.data / "index.json")
        catalog = publisher.load_json_document(self.data / index["catalog"])
        run = publisher.load_json_document(
            self.data / "runs" / "github-200-attempt-2.json"
        )
        self.assertTrue(catalog["isDemo"])
        self.assertEqual(run["environmentId"], "candidate-rocm-7.2")
        self.assertEqual(run["provenance"]["details"][0]["key"], "sourceRef")


if __name__ == "__main__":
    unittest.main()
