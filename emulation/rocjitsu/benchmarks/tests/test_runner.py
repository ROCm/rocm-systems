# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

from benchmarks import runner


class RunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.build = self.root / "build"
        (self.build / "tools" / "rocjitsu").mkdir(parents=True)
        (self.build / "benchmarks").mkdir()
        (self.build / "CMakeCache.txt").write_text(
            "CMAKE_BUILD_TYPE:STRING=Release\n"
            f"CMAKE_HOME_DIRECTORY:INTERNAL={runner.ROCJITSU_ROOT}\n",
            encoding="utf-8",
        )
        for path in (
            self.build / "tools" / "rocjitsu" / "rocjitsu",
            self.build / "benchmarks" / "rocjitsu-benchmark-hipblaslt",
        ):
            path.write_text("#!/bin/sh\n", encoding="utf-8")
            path.chmod(0o755)
        self.suite = runner.load_manifest()

    def _matrix(self, *cases: str, targets: tuple[str, ...] = ("gfx950",)):
        return runner.select_matrix(self.suite, cases=cases, targets=targets)

    @staticmethod
    def _payload(cell: runner.Cell, timings: list[int]) -> dict[str, object]:
        return {
            "schema": "rocjitsu.benchmark.workload.v1",
            "case": cell.case,
            "target": cell.target,
            "provider": cell.provider,
            "parameters": {"fixture": True},
            "timings_ns": timings,
        }

    def _successful_process(self, argv, **_kwargs):
        case = argv[argv.index("--case") + 1]
        target = argv[argv.index("--target") + 1]
        samples = int(argv[argv.index("--samples") + 1])
        output = Path(argv[argv.index("--output") + 1])
        output.write_text(
            json.dumps(
                self._payload(runner.Cell(case, target), list(range(1, samples + 1)))
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, "out", "")

    def _run(self, matrix, name: str, process=None, samples=None):
        output = self.root / name
        packages = {
            "rocm-sdk-devel": "7.2.0",
            "torch": "2.10.0",
            "triton": "3.6.0",
        }
        with (
            mock.patch.object(
                runner,
                "_source_info",
                return_value={
                    "commit_sha": "a" * 40,
                    "commit_timestamp": "2026-09-01T21:42:10Z",
                    "dirty": False,
                },
            ),
            mock.patch.object(
                runner,
                "_environment_info",
                return_value={
                    "hostname": "benchmark-host",
                    "platform": "Linux-test",
                    "kernel": "6.14.0",
                    "cpu": "test-cpu",
                    "python": "3.12.0",
                    "packages": packages,
                },
            ),
            mock.patch.object(
                runner,
                "_run_command",
                side_effect=process or self._successful_process,
            ),
        ):
            result = runner.run_suite(
                self.suite,
                matrix,
                build_dir=self.build,
                output=output,
                samples=samples,
            )
        return output, result

    def test_default_manifest_has_full_ordered_matrix(self) -> None:
        matrix = runner.select_matrix(self.suite)
        self.assertEqual(len(matrix), 18)
        self.assertEqual(set(runner.CASE_METADATA), set(self.suite.cases))
        self.assertEqual(
            matrix[:4],
            (
                runner.Cell("triton.softmax_fp16_aligned", "gfx950"),
                runner.Cell("triton.softmax_fp16_aligned", "gfx1250"),
                runner.Cell("triton.softmax_fp16_boundary", "gfx950"),
                runner.Cell("triton.softmax_fp16_boundary", "gfx1250"),
            ),
        )
        self.assertEqual(self.suite.warmups, 3)
        self.assertEqual(self.suite.samples, 21)
        self.assertEqual(self.suite.timeout_seconds, 300)

    def test_smoke_manifest_has_single_triton_case(self) -> None:
        smoke = runner.load_manifest(runner.BENCHMARK_ROOT / "suites" / "smoke.toml")
        self.assertEqual(smoke.name, "smoke")
        self.assertEqual(smoke.targets, ("gfx950",))
        self.assertEqual(smoke.cases, ("triton.rmsnorm_bf16",))
        self.assertEqual(smoke.warmups, 1)
        self.assertEqual(smoke.samples, 3)
        self.assertEqual(smoke.timeout_seconds, 60)

    def test_manifest_rejects_extra_fields(self) -> None:
        manifest = self.root / "suite.toml"
        text = runner.DEFAULT_MANIFEST.read_text(encoding="utf-8")
        manifest.write_text(text + "description = 'extra'\n", encoding="utf-8")
        with self.assertRaisesRegex(runner.RunnerError, "extra=.*description"):
            runner.load_manifest(manifest)

    def test_manifest_rejects_case_path_traversal(self) -> None:
        manifest = self.root / "suite.toml"
        text = runner.DEFAULT_MANIFEST.read_text(encoding="utf-8").replace(
            '"triton.softmax_fp16_aligned"', '"triton./../../../escaped"'
        )
        manifest.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(runner.RunnerError, "invalid benchmark case ID"):
            runner.load_manifest(manifest)

    def test_manifest_rejects_non_utf8_input(self) -> None:
        manifest = self.root / "suite.toml"
        manifest.write_bytes(b"name = \xff\n")
        with self.assertRaisesRegex(runner.RunnerError, "cannot read manifest"):
            runner.load_manifest(manifest)

    def test_subsets_keep_manifest_order_and_reject_unknowns(self) -> None:
        matrix = runner.select_matrix(
            self.suite,
            cases=("tensile.gemm_fp16", "triton.rmsnorm_bf16"),
            targets=("gfx1250",),
        )
        self.assertEqual(
            matrix,
            (
                runner.Cell("triton.rmsnorm_bf16", "gfx1250"),
                runner.Cell("tensile.gemm_fp16", "gfx1250"),
            ),
        )
        with self.assertRaisesRegex(runner.RunnerError, "not in the suite"):
            runner.select_matrix(self.suite, cases=("triton.unknown",))
        with self.assertRaisesRegex(runner.RunnerError, "not in the suite"):
            runner.select_matrix(self.suite, targets=("gfx9999",))

    def test_commands_are_derived_for_all_providers(self) -> None:
        output = self.root / "out"
        expected_programs = {
            "triton.rmsnorm_bf16": "-m",
            "tensile.gemm_fp16": "rocjitsu-benchmark-hipblaslt",
        }
        for case, marker in expected_programs.items():
            cell = runner.Cell(case, "gfx950")
            command = runner.prepare_command(
                self.build, output, cell, warmups=2, samples=5
            )
            self.assertTrue(
                any(
                    item == marker or item.endswith("/" + marker)
                    for item in command.argv
                )
            )
            self.assertEqual(
                command.argv[-10:],
                (
                    "--case",
                    case,
                    "--target",
                    "gfx950",
                    "--warmups",
                    "2",
                    "--samples",
                    "5",
                    "--output",
                    str(command.workload_path),
                ),
            )
            self.assertEqual(command.environment["PYTHONHASHSEED"], "0")
            self.assertNotIn("HIPBLASLT_TENSILE_LIBPATH", command.environment)
            self.assertTrue(
                command.environment["TRITON_CACHE_DIR"].endswith("triton/gfx950")
            )

    def test_validate_and_aggregate_workload(self) -> None:
        path = self.root / "workload.json"
        cell = runner.Cell("triton.rmsnorm_bf16", "gfx950")
        payload = self._payload(cell, [30, 10, 20])
        payload["parameters"] = {
            "threads_per_block": 64,
            "nested_values": [{"input_dtype": "fp16"}],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        result = runner.validate_workload(path, cell, 3)
        self.assertEqual(
            result["problem"],
            {
                "threadsPerBlock": 64,
                "nestedValues": [{"inputDtype": "fp16"}],
            },
        )
        self.assertEqual(result["durationSeconds"], 20 / 1_000_000_000)
        self.assertEqual(
            result["timing"],
            {
                "unit": "ns",
                "samples": [30, 10, 20],
                "minimum": 10,
                "median": 20,
                "maximum": 30,
            },
        )

    def test_dashboard_parameter_conversion_rejects_collisions(self) -> None:
        self.assertEqual(
            runner._dashboard_value(
                {
                    "input_dtype": "bf16",
                    "alreadyCamel": True,
                    "nested_list": [{"threads_per_block": 256}],
                }
            ),
            {
                "inputDtype": "bf16",
                "alreadyCamel": True,
                "nestedList": [{"threadsPerBlock": 256}],
            },
        )
        for parameters in (
            {"foo_bar": 1, "fooBar": 2},
            {"nested": {"foo__bar": 1, "foo_bar": 2}},
        ):
            with self.subTest(parameters=parameters):
                with self.assertRaisesRegex(runner.RunnerError, "collide as 'fooBar'"):
                    runner._dashboard_value(parameters)

    def test_success_writes_compact_artifacts(self) -> None:
        matrix = self._matrix("triton.rmsnorm_bf16")
        output, result = self._run(matrix, "success", samples=3)
        self.assertEqual(
            set(result),
            {
                "schemaVersion",
                "timestamp",
                "finishedAt",
                "status",
                "wallTimeSeconds",
                "benchmarkSuite",
                "targets",
                "measurement",
                "provenance",
                "environment",
                "tests",
            },
        )
        self.assertEqual(result["schemaVersion"], 1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["benchmarkSuite"], "nightly")
        self.assertEqual(result["targets"], ["gfx950"])
        self.assertEqual(
            result["measurement"],
            {"warmups": 3, "samples": 3, "timeoutSeconds": 300.0},
        )
        self.assertEqual(
            result["provenance"],
            {
                "rocjitsuCommitSha": "a" * 40,
                "rocjitsuCommitTimestamp": "2026-09-01T21:42:10Z",
                "dirty": False,
                "buildType": "Release",
                "rocmSdkVersion": "7.2.0",
                "pythonVersion": "3.12.0",
                "torchVersion": "2.10.0",
                "tritonVersion": "3.6.0",
                "tritonCommitSha": None,
                "tensileLiteCommitSha": None,
                "packages": {
                    "rocm-sdk-devel": "7.2.0",
                    "torch": "2.10.0",
                    "triton": "3.6.0",
                },
            },
        )
        self.assertEqual(
            result["environment"],
            {
                "hostname": "benchmark-host",
                "platform": "Linux-test",
                "kernel": "6.14.0",
                "cpu": "test-cpu",
            },
        )
        test = result["tests"][0]
        self.assertEqual(
            set(test),
            {
                "testId",
                "logicalTestId",
                "suite",
                "name",
                "target",
                "operation",
                "dataType",
                "problem",
                "execMode",
                "numThreads",
                "durationSeconds",
                "timing",
                "status",
                "exitCode",
                "timedOut",
                "error",
                "artifacts",
            },
        )
        self.assertEqual(test["testId"], "gfx950:triton.rmsnorm_bf16")
        self.assertEqual(test["logicalTestId"], "triton.rmsnorm_bf16")
        self.assertEqual(test["suite"], "Triton")
        self.assertEqual(test["name"], "BF16 RMSNorm")
        self.assertEqual(test["operation"], "RMSNorm")
        self.assertEqual(test["dataType"], "bf16")
        self.assertEqual(test["problem"], {"fixture": True})
        self.assertEqual(test["execMode"], "functional")
        self.assertEqual(test["numThreads"], 1)
        self.assertEqual(test["durationSeconds"], 2 / 1_000_000_000)
        self.assertEqual(
            test["timing"],
            {
                "unit": "ns",
                "samples": [1, 2, 3],
                "minimum": 1,
                "median": 2,
                "maximum": 3,
            },
        )
        self.assertEqual(test["status"], "completed")
        self.assertEqual(test["exitCode"], 0)
        self.assertFalse(test["timedOut"])
        self.assertIsNone(test["error"])
        self.assertNotIn("canonical", result["provenance"])
        self.assertTrue((output / "run.json").is_file())
        self.assertEqual(
            json.loads((output / "run.json").read_text(encoding="utf-8")), result
        )
        self.assertEqual(
            (output / "cases/triton.rmsnorm_bf16/gfx950/stdout.txt").read_text(),
            "out",
        )

    def test_initial_checkpoint_is_a_running_v1_run(self) -> None:
        observed = None

        def inspect_checkpoint(argv, **kwargs):
            nonlocal observed
            workload = Path(argv[argv.index("--output") + 1])
            observed = json.loads((workload.parents[3] / "run.json").read_text())
            return self._successful_process(argv, **kwargs)

        self._run(
            self._matrix("triton.rmsnorm_bf16"),
            "running-checkpoint",
            process=inspect_checkpoint,
            samples=1,
        )
        self.assertIsNotNone(observed)
        self.assertEqual(observed["schemaVersion"], 1)
        self.assertEqual(observed["status"], "running")
        self.assertIsNone(observed["finishedAt"])
        self.assertEqual(observed["tests"], [])

    def test_nonzero_exit_preserves_logs(self) -> None:
        def fail(argv, **_kwargs):
            return subprocess.CompletedProcess(argv, 7, "partial out", "failure text")

        output, result = self._run(
            self._matrix("triton.rmsnorm_bf16"), "nonzero", process=fail
        )
        self.assertEqual(result["status"], "failed")
        test = result["tests"][0]
        self.assertEqual(test["status"], "failed")
        self.assertEqual(test["exitCode"], 7)
        self.assertFalse(test["timedOut"])
        self.assertIn("status 7", test["error"])
        self.assertIsNone(test["problem"])
        self.assertIsNone(test["durationSeconds"])
        self.assertEqual(
            test["timing"],
            {
                "unit": "ns",
                "samples": [],
                "minimum": None,
                "median": None,
                "maximum": None,
            },
        )
        self.assertIsNone(test["artifacts"]["workload"])
        self.assertEqual(
            (output / "cases/triton.rmsnorm_bf16/gfx950/stderr.txt").read_text(),
            "failure text",
        )

    def test_rejects_build_from_another_worktree(self) -> None:
        (self.build / "CMakeCache.txt").write_text(
            "CMAKE_BUILD_TYPE:STRING=Release\n"
            "CMAKE_HOME_DIRECTORY:INTERNAL=/tmp/other-rocjitsu\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(runner.RunnerError, "belongs to"):
            runner.validate_build(self.build, self._matrix("triton.rmsnorm_bf16"))

    def test_target_metadata_is_read_from_selected_configuration(self) -> None:
        configuration = self.root / "target.json"
        configuration.write_text(
            json.dumps({"exec_mode": "parallel", "num_threads": 8}),
            encoding="utf-8",
        )
        with mock.patch.dict(
            runner.TARGET_CONFIGS, {"gfx950": configuration}, clear=True
        ):
            metadata = runner._target_metadata("gfx950")
            _, result = self._run(
                self._matrix("triton.rmsnorm_bf16"),
                "target-metadata",
                samples=1,
            )
        self.assertEqual(metadata, runner.TargetMetadata("parallel", 8))
        self.assertEqual(result["tests"][0]["execMode"], "parallel")
        self.assertEqual(result["tests"][0]["numThreads"], 8)

    def test_target_metadata_rejects_invalid_fields(self) -> None:
        configuration = self.root / "target.json"
        invalid_values = (
            [],
            {"exec_mode": "", "num_threads": 1},
            {"exec_mode": "functional", "num_threads": True},
            {"exec_mode": "functional", "num_threads": 0},
        )
        for value in invalid_values:
            with self.subTest(value=value):
                configuration.write_text(json.dumps(value), encoding="utf-8")
                with (
                    mock.patch.dict(
                        runner.TARGET_CONFIGS,
                        {"gfx950": configuration},
                        clear=True,
                    ),
                    self.assertRaises(runner.RunnerError),
                ):
                    runner._target_metadata("gfx950")
        configuration.write_text("{", encoding="utf-8")
        with (
            mock.patch.dict(
                runner.TARGET_CONFIGS, {"gfx950": configuration}, clear=True
            ),
            self.assertRaisesRegex(runner.RunnerError, "cannot read"),
        ):
            runner._target_metadata("gfx950")

    def test_commit_timestamp_is_normalized_to_utc(self) -> None:
        self.assertEqual(
            runner._normalize_timestamp("2026-09-02T09:30:00-07:00"),
            "2026-09-02T16:30:00Z",
        )
        self.assertEqual(
            runner._normalize_timestamp("2026-09-02T16:30:00+00:00"),
            "2026-09-02T16:30:00Z",
        )
        for value in ("", "not-a-timestamp", "2026-09-02T16:30:00"):
            with self.subTest(value=value):
                self.assertIsNone(runner._normalize_timestamp(value))

    def test_source_info_preserves_sha_when_timestamp_is_invalid(self) -> None:
        with mock.patch.object(
            runner.subprocess,
            "check_output",
            side_effect=["abc123\n", "invalid\n", " M local-file\n"],
        ) as check_output:
            result = runner._source_info()
        self.assertEqual(
            result,
            {
                "commit_sha": "abc123",
                "commit_timestamp": None,
                "dirty": True,
            },
        )
        self.assertEqual(
            check_output.call_args_list[1].args[0],
            ["git", "show", "-s", "--format=%cI", "abc123"],
        )
        for call in check_output.call_args_list:
            self.assertEqual(call.kwargs["cwd"], runner.ROCJITSU_ROOT)
            self.assertIs(call.kwargs["stderr"], subprocess.DEVNULL)

    def test_source_info_ties_timestamp_to_resolved_sha(self) -> None:
        with mock.patch.object(
            runner.subprocess,
            "check_output",
            side_effect=["abc123\n", "2026-09-02T09:30:00-07:00\n", ""],
        ) as check_output:
            result = runner._source_info()
        self.assertEqual(result["commit_sha"], "abc123")
        self.assertEqual(result["commit_timestamp"], "2026-09-02T16:30:00Z")
        self.assertEqual(
            check_output.call_args_list[1].args[0],
            ["git", "show", "-s", "--format=%cI", "abc123"],
        )

    def test_source_info_falls_back_when_git_is_unavailable(self) -> None:
        unavailable = subprocess.CalledProcessError(128, ["git"])
        with mock.patch.object(
            runner.subprocess,
            "check_output",
            side_effect=[unavailable, unavailable],
        ) as check_output:
            result = runner._source_info()
        self.assertEqual(
            result,
            {"commit_sha": None, "commit_timestamp": None, "dirty": None},
        )
        self.assertEqual(check_output.call_count, 2)

    def test_timeout_preserves_captured_output(self) -> None:
        def timeout(argv, **_kwargs):
            raise subprocess.TimeoutExpired(
                argv, 180, output=b"partial out", stderr=b"partial error"
            )

        output, result = self._run(
            self._matrix("triton.rmsnorm_bf16"), "timeout", process=timeout
        )
        test = result["tests"][0]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(test["status"], "timeout")
        self.assertIsNone(test["exitCode"])
        self.assertTrue(test["timedOut"])
        self.assertIn("timed out", test["error"])
        self.assertIsNone(test["problem"])
        self.assertIsNone(test["durationSeconds"])
        self.assertEqual(
            (output / "cases/triton.rmsnorm_bf16/gfx950/stdout.txt").read_text(),
            "partial out",
        )

    def test_zero_exit_without_workload_is_a_failed_test(self) -> None:
        def missing(argv, **_kwargs):
            return subprocess.CompletedProcess(argv, 0, "", "")

        _, result = self._run(
            self._matrix("triton.rmsnorm_bf16"),
            "missing-workload",
            process=missing,
        )
        test = result["tests"][0]
        self.assertEqual(test["status"], "failed")
        self.assertEqual(test["exitCode"], 0)
        self.assertIsNone(test["artifacts"]["workload"])
        self.assertIn("cannot read workload result", test["error"])

    def test_malformed_samples_fail_validation(self) -> None:
        def malformed(argv, **_kwargs):
            case = argv[argv.index("--case") + 1]
            target = argv[argv.index("--target") + 1]
            output = Path(argv[argv.index("--output") + 1])
            output.write_text(
                json.dumps(self._payload(runner.Cell(case, target), [1, 0])),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(argv, 0, "", "")

        _, result = self._run(
            self._matrix("triton.rmsnorm_bf16"),
            "malformed",
            process=malformed,
            samples=2,
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["tests"][0]["exitCode"], 0)
        self.assertFalse(result["tests"][0]["timedOut"])
        self.assertIn("positive integers", result["tests"][0]["error"])

    def test_nonfinite_parameters_fail_without_aborting_suite(self) -> None:
        def nonfinite(argv, **_kwargs):
            case = argv[argv.index("--case") + 1]
            target = argv[argv.index("--target") + 1]
            output = Path(argv[argv.index("--output") + 1])
            payload = self._payload(runner.Cell(case, target), [1])
            payload["parameters"] = {"invalid": float("nan")}
            output.write_text(json.dumps(payload), encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "", "")

        output, result = self._run(
            self._matrix("triton.rmsnorm_bf16"),
            "nonfinite",
            process=nonfinite,
            samples=1,
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("non-finite JSON value", result["tests"][0]["error"])
        persisted = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "failed")

    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    def test_success_cleans_up_workload_process_group(self) -> None:
        process = mock.Mock(pid=1234, returncode=0)
        process.communicate.return_value = ("out", "error")
        with (
            mock.patch.object(runner.subprocess, "Popen", return_value=process),
            mock.patch.object(runner.os, "killpg") as kill_group,
        ):
            completed = runner._run_command(
                ("workload",), cwd=self.root, env={}, timeout=1
            )
        kill_group.assert_called_once_with(1234, signal.SIGKILL)
        self.assertEqual(completed.stdout, "out")
        self.assertEqual(completed.stderr, "error")

    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    def test_timeout_kills_workload_process_group(self) -> None:
        process = mock.Mock(pid=1234, returncode=-signal.SIGKILL)
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(("workload",), 1),
            ("partial out", "partial error"),
        ]
        with (
            mock.patch.object(
                runner.subprocess, "Popen", return_value=process
            ) as popen,
            mock.patch.object(runner.os, "killpg") as kill_group,
            self.assertRaises(subprocess.TimeoutExpired) as raised,
        ):
            runner._run_command(("workload",), cwd=self.root, env={}, timeout=1)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(popen.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(popen.call_args.kwargs["errors"], "replace")
        kill_group.assert_called_once_with(1234, signal.SIGKILL)
        self.assertEqual(raised.exception.stdout, "partial out")
        self.assertEqual(raised.exception.stderr, "partial error")

    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    def test_interrupt_kills_workload_process_group(self) -> None:
        process = mock.Mock(pid=1234)
        process.communicate.side_effect = [KeyboardInterrupt(), ("", "")]
        with (
            mock.patch.object(runner.subprocess, "Popen", return_value=process),
            mock.patch.object(runner.os, "killpg") as kill_group,
            self.assertRaises(KeyboardInterrupt),
        ):
            runner._run_command(("workload",), cwd=self.root, env={}, timeout=1)
        kill_group.assert_called_once_with(1234, signal.SIGKILL)
        self.assertEqual(process.communicate.call_count, 2)

    def test_existing_output_is_never_overwritten(self) -> None:
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep"
        marker.write_text("unchanged", encoding="utf-8")
        with mock.patch.object(runner, "_run_command") as process:
            with self.assertRaisesRegex(runner.RunnerError, "already exists"):
                runner.run_suite(
                    self.suite,
                    self._matrix("triton.rmsnorm_bf16"),
                    build_dir=self.build,
                    output=output,
                )
        process.assert_not_called()
        self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")

    def test_failures_do_not_stop_later_cells(self) -> None:
        calls = 0

        def one_failure(argv, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return subprocess.CompletedProcess(argv, 9, "first", "failed")
            return self._successful_process(argv, **kwargs)

        matrix = self._matrix("triton.rmsnorm_bf16", "triton.gemm_bf16_aligned")
        output, result = self._run(matrix, "partial", process=one_failure, samples=2)
        self.assertEqual(
            [item["status"] for item in result["tests"]],
            ["failed", "completed"],
        )
        persisted = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(len(persisted["tests"]), 2)
        self.assertEqual(persisted["status"], "failed")

    def test_list_needs_no_build_or_dependency_metadata(self) -> None:
        stdout = io.StringIO()
        with (
            mock.patch.object(runner, "_environment_info", side_effect=AssertionError),
            mock.patch.object(runner, "_target_metadata", side_effect=AssertionError),
            contextlib.redirect_stdout(stdout),
        ):
            status = runner.main(
                [
                    "--list",
                    "--case",
                    "triton.rmsnorm_bf16",
                    "--target",
                    "gfx950",
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(stdout.getvalue(), "triton.rmsnorm_bf16\tgfx950\n")


if __name__ == "__main__":
    unittest.main()
