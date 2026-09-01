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
            self.build / "benchmarks" / "rocjitsu-benchmark-native-gfx950",
            self.build / "benchmarks" / "rocjitsu-benchmark-native-gfx1250",
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
        with (
            mock.patch.object(
                runner,
                "_source_info",
                return_value={"revision": "a", "dirty": False},
            ),
            mock.patch.object(runner, "_environment_info", return_value={}),
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
        self.assertEqual(len(matrix), 24)
        self.assertEqual(
            matrix[:4],
            (
                runner.Cell("hip.launch_noop", "gfx950"),
                runner.Cell("hip.launch_noop", "gfx1250"),
                runner.Cell("hip.copy_fp32_32m", "gfx950"),
                runner.Cell("hip.copy_fp32_32m", "gfx1250"),
            ),
        )
        self.assertEqual(self.suite.warmups, 3)
        self.assertEqual(self.suite.samples, 21)
        self.assertEqual(self.suite.timeout_seconds, 300)

    def test_manifest_rejects_extra_fields(self) -> None:
        manifest = self.root / "suite.toml"
        text = runner.DEFAULT_MANIFEST.read_text(encoding="utf-8")
        manifest.write_text(text + "description = 'extra'\n", encoding="utf-8")
        with self.assertRaisesRegex(runner.RunnerError, "extra=.*description"):
            runner.load_manifest(manifest)

    def test_manifest_rejects_case_path_traversal(self) -> None:
        manifest = self.root / "suite.toml"
        text = runner.DEFAULT_MANIFEST.read_text(encoding="utf-8").replace(
            '"hip.launch_noop"', '"hip./../../../escaped"'
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
            cases=("tensile.gemm_fp16", "hip.launch_noop"),
            targets=("gfx1250",),
        )
        self.assertEqual(
            matrix,
            (
                runner.Cell("hip.launch_noop", "gfx1250"),
                runner.Cell("tensile.gemm_fp16", "gfx1250"),
            ),
        )
        with self.assertRaisesRegex(runner.RunnerError, "not in the suite"):
            runner.select_matrix(self.suite, cases=("hip.unknown",))
        with self.assertRaisesRegex(runner.RunnerError, "not in the suite"):
            runner.select_matrix(self.suite, targets=("gfx9999",))

    def test_commands_are_derived_for_all_providers(self) -> None:
        output = self.root / "out"
        expected_programs = {
            "hip.launch_noop": "rocjitsu-benchmark-native-gfx950",
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
        cell = runner.Cell("hip.launch_noop", "gfx950")
        path.write_text(json.dumps(self._payload(cell, [30, 10, 20])), encoding="utf-8")
        result = runner.validate_workload(path, cell, 3)
        self.assertEqual(result["timings_ns"], [30, 10, 20])
        self.assertEqual(
            (result["min_ns"], result["median_ns"], result["max_ns"]),
            (10, 20, 30),
        )

    def test_success_writes_compact_artifacts(self) -> None:
        matrix = self._matrix("hip.launch_noop")
        output, result = self._run(matrix, "success", samples=3)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["results"][0]["timings_ns"], [1, 2, 3])
        self.assertIsNone(result["results"][0]["failure"])
        self.assertTrue((output / "run.json").is_file())
        self.assertEqual(
            (output / "cases/hip.launch_noop/gfx950/stdout.txt").read_text(),
            "out",
        )

    def test_nonzero_exit_preserves_logs(self) -> None:
        def fail(argv, **_kwargs):
            return subprocess.CompletedProcess(argv, 7, "partial out", "failure text")

        output, result = self._run(
            self._matrix("hip.launch_noop"), "nonzero", process=fail
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("status 7", result["results"][0]["failure"])
        self.assertIsNone(result["results"][0]["artifacts"]["workload"])
        self.assertEqual(
            (output / "cases/hip.launch_noop/gfx950/stderr.txt").read_text(),
            "failure text",
        )

    def test_rejects_build_from_another_worktree(self) -> None:
        (self.build / "CMakeCache.txt").write_text(
            "CMAKE_BUILD_TYPE:STRING=Release\n"
            "CMAKE_HOME_DIRECTORY:INTERNAL=/tmp/other-rocjitsu\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(runner.RunnerError, "belongs to"):
            runner.validate_build(self.build, self._matrix("hip.launch_noop"))

    def test_timeout_preserves_captured_output(self) -> None:
        def timeout(argv, **_kwargs):
            raise subprocess.TimeoutExpired(
                argv, 180, output=b"partial out", stderr=b"partial error"
            )

        output, result = self._run(
            self._matrix("hip.launch_noop"), "timeout", process=timeout
        )
        self.assertIn("timed out", result["results"][0]["failure"])
        self.assertEqual(
            (output / "cases/hip.launch_noop/gfx950/stdout.txt").read_text(),
            "partial out",
        )

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
            self._matrix("hip.launch_noop"),
            "malformed",
            process=malformed,
            samples=2,
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("positive integers", result["results"][0]["failure"])

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
            self._matrix("hip.launch_noop"),
            "nonfinite",
            process=nonfinite,
            samples=1,
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("non-finite JSON value", result["results"][0]["failure"])
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
                    self._matrix("hip.launch_noop"),
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

        matrix = self._matrix("hip.launch_noop", "hip.copy_fp32_32m")
        output, result = self._run(matrix, "partial", process=one_failure, samples=2)
        self.assertEqual(
            [item["status"] for item in result["results"]],
            ["failed", "passed"],
        )
        persisted = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(len(persisted["results"]), 2)
        self.assertEqual(persisted["status"], "failed")

    def test_list_needs_no_build_or_dependency_metadata(self) -> None:
        stdout = io.StringIO()
        with (
            mock.patch.object(runner, "_environment_info", side_effect=AssertionError),
            contextlib.redirect_stdout(stdout),
        ):
            status = runner.main(
                ["--list", "--case", "hip.launch_noop", "--target", "gfx950"]
            )
        self.assertEqual(status, 0)
        self.assertEqual(stdout.getvalue(), "hip.launch_noop\tgfx950\n")


if __name__ == "__main__":
    unittest.main()
