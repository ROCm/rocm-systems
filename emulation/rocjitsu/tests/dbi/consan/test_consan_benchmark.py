#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import consan_benchmark as benchmark
from consan_validation_catalog import PROFILE_IDS
from consan_validation_test_support import coverage, verdict


def _run(metric: str, value: float, *, wall_ms: float = 1000.0) -> dict:
    return {
        "wall_ms": wall_ms,
        "payload": {
            "result": {
                "passed": True,
                "metrics": {metric: value, "parameter_count": 123},
            },
            "phase_ms": {"run": wall_ms / 2},
            "peak_device_memory": {
                "allocated_bytes": 1024,
                "reserved_bytes": 2048,
            },
            "runtime": {
                "device_name": "fixture GPU",
                "architecture": "gfx1201",
                "torch": "fixture",
                "hip": "fixture",
            },
        },
    }


class ConSanBenchmarkTest(unittest.TestCase):
    @staticmethod
    def _runner_args(directory: str, *, resume: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            target="gfx1201",
            hook=None,
            output_dir=Path(directory),
            python=Path("/fixture/python"),
            aorta_dir=Path("/fixture/aorta"),
            timeout=7,
            resume=resume,
            run_identity={"fixture": "identity"},
        )

    def test_resume_reuses_only_a_fingerprinted_completed_cell(self) -> None:
        workload = benchmark.Workload("cell", "fixture", "latency_ms", {})
        output = benchmark.RESULT_MARKER + json.dumps(
            {"result": {"passed": True, "metrics": {"latency_ms": 1.0}}}
        )
        with tempfile.TemporaryDirectory() as directory:
            args = self._runner_args(directory)
            with mock.patch.object(
                benchmark.subprocess,
                "run",
                return_value=SimpleNamespace(stdout=output, stderr="", returncode=0),
            ) as run:
                first = benchmark._run_one(
                    args=args,
                    workload=workload,
                    mode=None,
                    audit_sites=False,
                    label="native",
                )
                run.assert_called_once()

            args.resume = True
            with mock.patch.object(benchmark.subprocess, "run") as run:
                second = benchmark._run_one(
                    args=args,
                    workload=workload,
                    mode=None,
                    audit_sites=False,
                    label="native",
                )
                run.assert_not_called()
            self.assertEqual(second, first)

    def test_timeout_preserves_partial_output(self) -> None:
        workload = benchmark.Workload("cell", "fixture", "latency_ms", {})
        with tempfile.TemporaryDirectory() as directory:
            args = self._runner_args(directory)
            timeout = subprocess.TimeoutExpired(
                cmd=["fixture"], timeout=args.timeout, output=b"partial stdout\n", stderr=b"partial stderr\n"
            )
            with mock.patch.object(benchmark.subprocess, "run", side_effect=timeout):
                with self.assertRaisesRegex(benchmark.BenchmarkError, "timed out"):
                    benchmark._run_one(
                        args=args,
                        workload=workload,
                        mode=None,
                        audit_sites=False,
                        label="native",
                    )
            self.assertEqual(
                (Path(directory) / "cell--native.log").read_text(),
                "partial stdout\npartial stderr\n",
            )

    def test_workloads_measure_one_bounded_operation_per_process(self) -> None:
        self.assertEqual(len(benchmark.WORKLOADS), 3)
        for workload in benchmark.WORKLOADS:
            self.assertEqual(workload.config["warmup_steps"], 0)
            self.assertEqual(workload.config["steps"], 1)

        prefill, decode, moe = benchmark.WORKLOADS
        self.assertEqual(prefill.config["request"]["generate_tokens"], 0)
        self.assertEqual(decode.config["mode"], "continuous_batch")
        self.assertEqual(decode.config["request"]["generate_tokens"], 1)
        self.assertEqual(decode.primary_metric, "decode_latency_ms")
        self.assertEqual(moe.config["request"]["generate_tokens"], 0)

    def test_site_audit_is_enabled_by_default_and_has_one_switch(self) -> None:
        common = [
            "--target",
            "gfx1201",
            "--aorta-dir",
            "/aorta",
            "--hook",
            "/hook",
            "--output-dir",
            "/output",
        ]
        self.assertTrue(benchmark._parse_args(common).audit_sites)
        self.assertFalse(
            benchmark._parse_args([*common, "--no-audit-sites"]).audit_sites
        )

    def test_clean_environment_removes_stale_sanitizer_controls(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "PATH": "/bin",
                "RJ_CONSAN_MODE": "stale",
                "HSA_TOOLS_LIB": "/stale",
                "HSA_MODEL_LIB": "/stale-model",
            },
            clear=True,
        ):
            native = benchmark._clean_environment("gfx1201", None, None, False)
            instrumented = benchmark._clean_environment(
                "gfx1201", Path("/new-hook"), "sampled", True, Path("/names.txt")
            )
        self.assertEqual(native, {"PATH": "/bin", "HIP_TARGET": "gfx1201"})
        self.assertEqual(instrumented["HSA_TOOLS_LIB"], "/new-hook")
        self.assertEqual(instrumented["RJ_CONSAN_MODE"], "sampled")
        self.assertEqual(instrumented["RJ_CONSAN_LOG"], "3")
        self.assertEqual(
            instrumented["RJ_CONSAN_KERNEL_ALLOWLIST_FILE"], "/names.txt"
        )
        self.assertNotIn("HSA_MODEL_LIB", instrumented)

    def test_payload_parser_requires_one_successful_machine_record(self) -> None:
        text = "noise\n" + benchmark.RESULT_MARKER + json.dumps(
            {"result": {"passed": True}}
        )
        self.assertTrue(benchmark._parse_payload(text)["result"]["passed"])
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark._parse_payload("noise only")
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark._parse_payload(
                benchmark.RESULT_MARKER + json.dumps({"result": {"passed": False}})
            )

    def test_coverage_summary_fails_closed_and_counts_sites(self) -> None:
        summary = benchmark._coverage_summary("\n".join((coverage(), verdict())))
        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["selected"], 27)
        self.assertEqual(summary["patched"], 27)
        self.assertEqual(summary["checked"], 27)
        self.assertEqual(summary["missed"], 0)
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark._coverage_summary("not coverage evidence")

    def test_ratio_uses_audit_off_latency_and_bracketed_native_median(self) -> None:
        workload = benchmark.Workload("id", "description", "latency_ms", {})
        modes = {}
        for index, mode in enumerate(PROFILE_IDS, 1):
            quick = _run("latency_ms", 20.0 + index)
            audit = _run("latency_ms", 9000.0, wall_ms=2000.0)
            audit["coverage"] = {"accepted": True}
            modes[mode] = {"quick": quick, "audit": audit}
        summary = benchmark._summarize_workload(
            workload,
            [_run("latency_ms", 9.0), _run("latency_ms", 11.0)],
            modes,
            {"payload": {"kernel_names": ["Cijk_fixture.kd"]}},
            ("Cijk_fixture.kd",),
        )
        self.assertEqual(summary["native_latency_ms"], 10.0)
        for index, mode in enumerate(PROFILE_IDS, 1):
            self.assertEqual(summary["modes"][mode]["ratio"], (20.0 + index) / 10.0)
            self.assertEqual(
                summary["modes"][mode]["site_audit"]["wall_delta_percent"],
                100.0,
            )

    def test_status_primary_table_has_all_four_modes(self) -> None:
        workload = benchmark.Workload("id", "description", "latency_ms", {})
        modes = {
            mode: {"quick": _run("latency_ms", 10.0), "audit": None}
            for mode in PROFILE_IDS
        }
        workload_summary = benchmark._summarize_workload(
            workload,
            [_run("latency_ms", 5.0), _run("latency_ms", 5.0)],
            modes,
            {"payload": {"kernel_names": ["Cijk_fixture.kd"]}},
            ("Cijk_fixture.kd",),
        )
        summary = {
            "completed_at": "now",
            "target": "gfx1201",
            "audit_sites": False,
            "suite_wall_seconds": 1.0,
            "artifact_dir": "/artifacts",
            "workloads": [workload_summary],
            "provenance": {
                "aorta": {"commit": "aorta", "dirty": False},
                "rocm_systems": {"commit": "source", "dirty": False},
                "hook": {"sha256": "hash"},
            },
        }
        text = benchmark._render_status(summary)
        for label in benchmark.MODE_LABELS.values():
            self.assertIn(label, text)
        self.assertIn("| description | 2.000× | 2.000× | 2.000× | 2.000× |", text)

    def test_workload_selection_includes_every_exact_observed_kernel(self) -> None:
        inventory = {
            "payload": {
                "kernel_names": [
                    "void templated<int, float>() [clone .kd]",
                    "Cijk_selected_b.kd",
                    "Cijk_selected_a.kd",
                ]
            }
        }
        self.assertEqual(
            benchmark._select_dispatched_kernels(inventory),
            (
                "Cijk_selected_a.kd",
                "Cijk_selected_b.kd",
                "void templated<int, float>() [clone .kd]",
            ),
        )
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark._select_dispatched_kernels({"payload": {"kernel_names": []}})


if __name__ == "__main__":
    unittest.main()
