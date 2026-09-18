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
from consan_benchmark import PROFILE_IDS
from consan_validation_test_support import coverage, log, verdict


def _run(
    metric: str,
    values: tuple[float, float],
    *,
    wall_ms: float = 1000.0,
    run_ms: tuple[float, float] | None = None,
    instrumentation_ms: tuple[float, float] = (0.0, 0.0),
    startup_ms: float = 0.0,
    direct_runtime: bool = False,
) -> dict:
    if run_ms is None:
        run_ms = (wall_ms / 2, wall_ms / 2)
    runs = []
    for index in range(2):
        operation = {
            "index": index + 1,
            "result": {
                "passed": True,
                "metrics": {metric: values[index], "parameter_count": 123},
            },
            "phase_ms": run_ms[index],
            "instrumentation_ms": instrumentation_ms[index],
        }
        if direct_runtime:
            operation["runtime_ms"] = run_ms[index]
        runs.append(operation)
    return {
        "wall_ms": wall_ms,
        "payload": {
            "result": runs[0]["result"],
            "runs": runs,
            "phase_ms": {"run": run_ms[0]},
            "instrumentation_ms": {
                "before_run": startup_ms - sum(instrumentation_ms),
                "during_run": sum(instrumentation_ms),
                "total": startup_ms,
            },
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
    def test_source_identity_tracks_dirty_code_but_not_markdown_or_commit_bookkeeping(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                subprocess.run(
                    (
                        "git",
                        "-C",
                        directory,
                        "-c",
                        "user.name=Test",
                        "-c",
                        "user.email=test@example.invalid",
                        *args,
                    ),
                    check=True,
                    capture_output=True,
                )

            git("init")
            (root / "code.py").write_text("value = 1\n")
            (root / "STATUS.md").write_text("pending\n")
            git("add", ".")
            git("commit", "-m", "initial")
            initial = benchmark._git_identity(root)
            (root / "STATUS.md").write_text("accepted\n")
            self.assertEqual(
                initial["execution_tree_sha256"],
                benchmark._git_identity(root)["execution_tree_sha256"],
            )
            (root / "code.py").write_text("value = 2\n")
            dirty = benchmark._git_identity(root)
            self.assertNotEqual(
                initial["execution_tree_sha256"], dirty["execution_tree_sha256"]
            )
            git("add", ".")
            git("commit", "-m", "record tested code")
            committed = benchmark._git_identity(root)
            self.assertEqual(
                dirty["execution_tree_sha256"], committed["execution_tree_sha256"]
            )
            self.assertNotEqual(initial["commit"], committed["commit"])

    def test_workload_identity_uses_executables_not_ledger_commit_or_selection(
        self,
    ) -> None:
        workload = benchmark.WORKLOADS[3]
        identity = {
            "source": {
                "commit": "before",
                "dirty": False,
                "execution_tree_sha256": "code",
            },
            "hook_sha256": "hook",
            "payload_sha256": {"gluon": "payload"},
        }
        expected = benchmark._workload_identity(identity, workload)
        changed = {
            **identity,
            "source": {
                "commit": "ledger-commit",
                "dirty": True,
                "execution_tree_sha256": "code",
            },
            "payload_sha256": {"gluon": "payload", "tokenspeed": "another-payload"},
        }
        self.assertEqual(expected, benchmark._workload_identity(changed, workload))
        self.assertNotEqual(
            expected,
            benchmark._workload_identity(
                {**identity, "source": {"execution_tree_sha256": "changed-code"}},
                workload,
            ),
        )
        for key, value in (
            ("hook_sha256", "new-hook"),
            ("payload_sha256", {"gluon": "new-payload"}),
            ("python_environment", {"torch": "changed"}),
        ):
            self.assertNotEqual(
                expected,
                benchmark._workload_identity({**identity, key: value}, workload),
            )
        different_input = benchmark.Workload(
            workload.id,
            workload.description,
            workload.primary_metric,
            {**workload.config, "elements": 42},
            workload.payload,
        )
        self.assertNotEqual(
            expected, benchmark._workload_identity(identity, different_input)
        )

    def test_target_corpus_preserves_rdna_and_preallocates_cdna_rows(self) -> None:
        self.assertEqual(benchmark._target_workloads("gfx1201"), benchmark.WORKLOADS)
        cdna = benchmark._target_workloads("gfx950")
        self.assertEqual(len(cdna), 13)
        self.assertEqual(len({workload.id for workload in cdna}), 13)
        self.assertEqual(cdna[:5], benchmark.WORKLOADS)

    def test_live_status_records_running_and_failure_without_removing_rows(
        self,
    ) -> None:
        workloads = benchmark._target_workloads("gfx950")
        with tempfile.TemporaryDirectory() as directory:
            args = self._runner_args(directory)
            args.status = Path(directory) / "STATUS.md"
            args.status_projection = {
                "target": "gfx950",
                "workloads": [
                    {
                        "id": w.id,
                        "description": w.description,
                        "modes": {},
                        "progress": {},
                    }
                    for w in workloads
                ],
            }

            def fail(*unused_args, **unused_kwargs):
                text = args.status.read_text()
                self.assertIn("native-reference-1: running", text)
                self.assertIn(workloads[-1].description, text)
                raise subprocess.TimeoutExpired(
                    "fixture", 7, output=b"partial evidence"
                )

            with mock.patch.object(benchmark.subprocess, "run", side_effect=fail):
                with self.assertRaises(benchmark.BenchmarkError):
                    benchmark._run_one(
                        args=args,
                        workload=workloads[0],
                        mode=None,
                        audit_sites=False,
                        label="native-reference-1",
                    )
            text = args.status.read_text()
            self.assertIn("native-reference-1: failed", text)
            self.assertIn("| failed | failed |", text)
            self.assertTrue(all(w.description in text for w in workloads))
            progress = json.loads((Path(directory) / "progress.json").read_text())
            self.assertIn(
                "timed out", progress["projection"]["workloads"][0]["progress"]["error"]
            )

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
            rocprofv3=Path("/fixture/rocprofv3"),
            hipblaslt_bench=None,
        )

    def test_resume_reuses_only_a_fingerprinted_completed_cell(self) -> None:
        workload = benchmark.Workload("cell", "fixture", "latency_ms", {})
        output = benchmark.RESULT_MARKER + json.dumps(
            {
                "result": {"passed": True, "metrics": {"latency_ms": 1.0}},
            }
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

    def test_rocprof_profiles_the_exact_payload_command(self) -> None:
        workload = benchmark.Workload("cell", "fixture", "latency_ms", {})
        output = benchmark.RESULT_MARKER + json.dumps(
            {"result": {"passed": True, "metrics": {"latency_ms": 1.0}}}
        )
        with tempfile.TemporaryDirectory() as directory:
            args = self._runner_args(directory)
            profile = Path(directory) / "profile"
            with mock.patch.object(
                benchmark.subprocess,
                "run",
                return_value=SimpleNamespace(stdout=output, stderr="", returncode=0),
            ) as run:
                benchmark._run_one(
                    args=args,
                    workload=workload,
                    mode=None,
                    audit_sites=False,
                    label="inventory",
                    profile_directory=profile,
                )
            command = run.call_args.args[0]
            separator = command.index("--")
            self.assertEqual(
                command[:separator],
                [
                    "/fixture/rocprofv3",
                    "--kernel-trace",
                    "--output-format",
                    "csv",
                    "--output-directory",
                    str(profile),
                ],
            )
            self.assertEqual(
                command[separator + 1 :], benchmark._payload_command(args, workload)
            )

    def test_allowlist_content_participates_in_cell_fingerprint(self) -> None:
        workload = benchmark.Workload("cell", "fixture", "latency_ms", {})
        output = benchmark.RESULT_MARKER + json.dumps(
            {"result": {"passed": True, "metrics": {"latency_ms": 1.0}}}
        )
        with tempfile.TemporaryDirectory() as directory:
            args = self._runner_args(directory)
            allowlist = Path(directory) / "allowlist.txt"
            allowlist.write_text("kernel-a\n", encoding="utf-8")
            with mock.patch.object(
                benchmark.subprocess,
                "run",
                return_value=SimpleNamespace(stdout=output, stderr="", returncode=0),
            ):
                benchmark._run_one(
                    args=args,
                    workload=workload,
                    mode=None,
                    audit_sites=False,
                    label="native",
                    kernel_allowlist_file=allowlist,
                )
            args.resume = True
            allowlist.write_text("kernel-b\n", encoding="utf-8")
            with mock.patch.object(
                benchmark.subprocess,
                "run",
                return_value=SimpleNamespace(stdout=output, stderr="", returncode=0),
            ) as run:
                benchmark._run_one(
                    args=args,
                    workload=workload,
                    mode=None,
                    audit_sites=False,
                    label="native",
                    kernel_allowlist_file=allowlist,
                )
            run.assert_called_once()

    def test_timeout_preserves_partial_output(self) -> None:
        workload = benchmark.Workload("cell", "fixture", "latency_ms", {})
        with tempfile.TemporaryDirectory() as directory:
            args = self._runner_args(directory)
            timeout = subprocess.TimeoutExpired(
                cmd=["fixture"],
                timeout=args.timeout,
                output=b"partial stdout\n",
                stderr=b"partial stderr\n",
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

    def test_workloads_measure_two_bounded_operations_per_process(self) -> None:
        self.assertEqual(len(benchmark.WORKLOADS), 5)
        for workload in benchmark.WORKLOADS[:3]:
            self.assertEqual(workload.config["warmup_steps"], 0)
            self.assertEqual(workload.config["steps"], 1)

        prefill, decode, moe, gluon, hipblaslt = benchmark.WORKLOADS
        self.assertEqual(prefill.config["request"]["generate_tokens"], 0)
        self.assertEqual(decode.config["mode"], "continuous_batch")
        self.assertEqual(decode.config["request"]["generate_tokens"], 1)
        self.assertEqual(decode.primary_metric, "decode_latency_ms")
        self.assertEqual(moe.config["request"]["generate_tokens"], 0)
        self.assertEqual(gluon.payload, "gluon")
        self.assertEqual(hipblaslt.payload, "hipblaslt")

    def test_site_audit_is_enabled_by_default_with_hidden_opt_out(self) -> None:
        common = [
            "--target",
            "gfx1201",
            "--aorta-dir",
            "/aorta",
            "--hook",
            "/hook",
            "--rocprofv3",
            "/rocprofv3",
            "--output-dir",
            "/output",
        ]
        self.assertTrue(benchmark._parse_args(common).audit_sites)
        self.assertFalse(
            benchmark._parse_args([*common, "--no-audit-sites"]).audit_sites
        )
        with self.assertRaises(SystemExit), mock.patch("sys.stdout") as stdout:
            benchmark._parse_args(["--help"])
        self.assertNotIn(
            "audit-sites",
            "".join(call.args[0] for call in stdout.write.call_args_list),
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
                "gfx1201", Path("/new-hook"), "default", True, Path("/names.txt")
            )
            manually_selected = benchmark._clean_environment(
                "gfx1201",
                Path("/new-hook"),
                "default",
                True,
                epoch_analysis="manual",
            )
            supercollider = benchmark._clean_environment(
                "gfx1201",
                Path("/new-hook"),
                "supercollider",
                True,
                epoch_analysis="manual",
            )
        self.assertEqual(native, {"PATH": "/bin", "HIP_TARGET": "gfx1201"})
        self.assertEqual(instrumented["HSA_TOOLS_LIB"], "/new-hook")
        self.assertEqual(instrumented["RJ_CONSAN_MODE"], "default")
        self.assertEqual(instrumented["RJ_CONSAN_LOG"], "3")
        self.assertEqual(instrumented["RJ_CONSAN_KERNEL_ALLOWLIST_FILE"], "/names.txt")
        self.assertEqual(instrumented["RJ_CONSAN_REQUIRE_RECORDS"], "0")
        self.assertEqual(instrumented["RJ_CONSAN_FORBID_DIAGNOSTICS"], "0")
        self.assertEqual(instrumented["RJ_CONSAN_FORBID_OVERFLOW"], "0")
        self.assertNotIn("HSA_MODEL_LIB", instrumented)
        self.assertEqual(manually_selected["RJ_CONSAN_EPOCH_ANALYSIS"], "manual")
        self.assertNotIn("RJ_CONSAN_EPOCH_ANALYSIS", supercollider)

    def test_high_variant_uses_default_mode_and_explicit_preset(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"RJ_CONSAN_PRESET": "max", "RJ_CONSAN_CELL_SAMPLE_STRIDE": "1"},
            clear=True,
        ):
            default = benchmark._clean_environment(
                "gfx950", Path("/hook"), "default", True
            )
            high = benchmark._clean_environment(
                "gfx950", Path("/hook"), "default-high", True, epoch_analysis="manual"
            )
        self.assertEqual(default["RJ_CONSAN_PRESET"], "default")
        self.assertEqual(high["RJ_CONSAN_MODE"], "default")
        self.assertEqual(high["RJ_CONSAN_PRESET"], "high")
        self.assertEqual(high["RJ_CONSAN_EPOCH_ANALYSIS"], "manual")
        self.assertNotIn("RJ_CONSAN_CELL_SAMPLE_STRIDE", high)

    def test_presets_are_explicit_and_do_not_leak_into_supercollider(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "RJ_CONSAN_PRESET": "max",
                "RJ_CONSAN_WORKGROUP_SAMPLE_STRIDE": "1",
            },
            clear=True,
        ):
            for configuration, preset in (
                ("default", "default"),
                ("default-high", "high"),
            ):
                environment = benchmark._clean_environment(
                    "gfx1201",
                    Path("/hook"),
                    configuration,
                    True,
                    epoch_analysis="manual",
                )
                self.assertEqual(environment["RJ_CONSAN_MODE"], "default")
                self.assertEqual(environment["RJ_CONSAN_PRESET"], preset)
                self.assertEqual(environment["RJ_CONSAN_EPOCH_ANALYSIS"], "manual")
                self.assertNotIn("RJ_CONSAN_WORKGROUP_SAMPLE_STRIDE", environment)
            environment = benchmark._clean_environment(
                "gfx1201",
                Path("/hook"),
                "supercollider",
                True,
            )
            self.assertNotIn("RJ_CONSAN_PRESET", environment)
        self.assertEqual(PROFILE_IDS, ("default", "default-high", "supercollider"))

    def test_payload_parser_requires_one_successful_machine_record(self) -> None:
        text = (
            "noise\n"
            + benchmark.RESULT_MARKER
            + json.dumps({"result": {"passed": True}})
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

    def test_coverage_summary_accepts_incomplete_dynamic_capture(self) -> None:
        summary = benchmark._coverage_summary(
            "\n".join(
                (
                    coverage(),
                    verdict(
                        analysis_complete="false",
                        dynamic_complete="false",
                        dynamic_incomplete="1",
                        window_saturation="1",
                    ),
                )
            )
        )
        self.assertTrue(summary["accepted"])
        self.assertFalse(summary["dynamic_complete"])

    def test_zero_site_workload_is_inapplicable_only_with_dispatch_proof(self) -> None:
        empty = coverage(**{
            f"{kind}_{field}": "0"
            for kind in benchmark.SITE_KINDS
            for field in ("discovered", "supported", "selected", "patched")
        })
        empty_verdict = verdict(
            applicable="false", analysis_complete="false", static_complete="false",
            applicable_code_objects="0", access="0/0", barrier="0/0",
            atomic="0/0", fence="0/0",
        )
        name = "void kernel<int, float>(int)"
        entry = (
            f"[rocjitsu-dbi-hooks] ConSan kernel allowlist entry name={name} "
            "loaded=true instrumented=false dispatches=2 visible_records=0 "
            "status=loaded-not-instrumented"
        )
        output = log(empty, empty_verdict, entry)
        result = benchmark._coverage_summary(output, (name,))
        self.assertFalse(result["accepted"])
        self.assertFalse(result["applicable"])
        self.assertEqual(result["dispatched_kernels"], [name])
        self.assertEqual(result["inventoried_code_objects"], 1)
        for bad_output, allowlist in (
            (output, ()),
            (output, (name, "missing")),
            (output, ("different",)),
            (log(empty, empty_verdict), (name,)),
            (output.replace("dispatches=2", "dispatches=0"), (name,)),
            (output.replace("loaded=true", "loaded=false"), (name,)),
            (output.replace("instrumented=false", "instrumented=true"), (name,)),
            (output.replace("expert_limit=false", "expert_limit=true"), (name,)),
            (output.replace("analysis_complete=true", "analysis_complete=false"), (name,)),
            (log(output, entry), (name,)),
            (log(output, "[rocjitsu-dbi-hooks] ConSan kernel allowlist entry malformed"), (name,)),
        ):
            with self.subTest(output=bad_output, allowlist=allowlist):
                with self.assertRaises(benchmark.BenchmarkError):
                    benchmark._coverage_summary(bad_output, allowlist)

    def test_applicable_sites_cannot_be_reclassified_as_inapplicable(self) -> None:
        # Even dispatch proof must not hide a real static instrumentation gap.
        output = log(
            coverage(analysis_complete="false", access_patched="19",
                     access_placement_or_lowering_failed="1"),
            verdict(analysis_complete="false", static_complete="false",
                    incomplete_code_objects="1", access="19/20"),
            "[rocjitsu-dbi-hooks] ConSan kernel allowlist entry name=kernel "
            "loaded=true instrumented=false dispatches=2 visible_records=0 "
            "status=loaded-not-instrumented",
        )
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark._coverage_summary(output, ("kernel",))

    def test_inapplicable_summary_and_status_do_not_claim_performance(self) -> None:
        result = benchmark._summarize_mode(
            {"coverage": {"accepted": False, "applicable": False}}, (1.0, 1.0)
        )
        self.assertNotIn("run_ratio", result)
        self.assertNotIn("startup_ms", result)
        status = benchmark._render_status({
            "target": "gfx950",
            "workloads": [{
                "description": "no LDS", "native_runtime_ms": [1.0, 1.0],
                "modes": {mode: result for mode in PROFILE_IDS},
            }],
        })
        self.assertEqual(status.count("N/A (no applicable sites)"), 6)
        self.assertEqual(status.count("×"), 1)  # Native baseline only.

    def test_coverage_summary_rejects_incomplete_static_instrumentation(self) -> None:
        with self.assertRaisesRegex(
            benchmark.BenchmarkError, "static analysis incomplete"
        ):
            benchmark._coverage_summary(
                log(
                    coverage(
                        analysis_complete="false",
                        access_patched="19",
                        access_placement_or_lowering_failed="1",
                    ),
                    verdict(
                        analysis_complete="false",
                        static_complete="false",
                        incomplete_code_objects="1",
                        access="19/20",
                    ),
                )
            )

    def test_summary_folds_first_run_into_startup_and_reports_steady_run(self) -> None:
        workload = benchmark.Workload("id", "description", "latency_ms", {})
        modes = {}
        for index, mode in enumerate(PROFILE_IDS, 1):
            run = _run(
                "latency_ms",
                (20.0 + index, 30.0 + index),
                run_ms=(1120.0 + 100.0 * index, 630.0),
                instrumentation_ms=(800.0 + 100.0 * index, 30.0),
                startup_ms=1100.0 + 100.0 * index,
            )
            run["coverage"] = {"accepted": True}
            modes[mode] = run
        summary = benchmark._summarize_workload(
            workload,
            [
                _run("latency_ms", (9.0, 29.0), run_ms=(19.0, 29.0)),
                _run("latency_ms", (11.0, 31.0), run_ms=(21.0, 31.0)),
            ],
            _run("latency_ms", (12.0, 33.0), run_ms=(22.0, 33.0)),
            modes,
            ("Cijk_fixture.kd",),
        )
        self.assertEqual(summary["native_latency_ms"], [10.0, 30.0])
        self.assertEqual(summary["native_runtime_ms"], [20.0, 30.0])
        self.assertEqual(summary["native_validation_runtime_ms"], [22.0, 33.0])
        self.assertAlmostEqual(summary["native_validation_drift_ratio"][0], 0.1)
        self.assertAlmostEqual(summary["native_validation_drift_ratio"][1], 0.1)
        for index, mode in enumerate(PROFILE_IDS, 1):
            self.assertEqual(summary["modes"][mode]["runtime_ratio"], [16.0, 20.0])
            self.assertEqual(
                summary["modes"][mode]["startup_ms"], 1420.0 + 100.0 * index
            )
            self.assertEqual(
                summary["modes"][mode]["instrumentation_ms"],
                1100.0 + 100.0 * index,
            )
            self.assertEqual(summary["modes"][mode]["run_ms"], 600.0)
            self.assertEqual(summary["modes"][mode]["run_ratio"], 20.0)
            self.assertEqual(summary["modes"][mode]["coverage"], {"accepted": True})

    def test_summary_accepts_a_non_model_payload_without_parameters(self) -> None:
        workload = benchmark.Workload("id", "description", "latency_ms", {})
        runs = [_run("latency_ms", (2.0, 1.0)) for _ in range(2)]
        for run in runs:
            run["payload"]["result"]["metrics"].pop("parameter_count")
        validation = _run("latency_ms", (2.0, 1.0))
        validation["payload"]["result"]["metrics"].pop("parameter_count")
        modes = {mode: _run("latency_ms", (2.0, 1.0)) for mode in PROFILE_IDS}
        summary = benchmark._summarize_workload(
            workload, runs, validation, modes, ("kernel",)
        )
        self.assertNotIn("parameter_count", summary)
        self.assertEqual(summary["payload_metrics"], {"latency_ms": 2.0})

    def test_direct_device_runtime_does_not_subtract_instrumentation(self) -> None:
        run = _run(
            "latency_ms",
            (2.0, 3.0),
            run_ms=(2.0, 3.0),
            instrumentation_ms=(100.0, 200.0),
            direct_runtime=True,
        )
        self.assertEqual(benchmark._runtime_ms(run, 0), 2.0)
        self.assertEqual(benchmark._runtime_ms(run, 1), 3.0)

    def test_status_primary_table_has_all_benchmark_variants(self) -> None:
        workload = benchmark.Workload("id", "description", "latency_ms", {})
        modes = {
            mode: _run(
                "latency_ms",
                (398.875, 399.0),
                run_ms=(1250.0, 1000.0),
                instrumentation_ms=(500.0, 0.0),
                startup_ms=750.0,
            )
            for mode in PROFILE_IDS
        }
        workload_summary = benchmark._summarize_workload(
            workload,
            [
                _run("latency_ms", (5.0, 5.0), run_ms=(5.0, 5.0)),
                _run("latency_ms", (5.0, 5.0), run_ms=(5.0, 5.0)),
            ],
            _run("latency_ms", (5.0, 5.0), run_ms=(5.0, 5.0)),
            modes,
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
        self.assertLess(
            text.index("Default Mode Startup"),
            text.index("Default Mode (high) Startup"),
        )
        self.assertLess(
            text.index("Default Mode (high) Startup"),
            text.index("SuperCollider Startup"),
        )
        self.assertNotIn("ConSan Startup", text)
        self.assertIn(
            "| description | 0.005 s | 0.005 s (1×) | 1.5 s | 1 s (200×) | "
            "1.5 s | 1 s (200×) | 1.5 s | 1 s (200×) |",
            text,
        )
        self.assertNotIn("Absolute latency", text)
        self.assertNotIn("Site-audit", text)
        self.assertEqual(text.count("| --- |"), 1)

    def test_status_uses_grouped_decimal_not_scientific_notation(self) -> None:
        self.assertEqual(benchmark._format_three_significant_digits(9296.37), "9,300")
        self.assertEqual(benchmark._format_three_significant_digits(12345.0), "12,300")
        self.assertEqual(benchmark._format_three_significant_digits(2.714), "2.71")
        self.assertEqual(benchmark._format_three_significant_digits(0.01234), "0.0123")

    def test_payload_command_selects_each_backend(self) -> None:
        args = self._runner_args("/tmp")
        args.hipblaslt_bench = Path("/fixture/hipblaslt-bench")
        for workload in benchmark.WORKLOADS:
            command = benchmark._payload_command(args, workload)
            self.assertIn(benchmark._payload_program(workload).name, command[1])
        self.assertEqual(
            command[-2:], ["--hipblaslt-bench", "/fixture/hipblaslt-bench"]
        )

    def test_allowlist_converter_output_is_the_exact_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "allowlist.txt"
            output.write_text("kernel one\nkernel,two\n", encoding="utf-8")
            log = root / "converter.log"
            with mock.patch.object(
                benchmark.subprocess,
                "run",
                return_value=SimpleNamespace(
                    stdout="wrote two kernels\n", stderr="", returncode=0
                ),
            ) as run:
                names = benchmark._generate_kernel_allowlist(root, output, log)
            self.assertEqual(names, ("kernel one", "kernel,two"))
            self.assertEqual(log.read_text(), "wrote two kernels\n")
            self.assertIn("rocjitsu_consan_allowlist.py", str(run.call_args.args[0][1]))


if __name__ == "__main__":
    unittest.main()
