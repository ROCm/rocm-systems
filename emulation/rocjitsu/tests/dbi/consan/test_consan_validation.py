#!/usr/bin/env python3

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest import mock

import consan_validation as validation
from consan_coverage_gate import _COVERAGE_COUNT_FIELDS
from consan_validation_test_support import temporary_root


class ConSanValidationTest(unittest.TestCase):
    def test_launcher_json_is_an_exact_argv_prefix(self) -> None:
        self.assertEqual(
            validation._launcher_from_json(
                '["env", "-u", "HSA_MODEL_LIB", "tool", "--"]'
            ),
            ["env", "-u", "HSA_MODEL_LIB", "tool", "--"],
        )
        self.assertEqual(validation._launcher_from_json(None), [])
        for malformed in ('"tool"', "[]", '["tool", ""]', "not-json"):
            with self.assertRaises(validation.ValidationError):
                validation._launcher_from_json(malformed)

    def test_run_process_timeout_contains_descendants(self) -> None:
        parent = (
            "import subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "print(child.pid, flush=True); time.sleep(60)"
        )
        with temporary_root() as root:
            returncode, _, output = validation._run_process(
                [sys.executable, "-c", parent], os.environ.copy(), root / "run.log", 1
            )
        self.assertEqual(returncode, 124)
        self.assertIn("validation timeout after 1s", output)
        child_pid = int(output.splitlines()[0])
        for _ in range(20):
            try:
                state = (
                    Path(f"/proc/{child_pid}/stat")
                    .read_text(encoding="utf-8")
                    .split()[2]
                )
            except FileNotFoundError:
                break
            if state == "Z":
                break
            time.sleep(0.05)
        else:
            self.fail(f"timed-out descendant {child_pid} remained runnable")

    def test_coverage_summary_rejects_static_analysis_incompleteness(self) -> None:
        counts = {name: 0 for name in _COVERAGE_COUNT_FIELDS}
        counts["access_discovered"] = 1
        counts["access_unsupported"] = 1
        coverage = " ".join(f"{name}={counts[name]}" for name in _COVERAGE_COUNT_FIELDS)
        log = "\n".join(
            (
                "[rocjitsu-dbi-hooks] ConSan coverage reader=1 flavor=moi engine=record_replay "
                f"analysis_complete=false expert_limit=false {coverage}",
                "[rocjitsu-dbi-hooks] ConSan coverage_site reader=1 kind=access "
                "disposition=unsupported reason=unsupported_mnemonic outcome=unsupported "
                "lowering_reason=semantic_unsupported resource_reason=none container=k "
                "scope=kernel text=0x0 mnemonic=ds_unknown",
                "[rocjitsu-dbi-hooks] ConSan analysis verdict applicable=true "
                "analysis_complete=false "
                "static_complete=false dynamic_complete=true applicable_code_objects=1 "
                "incomplete_code_objects=1 access=0/0 barrier=0/0 atomic=0/0 fence=0/0 "
                "visible_evidence=0 dynamic_incomplete=0 replay_unsupported_access=0 "
                "replay_unsupported_atomics=0 replay_unsupported_fences=0 "
                "replay_metadata_full=0",
            )
        )
        summary = validation._coverage_summary(log)
        self.assertFalse(summary["accepted"])
        self.assertIn("analysis incomplete", summary["reasons"])

    def test_manifest_is_the_complete_north_star_matrix(self) -> None:
        manifest = validation._manifest("gfx1201")
        self.assertEqual(len(manifest["workloads"]), 14)
        self.assertEqual(
            [profile["id"] for profile in manifest["profiles"]],
            list(validation.PROFILE_IDS),
        )
        self.assertEqual(
            len({workload["id"] for workload in manifest["workloads"]}), 14
        )
        workloads = {workload["id"]: workload for workload in manifest["workloads"]}
        self.assertEqual(
            workloads["pytorch-rdna4-compiled-softmax"]["targets"], ("gfx1201",)
        )
        self.assertEqual(
            workloads["pytorch-rdna4-llm-topk"]["targets"], ("gfx1201",)
        )
        self.assertEqual(
            workloads["pytorch-rdna4-sdpa"]["targets"], ("gfx1201",)
        )

    def test_text_manifest_filters_target_specific_workloads(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(validation.main(["--target", "gfx1201", "manifest"]), 0)
        text = output.getvalue()
        self.assertIn("pytorch-rdna4-compiled-softmax", text)
        self.assertIn("pytorch-rdna4-llm-topk", text)
        self.assertIn("pytorch-rdna4-sdpa", text)
        self.assertNotIn("pytorch-tdm-descriptor-add", text)
        self.assertNotIn("tensile-sk-mxf8gemm-explicit", text)

    def test_gfx950_manifest_resolves_cdna4_native_workloads(self) -> None:
        manifest = validation._manifest("gfx950")
        workloads = {workload["id"]: workload for workload in manifest["workloads"]}
        self.assertEqual(
            workloads["d128-block"]["relative_path"],
            (
                "hip-moi-build/tests/"
                "hip_moi_instrumented_cdna4_d128_attention_block_test"
            ),
        )
        self.assertEqual(
            workloads["wmma-attention"]["clean_filter"],
            "HipMoiCdna4MfmaAttentionBlock.*",
        )
        self.assertEqual(
            workloads["d128-block"]["overhead_filter"],
            ("HipMoiCdna4D128AttentionBlock." "SampledFastContextMatchesHostReference"),
        )
        self.assertIn(
            "HipMoiCdna4MfmaStreamKArrivalCounter",
            workloads["streamk-arrival"]["overhead_filter"],
        )
        self.assertEqual(
            workloads["jakub-attention"]["relative_path"],
            "hip-moi-build/tests/hip_moi_reference_cdna4_jakub_matmul",
        )
        native_spellings = json.dumps(
            [
                workloads[workload_id]
                for workload_id in validation.GFX950_WORKLOAD_OVERRIDES
            ]
        )
        self.assertNotIn("rdna4", native_spellings.lower())

    def test_gfx950_doctor_checks_resolved_cdna4_executables(self) -> None:
        with temporary_root() as workspace:
            with mock.patch.object(validation.shutil, "which", return_value="/tool"):
                doctor = validation._doctor(workspace, "gfx950")
        d128 = doctor["paths"]["workload:d128-block:executable"]
        self.assertTrue(d128["path"].endswith("cdna4_d128_attention_block_test"))
        jakub = doctor["paths"]["workload:jakub-attention:executable"]
        self.assertTrue(jakub["path"].endswith("hip_moi_reference_cdna4_jakub_matmul"))

    def test_gfx1250_manifest_resolves_target_native_workloads(self) -> None:
        manifest = validation._manifest("gfx1250")
        workloads = {workload["id"]: workload for workload in manifest["workloads"]}
        self.assertEqual(
            workloads["d128-block"]["relative_path"],
            (
                "hip-moi-build-gfx1250-tests/tests/"
                "hip_moi_instrumented_gfx1250_d128_attention_block_test"
            ),
        )
        self.assertEqual(
            workloads["wmma-attention"]["clean_filter"],
            "HipMoiGfx1250WmmaAttentionBlock.*",
        )
        self.assertIn(
            "HipMoiGfx1250WmmaStreamKArrivalCounter",
            workloads["streamk-arrival"]["overhead_filter"],
        )
        self.assertEqual(
            workloads["d128-block"]["fault_filter"],
            (
                "HipMoiGfx1250D128AttentionBlock."
                "SampledFastContextMatchesHostReference"
            ),
        )
        self.assertEqual(
            workloads["jakub-attention"]["relative_path"],
            (
                "hip-moi-build-gfx1250-tests/tests/"
                "hip_moi_reference_gfx1250_jakub_matmul"
            ),
        )

    def test_gfx1250_doctor_checks_target_native_executables(self) -> None:
        with temporary_root() as workspace:
            with mock.patch.object(validation.shutil, "which", return_value="/tool"):
                doctor = validation._doctor(workspace, "gfx1250")
        d128 = doctor["paths"]["workload:d128-block:executable"]
        self.assertTrue(d128["path"].endswith("gfx1250_d128_attention_block_test"))
        jakub = doctor["paths"]["workload:jakub-attention:executable"]
        self.assertTrue(
            jakub["path"].endswith("hip_moi_reference_gfx1250_jakub_matmul")
        )

    def test_main_doctor_all_uses_target_filtered_workloads(self) -> None:
        result = {
            "ok": True,
            "workspace": "/workspace",
            "target": "gfx1201",
            "paths": {},
            "tools": {},
        }
        with (
            mock.patch.object(
                validation,
                "_workspace_from_environment",
                return_value=Path("/workspace"),
            ),
            mock.patch.object(validation, "_doctor", return_value=result) as doctor,
        ):
            self.assertEqual(validation.main(["--target", "gfx1201", "doctor"]), 0)
        doctor.assert_called_once_with(Path("/workspace"), "gfx1201", None)

    def test_workload_doctor_requires_only_selected_corpus_and_tools(self) -> None:
        with temporary_root() as workspace:
            with mock.patch.object(validation.shutil, "which", return_value="/tool"):
                doctor = validation._doctor(workspace, "gfx950", ("d128-block",))
        self.assertEqual(doctor["workloads"], ["d128-block"])
        self.assertIn("hip-moi", doctor["paths"])
        self.assertIn("hip-moi-build", doctor["paths"])
        self.assertNotIn("iree-test-suites", doctor["paths"])
        self.assertNotIn("iree-test-suites-build", doctor["paths"])
        self.assertEqual(doctor["tools"], {"rocminfo": "/tool"})

    def test_health_smoke_falls_back_to_ready_selected_workload(self) -> None:
        workload = validation.WORKLOAD_BY_ID["d128-block"]
        with temporary_root() as workspace:
            command = validation._health_smoke_command(
                workspace, "gfx950", workload, workspace / "health.json"
            )
        self.assertTrue(command[0].endswith("cdna4_d128_attention_block_test"))
        self.assertEqual(command[1], "--gtest_filter=HipMoiCdna4D128AttentionBlock.*")

    def test_profile_environment_scrubs_controls_and_relies_on_sync_defaults(
        self,
    ) -> None:
        workload = validation.WORKLOAD_BY_ID["streamk-arrival"]
        with mock.patch.dict(
            os.environ,
            {
                "RJ_CONSAN_MAX_PATCHES": "1",
                "RJ_CONSAN_TMP_VGPR": "99",
                "HSA_TOOLS_LIB": "/stale/hook.so",
                "HSA_TOOLS_ROCPROFILER_V1_TOOLS": "0",
            },
            clear=False,
        ):
            environment = validation._clean_environment(
                "record-replay", workload, Path("/new/hook.so")
            )
        self.assertNotIn("RJ_CONSAN_MAX_PATCHES", environment)
        self.assertNotIn("RJ_CONSAN_TMP_VGPR", environment)
        self.assertEqual(environment["HSA_TOOLS_LIB"], "/new/hook.so")
        self.assertNotIn("HSA_TOOLS_ROCPROFILER_V1_TOOLS", environment)
        self.assertEqual(environment["RJ_CONSAN_MODE"], "record-replay")
        self.assertEqual(environment["RJ_CONSAN_POLICY"], "strict")
        self.assertNotIn("RJ_CONSAN_FLAVOR", environment)
        self.assertNotIn("RJ_CONSAN_MOI_ENGINE", environment)
        self.assertNotIn("RJ_CONSAN_MOI_TRACK_BARRIERS", environment)
        self.assertNotIn("RJ_CONSAN_MOI_TRACK_ATOMICS", environment)
        self.assertEqual(
            validation.ORDINARY_MOI_RUNTIME_DEFAULTS,
            {
                "RJ_CONSAN_MOI_TRACK_BARRIERS": "1",
                "RJ_CONSAN_MOI_TRACK_ATOMICS": "1",
            },
        )

    def test_pytorch_profile_enables_environment_hsa_tool_loading(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-torch-histc"]
        with mock.patch.dict(
            os.environ,
            {"HSA_TOOLS_ROCPROFILER_V1_TOOLS": "0"},
            clear=False,
        ):
            baseline = validation._clean_environment(
                None, workload, Path("/hook.so")
            )
            environment = validation._clean_environment(
                "record-replay", workload, Path("/hook.so")
            )
        self.assertNotIn("HSA_TOOLS_ROCPROFILER_V1_TOOLS", baseline)
        self.assertEqual(environment["HSA_TOOLS_ROCPROFILER_V1_TOOLS"], "1")
        setting = validation._audited_settings(environment)
        v1_tool = next(
            item
            for item in setting
            if item["name"] == "HSA_TOOLS_ROCPROFILER_V1_TOOLS"
        )
        self.assertEqual(v1_tool["category"], "runtime-plumbing")
        self.assertFalse(v1_tool["usability_exception"])

    def test_supercollider_does_not_receive_moi_tracking_controls(self) -> None:
        workload = validation.WORKLOAD_BY_ID["streamk-arrival"]
        environment = validation._clean_environment(
            "supercollider", workload, Path("/hook.so")
        )
        self.assertNotIn("RJ_CONSAN_MOI_TRACK_BARRIERS", environment)
        self.assertNotIn("RJ_CONSAN_MOI_TRACK_ATOMICS", environment)

    def test_scatter_disables_strict_record_requirement_for_inapplicable_lds(
        self,
    ) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-scatter-reduce"]
        for profile in ("record-replay", "inline-shadow"):
            environment = validation._clean_environment(
                profile, workload, Path("/hook.so")
            )
            self.assertEqual(environment["RJ_CONSAN_MOI_REQUIRE_RECORDS"], "0")

    def test_qwen_sampled_relies_on_the_standard_runtime_operating_point(self) -> None:
        qwen = validation.WORKLOAD_BY_ID["qwen-prefill"]
        tp1 = validation.WORKLOAD_BY_ID["tp1-prefill"]
        qwen_environment = validation._clean_environment(
            "sampled", qwen, Path("/hook.so")
        )
        tp1_environment = validation._clean_environment(
            "sampled", tp1, Path("/hook.so")
        )
        self.assertEqual(qwen_environment["RJ_CONSAN_MOI_REQUIRE_RECORDS"], "1")
        self.assertNotIn("RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE", qwen_environment)
        self.assertNotIn("RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET", qwen_environment)
        self.assertNotIn("RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE", tp1_environment)

    def test_qwen_gfx1250_overhead_uses_a_software_backend_median(self) -> None:
        qwen = validation.WORKLOAD_BY_ID["qwen-prefill"]
        output = Path("/tmp/qwen-overhead.json")
        gfx1250 = validation._workload_command(
            Path("/workspace"), "gfx1250", qwen, "overhead", output
        )
        gfx1201 = validation._workload_command(
            Path("/workspace"), "gfx1201", qwen, "overhead", output
        )
        self.assertIn("--benchmark_repetitions=1", gfx1250)
        self.assertIn("--benchmark_repetitions=10", gfx1201)

    def test_sharktank_gfx1250_overhead_uses_one_inner_repetition(self) -> None:
        tp1 = validation.WORKLOAD_BY_ID["tp1-prefill"]
        command = validation._workload_command(
            Path("/workspace"), "gfx1250", tp1, "overhead", Path("/unused")
        )
        self.assertEqual(command[command.index("--repetitions") + 1], "1")

    def test_pytorch_gfx1250_runs_both_variants_once(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-tdm-descriptor-add"]
        with mock.patch.dict(
            os.environ,
            {validation.PYTORCH_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            command = validation._workload_command(
                Path("/workspace"),
                "gfx1250",
                workload,
                "overhead",
                Path("/unused"),
            )
        self.assertEqual(command[0], "/workspace/venv/bin/python")
        self.assertEqual(command[command.index("--repetitions") + 1], "1")
        self.assertEqual(command[command.index("--workload") + 1], "tdm-descriptor-add")

    def test_pytorch_python_discovers_standard_workspace_environment(self) -> None:
        with temporary_root() as workspace:
            python = workspace / "consan-pytorch-venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            with mock.patch.dict(
                os.environ, {validation.PYTORCH_PYTHON_ENV: ""}, clear=False
            ):
                self.assertEqual(validation._pytorch_python(workspace), python)

    def test_pytorch_python_explicit_environment_overrides_workspace(self) -> None:
        with mock.patch.dict(
            os.environ,
            {validation.PYTORCH_PYTHON_ENV: "/custom/venv/bin/python"},
            clear=False,
        ):
            self.assertEqual(
                validation._pytorch_python(Path("/workspace")),
                Path("/custom/venv/bin/python"),
            )

    def test_pytorch_doctor_rejects_interpreter_with_broken_imports(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-rdna4-compiled-softmax"]
        with temporary_root() as workspace:
            python = workspace / "consan-pytorch-venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            completed = subprocess.CompletedProcess(
                [str(python)], 1, stdout="", stderr="No module named torch"
            )
            with (
                mock.patch.object(validation.shutil, "which", return_value="/tool"),
                mock.patch.object(
                    validation.subprocess, "run", return_value=completed
                ) as run,
            ):
                doctor = validation._doctor(workspace, "gfx1201", (workload.id,))
        self.assertFalse(doctor["ok"])
        self.assertFalse(doctor["runtimes"]["pytorch"]["ok"])
        self.assertIn("No module named torch", doctor["runtimes"]["pytorch"]["detail"])
        self.assertEqual(run.call_args.args[0][0], str(python))
        environment = run.call_args.kwargs["env"]
        self.assertEqual(environment["HSA_TOOLS_ROCPROFILER_V1_TOOLS"], "1")
        self.assertEqual(
            environment["RJ_CONSAN_TEST_KERNEL_FILTER"],
            "__consan_pytorch_runtime_probe_never_matches__",
        )

    def test_pytorch_doctor_rejects_runtime_that_skips_consan_hook(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-rdna4-compiled-softmax"]
        with temporary_root() as workspace:
            python = workspace / "consan-pytorch-venv" / "bin" / "python"
            hook = (
                workspace
                / "rocjitsu-build/lib/rocjitsu/src/rocjitsu/hooks/"
                "librocjitsu_dbi_hooks.so"
            )
            python.parent.mkdir(parents=True)
            hook.parent.mkdir(parents=True)
            python.touch()
            hook.touch()
            completed = subprocess.CompletedProcess(
                [str(python)],
                0,
                stdout=json.dumps(
                    {
                        "torch": "test",
                        "hip": "test",
                        "triton": "test",
                        "device": "test GPU",
                        "arch": "gfx1201",
                        "numeric_oracle": True,
                        "hook_loaded": False,
                    }
                ),
                stderr="",
            )
            with (
                mock.patch.object(validation.shutil, "which", return_value="/tool"),
                mock.patch.object(
                    validation.subprocess, "run", return_value=completed
                ),
            ):
                doctor = validation._doctor(workspace, "gfx1201", (workload.id,))
        runtime = doctor["runtimes"]["pytorch"]
        self.assertFalse(doctor["ok"])
        self.assertFalse(runtime["ok"])
        self.assertIn(
            "PyTorch HSA runtime did not load the ConSan hook", runtime["reasons"]
        )

    def test_pytorch_cluster_workload_runs_once(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-cluster-load-sync"]
        with mock.patch.dict(
            os.environ,
            {validation.PYTORCH_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            command = validation._workload_command(
                Path("/workspace"),
                "gfx1250",
                workload,
                "clean",
                Path("/unused"),
            )
        self.assertEqual(command[command.index("--repetitions") + 1], "1")
        self.assertEqual(command[command.index("--workload") + 1], "cluster-load-sync")

    def test_pytorch_rdna4_softmax_uses_native_compiled_client(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-rdna4-compiled-softmax"]
        with mock.patch.dict(
            os.environ,
            {validation.PYTORCH_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            command = validation._workload_command(
                Path("/workspace"),
                "gfx1201",
                workload,
                "clean",
                Path("/unused"),
            )
        self.assertEqual(command[0], "/workspace/venv/bin/python")
        self.assertEqual(command[command.index("--repetitions") + 1], "1")
        self.assertEqual(
            command[command.index("--workload") + 1], "rdna4-compiled-softmax"
        )

    def test_pytorch_rdna4_llm_topk_uses_native_client(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"]
        with mock.patch.dict(
            os.environ,
            {validation.PYTORCH_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            command = validation._workload_command(
                Path("/workspace"),
                "gfx1201",
                workload,
                "clean",
                Path("/unused"),
            )
        self.assertEqual(command[0], "/workspace/venv/bin/python")
        self.assertEqual(command[command.index("--repetitions") + 1], "1")
        self.assertEqual(command[command.index("--workload") + 1], "rdna4-llm-topk")

    def test_pytorch_rdna4_sdpa_uses_native_client(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-rdna4-sdpa"]
        with mock.patch.dict(
            os.environ,
            {validation.PYTORCH_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            command = validation._workload_command(
                Path("/workspace"),
                "gfx1201",
                workload,
                "clean",
                Path("/unused"),
            )
        self.assertEqual(command[0], "/workspace/venv/bin/python")
        self.assertEqual(command[command.index("--repetitions") + 1], "1")
        self.assertEqual(command[command.index("--workload") + 1], "rdna4-sdpa")

    def test_tensile_gfx1250_uses_numeric_runner_once(self) -> None:
        workload = validation.WORKLOAD_BY_ID["tensile-sk-mxf8gemm-explicit"]
        with mock.patch.dict(
            os.environ,
            {validation.TENSILE_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            command = validation._workload_command(
                Path("/workspace"),
                "gfx1250",
                workload,
                "clean",
                Path("/artifacts/benchmark.json"),
            )
        self.assertEqual(command[0], "/workspace/venv/bin/python")
        self.assertTrue(command[1].endswith("consan_tensile_validation.py"))
        self.assertEqual(command[command.index("--repetitions") + 1], "1")
        self.assertEqual(command[command.index("--config") + 1], workload.relative_path)
        self.assertEqual(
            command[command.index("--output-dir") + 1],
            "/artifacts/tensile-work",
        )

    def test_gfx1250_manifest_registers_f16_sb_tensile_closure(self) -> None:
        manifest = validation._manifest("gfx1250")
        workloads = {workload["id"]: workload for workload in manifest["workloads"]}
        workload = workloads["tensile-spmm-f16-sb"]
        self.assertEqual(workload["priority"], "P2")
        self.assertEqual(workload["kind"], "tensile")
        self.assertEqual(
            workload["relative_path"],
            (
                "corpus/tensile/configs/Tensile/Tests/common/sparse/gfx1250/"
                "spmm_f16_sb.yaml"
            ),
        )
        self.assertEqual(workload["fault_families"], ("barrier-drop",))

    def test_f16_sb_tensile_closure_uses_exact_runner_once(self) -> None:
        workload = validation.WORKLOAD_BY_ID["tensile-spmm-f16-sb"]
        with mock.patch.dict(
            os.environ,
            {validation.TENSILE_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            command = validation._workload_command(
                Path("/workspace"),
                "gfx1250",
                workload,
                "clean",
                Path("/artifacts/benchmark.json"),
            )
        self.assertEqual(command[0], "/workspace/venv/bin/python")
        self.assertTrue(command[1].endswith("consan_tensile_validation.py"))
        self.assertEqual(command[command.index("--repetitions") + 1], "1")
        self.assertEqual(command[command.index("--config") + 1], workload.relative_path)
        self.assertEqual(
            command[command.index("--output-dir") + 1],
            "/artifacts/tensile-work",
        )

    def test_pytorch_json_reports_independent_variant_medians(self) -> None:
        document = {
            "one-cta": {"median_ms": 4.0, "oracle_passed": True},
            "two-cta-cluster": {"median_ms": 7.0, "oracle_passed": True},
        }
        self.assertEqual(
            validation._json_medians(json.dumps(document), "Pytorch"),
            {"one-cta": 4.0, "two-cta-cluster": 7.0},
        )

    def test_single_qwen_iteration_is_its_median(self) -> None:
        with temporary_root() as root:
            result = root / "benchmark.json"
            result.write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {
                                "name": "BM_main/process_time/real_time",
                                "run_type": "iteration",
                                "repetitions": 1,
                                "real_time": 12.5,
                                "time_unit": "ms",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(validation._benchmark_median(result), 12.5)

    def test_explain_expands_commands_and_marks_only_real_tuning(self) -> None:
        audit = validation._explain_contract(
            Path("/workspace"),
            "gfx1201",
            ("qwen-prefill",),
            validation.PROFILE_IDS,
            None,
            allow_reference=False,
        )
        workload = audit["workloads"][0]
        expected = validation._workload_command(
            Path("/workspace"),
            "gfx1201",
            validation.WORKLOAD_BY_ID["qwen-prefill"],
            "clean",
            Path("$ARTIFACT_ROOT/qwen-prefill/clean/$PROFILE/benchmark-0.json"),
        )
        self.assertEqual(workload["commands"]["clean"]["payload_argv"], expected)
        settings = {
            item["name"]: item
            for profile in workload["profiles"]
            for item in profile["settings"]
        }
        self.assertNotIn("CTEST_PARALLEL_LEVEL", settings)
        for forbidden in validation.ORDINARY_FORBIDDEN_ENVIRONMENT:
            self.assertNotIn(forbidden, settings)
        sampled = next(
            profile for profile in workload["profiles"] if profile["id"] == "sampled"
        )
        self.assertEqual(
            {item["name"] for item in sampled["implicit_runtime_defaults"]},
            {
                "RJ_CONSAN_MOI_TRACK_BARRIERS",
                "RJ_CONSAN_MOI_TRACK_ATOMICS",
                "RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET",
                "RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE",
            },
        )
        self.assertEqual(sampled["usability_exceptions"], [])
        self.assertEqual(
            audit["usability_audit"]["coverage_limiting_controls_present"], []
        )
        self.assertEqual(
            audit["usability_audit"]["explicit_event_family_overrides"], []
        )
        self.assertEqual(audit["usability_audit"]["workload_specific_tuning"], [])
        sampled_defaults = next(
            item
            for item in audit["usability_audit"]["automatic_profile_defaults"]
            if item["profile"] == "sampled"
        )
        self.assertEqual(
            sampled_defaults["settings"]["RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE"],
            "16384",
        )

    def test_explain_audits_reference_fault_outcomes_and_trial_knobs(self) -> None:
        path = Path(__file__).with_name(
            "consan_validation_faults_gfx1201_reference.json"
        )
        audit = validation._explain_contract(
            Path("/workspace"),
            "gfx1201",
            ("qwen-prefill",),
            validation.PROFILE_IDS,
            path,
            allow_reference=True,
        )
        fault = audit["workloads"][0]["faults"][0]
        sampled = next(
            item
            for item in fault["profile_expectations"]
            if item["profile"] == "sampled"
        )
        self.assertEqual(sampled["detector"], "statistical")
        self.assertEqual(sampled["oracle"], "fail")
        self.assertEqual(sampled["trial_count"], 32)
        self.assertEqual(
            sampled["trials"][0]["overrides"][0]["name"],
            "RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET",
        )
        self.assertIn("at least 1", sampled["required_diagnostic"])
        inline = next(
            item
            for item in fault["profile_expectations"]
            if item["profile"] == "inline-shadow"
        )
        effective = {
            setting["name"]: setting["value"]
            for setting in inline["trials"][0]["effective_settings"]
        }
        implicit = {
            setting["name"]: setting["value"]
            for setting in inline["trials"][0]["implicit_runtime_defaults"]
        }
        self.assertEqual(effective["RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS"], "1")
        self.assertNotIn("RJ_CONSAN_MOI_FORBID_DIAGNOSTICS", effective)
        self.assertEqual(implicit, validation.ORDINARY_MOI_RUNTIME_DEFAULTS)
        self.assertIn("$FAULT_SPEC", fault["validator_argv_template"])

    def test_tp1_decode_row_does_not_repeat_prefill(self) -> None:
        workload = validation.WORKLOAD_BY_ID["tp1-decode-combined"]
        command = validation._workload_command(
            Path("/workspace"), "gfx1201", workload, "clean", Path("/unused")
        )
        self.assertEqual(command[command.index("--mode") + 1], "decode-combined")

    def test_gfx1250_d128_fault_uses_fast_oracle_variant(self) -> None:
        workload = validation.WORKLOAD_BY_ID["d128-block"]
        command = validation._workload_command(
            Path("/workspace"), "gfx1250", workload, "fault", Path("/unused")
        )
        self.assertEqual(
            command[1],
            (
                "--gtest_filter=HipMoiGfx1250D128AttentionBlock."
                "SampledFastContextMatchesHostReference"
            ),
        )
        rdna_command = validation._workload_command(
            Path("/workspace"), "gfx1201", workload, "fault", Path("/unused")
        )
        self.assertEqual(
            rdna_command[1], "--gtest_filter=HipMoiRdna4D128AttentionBlock.*"
        )

    def test_overhead_uses_bracketing_baseline_mean_and_maximum_mode(self) -> None:
        results = [
            {"profile": "baseline", "timing_median_ms": {"a": 2.0, "b": 4.0}},
            {"profile": "sampled", "timing_median_ms": {"a": 6.0, "b": 10.0}},
            {"profile": "baseline", "timing_median_ms": {"a": 4.0, "b": 6.0}},
        ]
        summary = validation._overhead_summary(results)
        self.assertEqual(summary["paired_baseline_median_ms"], {"a": 3.0, "b": 5.0})
        self.assertEqual(summary["profiles"]["sampled"]["cell_slowdown"], 2.0)

    def test_inventory_parser_deduplicates_exact_identities(self) -> None:
        output = "\n".join(
            (
                "ConSan fault site reader=1 identity=site-a kind=barrier "
                "sync_sequence=sequence-b",
                "ConSan fault site reader=2 identity=site-a kind=barrier",
                "ConSan sync sequence reader=1 identity=sequence-a kind=barrier",
                "ConSan barrier destination reader=1 identity=destination-a container=k",
            )
        )
        self.assertEqual(
            validation._inventory_records(output),
            {
                "sites": ["site-a"],
                "sequences": ["sequence-a", "sequence-b"],
                "destinations": ["destination-a"],
            },
        )

    def test_inventory_completion_requires_relevant_site_then_matching_coverage(
        self,
    ) -> None:
        unrelated = "\n".join(
            (
                "ConSan fault site reader=7 identity=a kind=atomic container=k",
                "ConSan coverage reader=7 analysis_complete=true",
            )
        )
        self.assertFalse(
            validation._inventory_collection_complete(unrelated, "barrier-drop")
        )
        wrong_reader = "\n".join(
            (
                "ConSan fault site reader=7 identity=a kind=barrier container=k",
                "ConSan coverage reader=8 analysis_complete=true",
            )
        )
        self.assertFalse(
            validation._inventory_collection_complete(wrong_reader, "barrier-drop")
        )
        complete = "\n".join(
            (
                "ConSan fault site reader=7 identity=a kind=barrier container=k",
                "ConSan sync sequence reader=7 identity=s kind=barrier",
                "ConSan coverage reader=7 analysis_complete=false",
            )
        )
        self.assertTrue(
            validation._inventory_collection_complete(complete, "barrier-drop")
        )

    def test_family_inventory_records_exclude_unrelated_sites(self) -> None:
        output = "\n".join(
            (
                "ConSan fault site reader=7 identity=h|kind=ordinary-memory|pc=1 "
                "kind=ordinary-memory sync_sequence=-",
                "ConSan fault site reader=7 identity=h|kind=barrier|pc=2 "
                "kind=barrier sync_sequence=h|event=barrier|pc=2",
                "ConSan sync sequence reader=7 identity=h|event=atomic|pc=3 kind=atomic",
            )
        )
        self.assertEqual(
            validation._inventory_records(output, "barrier-drop"),
            {
                "sites": ["h|kind=barrier|pc=2"],
                "sequences": ["h|event=barrier|pc=2"],
                "destinations": [],
            },
        )

    def test_atomic_inventory_completion_rejects_barrier_only_reader(self) -> None:
        output = "\n".join(
            (
                "ConSan fault site reader=3 identity=a kind=barrier container=k",
                "ConSan coverage reader=3 analysis_complete=true",
            )
        )
        self.assertFalse(
            validation._inventory_collection_complete(output, "atomic-weaken-order")
        )

    def test_inventory_runner_stops_after_static_collection(self) -> None:
        program = "; ".join(
            (
                "import time",
                "print('ConSan fault site reader=9 identity=a kind=barrier container=k', flush=True)",
                "print('ConSan coverage reader=9 analysis_complete=false', flush=True)",
                "time.sleep(30)",
            )
        )
        with temporary_root() as root:
            returncode, elapsed, output, complete, outcome = (
                validation._run_inventory_process(
                    [sys.executable, "-c", program],
                    os.environ.copy(),
                    root / "inventory.log",
                    5,
                    "barrier-drop",
                )
            )
        self.assertLess(elapsed, 3)
        self.assertNotEqual(returncode, 0)
        self.assertIn("ConSan coverage reader=9", output)
        self.assertTrue(complete)
        self.assertEqual(outcome, "static-inventory-complete")

    def test_inventory_runner_rejects_timeout_before_matching_coverage(self) -> None:
        program = "; ".join(
            (
                "import time",
                "print('ConSan fault site reader=9 identity=a kind=barrier container=k', flush=True)",
                "time.sleep(30)",
            )
        )
        with temporary_root() as root:
            returncode, elapsed, output, complete, outcome = (
                validation._run_inventory_process(
                    [sys.executable, "-c", program],
                    os.environ.copy(),
                    root / "inventory.log",
                    1,
                    "barrier-drop",
                )
            )
        self.assertLess(elapsed, 3)
        self.assertEqual(returncode, 124)
        self.assertIn("validation timeout after 1s", output)
        self.assertFalse(complete)
        self.assertEqual(outcome, "timeout")

    def test_fault_parser_accepts_paired_health_command_overrides(self) -> None:
        args = validation._parse_args(
            [
                "fault",
                "--workload",
                "jakub-attention",
                "--spec",
                "/tmp/spec.json",
                "--fault",
                "barrier-drop",
                "--artifact-root",
                "/tmp/artifacts",
                "--health-command-json",
                '["/bin/true"]',
                "--smoke-command-json",
                '["/tmp/smoke", "--short"]',
            ]
        )
        self.assertEqual(args.health_command_json, ["/bin/true"])
        self.assertEqual(args.smoke_command_json, ["/tmp/smoke", "--short"])

    def test_fault_parser_rejects_unpaired_health_command_override(self) -> None:
        with self.assertRaises(SystemExit):
            validation._parse_args(
                [
                    "fault",
                    "--workload",
                    "jakub-attention",
                    "--spec",
                    "/tmp/spec.json",
                    "--fault",
                    "barrier-drop",
                    "--artifact-root",
                    "/tmp/artifacts",
                    "--health-command-json",
                    '["/bin/true"]',
                ]
            )

    def test_marker_smoke_stops_after_independent_success_marker(self) -> None:
        script = Path(__file__).with_name("consan_marker_smoke.py")
        program = "; ".join(
            (
                "import time",
                "print('checked result: PASS', flush=True)",
                "time.sleep(30)",
            )
        )
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--success-marker",
                "checked result: PASS",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                program,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("checked result: PASS", result.stdout)

    def test_marker_smoke_rejects_natural_exit_without_marker(self) -> None:
        script = Path(__file__).with_name("consan_marker_smoke.py")
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--success-marker",
                "PASS",
                "--",
                sys.executable,
                "-c",
                "print('FAIL')",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 1)

    def test_fault_inventory_enables_family_analysis_without_a_selector(self) -> None:
        barrier = validation._fault_inventory_environment("barrier-drop")
        self.assertEqual(barrier, {"RJ_CONSAN_FAULT_DROP_BARRIER": "1"})
        atomic = validation._fault_inventory_environment("atomic-weaken-order")
        self.assertEqual(
            atomic,
            {
                "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER": "1",
                "RJ_CONSAN_FAULT_ATOMIC_ORDER_EDGE": "release",
            },
        )
        self.assertFalse(any(name.endswith("_IDENTITY") for name in atomic))

    def test_fault_template_matches_target_barrier_geometry(self) -> None:
        workload = validation.WORKLOAD_BY_ID["d128-block"]
        rdna_fault = validation._fault_template("gfx1201", workload)["faults"][0]
        cdna_fault = validation._fault_template("gfx950", workload)["faults"][0]
        self.assertIn(
            "RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY",
            rdna_fault["environment"],
        )
        self.assertNotIn(
            "RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY",
            cdna_fault["environment"],
        )
        self.assertEqual(
            cdna_fault["environment"]["RJ_CONSAN_FAULT_SITE_IDENTITY"],
            "REPLACE_FROM_INVENTORY",
        )

    def test_fault_acceptance_rejects_an_unattributed_process_signal(self) -> None:
        accepted, reasons = validation._fault_acceptance(
            {
                "mutation": {"requested": 1, "planned": 1, "applied": 1},
                "sanitizer": {"outcome": "not_detected"},
                "oracle": {"outcome": "pass"},
                "execution": {
                    "outcome": "signal",
                    "timed_out": False,
                    "health_before": {"healthy": True},
                    "health_after": {"healthy": True},
                },
            },
            {"detector": "not_detected", "oracle": "pass"},
        )
        self.assertFalse(accepted)
        self.assertIn("invalid execution outcome=signal", reasons)

    def test_fault_spec_requires_target_workload_and_exact_mutation(self) -> None:
        workload = validation.WORKLOAD_BY_ID["d128-block"]
        document = {
            "schema_version": 1,
            "target": "gfx1201",
            "workload": workload.id,
            "review_required": False,
            "faults": [
                {
                    "id": "drop",
                    "family": "barrier-drop",
                    "environment": {
                        "RJ_CONSAN_FAULT_DROP_BARRIER": "1",
                        "RJ_CONSAN_FAULT_SITE_IDENTITY": "site-a",
                        "RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY": "sequence-a",
                    },
                }
            ],
        }
        with temporary_root() as root:
            path = root / "faults.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = validation._load_fault(path, "gfx1201", workload, "drop")
        self.assertEqual(loaded["id"], "drop")

    def test_checked_in_gfx1201_fault_reference_is_a_valid_manifest_subset(self) -> None:
        path = Path(__file__).with_name(
            "consan_validation_faults_gfx1201_reference.json"
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(validation.ValidationError, "reference-only"):
            validation._load_fault(
                path,
                "gfx1201",
                validation.WORKLOAD_BY_ID["qwen-prefill"],
                "barrier-drop",
            )
        manifest_ids = {
            workload.id for workload in validation._workloads_for_target("gfx1201")
        }
        self.assertLessEqual(set(document["workloads"]), manifest_ids)
        for workload_id, workload_document in document["workloads"].items():
            workload = validation.WORKLOAD_BY_ID[workload_id]
            for fault_document in workload_document["faults"]:
                fault_id = fault_document["id"]
                fault = validation._load_fault(
                    path,
                    "gfx1201",
                    workload,
                    fault_id,
                    allow_reference=True,
                )
                for profile in validation.PROFILE_IDS:
                    policy, trials = validation._fault_trials(fault, profile)
                    self.assertTrue(trials)
                    self.assertIn(
                        policy.get("detector"),
                        {"detected", "not_detected", "statistical"},
                    )
        qwen = validation.WORKLOAD_BY_ID["qwen-prefill"]
        fault = validation._load_fault(
            path,
            "gfx1201",
            qwen,
            "barrier-drop",
            allow_reference=True,
        )
        policy, trials = validation._fault_trials(fault, "sampled")
        self.assertEqual(policy["minimum_detections"], 1)
        self.assertEqual(
            policy["environment"]["RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE"],
            "256",
        )
        self.assertEqual(len(trials), 32)
        self.assertEqual(trials[0], {"RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET": "0"})
        self.assertEqual(trials[-1], {"RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET": "31"})


if __name__ == "__main__":
    unittest.main()
