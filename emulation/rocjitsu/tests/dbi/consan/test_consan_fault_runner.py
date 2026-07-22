#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

import consan_fault_runner as runner
from consan_validation_support import RESULT_SCHEMA_VERSION, read_row_result
from consan_validation_test_support import temporary_root
from test_consan_coverage_gate import coverage as coverage_line, log, verdict


RUNNER = Path(__file__).with_name("consan_fault_runner.py")


class ConSanFaultRunnerTest(unittest.TestCase):
    def run_runner(
        self, root: Path, *args: str, parallel: str = "4"
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["CTEST_PARALLEL_LEVEL"] = parallel
        return subprocess.run(
            [sys.executable, str(RUNNER), "--artifact-root", str(root), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
        )

    def test_records_completing_row(self) -> None:
        with temporary_root() as root:
            completed = self.run_runner(
                root,
                "--name",
                "clean",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                "print('row output')",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = read_row_result(root, "clean")
            self.assertEqual(result["outcome"], "passed")
            self.assertEqual(result["ctest_parallel_level"], 4)
            self.assertEqual(result["schema_version"], RESULT_SCHEMA_VERSION)
            self.assertEqual(result["execution"]["outcome"], "passed")
            self.assertEqual(result["sanitizer"]["outcome"], "not_exercised")
            self.assertEqual(result["oracle"]["outcome"], "not_run")
            self.assertEqual(result["source_diagnostics"]["outcome"], "unknown")
            self.assertEqual(result["spec"]["row_role"], "unspecified")
            self.assertIn("git_revisions", result)
            self.assertEqual(result["stdout_stderr"], "command.log")
            self.assertIn("row output", (root / "clean" / "command.log").read_text())

    def test_retains_invalid_patch_outcome(self) -> None:
        parsed = runner._parse_consan_log(
            "[rocjitsu-dbi-hooks] ConSan patch end reader=7 visited=true "
            "modified=false outcome=invalid errors=1 warnings=2 patches=0"
        )
        self.assertEqual(parsed["coverage"]["patch_outcomes"], [{
            "reader": "7", "outcome": "invalid", "errors": 1,
            "warnings": 2, "patches": 0,
        }])

    def test_supercollider_marker_is_value_instability_diagnosis(self) -> None:
        parsed = runner._parse_consan_log(
            "\n".join((
                "[rocjitsu-dbi-hooks] ConSan SC auto report reader=7 "
                "outcome=complete marker=1 mismatch=true",
                "[rocjitsu-dbi-hooks] ConSan SC report summary buffers=1 "
                "mismatches=1 allocation_failures=0 read_failures=0 "
                "cleanup_failures=0 complete=true",
            ))
        )
        self.assertEqual(parsed["sanitizer"]["outcome"], "detected")
        self.assertEqual(parsed["sanitizer"]["supercollider_mismatches"], 1)
        self.assertEqual(parsed["sanitizer"]["supercollider_diagnostics"], 1)
        self.assertEqual(parsed["sanitizer"]["measured_instability_count"], 1)
        self.assertEqual(parsed["sanitizer"]["diagnostic_count"], 1)
        self.assertEqual(parsed["metrics"]["sc_report_buffer_count"], 1)
        self.assertEqual(parsed["metrics"]["sc_report_buffer_bytes"], 4)
        self.assertTrue(parsed["coverage"]["readers"][0]["supercollider_mismatch"])

    def test_retains_analysis_incomplete_verdict(self) -> None:
        parsed = runner._parse_consan_log(log(
            coverage_line(
                analysis_complete="false", access_supported="19",
                access_unsupported="1", access_patched="19",
            ),
            verdict(
                analysis_complete="false", static_complete="false",
                incomplete_code_objects="1", access="19/19",
            ),
        ))
        self.assertFalse(parsed["coverage"]["analysis_complete"])
        self.assertFalse(parsed["coverage"]["analysis_verdict"]["analysis_complete"])

    def test_illegal_shader_instruction_is_classified_as_trap(self) -> None:
        with temporary_root() as root:
            completed = self.run_runner(
                root, "--name", "illegal-instruction", "--timeout", "5", "--",
                sys.executable, "-c",
                "import sys; print('HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION: illegal shader instruction'); sys.exit(1)",
            )
            self.assertEqual(completed.returncode, 1)
            result = read_row_result(root, "illegal-instruction")
            self.assertEqual(result["execution"]["outcome"], "trap")
            self.assertEqual(result["sanitizer"]["trap_attribution"], "unattributed")

    def test_recognizes_iree_expected_output_oracle_without_result_file(self) -> None:
        with temporary_root() as root:
            cases = {
                "pass": "[SUCCESS] all function outputs matched their expected values.",
                "fail": "[FAILED] result[0]: element at index 0 does not match",
            }
            for expected, marker in cases.items():
                completed = self.run_runner(
                    root,
                    "--name",
                    expected,
                    "--timeout",
                    "5",
                    "--",
                    sys.executable,
                    "-c",
                    f"print({marker!r})",
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = read_row_result(root, expected)
                self.assertEqual(result["oracle"]["outcome"], expected)
                self.assertEqual(
                    result["oracle"]["source"], "iree_expected_output_log"
                )

    def test_recognizes_gtest_assertion_oracle_without_result_file(self) -> None:
        with temporary_root() as root:
            cases = {
                "pass": "[  PASSED  ] 1 test.",
                "fail": (
                    "[  PASSED  ] 0 tests.\n"
                    "[  FAILED  ] 1 test, listed below:"
                ),
            }
            for expected, marker in cases.items():
                completed = self.run_runner(
                    root,
                    "--name",
                    f"gtest-{expected}",
                    "--timeout",
                    "5",
                    "--",
                    sys.executable,
                    "-c",
                    f"print({marker!r})",
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = read_row_result(root, f"gtest-{expected}")
                self.assertEqual(result["oracle"]["outcome"], expected)
                self.assertEqual(result["oracle"]["source"], "gtest_assertions")

    def test_accepts_inventory_and_baseline_row_roles(self) -> None:
        with temporary_root() as root:
            for role in ("inventory", "baseline"):
                completed = self.run_runner(
                    root,
                    "--name",
                    role,
                    "--row-role",
                    role,
                    "--timeout",
                    "5",
                    "--",
                    sys.executable,
                    "-c",
                    "print('row output')",
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = read_row_result(root, role)
                self.assertEqual(result["spec"]["row_role"], role)

    def test_serialized_row_records_lock_and_command_only_timing(self) -> None:
        with temporary_root() as root:
            probe = [sys.executable, "-c", "import time; time.sleep(0.1)"]
            completed = self.run_runner(
                root,
                "--name",
                "serialized",
                "--timeout",
                "5",
                "--serialize-gpu",
                "--health-command-json",
                json.dumps(probe),
                "--smoke-command-json",
                json.dumps(probe),
                "--",
                "/bin/true",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = read_row_result(root, "serialized")
            self.assertTrue(result["gpu_serialized"])
            self.assertEqual(
                result["gpu_lock"], runner.DEFAULT_GLOBAL_DESTRUCTIVE_LOCK
            )
            self.assertTrue(result["health_before"]["healthy"])
            self.assertTrue(result["health_after"]["healthy"])
            self.assertGreater(result["metrics"]["elapsed_seconds"], 0.35)
            self.assertLess(result["metrics"]["command_elapsed_seconds"], 0.2)

    def test_rejects_unpaired_health_probe_contract(self) -> None:
        with temporary_root() as root:
            completed = self.run_runner(
                root,
                "--name",
                "unpaired",
                "--timeout",
                "5",
                "--health-command-json",
                json.dumps(["/bin/true"]),
                "--",
                "/bin/true",
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("must be provided together", completed.stderr)

    def test_records_manifest_and_replays_it(self) -> None:
        with temporary_root() as root:
            input_path = root / "input.co"
            input_path.write_bytes(b"code object")
            completed = self.run_runner(
                root,
                "--name",
                "manifested",
                "--pair-id",
                "atomic-handoff-1",
                "--row-role",
                "fault",
                "--corpus",
                "hip-moi",
                "--workload",
                "atomic-add",
                "--flavor",
                "moi",
                "--engine",
                "inline_shadow",
                "--fault-family",
                "wrong-address",
                "--timeout",
                "5",
                "--serialize-gpu",
                "--env",
                "RJ_TEST_MANIFEST=value",
                "--hash-file",
                f"input={input_path}",
                "--site-id",
                "object:symbol:0x10",
                "--",
                sys.executable,
                "-c",
                "import json,os,pathlib; "
                "pathlib.Path(os.environ['CONSAN_ROW_RESULT_PATH']).write_text("
                "json.dumps({'schema_version':1,'oracle':'pass','detail':'matched'})); "
                "print('[rocjitsu-dbi-hooks] ConSan proof patch anchor=0x10')",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest_path = root / "manifested" / "result.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["environment"]["RJ_TEST_MANIFEST"], "value")
            self.assertEqual(manifest["spec"]["pair_id"], "atomic-handoff-1")
            self.assertEqual(manifest["spec"]["row_role"], "fault")
            self.assertEqual(manifest["spec"]["flavor"], "moi")
            self.assertEqual(manifest["spec"]["engine"], "inline_shadow")
            self.assertEqual(manifest["oracle"]["outcome"], "pass")
            self.assertEqual(manifest["oracle"]["detail"], "matched")
            self.assertTrue(manifest["gpu_serialized"])
            self.assertEqual(manifest["site_identities"], ["object:symbol:0x10"])
            self.assertEqual(
                manifest["hashes_before"][0]["sha256"],
                manifest["hashes_after"][0]["sha256"],
            )
            self.assertEqual(len(manifest["patch_inventory"]), 1)

            replay = subprocess.run(
                [sys.executable, str(RUNNER), "replay", str(manifest_path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(replay.returncode, 0, replay.stderr)
            replayed = read_row_result(root, "replays", "manifested-replay")
            self.assertEqual(replayed["schema_version"], RESULT_SCHEMA_VERSION)
            self.assertEqual(replayed["command"], manifest["command"])
            self.assertEqual(replayed["environment"]["RJ_TEST_MANIFEST"], "value")
            self.assertEqual(replayed["spec"], manifest["spec"])
            self.assertEqual(replayed["oracle"]["outcome"], "pass")
            self.assertTrue(replayed["gpu_serialized"])
            self.assertNotEqual(
                replayed["environment"]["CONSAN_ROW_RESULT_PATH"],
                manifest["environment"]["CONSAN_ROW_RESULT_PATH"],
            )

            legacy_manifest = dict(manifest)
            legacy_manifest["schema_version"] = 2
            legacy_manifest.pop("spec")
            legacy_manifest.pop("oracle")
            legacy_manifest["environment"] = dict(legacy_manifest["environment"])
            legacy_manifest["environment"].pop("CONSAN_ROW_RESULT_PATH")
            legacy_path = root / "legacy-result.json"
            legacy_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
            legacy_replay = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "replay",
                    str(legacy_path),
                    "--artifact-root",
                    str(root / "replays"),
                    "--name",
                    "manifested-v2-replay",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(legacy_replay.returncode, 0, legacy_replay.stderr)
            legacy_replayed = read_row_result(root, "replays", "manifested-v2-replay")
            self.assertEqual(legacy_replayed["schema_version"], RESULT_SCHEMA_VERSION)
            self.assertEqual(legacy_replayed["spec"]["row_role"], "unspecified")
            self.assertEqual(legacy_replayed["spec"]["flavor"], "unspecified")
            self.assertEqual(legacy_replayed["oracle"]["outcome"], "pass")

    def test_oracle_file_is_explicit_and_malformed_data_is_unknown(self) -> None:
        with temporary_root() as root:
            oracle_failure = self.run_runner(
                root,
                "--name",
                "oracle-failure",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                "import json,os,pathlib; "
                "pathlib.Path(os.environ['CONSAN_ROW_RESULT_PATH']).write_text("
                "json.dumps({'schema_version':1,'oracle':'fail','detail':{'index':3}}))",
            )
            self.assertEqual(oracle_failure.returncode, 0, oracle_failure.stderr)
            failed_result = read_row_result(root, "oracle-failure")
            self.assertEqual(failed_result["execution"]["outcome"], "passed")
            self.assertEqual(failed_result["oracle"]["outcome"], "fail")
            self.assertEqual(failed_result["oracle"]["detail"], {"index": 3})

            malformed = self.run_runner(
                root,
                "--name",
                "malformed-oracle",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                "import os,pathlib; "
                "pathlib.Path(os.environ['CONSAN_ROW_RESULT_PATH']).write_text('{')",
            )
            self.assertEqual(malformed.returncode, 0, malformed.stderr)
            malformed_result = read_row_result(root, "malformed-oracle")
            self.assertEqual(malformed_result["oracle"]["outcome"], "unknown")
            self.assertEqual(malformed_result["oracle"]["source"], "malformed")

    def test_source_diagnostics_are_explicit_and_separate_from_consan_logs(self) -> None:
        with temporary_root() as root:
            explicit = self.run_runner(
                root,
                "--name",
                "explicit-source",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                "import json,os,pathlib; "
                "pathlib.Path(os.environ['CONSAN_ROW_RESULT_PATH']).write_text("
                "json.dumps({'schema_version':1,'oracle':'pass','source_diagnostics':"
                "{'count':3,'expectation':'nonzero'}})); "
                "print('[  FAILED  ] unrelated gtest text')",
            )
            self.assertEqual(explicit.returncode, 0, explicit.stderr)
            result = read_row_result(root, "explicit-source")
            self.assertEqual(result["source_diagnostics"]["outcome"], "matched")
            self.assertEqual(result["source_diagnostics"]["count"], 3)

            for name, source in (
                ("malformed-source-count", {"count": "3", "expectation": "nonzero"}),
                ("malformed-source-expectation", {"count": 3, "expectation": "positive"}),
            ):
                malformed = self.run_runner(
                    root,
                    "--name",
                    name,
                    "--timeout",
                    "5",
                    "--",
                    sys.executable,
                    "-c",
                    "import json,os,pathlib; "
                    "pathlib.Path(os.environ['CONSAN_ROW_RESULT_PATH']).write_text("
                    f"json.dumps({{'schema_version':1,'oracle':'pass','source_diagnostics':{source!r}}}))",
                )
                self.assertEqual(malformed.returncode, 0, malformed.stderr)
                malformed_result = read_row_result(root, name)
                self.assertEqual(malformed_result["source_diagnostics"]["outcome"], "unknown")

            text_only = self.run_runner(
                root,
                "--name",
                "text-only-source",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                "print('[  FAILED  ] source diagnostics=7'); "
                "print('[rocjitsu-dbi-hooks] ConSan MOI auto report reader=10 "
                "visible_diagnostics=4')",
            )
            self.assertEqual(text_only.returncode, 0, text_only.stderr)
            result = read_row_result(root, "text-only-source")
            self.assertEqual(result["source_diagnostics"]["outcome"], "unknown")
            self.assertEqual(result["sanitizer"]["diagnostic_count"], 4)

    def test_summarizes_every_conservative_pair_classification(self) -> None:
        with temporary_root() as root:

            def write_row(
                pair_id: str,
                role: str,
                *,
                oracle: str = "pass",
                sanitizer: str = "not_detected",
                requested: int = 0,
                applied: int = 0,
                applicability: str = "not_requested",
                overflowed: bool = False,
                execution_outcome: str = "passed",
                completed: bool = True,
                source_outcome: str = "unknown",
            ) -> None:
                row_dir = root / f"{pair_id}-{role}"
                row_dir.mkdir()
                manifest = {
                    "schema_version": 3,
                    "state": "complete",
                    "name": row_dir.name,
                    "spec": {
                        "pair_id": pair_id,
                        "row_role": role,
                        "corpus": "focused",
                        "workload": "handoff",
                        "flavor": "moi",
                        "engine": "inline_shadow",
                        "fault_family": "atomic",
                    },
                    "execution": {
                        "outcome": execution_outcome,
                        "completed": completed,
                    },
                    "oracle": {"outcome": oracle},
                    "source_diagnostics": {"outcome": source_outcome},
                    "mutation": {
                        "requested": requested,
                        "applied": applied,
                        "applicability": applicability,
                    },
                    "sanitizer": {"outcome": sanitizer},
                    "coverage": {"overflowed": overflowed},
                }
                (row_dir / "result.json").write_text(json.dumps(manifest), encoding="utf-8")

            cases = {
                "detected": "qualified_detected",
                "undetected": "qualified_undetected",
                "not-applied": "fault_not_applied",
                "clean-failed": "clean_control_failed",
                "unsupported": "unsupported",
                "overflow": "overflowed",
                "indeterminate": "indeterminate_execution",
                "calibration": "source_calibration_failed",
                "source-positive": "qualified_detected",
            }
            for pair_id in cases:
                write_row(
                    pair_id,
                    "clean",
                    oracle="fail" if pair_id == "clean-failed" else "pass",
                    source_outcome=(
                        "mismatch" if pair_id == "calibration" else
                        "matched" if pair_id == "source-positive" else "unknown"
                    ),
                )
                fault_options = {
                    "requested": 1,
                    "applied": 1,
                    "applicability": "applied",
                    "oracle": "fail",
                    "sanitizer": "detected" if pair_id == "detected" else "not_detected",
                    "overflowed": pair_id == "overflow",
                    "execution_outcome": "timeout" if pair_id == "indeterminate" else "passed",
                    "completed": pair_id != "indeterminate",
                }
                if pair_id == "source-positive":
                    fault_options.update(sanitizer="detected", source_outcome="matched")
                if pair_id == "not-applied":
                    fault_options.update(applied=0, applicability="planned_not_applied")
                if pair_id == "unsupported":
                    fault_options.update(applied=0, applicability="not_applicable")
                write_row(pair_id, "fault", **fault_options)

            legacy_dir = root / "legacy"
            legacy_dir.mkdir()
            (legacy_dir / "result.json").write_text(
                json.dumps({"schema_version": 2, "state": "complete", "name": "legacy"}),
                encoding="utf-8",
            )
            json_out = root / "summary.json"
            csv_out = root / "summary.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "summarize",
                    str(root),
                    "--json-out",
                    str(json_out),
                    "--csv-out",
                    str(csv_out),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary, json.loads(json_out.read_text()))
            classifications = {
                group["pair_id"]: group["classification"] for group in summary["groups"]
            }
            groups_by_pair = {group["pair_id"]: group for group in summary["groups"]}
            for pair_id, expected in cases.items():
                self.assertEqual(classifications[pair_id], expected)
            self.assertEqual(
                groups_by_pair["unsupported"]["mutation_applicability"], "not_applicable"
            )
            self.assertEqual(
                groups_by_pair["not-applied"]["mutation_applicability"],
                "planned_not_applied",
            )
            self.assertEqual(classifications["unspecified"], "indeterminate_execution")
            self.assertEqual(summary["manifest_count"], 19)
            with csv_out.open(newline="", encoding="utf-8") as source:
                csv_rows = list(csv.DictReader(source))
            self.assertEqual(len(csv_rows), 10)
            self.assertEqual(
                {row["pair_id"]: row["classification"] for row in csv_rows}, classifications
            )

    def test_rejects_unsafe_parallel_level(self) -> None:
        with temporary_root() as root:
            completed = self.run_runner(
                root,
                "--name",
                "unsafe",
                "--timeout",
                "5",
                "--",
                "/bin/true",
                parallel="5",
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("from 1 through 4", completed.stderr)

    def test_destructive_mode_requires_explicit_opt_in(self) -> None:
        with temporary_root() as root:
            completed = self.run_runner(
                root,
                "--name",
                "destructive",
                "--timeout",
                "5",
                "--destructive",
                "--",
                "/bin/true",
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("requires --allow-destructive", completed.stderr)

    def test_incomplete_barrier_drop_opt_in_requires_destructive_containment(self) -> None:
        with temporary_root() as root:
            completed = self.run_runner(
                root,
                "--name",
                "unsafe-incomplete-drop",
                "--timeout",
                "5",
                "--env",
                "RJ_CONSAN_FAULT_ALLOW_DESTRUCTIVE_INCOMPLETE_BARRIER_DROP=1",
                "--",
                "/bin/true",
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("requires --destructive containment", completed.stderr)

    def test_divergent_barrier_move_opt_in_requires_destructive_containment(self) -> None:
        with temporary_root() as root:
            completed = self.run_runner(
                root,
                "--name",
                "unsafe-divergent-move",
                "--timeout",
                "5",
                "--env",
                "RJ_CONSAN_FAULT_ALLOW_DESTRUCTIVE_DIVERGENT_BARRIER_MOVE=1",
                "--",
                "/bin/true",
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("requires --destructive containment", completed.stderr)

    def test_timeout_kills_complete_process_group(self) -> None:
        with temporary_root() as root:
            child_survived = root / "child-survived"
            child_code = (
                "import pathlib,time; time.sleep(0.5); "
                f"pathlib.Path({str(child_survived)!r}).touch()"
            )
            started = time.monotonic()
            completed = self.run_runner(
                root,
                "--name",
                "timeout",
                "--timeout",
                "0.1",
                "--destructive",
                "--allow-destructive",
                "--health-command-json",
                json.dumps(["/bin/true"]),
                "--smoke-command-json",
                json.dumps(["/bin/true"]),
                "--",
                sys.executable,
                "-c",
                "import subprocess,time; "
                f"subprocess.Popen([{sys.executable!r}, '-c', {child_code!r}]); "
                "time.sleep(30)",
            )
            self.assertEqual(completed.returncode, 124, completed.stderr)
            self.assertLess(time.monotonic() - started, 5)
            result = read_row_result(root, "timeout")
            self.assertEqual(result["outcome"], "timeout")
            self.assertTrue(result["timed_out"])
            self.assertFalse(result["execution"]["completed"])
            self.assertNotEqual(result["sanitizer"]["outcome"], "detected")
            self.assertTrue((root / ".gpu-quarantine.json").is_file())
            time.sleep(0.6)
            self.assertFalse(child_survived.exists())

    def test_signal_is_typed_separately_from_timeout_and_failure(self) -> None:
        with temporary_root() as root:
            completed = self.run_runner(
                root,
                "--name",
                "signal",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                "import os,signal; os.kill(os.getpid(), signal.SIGTERM)",
            )
            self.assertEqual(completed.returncode, 256 - 15)
            result = read_row_result(root, "signal")
            self.assertEqual(result["execution"]["outcome"], "signal")
            self.assertEqual(result["execution"]["signal"], 15)
            self.assertFalse(result["execution"]["timed_out"])
            self.assertNotEqual(result["sanitizer"]["outcome"], "detected")

    def test_preflight_probe_timeout_quarantines_without_launching_row(self) -> None:
        with temporary_root() as root:
            launched = root / "launched"
            completed = self.run_runner(
                root,
                "--name",
                "unhealthy-before",
                "--timeout",
                "5",
                "--health-timeout",
                "0.1",
                "--destructive",
                "--allow-destructive",
                "--health-command-json",
                json.dumps([sys.executable, "-c", "import time; time.sleep(30)"]),
                "--smoke-command-json",
                json.dumps(["/bin/true"]),
                "--",
                sys.executable,
                "-c",
                f"import pathlib; pathlib.Path({str(launched)!r}).touch()",
            )
            self.assertEqual(completed.returncode, 70, completed.stderr)
            result = read_row_result(root, "unhealthy-before")
            self.assertEqual(result["execution"]["outcome"], "preflight_device_unhealthy")
            self.assertFalse(result["execution"]["command_ran"])
            self.assertTrue(result["health_before"]["rocminfo_timed_out"])
            self.assertFalse(launched.exists())
            self.assertTrue((root / ".gpu-quarantine.json").is_file())

    def test_health_smoke_cannot_overwrite_workload_oracle(self) -> None:
        with temporary_root() as root:
            workload = (
                "import os,pathlib;"
                "pathlib.Path(os.environ['CONSAN_WORKLOAD_RESULT_PATH']).write_text("
                "'{\"schema_version\":1,\"oracle\":\"pass\",\"detail\":{\"source\":\"row\"}}')"
            )
            smoke = [
                sys.executable,
                "-c",
                "import os; "
                "assert 'CONSAN_WORKLOAD_RESULT_PATH' not in os.environ; "
                "assert 'HSA_TOOLS_LIB' not in os.environ; "
                "assert 'RJ_CONSAN_MOI_REQUIRE_RECORDS' not in os.environ",
            ]
            completed = self.run_runner(
                root,
                "--name",
                "isolated-oracle",
                "--timeout",
                "5",
                "--destructive",
                "--allow-destructive",
                "--health-command-json",
                json.dumps(["/bin/true"]),
                "--smoke-command-json",
                json.dumps(smoke),
                "--env",
                f"CONSAN_WORKLOAD_RESULT_PATH={root / 'isolated-oracle' / 'oracle.json'}",
                "--env",
                "HSA_TOOLS_LIB=/does/not/exist.so",
                "--env",
                "RJ_CONSAN_MOI_REQUIRE_RECORDS=1",
                "--",
                sys.executable,
                "-c",
                workload,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = read_row_result(root, "isolated-oracle")
            self.assertEqual(result["oracle"]["detail"]["source"], "row")

    def test_parses_consan_qualification_evidence_and_metrics(self) -> None:
        with temporary_root() as root:
            script = "\n".join(
                [
                    "import os, pathlib",
                    "dump = pathlib.Path(os.environ['RJ_CONSAN_DUMP_DIR'])",
                    "(dump / 'rj-dbi-000001-reader-7-original.hsaco').write_bytes(b'o' * 10)",
                    "(dump / 'rj-dbi-000001-reader-7-patched.hsaco').write_bytes(b'p' * 26)",
                    "print('[rocjitsu-dbi-hooks] ConSan fault summary process=123 reader=7 requested=1 planned=1 applied=1 require_exactly_one=true')",
                    "print('[rocjitsu-dbi-hooks] ConSan fault site reader=7 identity=site kind=atomic')",
                    "print('[rocjitsu-dbi-hooks] ConSan proof patch reader=7 kind=inline-atomic-address-rewrite anchor=0x10')",
                    "print('[rocjitsu-dbi-hooks] ConSan proof patch reader=7 kind=trampoline-moi-atomic-record anchor=0x20 spilled_vgprs=2 private_bytes=96 workgroup_shadow_bytes=128 group_bytes=512')",
                    "print('[rocjitsu-dbi-hooks] ConSan patch end reader=7 patches=2')",
                    "print('[rocjitsu-dbi-hooks] ConSan summary reader=7 supported_lds_sites=3 function_supported_lds_sites=2 skips=4 rejects=1')",
                    "print('[rocjitsu-dbi-hooks] ConSan coverage_site reader=7 kind=atomic disposition=supported reason=none outcome=patched lowering_reason=none resource_reason=none container=atomic_kernel scope=kernel text=0x20 mnemonic=global_atomic_add')",
                    "print('[rocjitsu-dbi-hooks] ConSan coverage_site reader=7 kind=access disposition=unsupported reason=unsupported_mnemonic outcome=unsupported lowering_reason=semantic_unsupported resource_reason=none container=unsupported_helper scope=function text=0x30 mnemonic=ds_load_b96')",
                    "print('[rocjitsu-dbi-hooks] ConSan MOI resources reader=7 emitted_spill_patches=2 emitted_spill_slot_bytes=24')",
                    "print('[rocjitsu-dbi-hooks] ConSan MOI auto report buffer skipped reader=6: no MOI report sites')",
                    "print('[rocjitsu-dbi-hooks] ConSan MOI auto report plan reader=7 outcome=complete reason=none required_bytes=4096 cap_bytes=16777216 per_buffer_ceiling=16777216 process_ceiling=268435456 access_ranges=5 barriers=2 atomics=4 fences=1 diagnostics=4 sampled_banks=3 sampled_watchpoints=3 inline_lds_bytes=128 inline_releases=64 inline_snapshots=64 inline_tokens=64')",
                    "print('[rocjitsu-dbi-hooks] ConSan MOI auto report buffer reader=7 bytes=4096 required_bytes=4096 cap_bytes=16777216 process_current_bytes=4096 process_peak_bytes=4096 process_ceiling_bytes=268435456 allocation_outcome=allocated access_record_capacity=5 barrier_record_capacity=2 atomic_record_capacity=4 fence_record_capacity=1 exact_shadow_entry_capacity=32 diagnostic_capacity=4 inline_atomic_release_capacity=64 inline_acquired_epoch_token_capacity=64 inline_causal_snapshot_capacity=64 sampled_watchpoint_capacity=3 sampled_causal_window_capacity=6 sampled_sync_metadata_capacity=6 sampled_pending_acquire_capacity=6')",
                    "print('[rocjitsu-dbi-hooks] ConSan MOI auto report reader=7 visible_records=5 dropped_records=1 visible_barriers=2 dropped_barriers=0 visible_atomics=4 dropped_atomics=0 visible_diagnostics=1 dropped_diagnostics=2 visible_exact_shadow=7 exact_incomplete_snapshots=5 exact_changed_snapshots=6 exact_malformed_snapshots=7 inline_undercoverage=8 inline_overflow=9 inline_unsupported=10 inline_malformed=11 visible_inline_atomic_releases=3 visible_inline_acquired_tokens=0 release_incomplete_snapshots=0 release_changed_snapshots=0 release_overflow_snapshots=0 release_source_incomplete_snapshots=0 release_malformed_snapshots=0 token_incomplete_snapshots=0 token_changed_snapshots=0 token_malformed_snapshots=0 visible_sampled=3 sampled_conflicts=1 sampled_immediate_conflicts=2 sampled_claimed_windows=3 sampled_dropped_windows=1 sampled_stale_snapshots=1 sampled_incomplete_snapshots=2 sampled_changed_snapshots=3 sampled_malformed_snapshots=4')",
                    "print('[rocjitsu-dbi-hooks] ConSan MOI auto inline-atomic-release reader=7 index=0 version=2 owner=1 epoch_plus_one=2 workgroup=3 address=0x4000 dispatch=0x5000 snapshot_count=1 snapshot_flags=0 snapshot0_owner=4 snapshot0_epoch_plus_one=5')",
                    "print('[rocjitsu-dbi-hooks] ConSan MOI auto inline-atomic-release reader=7 index=1 version=4 owner=2 epoch_plus_one=3 workgroup=3 address=0x4010 dispatch=0x5000 snapshot_count=0 snapshot_flags=0')",
                    "print('[rocjitsu-dbi-hooks] ConSan MOI auto inline-atomic-release reader=7 index=2 version=6 owner=3 epoch_plus_one=4 workgroup=3 address=0x4020 dispatch=0x5000 snapshot_count=0 snapshot_flags=0')",
                    "print('[rocjitsu-dbi-hooks] ConSan MOI auto sampled reader=7 index=0 kind=1')",
                    "print('[rocjitsu-dbi-hooks] ConSan MOI auto sampled reader=7 index=1 kind=2')",
                    "print('[rocjitsu-dbi-hooks] ConSan MOI auto sampled reader=7 index=2 kind=3')",
                    "print('[rocjitsu-dbi-hooks] ConSan MOI auto replay reader=7 diagnostics=3')",
                    "print('[rocjitsu-dbi-hooks] ConSan MOI report memory required_bytes=4096 allocated_bytes=4096 live_before_cleanup=4096 live_after_cleanup=0 peak_live_bytes=4096 per_buffer_ceiling=16777216 process_ceiling=268435456 allocation_failures=0 capacity_failures=0 cleanup_failures=0')",
                ]
            )
            completed = self.run_runner(
                root,
                "--name",
                "evidence",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                script,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = read_row_result(root, "evidence")
            self.assertEqual(result["mutation"]["applied"], 1)
            self.assertEqual(result["mutation"]["applicability"], "applied")
            self.assertEqual(result["mutation"]["inventoried_sites"], 1)
            self.assertEqual(result["mutation"]["accounting_schema_version"], 1)
            self.assertEqual(result["mutation"]["applied_readers"], 1)
            self.assertEqual(result["mutation"]["fault_patch_applications"], 1)
            self.assertEqual(
                result["mutation"]["fault_patch_kinds"],
                ["inline-atomic-address-rewrite"],
            )
            self.assertTrue(result["mutation"]["process_evidence_complete"])
            self.assertEqual(
                result["mutation"]["processes"],
                [
                    {
                        "process": "123",
                        "requested": 1,
                        "planned": 1,
                        "applied": 1,
                        "readers": [
                            {
                                "reader": "7",
                                "requested": 1,
                                "planned": 1,
                                "applied": 1,
                            }
                        ],
                    }
                ],
            )
            self.assertEqual(result["sanitizer"]["outcome"], "detected")

            self.assertEqual(result["sanitizer"]["diagnostic_count"], 7)
            self.assertEqual(result["coverage"]["supported_sites"], 5)
            self.assertTrue(result["coverage"]["site_dispositions_complete"])
            self.assertIsNone(result["coverage"]["site_disposition_parse_error"])
            self.assertEqual(
                result["coverage"]["site_dispositions"],
                [
                    {
                        "reader": 7,
                        "load": None,
                        "kind": "atomic",
                        "disposition": "supported",
                        "reason": "none",
                        "outcome": "patched",
                        "lowering_reason": "none",
                        "resource_reason": "none",
                        "container": "atomic_kernel",
                        "scope": "kernel",
                        "text_offset": 0x20,
                        "mnemonic": "global_atomic_add",
                    },
                    {
                        "reader": 7,
                        "load": None,
                        "kind": "access",
                        "disposition": "unsupported",
                        "reason": "unsupported_mnemonic",
                        "outcome": "unsupported",
                        "lowering_reason": "semantic_unsupported",
                        "resource_reason": "none",
                        "container": "unsupported_helper",
                        "scope": "function",
                        "text_offset": 0x30,
                        "mnemonic": "ds_load_b96",
                    },
                ],
            )
            self.assertEqual(result["coverage"]["instrumentation_patches"], 1)
            self.assertEqual(
                result["coverage"]["instrumentation_patch_kinds"],
                ["trampoline-moi-atomic-record"],
            )
            self.assertTrue(result["coverage"]["overflowed"])
            self.assertEqual(result["coverage"]["event_counts"]["atomic"], 4)
            self.assertEqual(result["coverage"]["event_counts"]["inline_atomic_release"], 3)
            self.assertEqual(result["coverage"]["selected_watchpoints"], 3)
            self.assertEqual(result["coverage"]["sampled_claimed_windows"], 3)
            self.assertEqual(result["coverage"]["reader_access_events"], 2)
            self.assertEqual(result["coverage"]["writer_access_events"], 2)
            reader = result["coverage"]["readers"][0]
            self.assertEqual(reader["reader"], "7")
            self.assertEqual(reader["patches"], 2)
            self.assertEqual(reader["event_counts"]["inline_atomic_release"], 3)
            self.assertEqual(reader["event_counts"]["inline_acquired_token"], 0)
            self.assertEqual(
                reader["inline_evidence_capacities"],
                {"release": 64, "snapshot": 64, "token": 64},
            )
            self.assertEqual(len(reader["inline_release_evidence"]), 3)
            self.assertEqual(
                reader["inline_release_evidence"][0]["snapshot"],
                [{"owner": 4, "epoch_plus_one": 5}],
            )
            self.assertEqual(reader["inline_evidence_counts"]["malformed_records"], 0)
            self.assertEqual(reader["inline_evidence_counts"]["duplicate_records"], 0)
            self.assertEqual(reader["inline_evidence_counts"]["count_mismatches"], 0)
            self.assertEqual(reader["inline_evidence_counts"]["capacity_violations"], 0)
            self.assertTrue(reader["overflowed"])
            self.assertEqual(
                result["coverage"]["sampled_snapshot_counts"],
                {"stale": 1, "incomplete": 2, "changed": 3, "malformed": 4},
            )
            self.assertEqual(
                result["coverage"]["exact_snapshot_counts"],
                {"incomplete": 5, "changed": 6, "malformed": 7},
            )
            self.assertEqual(
                result["coverage"]["inline_coverage_counts"],
                {
                    "undercoverage": 8,
                    "overflow": 9,
                    "unsupported": 10,
                    "malformed": 11,
                },
            )
            self.assertEqual(result["metrics"]["report_buffer_bytes"], 4096)
            self.assertEqual(result["metrics"]["report_buffer_count"], 1)
            self.assertEqual(result["metrics"]["report_plan_count"], 1)
            self.assertEqual(
                result["metrics"]["report_plans"][0],
                {
                    "reader": "7",
                    "outcome": "complete",
                    "reason": "none",
                    "required_bytes": 4096,
                    "cap_bytes": 16777216,
                    "per_buffer_ceiling": 16777216,
                    "process_ceiling": 268435456,
                    "access_ranges": 5,
                    "barriers": 2,
                    "atomics": 4,
                    "fences": 1,
                    "diagnostics": 4,
                    "sampled_banks": 3,
                    "sampled_watchpoints": 3,
                    "inline_lds_bytes": 128,
                    "inline_releases": 64,
                    "inline_snapshots": 64,
                    "inline_tokens": 64,
                },
            )
            self.assertEqual(
                result["metrics"]["report_region_capacity_entries"],
                {
                    "access": 5,
                    "barrier": 2,
                    "atomic": 4,
                    "fence": 1,
                    "diagnostic": 4,
                    "exact_shadow": 32,
                    "inline_atomic_release": 64,
                    "inline_acquired_token": 64,
                    "inline_causal_snapshot": 64,
                    "sampled_watchpoint": 3,
                    "sampled_causal_window": 6,
                    "sampled_sync_metadata": 6,
                    "sampled_pending_acquire": 6,
                },
            )
            self.assertEqual(result["metrics"]["report_memory_summary_count"], 1)
            self.assertEqual(result["metrics"]["report_required_bytes"], 4096)
            self.assertEqual(result["metrics"]["report_allocated_bytes"], 4096)
            self.assertEqual(result["metrics"]["report_live_before_cleanup_bytes"], 4096)
            self.assertEqual(result["metrics"]["report_live_after_cleanup_bytes"], 0)
            self.assertEqual(result["metrics"]["report_peak_live_bytes"], 4096)
            self.assertEqual(result["metrics"]["report_allocation_failures"], 0)
            self.assertEqual(result["metrics"]["report_capacity_failures"], 0)
            self.assertEqual(result["metrics"]["report_cleanup_failures"], 0)
            self.assertEqual(result["metrics"]["shadow_capacity_entries"], 32)
            self.assertEqual(result["metrics"]["diagnostic_capacity_entries"], 4)
            self.assertEqual(result["metrics"]["inline_atomic_release_capacity_entries"], 64)
            self.assertEqual(result["metrics"]["inline_acquired_token_capacity_entries"], 64)
            self.assertEqual(result["metrics"]["inline_causal_snapshot_capacity_entries"], 64)
            self.assertEqual(result["metrics"]["spill_slot_bytes"], 24)
            self.assertEqual(result["metrics"]["private_segment_bytes"], 96)
            self.assertEqual(result["metrics"]["workgroup_shadow_bytes"], 128)
            self.assertEqual(result["metrics"]["group_segment_bytes"], 512)
            self.assertEqual(reader["spilled_vgpr_count"], 2)
            self.assertEqual(reader["private_segment_bytes"], 96)
            self.assertEqual(reader["workgroup_shadow_bytes"], 128)
            self.assertEqual(reader["group_segment_bytes"], 512)
            self.assertEqual(result["metrics"]["modified_code_object_count"], 1)
            self.assertEqual(result["metrics"]["code_growth_bytes"], 16)

    def test_strict_site_disposition_parse_failure_is_retained(self) -> None:
        parsed = runner._parse_consan_log(
            "[rocjitsu-dbi-hooks] ConSan coverage_site reader=7 kind=access "
            "disposition=supported reason=none outcome=resource_failed "
            "lowering_reason=unsupported_resource_plan resource_reason=none "
            "container=kernel scope=kernel text=0x10 mnemonic=ds_load_b32"
        )
        self.assertFalse(parsed["coverage"]["site_dispositions_complete"])
        self.assertEqual(parsed["coverage"]["site_dispositions"], [])
        self.assertIn(
            "resource_failed requires a detailed resource reason",
            parsed["coverage"]["site_disposition_parse_error"],
        )

        with temporary_root() as root:
            completed = self.run_runner(
                root,
                "--name",
                "malformed-site-evidence",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                "print('[rocjitsu-dbi-hooks] ConSan coverage_site reader=7 "
                "kind=access disposition=supported reason=none "
                "outcome=resource_failed lowering_reason=unsupported_resource_plan "
                "resource_reason=none container=kernel scope=kernel text=0x10 "
                "mnemonic=ds_load_b32')",
            )
            self.assertEqual(completed.returncode, 1, completed.stderr)
            result = read_row_result(root, "malformed-site-evidence")
            self.assertEqual(result["outcome"], "evidence_incomplete")
            self.assertFalse(result["coverage"]["evidence_complete"])
            self.assertFalse(result["coverage"]["site_dispositions_complete"])
            self.assertIn(
                "resource_failed requires a detailed resource reason",
                result["coverage"]["site_disposition_parse_error"],
            )

    def test_static_malformed_barrier_abort_is_a_typed_diagnostic(self) -> None:
        with temporary_root() as root:
            completed = self.run_runner(
                root,
                "--name",
                "malformed-wait",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                "print('[rocjitsu-dbi-hooks] ConSan proof patch reader=7 "
                "kind=inline-malformed-barrier-abort anchor=0x10')",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = read_row_result(root, "malformed-wait")
            self.assertEqual(result["sanitizer"]["outcome"], "detected")
            self.assertEqual(result["sanitizer"]["diagnostic_count"], 1)
            self.assertEqual(result["sanitizer"]["static_diagnostics"], 1)
            self.assertEqual(
                result["coverage"]["instrumentation_patch_kinds"],
                ["inline-malformed-barrier-abort"],
            )

    def test_retains_complete_inline_release_snapshot_and_token_evidence(self):
        parsed = runner._parse_consan_log(
            "\n".join(
                (
                    "[rocjitsu-dbi-hooks] ConSan MOI auto report buffer reader=7 "
                    "bytes=4096 allocation_outcome=allocated inline_atomic_release_capacity=2 "
                    "inline_causal_snapshot_capacity=2 "
                    "inline_acquired_epoch_token_capacity=2",
                    "[rocjitsu-dbi-hooks] ConSan MOI auto report reader=7 "
                    "visible_inline_atomic_releases=1 visible_inline_acquired_tokens=2 "
                    "release_incomplete_snapshots=0 release_changed_snapshots=0 "
                    "release_overflow_snapshots=0 release_source_incomplete_snapshots=0 "
                    "release_malformed_snapshots=0 token_incomplete_snapshots=0 "
                    "token_changed_snapshots=0 token_malformed_snapshots=0",
                    "[rocjitsu-dbi-hooks] ConSan MOI auto inline-atomic-release reader=7 "
                    "index=0 version=4 owner=2 epoch_plus_one=7 workgroup=0x30 "
                    "address=0x4000 dispatch=0x500000006 snapshot_count=1 "
                    "snapshot_flags=0 snapshot0_owner=9 snapshot0_epoch_plus_one=3 "
                    "snapshot1_owner=0 snapshot1_epoch_plus_one=0 "
                    "snapshot2_owner=0 snapshot2_epoch_plus_one=0 "
                    "snapshot3_owner=0 snapshot3_epoch_plus_one=0",
                    "[rocjitsu-dbi-hooks] ConSan MOI auto inline-acquired-token reader=7 "
                    "index=0 version=2 kind=direct consumer=4 producer=2 "
                    "epoch_plus_one=7 workgroup=0x30 dispatch=0x500000006 "
                    "source_address=0x4000 source_version=4",
                    "[rocjitsu-dbi-hooks] ConSan MOI auto inline-acquired-token reader=7 "
                    "index=1 version=6 kind=inherited consumer=4 producer=9 "
                    "epoch_plus_one=3 workgroup=0x30 dispatch=0x500000006 "
                    "source_address=0x4000 source_version=4",
                )
            )
        )
        coverage = parsed["coverage"]
        self.assertFalse(coverage["overflowed"])
        self.assertEqual(coverage["event_counts"]["inline_acquired_token"], 2)
        self.assertEqual(
            [token["kind"] for token in coverage["inline_token_evidence"]],
            ["direct", "inherited"],
        )
        self.assertEqual(
            coverage["inline_release_evidence"][0]["snapshot"],
            [{"owner": 9, "epoch_plus_one": 3}],
        )
        self.assertEqual(coverage["inline_evidence_counts"]["malformed_records"], 0)
        self.assertEqual(coverage["inline_evidence_counts"]["duplicate_records"], 0)
        self.assertEqual(coverage["inline_evidence_counts"]["count_mismatches"], 0)
        self.assertEqual(coverage["inline_evidence_counts"]["capacity_violations"], 0)

    def test_inline_evidence_fails_closed_on_malformed_duplicate_and_capacity(self):
        parsed = runner._parse_consan_log(
            "\n".join(
                (
                    "[rocjitsu-dbi-hooks] ConSan MOI auto report buffer reader=8 "
                    "bytes=4096 allocation_outcome=allocated inline_atomic_release_capacity=1 "
                    "inline_causal_snapshot_capacity=1 "
                    "inline_acquired_epoch_token_capacity=1",
                    "[rocjitsu-dbi-hooks] ConSan MOI auto report reader=8 "
                    "visible_inline_atomic_releases=1 visible_inline_acquired_tokens=1 "
                    "token_changed_snapshots=1",
                    "[rocjitsu-dbi-hooks] ConSan MOI auto inline-atomic-release reader=8 "
                    "index=1 version=2 owner=1 epoch_plus_one=2 workgroup=3 "
                    "address=0x4000 dispatch=0x5000 snapshot_count=0 snapshot_flags=0",
                    "[rocjitsu-dbi-hooks] ConSan MOI auto inline-atomic-release reader=8 "
                    "index=1 version=2 owner=1 epoch_plus_one=2 workgroup=3 "
                    "address=0x4000 dispatch=0x5000 snapshot_count=0 snapshot_flags=0",
                    "[rocjitsu-dbi-hooks] ConSan MOI auto inline-atomic-release reader=8 "
                    "index=0 version=2 owner=1 epoch_plus_one=2 address=0x4000 "
                    "dispatch=0x5000 snapshot_count=0 snapshot_flags=0",
                    "[rocjitsu-dbi-hooks] ConSan MOI auto inline-acquired-token reader=8 "
                    "index=2 version=2 kind=direct consumer=2 producer=1 "
                    "epoch_plus_one=2 workgroup=3 dispatch=0x5000 "
                    "source_address=0x4000 source_version=2",
                )
            )
        )
        coverage = parsed["coverage"]
        self.assertTrue(coverage["overflowed"])
        self.assertEqual(coverage["inline_token_snapshot_counts"]["changed"], 1)
        self.assertGreaterEqual(coverage["inline_evidence_counts"]["malformed_records"], 1)
        self.assertEqual(coverage["inline_evidence_counts"]["duplicate_records"], 1)
        self.assertEqual(coverage["inline_evidence_counts"]["capacity_violations"], 2)

    def test_inline_tokens_require_exact_direct_or_inherited_release_provenance(self):
        parsed = runner._parse_consan_log(
            "\n".join(
                (
                    "[rocjitsu-dbi-hooks] ConSan MOI auto report buffer reader=9 "
                    "bytes=4096 allocation_outcome=allocated inline_atomic_release_capacity=2 "
                    "inline_causal_snapshot_capacity=2 "
                    "inline_acquired_epoch_token_capacity=2",
                    "[rocjitsu-dbi-hooks] ConSan MOI auto report reader=9 "
                    "visible_inline_atomic_releases=1 visible_inline_acquired_tokens=2 "
                    "release_incomplete_snapshots=0 release_changed_snapshots=0 "
                    "release_overflow_snapshots=0 release_source_incomplete_snapshots=0 "
                    "release_malformed_snapshots=0 token_incomplete_snapshots=0 "
                    "token_changed_snapshots=0 token_malformed_snapshots=0",
                    "[rocjitsu-dbi-hooks] ConSan MOI auto inline-atomic-release reader=9 "
                    "index=0 version=4 owner=2 epoch_plus_one=7 workgroup=0x30 "
                    "address=0x4000 dispatch=0x500000006 snapshot_count=1 "
                    "snapshot_flags=0 snapshot0_owner=9 snapshot0_epoch_plus_one=3 "
                    "snapshot1_owner=0 snapshot1_epoch_plus_one=0 "
                    "snapshot2_owner=0 snapshot2_epoch_plus_one=0 "
                    "snapshot3_owner=0 snapshot3_epoch_plus_one=0",
                    # A direct token must name the release owner/epoch, not an ancestor.
                    "[rocjitsu-dbi-hooks] ConSan MOI auto inline-acquired-token reader=9 "
                    "index=0 version=2 kind=direct consumer=4 producer=9 "
                    "epoch_plus_one=3 workgroup=0x30 dispatch=0x500000006 "
                    "source_address=0x4000 source_version=4",
                    # An inherited token must appear exactly in the immutable snapshot.
                    "[rocjitsu-dbi-hooks] ConSan MOI auto inline-acquired-token reader=9 "
                    "index=1 version=6 kind=inherited consumer=4 producer=8 "
                    "epoch_plus_one=3 workgroup=0x30 dispatch=0x500000006 "
                    "source_address=0x4000 source_version=4",
                )
            )
        )
        coverage = parsed["coverage"]
        self.assertTrue(coverage["overflowed"])
        self.assertEqual(coverage["inline_evidence_counts"]["malformed_records"], 2)

    def test_inline_summary_count_capacity_and_state_mismatches_fail_closed(self):
        parsed = runner._parse_consan_log(
            "\n".join(
                (
                    "[rocjitsu-dbi-hooks] ConSan MOI auto report buffer reader=10 "
                    "bytes=4096 allocation_outcome=allocated inline_atomic_release_capacity=1 "
                    "inline_causal_snapshot_capacity=1 "
                    "inline_acquired_epoch_token_capacity=1",
                    # Missing token_malformed_snapshots is an incomplete state summary;
                    # stable+changed releases also exceed the one-slot capacity.
                    "[rocjitsu-dbi-hooks] ConSan MOI auto report reader=10 "
                    "visible_inline_atomic_releases=1 visible_inline_acquired_tokens=2 "
                    "release_incomplete_snapshots=0 release_changed_snapshots=1 "
                    "release_overflow_snapshots=0 release_source_incomplete_snapshots=0 "
                    "release_malformed_snapshots=0 token_incomplete_snapshots=0 "
                    "token_changed_snapshots=0",
                    "[rocjitsu-dbi-hooks] ConSan MOI auto inline-atomic-release reader=10 "
                    "index=0 version=4 owner=2 epoch_plus_one=7 workgroup=0x30 "
                    "address=0x4000 dispatch=0x500000006 snapshot_count=0 "
                    "snapshot_flags=0",
                    "[rocjitsu-dbi-hooks] ConSan MOI auto inline-acquired-token reader=10 "
                    "index=0 version=2 kind=direct consumer=4 producer=2 "
                    "epoch_plus_one=7 workgroup=0x30 dispatch=0x500000006 "
                    "source_address=0x4000 source_version=4",
                )
            )
        )
        coverage = parsed["coverage"]
        self.assertTrue(coverage["overflowed"])
        self.assertEqual(coverage["inline_evidence_counts"]["count_mismatches"], 1)
        self.assertEqual(coverage["inline_evidence_counts"]["state_mismatches"], 2)

    def test_inline_report_cleanup_is_not_parsed_as_a_state_summary(self):
        parsed = runner._parse_consan_log(
            "\n".join(
                (
                    "[rocjitsu-dbi-hooks] ConSan MOI auto report buffer reader=10 "
                    "bytes=4096 allocation_outcome=allocated inline_atomic_release_capacity=1 "
                    "inline_causal_snapshot_capacity=1 "
                    "inline_acquired_epoch_token_capacity=1",
                    "[rocjitsu-dbi-hooks] ConSan MOI auto report reader=10 "
                    "visible_records=0 visible_barriers=0 visible_atomics=0 "
                    "visible_diagnostics=0 visible_exact_shadow=0 visible_sampled=0 "
                    "visible_inline_atomic_releases=0 visible_inline_acquired_tokens=0 "
                    "release_incomplete_snapshots=0 release_changed_snapshots=0 "
                    "release_overflow_snapshots=0 release_source_incomplete_snapshots=0 "
                    "release_malformed_snapshots=0 token_incomplete_snapshots=0 "
                    "token_changed_snapshots=0 token_malformed_snapshots=0",
                    "[rocjitsu-dbi-hooks] ConSan MOI auto report cleanup reader=10 "
                    "bytes=4096 outcome=freed status=0",
                )
            )
        )
        coverage = parsed["coverage"]
        self.assertFalse(coverage["overflowed"])
        self.assertEqual(coverage["inline_evidence_counts"]["state_mismatches"], 0)

    def test_fault_and_instrumentation_patches_have_separate_reader_accounting(self):
        parsed = runner._parse_consan_log(
            "\n".join(
                (
                    "[rocjitsu-dbi-hooks] ConSan fault summary reader=1 requested=1 planned=0 applied=1",
                    "[rocjitsu-dbi-hooks] ConSan proof patch reader=1 kind=inline-barrier-nop-rewrite anchor=0x10",
                    "[rocjitsu-dbi-hooks] ConSan proof patch reader=1 kind=trampoline-moi-access-record-store anchor=0x20",
                    "[rocjitsu-dbi-hooks] ConSan fault summary reader=2 requested=1 planned=0 applied=1",
                    "[rocjitsu-dbi-hooks] ConSan proof patch reader=2 kind=inline-barrier-nop-rewrite anchor=0x10",
                    "[rocjitsu-dbi-hooks] ConSan proof patch reader=2 kind=trampoline-moi-access-record-store anchor=0x20",
                    "[rocjitsu-dbi-hooks] ConSan fault load selection reader=1 site=stable matched=true requested_occurrence=1 observed_occurrence=1 selected=true overflow=false",
                    "[rocjitsu-dbi-hooks] ConSan fault load selection reader=2 site=stable matched=true requested_occurrence=1 observed_occurrence=2 selected=false overflow=false",
                    "[rocjitsu-dbi-hooks] ConSan fault load summary requested_occurrence=1 observed=2 selected=1 overflow=false accepted=true",
                )
            )
        )
        self.assertEqual(parsed["mutation"]["applied"], 2)
        self.assertEqual(parsed["mutation"]["applied_readers"], 2)
        self.assertEqual(parsed["mutation"]["fault_patch_applications"], 2)
        self.assertEqual(parsed["coverage"]["instrumentation_patches"], 2)
        self.assertEqual(
            parsed["mutation"]["load_selection"],
            {
                "schema_version": 1,
                "requested_occurrences": [1],
                "matching_readers": 2,
                "observed_occurrences": [1, 2],
                "readers": [
                    {
                        "reader": "1",
                        "matched": True,
                        "observed_occurrence": 1,
                        "selected": True,
                        "overflowed": False,
                    },
                    {
                        "reader": "2",
                        "matched": True,
                        "observed_occurrence": 2,
                        "selected": False,
                        "overflowed": False,
                    },
                ],
                "selected_loads": 1,
                "overflowed": False,
                "accepted_summaries": 1,
                "summary_count": 1,
            },
        )

    def test_retains_duplicate_fault_applications_within_one_process(self) -> None:
        with temporary_root() as root:
            script = "; ".join(
                [
                    "print('[rocjitsu-dbi-hooks] ConSan fault summary process=456 reader=7 requested=1 planned=1 applied=1')",
                    "print('[rocjitsu-dbi-hooks] ConSan fault summary process=456 reader=8 requested=1 planned=1 applied=1')",
                ]
            )
            completed = self.run_runner(
                root,
                "--name",
                "duplicate",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                script,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            mutation = read_row_result(root, "duplicate")["mutation"]
            self.assertEqual(mutation["applied"], 2)
            self.assertTrue(mutation["process_evidence_complete"])
            self.assertEqual(len(mutation["processes"]), 1)
            self.assertEqual(mutation["processes"][0]["applied"], 2)
            self.assertEqual(
                [reader["reader"] for reader in mutation["processes"][0]["readers"]],
                ["7", "8"],
            )

    def test_marks_fault_process_evidence_incomplete_without_process_id(self) -> None:
        with temporary_root() as root:
            completed = self.run_runner(
                root,
                "--name",
                "legacy",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                "print('[rocjitsu-dbi-hooks] ConSan fault summary reader=7 requested=1 planned=1 applied=1')",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            mutation = read_row_result(root, "legacy")["mutation"]
            self.assertFalse(mutation["process_evidence_complete"])
            self.assertEqual(mutation["processes"][0]["process"], "unspecified")

    def test_sampled_unusable_snapshot_is_incomplete_coverage(self) -> None:
        with temporary_root() as root:
            script = (
                "print('[rocjitsu-dbi-hooks] ConSan MOI auto report reader=7 "
                "visible_sampled=0 sampled_stale_snapshots=0 "
                "sampled_incomplete_snapshots=1 sampled_changed_snapshots=0 "
                "sampled_malformed_snapshots=0')"
            )
            completed = self.run_runner(
                root,
                "--name",
                "sampled-incomplete",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                script,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = read_row_result(root, "sampled-incomplete")
            self.assertTrue(result["coverage"]["overflowed"])
            self.assertEqual(
                result["coverage"]["overflow_counts"],
                {
                    "access": 0,
                    "barrier": 0,
                    "atomic": 0,
                    "diagnostic": 0,
                    "sampled": 0,
                },
            )
            self.assertEqual(result["coverage"]["sampled_snapshot_counts"]["incomplete"], 1)

    def test_workload_mismatch_and_nonzero_exit_are_not_sanitizer_detection(self) -> None:
        with temporary_root() as root:
            mismatch = self.run_runner(
                root,
                "--name",
                "mismatch",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                "print('output mismatch')",
            )
            self.assertEqual(mismatch.returncode, 1)
            mismatch_result = read_row_result(root, "mismatch")
            self.assertEqual(mismatch_result["execution"]["outcome"], "output_mismatch")
            self.assertNotEqual(mismatch_result["sanitizer"]["outcome"], "detected")

            failed = self.run_runner(
                root,
                "--name",
                "failed",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(7)",
            )
            self.assertEqual(failed.returncode, 7)
            failed_result = read_row_result(root, "failed")
            self.assertEqual(failed_result["execution"]["outcome"], "failed")
            self.assertNotEqual(failed_result["sanitizer"]["outcome"], "detected")

    def test_device_loss_quarantines_future_rows_until_healthy_clear(self) -> None:
        with temporary_root() as root:
            lost_marker = root / "device-lost"
            health_command = [
                sys.executable,
                "-c",
                f"import pathlib,sys; sys.exit(pathlib.Path({str(lost_marker)!r}).exists())",
            ]
            failed = self.run_runner(
                root,
                "--name",
                "loses-device",
                "--timeout",
                "5",
                "--destructive",
                "--allow-destructive",
                "--health-command-json",
                json.dumps(health_command),
                "--smoke-command-json",
                json.dumps(["/bin/true"]),
                "--",
                sys.executable,
                "-c",
                f"import pathlib; pathlib.Path({str(lost_marker)!r}).touch()",
            )
            self.assertEqual(failed.returncode, 70, failed.stderr)
            result = read_row_result(root, "loses-device")
            self.assertEqual(result["outcome"], "device_lost")
            self.assertNotEqual(result["sanitizer"]["outcome"], "detected")
            self.assertTrue((root / ".gpu-quarantine.json").is_file())

            refused = self.run_runner(
                root, "--name", "refused", "--timeout", "5", "--", "/bin/true"
            )
            self.assertEqual(refused.returncode, 75)
            environment = os.environ.copy()
            environment["CTEST_PARALLEL_LEVEL"] = "4"
            clear_command = [
                sys.executable,
                str(RUNNER),
                "clear-quarantine",
                "--artifact-root",
                str(root),
                "--health-command-json",
                json.dumps(health_command),
                "--smoke-command-json",
                json.dumps(["/bin/true"]),
            ]
            retained = subprocess.run(
                clear_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                check=False,
            )
            self.assertEqual(retained.returncode, 70)
            self.assertTrue((root / ".gpu-quarantine.json").is_file())

            lost_marker.unlink()
            cleared = subprocess.run(
                clear_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                check=False,
            )
            self.assertEqual(cleared.returncode, 0, cleared.stderr)
            self.assertFalse((root / ".gpu-quarantine.json").exists())

    def test_queued_serialized_row_rechecks_quarantine_after_lock(self) -> None:
        with temporary_root() as root:
            started_marker = root / "first-started"
            lost_marker = root / "device-lost"
            health_command = [
                sys.executable,
                "-c",
                f"import pathlib,sys; sys.exit(pathlib.Path({str(lost_marker)!r}).exists())",
            ]
            environment = os.environ.copy()
            environment["CTEST_PARALLEL_LEVEL"] = "4"
            common = [
                "--artifact-root",
                str(root),
                "--timeout",
                "5",
                "--health-command-json",
                json.dumps(health_command),
                "--smoke-command-json",
                json.dumps(["/bin/true"]),
            ]
            first = subprocess.Popen(
                [
                    sys.executable,
                    str(RUNNER),
                    *common,
                    "--name",
                    "first",
                    "--destructive",
                    "--allow-destructive",
                    "--",
                    sys.executable,
                    "-c",
                    "import pathlib,time; "
                    f"pathlib.Path({str(started_marker)!r}).touch(); "
                    "time.sleep(0.3); "
                    f"pathlib.Path({str(lost_marker)!r}).touch()",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
            deadline = time.monotonic() + 10
            while not started_marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(started_marker.exists())

            queued = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    *common,
                    "--name",
                    "queued",
                    "--serialize-gpu",
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit('must not run')",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                check=False,
            )
            first_stdout, first_stderr = first.communicate(timeout=2)
            self.assertEqual(first.returncode, 70, first_stderr or first_stdout)
            self.assertEqual(queued.returncode, 75, queued.stderr)
            result = read_row_result(root, "queued")
            self.assertEqual(result["outcome"], "preflight_device_quarantined")
            self.assertFalse(result["execution"]["command_ran"])
            self.assertTrue(result["execution"]["quarantined_after_lock"])


if __name__ == "__main__":
    unittest.main()
