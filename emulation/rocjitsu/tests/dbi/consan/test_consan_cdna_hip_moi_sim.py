# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import consan_cdna_hip_moi_sim as simulator


class CdnaHipMoiSimulatorTest(unittest.TestCase):
    def test_registry_uses_exact_target_build_roots_and_totals(self) -> None:
        self.assertEqual(
            sum(suite.expected_tests for suite in simulator.SUITES),
            simulator.EXPECTED_TESTS,
        )
        self.assertEqual(simulator.EXPECTED_TESTS, 33)
        self.assertEqual(len(simulator.SUITES), 13)
        self.assertEqual(
            {
                target_id: (target.build_dir_name, target.default_config_name)
                for target_id, target in simulator.TARGETS.items()
            },
            {
                "gfx942": (
                    "hip-moi-build-gfx942-tests",
                    "gfx942_cdna3_kmd.json",
                ),
                "gfx950": ("hip-moi-build-gfx950-tests", "gfx950_cdna4.json"),
            },
        )
        expected_suffixes = {
            "gfx942": {
                "jakub-matmul": "tests/hip_moi_reference_cdna3_jakub_matmul",
                "mfma-attention": (
                    "tests/hip_moi_instrumented_cdna3_mfma_attention_block_test"
                ),
                "d128-block": (
                    "tests/hip_moi_instrumented_cdna3_d128_attention_block_test"
                ),
                "d128-pressure": (
                    "tests/hip_moi_instrumented_cdna3_d128_attention_pressure_test"
                ),
                "mfma-register-handoff": (
                    "tests/hip_moi_instrumented_cdna3_mfma_register_handoff_test"
                ),
                "mfma-no-score": (
                    "tests/"
                    "hip_moi_instrumented_cdna3_mfma_no_score_lds_attention_test"
                ),
                "d128-no-score": (
                    "tests/"
                    "hip_moi_instrumented_cdna3_d128_no_score_lds_attention_test"
                ),
                "pingpong-private": (
                    "tests/hip_moi_instrumented_cdna3_pingpong_private_lds_test"
                ),
                "pingpong-cooperative": (
                    "tests/" "hip_moi_instrumented_cdna3_pingpong_cooperative_lds_test"
                ),
                "pingpong-wide": (
                    "tests/"
                    "hip_moi_instrumented_cdna3_pingpong_wide_cooperative_lds_test"
                ),
                "streamk-arrival": (
                    "tests/"
                    "hip_moi_instrumented_cdna3_mfma_streamk_arrival_counter_test"
                ),
                "tree-atomic-or": (
                    "tests/"
                    "hip_moi_instrumented_cdna3_mfma_streamk_tree_atomic_or_test"
                ),
                "mfma-lds-alias-handoff": (
                    "tests/"
                    "hip_moi_instrumented_cdna3_mfma_attention_lds_alias_handoff_test"
                ),
            },
            "gfx950": {
                "jakub-matmul": "tests/hip_moi_reference_cdna4_jakub_matmul",
                "mfma-attention": (
                    "tests/hip_moi_instrumented_cdna4_mfma_attention_block_test"
                ),
                "d128-block": (
                    "tests/hip_moi_instrumented_cdna4_d128_attention_block_test"
                ),
                "d128-pressure": (
                    "tests/hip_moi_instrumented_cdna4_d128_attention_pressure_test"
                ),
                "mfma-register-handoff": (
                    "tests/hip_moi_instrumented_cdna4_mfma_register_handoff_test"
                ),
                "mfma-no-score": (
                    "tests/"
                    "hip_moi_instrumented_cdna4_mfma_no_score_lds_attention_test"
                ),
                "d128-no-score": (
                    "tests/"
                    "hip_moi_instrumented_cdna4_d128_no_score_lds_attention_test"
                ),
                "pingpong-private": (
                    "tests/hip_moi_instrumented_cdna4_pingpong_private_lds_test"
                ),
                "pingpong-cooperative": (
                    "tests/" "hip_moi_instrumented_cdna4_pingpong_cooperative_lds_test"
                ),
                "pingpong-wide": (
                    "tests/"
                    "hip_moi_instrumented_cdna4_pingpong_wide_cooperative_lds_test"
                ),
                "streamk-arrival": (
                    "tests/"
                    "hip_moi_instrumented_cdna4_mfma_streamk_arrival_counter_test"
                ),
                "tree-atomic-or": (
                    "tests/"
                    "hip_moi_instrumented_cdna4_mfma_streamk_tree_atomic_or_test"
                ),
                "mfma-lds-alias-handoff": (
                    "tests/"
                    "hip_moi_instrumented_cdna4_mfma_attention_lds_alias_handoff_test"
                ),
            },
        }
        for target_id, expected in expected_suffixes.items():
            target = simulator.TARGETS[target_id]
            with self.subTest(target=target_id):
                self.assertEqual(
                    {
                        suite.id: str(simulator._executable_suffix(target, suite))
                        for suite in simulator.SUITES
                    },
                    expected,
                )

    def test_registry_rejects_mismatched_campaign_workload_mapping(self) -> None:
        simulator._validate_registry()
        mismatched_suite = replace(
            simulator.SUITE_BY_ID["d128-block"],
            validation_workload_id="d128-pressure",
        )
        with mock.patch.object(simulator, "SUITES", (mismatched_suite,)):
            with self.assertRaisesRegex(
                simulator.registry.RegistryError,
                "but validation workload",
            ):
                simulator._validate_registry()

    def test_registry_validation_rejects_structural_drift(self) -> None:
        registry = simulator.registry

        too_few_suites = registry.SUITES[:-1]
        with (
            mock.patch.object(registry, "SUITES", too_few_suites),
            mock.patch.object(
                registry,
                "SUITE_BY_ID",
                {suite.id: suite for suite in too_few_suites},
            ),
            self.assertRaisesRegex(registry.RegistryError, "13 suites"),
        ):
            registry.validate()

        duplicate_ctest_name = (
            registry.SUITES[0],
            replace(
                registry.SUITES[1],
                ctest_name=registry.SUITES[0].ctest_name,
            ),
            *registry.SUITES[2:],
        )
        with (
            mock.patch.object(registry, "SUITES", duplicate_ctest_name),
            mock.patch.object(
                registry,
                "SUITE_BY_ID",
                {suite.id: suite for suite in duplicate_ctest_name},
            ),
            self.assertRaisesRegex(registry.RegistryError, "CTest names"),
        ):
            registry.validate()

        wrong_test_count = (
            replace(
                registry.SUITES[0],
                expected_tests=registry.SUITES[0].expected_tests + 1,
            ),
            *registry.SUITES[1:],
        )
        with (
            mock.patch.object(registry, "SUITES", wrong_test_count),
            mock.patch.object(
                registry,
                "SUITE_BY_ID",
                {suite.id: suite for suite in wrong_test_count},
            ),
            self.assertRaisesRegex(registry.RegistryError, "33 tests"),
        ):
            registry.validate()

    def test_registry_rejects_unknown_target_and_suite_keys(self) -> None:
        registry = simulator.registry
        for target_id, suite_id, unknown_key in (
            ("gfx999", "d128-block", "gfx999"),
            ("gfx942", "unknown-suite", "unknown-suite"),
        ):
            with (
                self.subTest(target=target_id, suite=suite_id),
                self.assertRaisesRegex(registry.RegistryError, unknown_key),
            ):
                registry.relative_executable_path(target_id, suite_id)

    def test_executable_suffix_rejects_paths_outside_the_target_build(self) -> None:
        target = simulator.TARGETS["gfx942"]
        suite = simulator.SUITE_BY_ID["d128-block"]
        with (
            mock.patch.object(
                simulator.registry,
                "relative_executable_path",
                return_value=Path("/outside"),
            ),
            self.assertRaisesRegex(
                simulator.consan_validation.ValidationError,
                "gfx942 d128-block must resolve beneath",
            ),
        ):
            simulator._executable_suffix(target, suite)

    def test_parser_requires_execution_arguments_outside_listing_mode(self) -> None:
        errors = io.StringIO()
        with (
            contextlib.redirect_stderr(errors),
            self.assertRaises(SystemExit) as exit_status,
        ):
            simulator._parse_args(["--suite", "d128-block"])

        self.assertEqual(exit_status.exception.code, 2)
        self.assertIn(
            "the following arguments are required: "
            "--target, --rocjitsu, --hip-moi-build",
            errors.getvalue(),
        )

    def test_list_suites_emits_the_cmake_registration_rows(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = simulator.main(["--list-suites"])

        self.assertEqual(result, 0)
        self.assertEqual(
            output.getvalue().splitlines(),
            [f"{suite.ctest_name}|{suite.id}" for suite in simulator.SUITES],
        )

    def test_suite_runs_through_rocjitsu_and_checks_passed_count(self) -> None:
        target = simulator.TARGETS["gfx942"]
        suite = simulator.SUITE_BY_ID["d128-block"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rocjitsu = root / "rocjitsu"
            config = root / "gfx942.json"
            build = root / target.build_dir_name
            executable = build / simulator._executable_suffix(target, suite)
            executable.parent.mkdir(parents=True)
            executable.touch()
            rocjitsu.touch()
            config.touch()
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="[  PASSED  ] 2 tests.\n",
                stderr="",
            )
            with mock.patch.object(
                simulator.subprocess, "run", return_value=completed
            ) as run:
                result = simulator._run_suite(
                    target, suite, rocjitsu, config, build, timeout=60.0
                )

        self.assertTrue(result.accepted)
        run.assert_called_once_with(
            [
                str(rocjitsu),
                "--config",
                str(config),
                "--",
                str(executable),
                "--gtest_brief=1",
            ],
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
        )

    def test_suite_rejects_unexpected_test_count(self) -> None:
        target = simulator.TARGETS["gfx942"]
        suite = simulator.SUITE_BY_ID["d128-pressure"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / target.build_dir_name
            executable = build / simulator._executable_suffix(target, suite)
            executable.parent.mkdir(parents=True)
            executable.touch()
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="[  PASSED  ] 3 tests.\n",
                stderr="",
            )
            with mock.patch.object(simulator.subprocess, "run", return_value=completed):
                result = simulator._run_suite(
                    target,
                    suite,
                    root / "rocjitsu",
                    root / "gfx942.json",
                    build,
                    timeout=60.0,
                )

        self.assertFalse(result.accepted)
        self.assertEqual(result.passed_tests, 3)
        self.assertEqual(result.error, "expected 4 tests, observed 3")

    def test_all_suites_run_and_report_the_aggregate(self) -> None:
        target = simulator.TARGETS["gfx950"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rocjitsu = root / "rocjitsu"
            config = root / "gfx950.json"
            build = root / target.build_dir_name
            rocjitsu.touch()
            config.touch()
            for suite in simulator.SUITES:
                executable = build / simulator._executable_suffix(target, suite)
                executable.parent.mkdir(parents=True, exist_ok=True)
                executable.touch()

            completed = [
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout=f"[  PASSED  ] {suite.expected_tests} tests.\n",
                    stderr="",
                )
                for suite in simulator.SUITES
            ]
            output = io.StringIO()
            with (
                mock.patch.object(
                    simulator.subprocess, "run", side_effect=completed
                ) as run,
                contextlib.redirect_stdout(output),
            ):
                result = simulator.main(
                    [
                        "--target",
                        target.id,
                        "--rocjitsu",
                        str(rocjitsu),
                        "--config",
                        str(config),
                        "--hip-moi-build",
                        str(build),
                    ]
                )

        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, 13)
        self.assertIn("== gfx950 hip-moi simulator summary ==", output.getvalue())
        self.assertIn("total: 33/33 tests", output.getvalue())

    def test_single_suite_count_mismatch_fails_the_ctest_entry(self) -> None:
        target = simulator.TARGETS["gfx942"]
        suite = simulator.SUITE_BY_ID["d128-pressure"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rocjitsu = root / "rocjitsu"
            config = root / "gfx942.json"
            build = root / target.build_dir_name
            executable = build / simulator._executable_suffix(target, suite)
            executable.parent.mkdir(parents=True)
            executable.touch()
            rocjitsu.touch()
            config.touch()
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="[  PASSED  ] 3 tests.\n",
                stderr="",
            )
            output = io.StringIO()
            with (
                mock.patch.object(
                    simulator.subprocess,
                    "run",
                    return_value=completed,
                ) as run,
                contextlib.redirect_stdout(output),
            ):
                result = simulator.main(
                    [
                        "--target",
                        target.id,
                        "--rocjitsu",
                        str(rocjitsu),
                        "--config",
                        str(config),
                        "--hip-moi-build",
                        str(build),
                        "--suite",
                        suite.id,
                    ]
                )

        self.assertEqual(result, 1)
        self.assertEqual(run.call_count, 1)
        self.assertIn(
            "FAIL d128-pressure: expected 4 tests, observed 3",
            output.getvalue(),
        )

    def test_main_uses_each_target_default_config(self) -> None:
        suite = simulator.SUITE_BY_ID["jakub-matmul"]
        for target in simulator.TARGETS.values():
            with self.subTest(target=target.id):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    rocjitsu = root / "rocjitsu"
                    build = root / target.build_dir_name
                    executable = build / simulator._executable_suffix(target, suite)
                    executable.parent.mkdir(parents=True)
                    executable.touch()
                    rocjitsu.touch()
                    completed = subprocess.CompletedProcess(
                        args=[],
                        returncode=0,
                        stdout="[  PASSED  ] 2 tests.\n",
                        stderr="",
                    )
                    with (
                        mock.patch.object(
                            simulator.subprocess, "run", return_value=completed
                        ) as run,
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        result = simulator.main(
                            [
                                "--target",
                                target.id,
                                "--rocjitsu",
                                str(rocjitsu),
                                "--hip-moi-build",
                                str(build),
                                "--suite",
                                suite.id,
                            ]
                        )

                self.assertEqual(result, 0)
                command = run.call_args.args[0]
                config = Path(command[command.index("--config") + 1])
                self.assertEqual(
                    config,
                    Path(simulator.__file__).resolve().parents[3]
                    / "configs"
                    / target.default_config_name,
                )

    def test_main_rejects_the_other_targets_build_directory(self) -> None:
        target = simulator.TARGETS["gfx950"]
        wrong_target = simulator.TARGETS["gfx942"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rocjitsu = root / "rocjitsu"
            wrong_build = root / wrong_target.build_dir_name
            rocjitsu.touch()
            wrong_build.mkdir()
            errors = io.StringIO()
            with (
                mock.patch.object(simulator.subprocess, "run") as run,
                contextlib.redirect_stderr(errors),
            ):
                result = simulator.main(
                    [
                        "--target",
                        target.id,
                        "--rocjitsu",
                        str(rocjitsu),
                        "--hip-moi-build",
                        str(wrong_build),
                        "--suite",
                        "jakub-matmul",
                    ]
                )

        self.assertEqual(result, 2)
        run.assert_not_called()
        self.assertIn(
            "--hip-moi-build must name an existing "
            "hip-moi-build-gfx950-tests directory",
            errors.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
