# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Unit tests for GPU-specific test counter selection.
"""

from __future__ import annotations
from pathlib import Path

import pytest
from conftest import RocprofsysTest, _validate_rocpd_candidates
from rocprofsys import GPUInfo, TestResult as RocprofsysTestResult, ValidationResult

pytestmark = [pytest.mark.pytest_impl]


@pytest.mark.class_name("gpu-info")
class TestGPUInfo(RocprofsysTest):
    def test_gfx1250_uses_gfx1250_counter_set(self):
        gpu_info = GPUInfo(
            available=True,
            architectures=["gfx1250"],
            device_count=1,
            categories={"instinct"},
        )

        assert (
            gpu_info.rocm_events_for_test
            == "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TX_VCA_VCA_BUSY"
        )
        assert gpu_info.counter_names == [
            "GRBM_COUNT",
            "SQ_WAVES",
            "SQ_INSTS_VALU",
            "TX_VCA_VCA_BUSY",
        ]
        assert gpu_info.expected_counter_files == [
            "rocprof-device-[0-9]-GRBM_COUNT*.txt",
            "rocprof-device-[0-9]-SQ_WAVES*.txt",
            "rocprof-device-[0-9]-SQ_INSTS_VALU*.txt",
            "rocprof-device-[0-9]-TX_VCA_VCA_BUSY*.txt",
        ]

    def test_mi300_and_later_keep_ta_ta_busy(self):
        gpu_info = GPUInfo(
            available=True,
            architectures=["gfx942"],
            device_count=1,
            categories={"instinct"},
        )

        assert (
            gpu_info.rocm_events_for_test
            == "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TA_TA_BUSY"
        )
        assert gpu_info.counter_names == [
            "GRBM_COUNT",
            "SQ_WAVES",
            "SQ_INSTS_VALU",
            "TA_TA_BUSY",
        ]

    def test_non_mi300_non_gfx1250_keep_single_counter(self):
        gpu_info = GPUInfo(
            available=True,
            architectures=["gfx1201"],
            device_count=1,
            categories={"radeon"},
        )

        assert gpu_info.rocm_events_for_test == "SQ_WAVES"
        assert gpu_info.counter_names == ["SQ_WAVES"]


@pytest.mark.class_name("test-result")
class TestTestResult(RocprofsysTest):
    def test_rocpd_files_prefers_default_database(self, tmp_path):
        default_db = tmp_path / "rocpd.db"
        rank_db = tmp_path / "rocpd-2-0.db"
        default_db.touch()
        rank_db.touch()

        result = RocprofsysTestResult(0, "", tmp_path, [], {})

        assert result.rocpd_files == [default_db]

    def test_rocpd_files_returns_sorted_rank_databases(self, tmp_path):
        higher_pid_db = tmp_path / "rocpd-66607-0.db"
        lower_pid_db = tmp_path / "rocpd-66606-0.db"
        higher_pid_db.touch()
        lower_pid_db.touch()

        result = RocprofsysTestResult(0, "", tmp_path, [], {})

        assert result.rocpd_files == [lower_pid_db, higher_pid_db]

    def test_rocpd_candidates_accepts_later_valid_candidate(
        self,
        tmp_path: Path,
    ) -> None:
        invalid_db = tmp_path / "rocpd-66606-0.db"
        valid_db = tmp_path / "rocpd-66607-0.db"
        invalid_db.touch()
        valid_db.touch()
        calls: list[Path] = []

        def validate_rocpd_database(db_path: Path) -> ValidationResult:
            calls.append(db_path)
            if db_path == valid_db:
                return ValidationResult(
                    True,
                    "valid candidate",
                    stdout="rocpd validated",
                    command=f"validate {db_path.name}",
                )
            return ValidationResult(
                False,
                "missing GPU counter rows",
                stdout="validation failed",
                command=f"validate {db_path.name}",
            )

        passing_output, failures, global_failure = _validate_rocpd_candidates(
            [invalid_db, valid_db],
            validate_rocpd_database,
            pass_regex=[r"rocpd validated"],
        )

        assert calls == [invalid_db, valid_db]
        assert passing_output == f"Command: validate {valid_db.name}\n\nvalid candidate"
        assert global_failure is None
        assert failures == [
            f"Command: validate {invalid_db.name}\n\nmissing GPU counter rows"
        ]

    def test_rocpd_candidates_reports_all_candidate_failures(
        self,
        tmp_path: Path,
    ) -> None:
        first_db = tmp_path / "rocpd-66606-0.db"
        second_db = tmp_path / "rocpd-7-0.db"
        first_db.touch()
        second_db.touch()

        def validate_rocpd_database(db_path: Path) -> ValidationResult:
            return ValidationResult(
                False,
                f"{db_path.name} is missing GPU counter rows",
                stdout="validation failed",
                command=f"validate {db_path.name}",
            )

        passing_output, failures, global_failure = _validate_rocpd_candidates(
            [first_db, second_db],
            validate_rocpd_database,
        )

        assert passing_output is None
        assert global_failure is None
        message = "\n\n--- Next ROCpd candidate ---\n\n".join(failures)
        assert "rocpd-66606-0.db is missing GPU counter rows" in message
        assert "rocpd-7-0.db is missing GPU counter rows" in message
        assert "--- Next ROCpd candidate ---" in message

    def test_rocpd_candidates_fail_regex_is_global(
        self,
        tmp_path: Path,
    ) -> None:
        invalid_db = tmp_path / "rocpd-66606-0.db"
        valid_db = tmp_path / "rocpd-66607-0.db"
        invalid_db.touch()
        valid_db.touch()
        calls: list[Path] = []

        def validate_rocpd_database(db_path: Path) -> ValidationResult:
            calls.append(db_path)
            if db_path == invalid_db:
                return ValidationResult(
                    False,
                    "invalid candidate",
                    stdout="validation failed with FORBIDDEN marker",
                    command=f"validate {db_path.name}",
                )
            return ValidationResult(
                True,
                "valid candidate",
                stdout="rocpd validated",
                command=f"validate {db_path.name}",
            )

        passing_output, failures, global_failure = _validate_rocpd_candidates(
            [invalid_db, valid_db],
            validate_rocpd_database,
            fail_regex=[r"FORBIDDEN"],
        )

        assert calls == [invalid_db]
        assert passing_output is None
        assert failures == []
        assert global_failure is not None
        assert "Fail regex found: FORBIDDEN" in global_failure
        assert f"validate {invalid_db.name}" in global_failure
