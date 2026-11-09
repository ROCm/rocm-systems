# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the rocjitsu AFL++ result collector."""

import os
import subprocess
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
COLLECTOR = SCRIPTS_DIR / "collect_rocjitsu_afl_results.py"


def test_collects_partial_results_when_second_lane_is_missing(tmp_path: Path) -> None:
    asan_output = tmp_path / "asan-output"
    asan_instance = asan_output / "worker-00"
    (asan_instance / "crashes").mkdir(parents=True)
    (asan_instance / "hangs").mkdir()
    (asan_instance / "crashes" / "id:000000").write_bytes(b"crash")
    (asan_instance / "fuzzer_stats").write_text(
        "execs_done : 123\nexecs_per_sec : 4.5\n"
    )
    second_instance = asan_output / "worker-01"
    (second_instance / "crashes").mkdir(parents=True)
    (second_instance / "hangs").mkdir()
    (second_instance / "crashes" / "id:000001").write_bytes(b"crash")
    (second_instance / "hangs" / "id:000000").write_bytes(b"hang")
    (second_instance / "fuzzer_stats").write_text(
        "execs_done : 7\nexecs_per_sec : 0.5\n"
    )
    asan_logs = Path(f"{asan_output}-logs")
    asan_logs.mkdir()
    (asan_logs / "worker-00.log").write_text("saved a crash\n")
    tsan_output = tmp_path / "missing-tsan-output"
    tsan_logs = Path(f"{tsan_output}-logs")
    tsan_logs.mkdir()
    (tsan_logs / "worker-00.log").write_text("startup failed\n")
    artifact = tmp_path / "artifact"
    github_output = tmp_path / "github-output"
    environment = os.environ | {
        "ASAN_UBSAN_CAMPAIGN_OUTCOME": "failure",
        "GITHUB_OUTPUT": os.fspath(github_output),
        "GITHUB_SHA": "0123456789abcdef",
        "ROCJITSU_ARTIFACT_DIR": os.fspath(artifact),
        "ROCJITSU_ASAN_UBSAN_FINDINGS_DIR": os.fspath(asan_output),
        "ROCJITSU_FUZZ_CAMPAIGN_SECONDS": "600",
        "ROCJITSU_FUZZ_WORKERS": "64",
        "ROCJITSU_TSAN_FINDINGS_DIR": os.fspath(tsan_output),
        "TSAN_CAMPAIGN_OUTCOME": "failure",
    }

    completed = subprocess.run([COLLECTOR], env=environment, check=False)

    assert completed.returncode == 0
    outputs = dict(
        line.split("=", 1) for line in github_output.read_text().splitlines()
    )
    assert outputs["asan-ubsan-crash-count"] == "1"
    assert outputs["asan-ubsan-execution-count"] == "130"
    assert outputs["asan-ubsan-executions-per-second"] == "5.00"
    assert outputs["asan-ubsan-hang-count"] == "1"
    assert outputs["tsan-crash-count"] == "0"
    assert outputs["tsan-execution-count"] == "0"
    assert outputs["campaign-failed"] == "true"
    saved_crash = artifact / "asan-ubsan/worker-00/crashes/id%3A000000"
    assert saved_crash.read_bytes() == b"crash"
    upload_unsafe_characters = set('":<>|*?\r\n')
    assert all(
        not upload_unsafe_characters.intersection(path.name)
        for path in artifact.rglob("*")
    )
    assert (artifact / "asan-ubsan/logs/worker-00.log").is_file()
    assert (artifact / "tsan/logs/worker-00.log").is_file()
    assert "tsan_campaign_outcome=failure" in (artifact / "summary.txt").read_text()


def test_clean_campaign_does_not_create_an_artifact(tmp_path: Path) -> None:
    github_output = tmp_path / "github-output"
    artifact = tmp_path / "artifact"
    environment = os.environ | {
        "ASAN_UBSAN_CAMPAIGN_OUTCOME": "success",
        "GITHUB_OUTPUT": os.fspath(github_output),
        "GITHUB_SHA": "0123456789abcdef",
        "ROCJITSU_ARTIFACT_DIR": os.fspath(artifact),
        "ROCJITSU_ASAN_UBSAN_FINDINGS_DIR": os.fspath(tmp_path / "missing-asan"),
        "ROCJITSU_FUZZ_CAMPAIGN_SECONDS": "600",
        "ROCJITSU_FUZZ_WORKERS": "64",
        "ROCJITSU_TSAN_FINDINGS_DIR": os.fspath(tmp_path / "missing-tsan"),
        "TSAN_CAMPAIGN_OUTCOME": "success",
    }

    completed = subprocess.run([COLLECTOR], env=environment, check=False)

    assert completed.returncode == 0
    outputs = dict(
        line.split("=", 1) for line in github_output.read_text().splitlines()
    )
    assert outputs["finding-count"] == "0"
    assert outputs["campaign-failed"] == "false"
    assert not artifact.exists()


def test_build_failure_records_skipped_lanes_and_zero_counts(tmp_path: Path) -> None:
    github_output = tmp_path / "github-output"
    artifact = tmp_path / "artifact"
    environment = os.environ | {
        "ASAN_UBSAN_CAMPAIGN_OUTCOME": "skipped",
        "GITHUB_OUTPUT": os.fspath(github_output),
        "GITHUB_SHA": "0123456789abcdef",
        "ROCJITSU_ARTIFACT_DIR": os.fspath(artifact),
        "ROCJITSU_ASAN_UBSAN_FINDINGS_DIR": os.fspath(tmp_path / "missing-asan"),
        "ROCJITSU_FUZZ_CAMPAIGN_SECONDS": "360",
        "ROCJITSU_FUZZ_WORKERS": "32",
        "ROCJITSU_TSAN_FINDINGS_DIR": os.fspath(tmp_path / "missing-tsan"),
        "TSAN_CAMPAIGN_OUTCOME": "skipped",
    }

    completed = subprocess.run([COLLECTOR], env=environment, check=False)

    assert completed.returncode == 0
    outputs = dict(
        line.split("=", 1) for line in github_output.read_text().splitlines()
    )
    assert outputs["asan-ubsan-campaign-outcome"] == "skipped"
    assert outputs["tsan-campaign-outcome"] == "skipped"
    assert outputs["crash-count"] == "0"
    assert outputs["finding-count"] == "0"
    assert outputs["hang-count"] == "0"
    assert outputs["campaign-failed"] == "true"
    summary = (artifact / "summary.txt").read_text()
    assert "asan_ubsan_campaign_outcome=skipped" in summary
    assert "tsan_campaign_outcome=skipped" in summary
