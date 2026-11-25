#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Collect complete or partial rocjitsu AFL++ results for GitHub Actions."""

import hashlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

REQUIRED_ENVIRONMENT = (
    "ASAN_UBSAN_CAMPAIGN_OUTCOME",
    "GITHUB_OUTPUT",
    "GITHUB_SHA",
    "ROCJITSU_ARTIFACT_DIR",
    "ROCJITSU_ASAN_UBSAN_FINDINGS_DIR",
    "ROCJITSU_FUZZ_CAMPAIGN_SECONDS",
    "ROCJITSU_FUZZ_WORKERS",
    "ROCJITSU_TSAN_FINDINGS_DIR",
    "TSAN_CAMPAIGN_OUTCOME",
)


@dataclass(frozen=True)
class LaneResults:
    outcome: str
    crashes: int
    executions: int
    executions_per_second: float
    hangs: int


def instance_directories(source_dir: Path) -> list[Path]:
    """List AFL++ instance directories in a stable order."""
    if not source_dir.is_dir():
        return []
    return sorted(path for path in source_dir.iterdir() if path.is_dir())


def count_findings(source_dir: Path, kind: str) -> int:
    """Count findings with distinct contents across every worker in one lane."""
    digests: set[bytes] = set()
    for instance_dir in instance_directories(source_dir):
        finding_dir = instance_dir / kind
        if not finding_dir.is_dir():
            continue
        for finding in finding_dir.iterdir():
            if finding.is_file() and finding.name != "README.txt":
                digests.add(hashlib.sha256(finding.read_bytes()).digest())
    return len(digests)


def read_fuzzer_stat(path: Path, key: str) -> float:
    """Read one numeric value from an AFL++ fuzzer_stats file."""
    try:
        for line in path.read_text().splitlines():
            name, separator, value = line.partition(":")
            if separator and name.strip() == key:
                try:
                    return float(value)
                except ValueError:
                    return 0
    except OSError:
        pass
    return 0


def sum_fuzzer_stat(source_dir: Path, key: str) -> float:
    return sum(
        read_fuzzer_stat(instance_dir / "fuzzer_stats", key)
        for instance_dir in instance_directories(source_dir)
    )


def collect_lane(source_dir: Path, outcome: str) -> LaneResults:
    return LaneResults(
        outcome=outcome,
        crashes=count_findings(source_dir, "crashes"),
        executions=round(sum_fuzzer_stat(source_dir, "execs_done")),
        executions_per_second=sum_fuzzer_stat(source_dir, "execs_per_sec"),
        hangs=count_findings(source_dir, "hangs"),
    )


def copy_lane(source_dir: Path, destination: Path) -> None:
    """Copy findings and statistics from every available worker."""
    for instance_dir in instance_directories(source_dir):
        instance_destination = destination / instance_dir.name
        instance_destination.mkdir(parents=True, exist_ok=True)
        for kind in ("crashes", "hangs"):
            source = instance_dir / kind
            if source.is_dir():
                finding_destination = instance_destination / kind
                finding_destination.mkdir(exist_ok=True)
                for finding in source.iterdir():
                    if finding.is_file():
                        # AFL++ uses metadata characters that artifact uploads reject.
                        encoded_name = quote(finding.name, safe=",+=._-")
                        shutil.copy2(finding, finding_destination / encoded_name)
        stats = instance_dir / "fuzzer_stats"
        if stats.is_file():
            shutil.copy2(stats, instance_destination / "fuzzer_stats")


def copy_logs(source_dir: Path, destination: Path) -> None:
    if source_dir.is_dir():
        shutil.copytree(source_dir, destination, dirs_exist_ok=True)


def format_rate(rate: float) -> str:
    return f"{rate:.2f}" if rate else "0"


def lane_outputs(prefix: str, results: LaneResults) -> list[tuple[str, str]]:
    return [
        (f"{prefix}-campaign-outcome", results.outcome),
        (f"{prefix}-crash-count", str(results.crashes)),
        (f"{prefix}-execution-count", str(results.executions)),
        (f"{prefix}-executions-per-second", format_rate(results.executions_per_second)),
        (f"{prefix}-hang-count", str(results.hangs)),
    ]


def require_environment() -> dict[str, str]:
    missing = [name for name in REQUIRED_ENVIRONMENT if name not in os.environ]
    if missing:
        raise ValueError(f"missing required environment: {', '.join(missing)}")
    return {name: os.environ[name] for name in REQUIRED_ENVIRONMENT}


def main() -> int:
    try:
        environment = require_environment()
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    asan_ubsan_source = Path(environment["ROCJITSU_ASAN_UBSAN_FINDINGS_DIR"])
    tsan_source = Path(environment["ROCJITSU_TSAN_FINDINGS_DIR"])
    asan_ubsan = collect_lane(
        asan_ubsan_source, environment["ASAN_UBSAN_CAMPAIGN_OUTCOME"]
    )
    tsan = collect_lane(tsan_source, environment["TSAN_CAMPAIGN_OUTCOME"])
    crash_count = asan_ubsan.crashes + tsan.crashes
    hang_count = asan_ubsan.hangs + tsan.hangs
    finding_count = crash_count + hang_count
    campaign_failed = asan_ubsan.outcome != "success" or tsan.outcome != "success"

    outputs = [
        *lane_outputs("asan-ubsan", asan_ubsan),
        *lane_outputs("tsan", tsan),
        ("crash-count", str(crash_count)),
        ("finding-count", str(finding_count)),
        ("hang-count", str(hang_count)),
        ("campaign-failed", str(campaign_failed).lower()),
    ]
    with Path(environment["GITHUB_OUTPUT"]).open("a") as github_output:
        for name, value in outputs:
            print(f"{name}={value}", file=github_output)

    if finding_count == 0 and not campaign_failed:
        return 0

    artifact_dir = Path(environment["ROCJITSU_ARTIFACT_DIR"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    copy_lane(asan_ubsan_source, artifact_dir / "asan-ubsan")
    copy_lane(tsan_source, artifact_dir / "tsan")
    copy_logs(Path(f"{asan_ubsan_source}-logs"), artifact_dir / "asan-ubsan/logs")
    copy_logs(Path(f"{tsan_source}-logs"), artifact_dir / "tsan/logs")
    summary = "\n".join(
        (
            f"commit={environment['GITHUB_SHA']}",
            f"workers={environment['ROCJITSU_FUZZ_WORKERS']}",
            "campaign_seconds_per_lane="
            f"{environment['ROCJITSU_FUZZ_CAMPAIGN_SECONDS']}",
            f"asan_ubsan_campaign_outcome={asan_ubsan.outcome}",
            f"asan_ubsan_crashes={asan_ubsan.crashes}",
            f"asan_ubsan_executions={asan_ubsan.executions}",
            "asan_ubsan_executions_per_second="
            f"{format_rate(asan_ubsan.executions_per_second)}",
            f"asan_ubsan_hangs={asan_ubsan.hangs}",
            f"tsan_campaign_outcome={tsan.outcome}",
            f"tsan_crashes={tsan.crashes}",
            f"tsan_executions={tsan.executions}",
            f"tsan_executions_per_second={format_rate(tsan.executions_per_second)}",
            f"tsan_hangs={tsan.hangs}",
        )
    )
    (artifact_dir / "summary.txt").write_text(f"{summary}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
