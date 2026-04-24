###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

from __future__ import annotations

import subprocess
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PACKAGE_ROOT / "tests" / "fixtures"
MAX_DB_FIXTURE_BYTES = 10 * 1024 * 1024
MAX_TRACKED_DB_BYTES = 32 * 1024 * 1024


def _tracked_fixture_dbs() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--", "tests/fixtures"],
        cwd=PACKAGE_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [
        PACKAGE_ROOT / line
        for line in result.stdout.splitlines()
        if line.endswith(".db")
    ]


def test_tracked_db_fixtures_stay_within_reviewable_size_budget():
    db_paths = _tracked_fixture_dbs()
    oversized = [
        f"{path.relative_to(PACKAGE_ROOT)} ({path.stat().st_size:,} bytes)"
        for path in db_paths
        if path.stat().st_size > MAX_DB_FIXTURE_BYTES
    ]
    total_bytes = sum(path.stat().st_size for path in db_paths)

    assert not oversized, (
        "Large trace fixtures should live in release artifacts, Git LFS, or "
        "local generated data, not regular git blobs:\n" + "\n".join(oversized)
    )
    assert total_bytes <= MAX_TRACKED_DB_BYTES, (
        f"Tracked DB fixtures use {total_bytes:,} bytes; "
        f"budget is {MAX_TRACKED_DB_BYTES:,} bytes"
    )


def test_fixture_packaging_policy_excludes_db_files_from_sdist():
    manifest = (PACKAGE_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "recursive-exclude tests/fixtures *.db" in manifest


def test_fixture_db_files_are_marked_for_lfs():
    attributes = (FIXTURE_ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "*.db filter=lfs diff=lfs merge=lfs -text" in attributes
