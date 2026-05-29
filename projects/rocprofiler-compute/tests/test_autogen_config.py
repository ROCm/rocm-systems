# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import hashlib
import json
from pathlib import Path

import common
import pytest

HASH_DB = Path(common.SRC) / "utils/.config_hashes.json"
ANALYSIS_CONFIGS = Path(common.SRC) / "rocprof_compute_soc/analysis_configs"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_config_hashes_match_files() -> None:
    assert HASH_DB.exists(), f"Missing hash DB: {HASH_DB}"
    assert ANALYSIS_CONFIGS.exists(), (
        f"Missing analysis configs dir: {ANALYSIS_CONFIGS}"
    )

    with HASH_DB.open() as f:
        data = json.load(f)

    assert "archs" in data, "Hash DB missing 'archs' key"
    assert isinstance(data["archs"], dict)

    failures = []

    for arch, arch_data in data["archs"].items():
        arch_dir = ANALYSIS_CONFIGS / arch
        if not arch_dir.exists():
            failures.append(f"Arch directory missing: {arch_dir}")
            continue

        files = arch_data.get("files", {})
        if not isinstance(files, dict):
            failures.append(f"'files' for {arch} is not a dict")
            continue

        for rel_path, expected_hash in files.items():
            panel_path = arch_dir / rel_path
            if not panel_path.exists():
                failures.append(f"Missing panel file: {panel_path}")
                continue

            actual_hash = md5(panel_path)
            if actual_hash != expected_hash:
                failures.append(
                    f"[{arch}] Panel hash mismatch: {panel_path}\n"
                    f"  expected: {expected_hash}\n"
                    f"  actual:   {actual_hash}"
                )

    if failures:
        pytest.fail("Hash consistency failures:\n\n" + "\n".join(failures))
