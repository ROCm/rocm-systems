"""Shared fixtures for red-team tests.

Key safety invariants:
  * Every adversarial subprocess runs under a bounded `timeout` (≤ 5s).
  * Every adversarial filesystem op runs under pytest's `tmp_path` (confined).
  * Every adversarial env var is whitelisted (PATH, ROCM_PATH, HIP_PATH) — no
    API keys forwarded.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict

import pytest


@pytest.fixture
def outcomes_dir() -> Path:
    """The shared _attack_outcomes directory — written by each attack test."""
    root = Path(__file__).parent / "_attack_outcomes"
    root.mkdir(parents=True, exist_ok=True)
    return root


def record_outcome(
    outcomes_dir: Path, attack_id: str, *, status: str, details: Dict[str, object]
) -> None:
    """Write a JSON record of one attack outcome. status ∈ {defeated, succeeded}."""
    (outcomes_dir / f"{attack_id}.json").write_text(
        json.dumps({"attack_id": attack_id, "status": status, "details": details}, indent=2)
    )


@pytest.fixture
def malicious_db_factory(tmp_path):
    """Factory to build a minimal synthetic rocpd DB with crafted metadata.

    Used by shell-metachar, path-traversal, prompt-injection-in-kernel-name
    attacks. Generates a UUID-based schema that `analyze_database()` accepts.
    """

    def _factory(*, kernel_name: str, extra_meta: Dict[str, str] | None = None) -> Path:
        import uuid

        db_path = tmp_path / f"malicious_{uuid.uuid4().hex[:8]}.db"
        conn = sqlite3.connect(db_path)
        uuid_hex = uuid.uuid4().hex
        # Minimal rocpd schema (UUID-based) — enough for analyze_database to load
        conn.executescript(
            f"""
            CREATE TABLE rocpd_metadata (
                uuid TEXT PRIMARY KEY,
                created_at TEXT
            );
            INSERT INTO rocpd_metadata VALUES ('{uuid_hex}', datetime('now'));

            CREATE TABLE rocpd_kernel_dispatch_{uuid_hex} (
                id INTEGER PRIMARY KEY,
                kernel_name TEXT,
                start_ns INTEGER,
                end_ns INTEGER,
                duration_ns INTEGER,
                grid_x INTEGER, grid_y INTEGER, grid_z INTEGER,
                workgroup_x INTEGER, workgroup_y INTEGER, workgroup_z INTEGER
            );
            """
        )
        conn.execute(
            f"INSERT INTO rocpd_kernel_dispatch_{uuid_hex} "
            f"(id, kernel_name, start_ns, end_ns, duration_ns, "
            f"grid_x, grid_y, grid_z, workgroup_x, workgroup_y, workgroup_z) "
            f"VALUES (1, ?, 0, 1000, 1000, 1, 1, 1, 64, 1, 1)",
            (kernel_name,),
        )
        if extra_meta:
            for k, v in extra_meta.items():
                conn.execute(
                    "INSERT INTO rocpd_metadata VALUES (?, ?)", (f"{k}_{uuid_hex}", v)
                )
        conn.commit()
        conn.close()
        return db_path

    return _factory
