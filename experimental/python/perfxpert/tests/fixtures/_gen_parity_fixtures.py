"""Generate small hermetic parity fixtures for the Phase 5 parity lane."""

from __future__ import annotations

import sqlite3
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent


def _connect(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE kernels (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            duration INTEGER NOT NULL
        );
        CREATE TABLE memory_copies (
            id INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            size INTEGER NOT NULL,
            duration INTEGER NOT NULL
        );
        CREATE TABLE regions (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            duration INTEGER NOT NULL
        );
        CREATE TABLE pmc_events (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            dispatch_id INTEGER NOT NULL,
            counter_name TEXT NOT NULL,
            counter_value REAL NOT NULL
        );
        """
    )
    return conn


def create_compute_bound(path: Path) -> None:
    conn = _connect(path)
    for dispatch_id in range(1, 51):
        start = (dispatch_id - 1) * 100_000
        duration = 100_000
        conn.execute(
            "INSERT INTO kernels (id, name, start, end, duration) VALUES (?, ?, ?, ?, ?)",
            (dispatch_id, "matmul", start, start + duration, duration),
        )
        for counter_name, value in (
            ("GRBM_COUNT", 100.0),
            ("GRBM_GUI_ACTIVE", 95.0),
            ("SQ_WAVES", 24.0),
        ):
            conn.execute(
                "INSERT INTO pmc_events (name, dispatch_id, counter_name, counter_value) "
                "VALUES (?, ?, ?, ?)",
                ("matmul", dispatch_id, counter_name, value),
            )
    conn.commit()
    conn.close()


def create_memory_transfer(path: Path) -> None:
    conn = _connect(path)
    conn.execute(
        "INSERT INTO kernels (id, name, start, end, duration) VALUES (?, ?, ?, ?, ?)",
        (1, "postprocess", 0, 100_000, 100_000),
    )
    conn.execute(
        "INSERT INTO memory_copies (id, category, start, end, size, duration) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (1, "HostToDevice", 100_000, 500_000, 4 * 1024 * 1024, 400_000),
    )
    conn.commit()
    conn.close()


def create_launch_overhead(path: Path) -> None:
    conn = _connect(path)
    for dispatch_id in range(1, 1201):
        start = (dispatch_id - 1) * 20_000
        kernel_duration = 5_000
        conn.execute(
            "INSERT INTO kernels (id, name, start, end, duration) VALUES (?, ?, ?, ?, ?)",
            (dispatch_id, "tiny", start, start + kernel_duration, kernel_duration),
        )
        conn.execute(
            "INSERT INTO regions (id, name, category, start, end, duration) VALUES (?, ?, ?, ?, ?, ?)",
            (
                dispatch_id,
                "hipLaunchKernelGGL",
                "HIP_RUNTIME_API_EXT",
                start - 10_000,
                start,
                10_000,
            ),
        )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(exist_ok=True)
    create_compute_bound(FIXTURES_DIR / "parity_compute_bound.db")
    create_memory_transfer(FIXTURES_DIR / "parity_memory_transfer.db")
    create_launch_overhead(FIXTURES_DIR / "parity_launch_overhead.db")
    print("Parity fixtures generated.")
