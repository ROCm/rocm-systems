#!/usr/bin/env python3
"""Generate a minimal RCCL-capable rocpd fixture.

Creates ``tests/fixtures/rccl_allreduce.db`` with:
  - 4 AllReduce regions (category='RCCL') across 4 distinct (nid, pid)
    rank tuples.
  - 1 GEMM kernel overlapping the AllReduce spans (for overlap ratio test).
  - rocpd_arg rows carrying (count, datatype) for each AllReduce.

The fixture uses simplified VIEWs (``regions``, ``kernels``, ``rocpd_arg``,
``rocpd_region``, ``processes``, ``rocpd_info_agent``) matching what
``perfxpert/tools/rccl_analysis.py`` actually queries. It does NOT reproduce
the full rocprofiler-sdk schema — that is out of scope for unit tests.

Run once to (re)generate:
    python3 tests/fixtures/_generate_rccl_fixture.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


_OUT = Path(__file__).resolve().parent / "rccl_allreduce.db"


def _build(path: Path) -> None:
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()

    # Minimal schema — just the handles rccl_analysis.py queries.
    cur.executescript(
        """
        CREATE TABLE regions (
            id INTEGER PRIMARY KEY,
            category TEXT,
            name TEXT,
            start INTEGER,
            end INTEGER,
            nid INTEGER,
            pid INTEGER,
            tid INTEGER,
            event_id INTEGER
        );

        CREATE TABLE rocpd_region (
            id INTEGER PRIMARY KEY,
            nid INTEGER,
            pid INTEGER,
            tid INTEGER,
            start INTEGER,
            end INTEGER,
            name_id INTEGER,
            event_id INTEGER
        );

        CREATE TABLE rocpd_string (
            id INTEGER PRIMARY KEY,
            string TEXT
        );

        CREATE TABLE rocpd_arg (
            id INTEGER PRIMARY KEY,
            event_id INTEGER,
            position INTEGER,
            type TEXT,
            name TEXT,
            value TEXT
        );

        CREATE TABLE kernels (
            id INTEGER PRIMARY KEY,
            name TEXT,
            start INTEGER,
            end INTEGER,
            duration INTEGER
        );

        CREATE TABLE processes (
            id INTEGER PRIMARY KEY,
            nid INTEGER,
            pid INTEGER,
            command TEXT
        );

        CREATE TABLE rocpd_info_agent (
            id INTEGER PRIMARY KEY,
            type TEXT,
            name TEXT
        );
        """
    )

    # 1 MB AllReduce per rank, 4 ranks. The spec's busBW calculation for
    # AllReduce is (msg_bytes * 2 * (N-1) / N) / duration_s; we size the
    # duration so the test can assert a specific busBW.
    #
    # msg_bytes   = 1 MiB = 1048576
    # factor      = 2*(4-1)/4 = 1.5
    # duration    = 1 ms   = 1e6 ns
    # => busBW    = 1048576 * 1.5 / 1e-3 / 1e9 ~= 1.572864 GB/s
    one_mib_count = 1024 * 1024  # 1 MiB of int8 => 1 MiB bytes
    # Duration 1 ms on each region so the busBW is deterministic.
    dur_ns = 1_000_000
    for rank in range(4):
        eid = 1000 + rank
        # region table (joined view)
        cur.execute(
            "INSERT INTO regions (id, category, name, start, end, nid, pid, "
            "tid, event_id) VALUES (?, 'RCCL', ?, ?, ?, ?, ?, ?, ?)",
            (rank + 1, "AllReduce", rank * 10_000,
             rank * 10_000 + dur_ns, 0, 1000 + rank, 100, eid),
        )
        # rocpd_region (raw) — lets _count_ranks fall back if needed.
        cur.execute(
            "INSERT INTO rocpd_region (id, nid, pid, tid, start, end, "
            "name_id, event_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rank + 1, 0, 1000 + rank, 100,
             rank * 10_000, rank * 10_000 + dur_ns, 1, eid),
        )
        cur.execute(
            "INSERT INTO rocpd_arg (event_id, position, type, name, value) "
            "VALUES (?, 0, 'uint64', 'count', ?)",
            (eid, str(one_mib_count)),
        )
        cur.execute(
            "INSERT INTO rocpd_arg (event_id, position, type, name, value) "
            "VALUES (?, 1, 'string', 'datatype', 'int8')",
            (eid,),
        )
        cur.execute(
            "INSERT INTO processes (id, nid, pid, command) VALUES (?, 0, ?, ?)",
            (rank + 1, 1000 + rank, f"rank{rank}"),
        )

    cur.execute("INSERT INTO rocpd_string (id, string) VALUES (1, 'AllReduce')")

    # One GEMM kernel overlapping the first AllReduce span (to exercise
    # the comm/compute overlap path).
    cur.execute(
        "INSERT INTO kernels (id, name, start, end, duration) "
        "VALUES (1, 'rocblas_gemm_f32', 0, ?, ?)",
        (dur_ns // 2, dur_ns // 2),
    )
    # And an RCCL-named kernel firing inside each AllReduce (to feed
    # NCCL categorization in tracelens_port).
    for rank in range(4):
        start = rank * 10_000
        cur.execute(
            "INSERT INTO kernels (id, name, start, end, duration) "
            "VALUES (?, 'ncclKernel_AllReduce_RING_LL_Sum_int8', ?, ?, ?)",
            (100 + rank, start, start + dur_ns, dur_ns),
        )

    cur.execute(
        "INSERT INTO rocpd_info_agent (id, type, name) VALUES (1, 'GPU', 'gfx942')"
    )

    conn.commit()
    conn.close()


def _build_fallback(path: Path) -> None:
    """Fallback fixture: kernels match RCCL regex but no category='RCCL' regions."""
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE regions (
            id INTEGER PRIMARY KEY,
            category TEXT,
            name TEXT,
            start INTEGER,
            end INTEGER,
            nid INTEGER,
            pid INTEGER,
            tid INTEGER,
            event_id INTEGER
        );
        CREATE TABLE rocpd_region (
            id INTEGER PRIMARY KEY,
            nid INTEGER,
            pid INTEGER,
            tid INTEGER,
            start INTEGER,
            end INTEGER,
            name_id INTEGER,
            event_id INTEGER
        );
        CREATE TABLE rocpd_string (
            id INTEGER PRIMARY KEY,
            string TEXT
        );
        CREATE TABLE rocpd_arg (
            id INTEGER PRIMARY KEY,
            event_id INTEGER,
            position INTEGER,
            type TEXT,
            name TEXT,
            value TEXT
        );
        CREATE TABLE kernels (
            id INTEGER PRIMARY KEY,
            name TEXT,
            start INTEGER,
            end INTEGER,
            duration INTEGER
        );
        CREATE TABLE processes (
            id INTEGER PRIMARY KEY,
            nid INTEGER,
            pid INTEGER,
            command TEXT
        );
        """
    )
    # No regions. Only kernels named like RCCL/nccl.
    for i in range(2):
        cur.execute(
            "INSERT INTO kernels (id, name, start, end, duration) "
            "VALUES (?, ?, ?, ?, ?)",
            (i + 1, "ncclKernel_AllReduce_RING_LL_Sum_f32",
             i * 1_000_000, i * 1_000_000 + 500_000, 500_000),
        )
    for pid in range(2):
        cur.execute(
            "INSERT INTO processes (id, nid, pid, command) VALUES (?, 0, ?, ?)",
            (pid + 1, 2000 + pid, f"rank{pid}"),
        )
    conn.commit()
    conn.close()


def _build_empty(path: Path) -> None:
    """DB with neither RCCL regions nor RCCL kernels."""
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE regions (
            id INTEGER PRIMARY KEY,
            category TEXT,
            name TEXT,
            start INTEGER,
            end INTEGER,
            nid INTEGER,
            pid INTEGER,
            tid INTEGER,
            event_id INTEGER
        );
        CREATE TABLE rocpd_region (
            id INTEGER PRIMARY KEY,
            nid INTEGER,
            pid INTEGER,
            tid INTEGER,
            start INTEGER,
            end INTEGER,
            name_id INTEGER,
            event_id INTEGER
        );
        CREATE TABLE rocpd_string (
            id INTEGER PRIMARY KEY,
            string TEXT
        );
        CREATE TABLE rocpd_arg (
            id INTEGER PRIMARY KEY,
            event_id INTEGER,
            position INTEGER,
            type TEXT,
            name TEXT,
            value TEXT
        );
        CREATE TABLE kernels (
            id INTEGER PRIMARY KEY,
            name TEXT,
            start INTEGER,
            end INTEGER,
            duration INTEGER
        );
        CREATE TABLE processes (
            id INTEGER PRIMARY KEY,
            nid INTEGER,
            pid INTEGER,
            command TEXT
        );
        """
    )
    # Only a plain GEMM kernel - no RCCL signal at all.
    cur.execute(
        "INSERT INTO kernels (id, name, start, end, duration) "
        "VALUES (1, 'rocblas_gemm_f32', 0, 1000000, 1000000)"
    )
    conn.commit()
    conn.close()


def main() -> None:
    _build(_OUT)
    _build_fallback(_OUT.parent / "rccl_fallback.db")
    _build_empty(_OUT.parent / "rccl_empty.db")
    print(f"wrote {_OUT}")
    print(f"wrote {_OUT.parent / 'rccl_fallback.db'}")
    print(f"wrote {_OUT.parent / 'rccl_empty.db'}")


if __name__ == "__main__":
    main()
