"""Generate small SQLite fixtures simulating rocpd kernel-dispatch tables."""

import sqlite3
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent


def create_rocpd_like(path: Path, kernels: list[tuple[str, int, int]]):
    """kernels = [(name, call_count, duration_ns_per_call)]"""
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE rocpd_kernel_dispatch (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            duration_ns INTEGER NOT NULL
        );
    """)
    for name, calls, dur in kernels:
        for _ in range(calls):
            conn.execute(
                "INSERT INTO rocpd_kernel_dispatch (name, duration_ns) VALUES (?, ?)",
                (name, dur),
            )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(exist_ok=True)

    # Baseline: hot kernel matmul (70% of time), medium conv (20%), small add (10%)
    create_rocpd_like(FIXTURES_DIR / "regression_baseline.db", [
        ("matmul", 100, 700_000),   # 70s total
        ("conv2d", 50, 400_000),    # 20s
        ("add",    20, 500_000),    # 10s
    ])

    # Improved: matmul is 20% faster, others same
    create_rocpd_like(FIXTURES_DIR / "regression_improved.db", [
        ("matmul", 100, 560_000),   # 56s (-20%)
        ("conv2d", 50, 400_000),
        ("add",    20, 500_000),
    ])

    # Tail-hurt: matmul unchanged; small kernels collectively regress 15%
    create_rocpd_like(FIXTURES_DIR / "regression_tail_hurt.db", [
        ("matmul", 100, 700_000),
        ("conv2d", 50, 460_000),   # +15%
        ("add",    20, 575_000),   # +15%
    ])

    print("Fixtures generated.")
