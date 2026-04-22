"""Generate small SQLite fixtures simulating rocpd kernel-dispatch tables."""

import sqlite3
import shutil
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent
PROVEN_OPTIMIZATIONS_DIR = FIXTURES_DIR / "proven_optimizations"


def create_rocpd_like(path: Path, kernels: list[tuple[str, int, int]]):
    """kernels = [(name, call_count, duration_ns_per_call)]"""
    path.parent.mkdir(parents=True, exist_ok=True)
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


PROVEN_OPTIMIZATION_CASES = [
    (
        "vgpr_reduction_compute_bound",
        [
            ("dense_gemm", 120, 780_000),
            ("epilogue", 80, 140_000),
            ("vector_add", 40, 90_000),
        ],
        [
            ("dense_gemm", 120, 650_000),
            ("epilogue", 80, 140_000),
            ("vector_add", 40, 90_000),
        ],
    ),
    (
        "mfma_enablement",
        [
            ("mfma_candidate", 96, 1_100_000),
            ("layout_convert", 32, 250_000),
        ],
        [
            ("mfma_candidate", 96, 610_000),
            ("layout_convert", 32, 240_000),
        ],
    ),
    (
        "fast_math_compiler_flag",
        [
            ("elementwise_poly", 180, 180_000),
            ("reduction", 24, 500_000),
        ],
        [
            ("elementwise_poly", 180, 155_000),
            ("reduction", 24, 500_000),
        ],
    ),
    (
        "memory_coalescing_stride_fix",
        [
            ("transpose_pass", 220, 260_000),
            ("staging_copy", 220, 220_000),
        ],
        [
            ("transpose_pass", 220, 215_000),
            ("staging_copy", 220, 180_000),
        ],
    ),
    (
        "lds_tiling_matmul",
        [
            ("blocked_matmul", 150, 540_000),
            ("global_loads", 150, 180_000),
        ],
        [
            ("blocked_matmul", 150, 430_000),
            ("global_loads", 150, 150_000),
        ],
    ),
    (
        "hip_stream_overlap",
        [
            ("memcpy_h2d", 64, 500_000),
            ("memcpy_d2h", 64, 480_000),
            ("compute_stage", 64, 140_000),
        ],
        [
            ("memcpy_h2d", 64, 395_000),
            ("memcpy_d2h", 64, 380_000),
            ("compute_stage", 64, 140_000),
        ],
    ),
    (
        "kernel_fusion_small_launches",
        [
            ("tiny_kernel_a", 900, 12_000),
            ("tiny_kernel_b", 900, 11_000),
            ("tiny_kernel_c", 900, 10_000),
        ],
        [
            ("tiny_kernel_a", 900, 8_500),
            ("tiny_kernel_b", 900, 8_000),
            ("tiny_kernel_c", 900, 7_500),
        ],
    ),
    (
        "device_sync_removal",
        [
            ("sync_heavy_step", 200, 85_000),
            ("followup_kernel", 200, 62_000),
        ],
        [
            ("sync_heavy_step", 200, 72_000),
            ("followup_kernel", 200, 58_000),
        ],
    ),
    (
        "warp_primitives_reduction",
        [
            ("block_reduce", 400, 44_000),
            ("tail_reduce", 80, 95_000),
        ],
        [
            ("block_reduce", 400, 38_000),
            ("tail_reduce", 80, 82_000),
        ],
    ),
    (
        "cache_blocking_kernel",
        [
            ("stencil_step", 140, 390_000),
            ("halo_exchange", 140, 170_000),
        ],
        [
            ("stencil_step", 140, 330_000),
            ("halo_exchange", 140, 155_000),
        ],
    ),
]


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

    if PROVEN_OPTIMIZATIONS_DIR.exists():
        shutil.rmtree(PROVEN_OPTIMIZATIONS_DIR)

    for case_id, baseline_kernels, optimized_kernels in PROVEN_OPTIMIZATION_CASES:
        case_dir = PROVEN_OPTIMIZATIONS_DIR / case_id
        create_rocpd_like(case_dir / "baseline.db", baseline_kernels)
        create_rocpd_like(case_dir / "optimized.db", optimized_kernels)

    print("Fixtures generated.")
