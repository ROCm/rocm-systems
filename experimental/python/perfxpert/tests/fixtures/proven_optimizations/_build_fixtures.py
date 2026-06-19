"""Generate the 10 proven-optimization fixture DB pairs.

Run once at PR time — outputs are committed. Regeneration:
    python3 tests/fixtures/proven_optimizations/_build_fixtures.py

Schema follows rocpd UUID-based format (CLAUDE.md §SQLite + rocpd):
    - rocpd_metadata
    - rocpd_kernel_dispatch_<uuid>
    - rocpd_api_<uuid>
    - rocpd_memory_copy_<uuid>
    - pmc_events_<uuid> (optional, only when case requires counters)
    - rocpd_agent_<uuid>

Every fixture is < 256 KB (no GB-scale counter tables); we hand-pick
only the counters that the integration test reads.
"""

import sqlite3
import uuid
from pathlib import Path
from typing import Dict, List, Any

FIXTURE_DIR = Path(__file__).parent


def _new_uuid() -> str:
    return uuid.uuid4().hex[:16]


def _init_schema(conn: sqlite3.Connection, uid: str) -> None:
    cur = conn.cursor()
    cur.executescript(f"""
    CREATE TABLE IF NOT EXISTS rocpd_metadata (
        key TEXT PRIMARY KEY, value TEXT
    );
    CREATE TABLE IF NOT EXISTS rocpd_agent_{uid} (
        id INTEGER PRIMARY KEY, gfx_id TEXT, name TEXT, cu_count INTEGER
    );
    CREATE TABLE IF NOT EXISTS rocpd_kernel_dispatch_{uid} (
        id INTEGER PRIMARY KEY, name TEXT, duration_ns INTEGER,
        grid_size_x INTEGER, block_size_x INTEGER, vgpr_per_thread INTEGER,
        lds_usage_bytes INTEGER, start_ns INTEGER
    );
    CREATE TABLE IF NOT EXISTS rocpd_api_{uid} (
        id INTEGER PRIMARY KEY, name TEXT, duration_ns INTEGER
    );
    CREATE TABLE IF NOT EXISTS rocpd_memory_copy_{uid} (
        id INTEGER PRIMARY KEY, direction TEXT, duration_ns INTEGER, bytes INTEGER
    );
    CREATE TABLE IF NOT EXISTS pmc_events_{uid} (
        id INTEGER PRIMARY KEY, counter_name TEXT, kernel_id INTEGER, value INTEGER
    );
    """)
    cur.execute("INSERT OR REPLACE INTO rocpd_metadata(key,value) VALUES (?,?)",
                ("active_uuid", uid))
    cur.execute("INSERT OR REPLACE INTO rocpd_metadata(key,value) VALUES (?,?)",
                ("schema_version", "rocpd_v3"))
    cur.execute(
        f"INSERT INTO rocpd_agent_{uid}(id,gfx_id,name,cu_count) VALUES (0,'gfx942','MI300X',304)"
    )
    conn.commit()


def _write_fixture(path: Path, kernels: List[Dict[str, Any]],
                   memcpy: List[Dict[str, Any]] = (),
                   api: List[Dict[str, Any]] = (),
                   counters: List[Dict[str, Any]] = ()) -> None:
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    uid = _new_uuid()
    _init_schema(conn, uid)
    cur = conn.cursor()
    for k in kernels:
        cur.execute(
            f"INSERT INTO rocpd_kernel_dispatch_{uid}(name,duration_ns,grid_size_x,block_size_x,vgpr_per_thread,lds_usage_bytes,start_ns) "
            f"VALUES (?,?,?,?,?,?,?)",
            (k["name"], k["duration_ns"], k.get("grid", 1024), k.get("block", 256),
             k.get("vgpr", 80), k.get("lds", 0), k.get("start_ns", 0)))
    for m in memcpy:
        cur.execute(
            f"INSERT INTO rocpd_memory_copy_{uid}(direction,duration_ns,bytes) VALUES (?,?,?)",
            (m["direction"], m["duration_ns"], m["bytes"]))
    for a in api:
        cur.execute(
            f"INSERT INTO rocpd_api_{uid}(name,duration_ns) VALUES (?,?)",
            (a["name"], a["duration_ns"]))
    for cnt in counters:
        cur.execute(
            f"INSERT INTO pmc_events_{uid}(counter_name,kernel_id,value) VALUES (?,?,?)",
            (cnt["counter_name"], cnt["kernel_id"], cnt["value"]))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Fixture definitions — one function per case id.
# Each returns (baseline_spec, optimized_spec) dicts suitable for _write_fixture.
# ---------------------------------------------------------------------------

def case_vgpr_reduction_compute_bound():
    # Baseline: 1 dominant kernel, 128 VGPR, waves/EU=4, compute-bound signature
    baseline = dict(
        kernels=[{"name": "matmul_kernel", "duration_ns": 1_000_000_000,
                  "vgpr": 128, "lds": 0}],
        counters=[
            {"counter_name": "SQ_WAVES",        "kernel_id": 1, "value": 1024},
            {"counter_name": "GRBM_GUI_ACTIVE", "kernel_id": 1, "value": 950_000},
            {"counter_name": "GRBM_COUNT",      "kernel_id": 1, "value": 1_000_000},
        ])
    # Optimized: same kernel, 64 VGPR, waves/EU=8, ~30% faster
    optimized = dict(
        kernels=[{"name": "matmul_kernel", "duration_ns": 700_000_000,
                  "vgpr": 64, "lds": 0}],
        counters=[
            {"counter_name": "SQ_WAVES",        "kernel_id": 1, "value": 2048},
            {"counter_name": "GRBM_GUI_ACTIVE", "kernel_id": 1, "value": 680_000},
            {"counter_name": "GRBM_COUNT",      "kernel_id": 1, "value": 700_000},
        ])
    return baseline, optimized


def case_memory_coalescing_stride_fix():
    # Baseline: high L1 miss, low hbm util, one dominant stride-bad kernel
    baseline = dict(
        kernels=[{"name": "strided_copy", "duration_ns": 1_000_000_000, "vgpr": 48}],
        counters=[
            {"counter_name": "FETCH_SIZE",  "kernel_id": 1, "value": 800_000},
            {"counter_name": "TCP_TCC_READ_REQ_sum", "kernel_id": 1, "value": 900_000},
        ])
    optimized = dict(
        kernels=[{"name": "coalesced_copy", "duration_ns": 250_000_000, "vgpr": 48}],
        counters=[
            {"counter_name": "FETCH_SIZE",  "kernel_id": 1, "value": 200_000},
            {"counter_name": "TCP_TCC_READ_REQ_sum", "kernel_id": 1, "value": 200_000},
        ])
    return baseline, optimized


def case_mfma_enablement():
    baseline = dict(
        kernels=[{"name": "gemm_vectors", "duration_ns": 5_000_000_000, "vgpr": 96}],
        counters=[
            {"counter_name": "SQ_INSTS_VALU", "kernel_id": 1, "value": 10_000_000},
            {"counter_name": "SQ_INSTS_MFMA", "kernel_id": 1, "value": 0},
        ])
    optimized = dict(
        kernels=[{"name": "gemm_mfma", "duration_ns": 800_000_000, "vgpr": 96}],
        counters=[
            {"counter_name": "SQ_INSTS_VALU", "kernel_id": 1, "value": 1_000_000},
            {"counter_name": "SQ_INSTS_MFMA", "kernel_id": 1, "value": 9_000_000},
        ])
    return baseline, optimized


def case_fast_math_compiler_flag():
    baseline = dict(kernels=[{"name": "transcendental_heavy",
                              "duration_ns": 1_000_000_000, "vgpr": 64}])
    optimized = dict(kernels=[{"name": "transcendental_heavy",
                               "duration_ns": 800_000_000, "vgpr": 64}])
    return baseline, optimized


def case_lds_tiling_matmul():
    baseline = dict(kernels=[{"name": "matmul_global",
                              "duration_ns": 3_000_000_000, "vgpr": 96, "lds": 0}],
                    counters=[{"counter_name": "FETCH_SIZE", "kernel_id": 1, "value": 5_000_000}])
    optimized = dict(kernels=[{"name": "matmul_lds",
                               "duration_ns": 900_000_000, "vgpr": 96, "lds": 16384}],
                     counters=[{"counter_name": "FETCH_SIZE", "kernel_id": 1, "value": 600_000}])
    return baseline, optimized


def case_hip_stream_overlap():
    baseline = dict(
        kernels=[{"name": "compute_stage", "duration_ns": 500_000_000, "start_ns": 500_000_000}],
        memcpy=[{"direction": "H2D", "duration_ns": 500_000_000, "bytes": 64_000_000}])
    optimized = dict(
        kernels=[{"name": "compute_stage", "duration_ns": 500_000_000, "start_ns": 0}],
        memcpy=[{"direction": "H2D", "duration_ns": 500_000_000, "bytes": 64_000_000}])
    return baseline, optimized


def case_kernel_fusion_small_launches():
    baseline = dict(kernels=[{"name": f"pw_op_{i}", "duration_ns": 8_000} for i in range(2000)],
                    api=[{"name": "hipLaunchKernel", "duration_ns": 2_000}
                         for _ in range(2000)])
    optimized = dict(kernels=[{"name": "fused_pw", "duration_ns": 8_000_000}],
                     api=[{"name": "hipLaunchKernel", "duration_ns": 2_000}])
    return baseline, optimized


def case_device_sync_removal():
    baseline = dict(
        kernels=[{"name": "k", "duration_ns": 10_000_000} for _ in range(20)],
        api=[{"name": "hipDeviceSynchronize", "duration_ns": 50_000_000} for _ in range(20)])
    optimized = dict(
        kernels=[{"name": "k", "duration_ns": 10_000_000} for _ in range(20)],
        api=[{"name": "hipDeviceSynchronize", "duration_ns": 50_000_000} for _ in range(2)])
    return baseline, optimized


def case_warp_primitives_reduction():
    baseline = dict(kernels=[{"name": "lds_reduction",
                              "duration_ns": 1_000_000_000, "lds": 4096}],
                    counters=[{"counter_name": "SQ_LDS_BANK_CONFLICT",
                               "kernel_id": 1, "value": 1_000_000}])
    optimized = dict(kernels=[{"name": "shfl_reduction",
                               "duration_ns": 600_000_000, "lds": 0}],
                     counters=[{"counter_name": "SQ_LDS_BANK_CONFLICT",
                                "kernel_id": 1, "value": 0}])
    return baseline, optimized


def case_cache_blocking_kernel():
    baseline = dict(kernels=[{"name": "stencil_bad_block", "duration_ns": 1_500_000_000}],
                    counters=[
                        {"counter_name": "TCC_HIT_sum",  "kernel_id": 1, "value": 300_000},
                        {"counter_name": "TCC_MISS_sum", "kernel_id": 1, "value": 700_000},
                    ])
    optimized = dict(kernels=[{"name": "stencil_tuned_block", "duration_ns": 900_000_000}],
                     counters=[
                         {"counter_name": "TCC_HIT_sum",  "kernel_id": 1, "value": 800_000},
                         {"counter_name": "TCC_MISS_sum", "kernel_id": 1, "value": 200_000},
                     ])
    return baseline, optimized


CASES = [
    ("vgpr_reduction_compute_bound",    case_vgpr_reduction_compute_bound),
    ("memory_coalescing_stride_fix",    case_memory_coalescing_stride_fix),
    ("mfma_enablement",                 case_mfma_enablement),
    ("fast_math_compiler_flag",         case_fast_math_compiler_flag),
    ("lds_tiling_matmul",               case_lds_tiling_matmul),
    ("hip_stream_overlap",              case_hip_stream_overlap),
    ("kernel_fusion_small_launches",    case_kernel_fusion_small_launches),
    ("device_sync_removal",             case_device_sync_removal),
    ("warp_primitives_reduction",       case_warp_primitives_reduction),
    ("cache_blocking_kernel",           case_cache_blocking_kernel),
]


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for case_id, builder in CASES:
        baseline_spec, optimized_spec = builder()
        _write_fixture(FIXTURE_DIR / f"{case_id}.baseline.db", **baseline_spec)
        _write_fixture(FIXTURE_DIR / f"{case_id}.optimized.db", **optimized_spec)
        print(f"wrote {case_id}.{{baseline,optimized}}.db")


if __name__ == "__main__":
    main()
