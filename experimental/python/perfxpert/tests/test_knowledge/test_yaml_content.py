"""Cross-YAML consistency: references in one YAML exist in the other.

These tests catch stale references before agents hit runtime errors.
"""

from perfxpert.knowledge import load_yaml


def test_every_arch_in_vgpr_occupancy_has_gpu_specs_entry():
    specs = load_yaml("gpu_specs")
    vgpr = load_yaml("vgpr_occupancy_tables")
    for gfx in vgpr:
        assert gfx in specs, (
            f"vgpr_occupancy_tables has gfx={gfx!r} not in gpu_specs"
        )


def test_gpu_specs_runtime_caps_cover_all_vgpr_table_archs():
    specs = load_yaml("gpu_specs")
    vgpr = load_yaml("vgpr_occupancy_tables")
    for gfx in vgpr:
        entry = specs[gfx]
        for key in ("vgprs_per_simd", "simds_per_cu", "max_waves_per_simd", "lds_per_cu_kb"):
            assert key in entry, f"gpu_specs missing {key!r} for {gfx!r}"


def test_counter_catalog_blocks_appear_in_pmc_limits():
    """Every block referenced in counter_catalog should have a per-block limit."""
    catalog = load_yaml("counter_catalog")
    limits = load_yaml("pmc_limits")["per_block_limits"]

    blocks_in_catalog = {entry["block"] for entry in catalog}
    for block in blocks_in_catalog:
        assert block in limits, (
            f"counter_catalog references block {block!r} not in pmc_limits"
        )


def test_bottleneck_types_metrics_documented():
    """Every metric referenced in bottleneck signatures is one we can produce."""
    types = load_yaml("bottleneck_types")
    referenced = set()
    for entry in types:
        for sig in entry.get("signatures", []):
            referenced.add(sig["metric"])

    # Metrics we can produce via analysis tools
    producible = {
        "valu_util_pct", "mfma_util_pct",
        "arithmetic_intensity_above_ridge", "arithmetic_intensity_below_ridge",
        "memcpy_pct", "hbm_bw_utilization",
        "avg_waves_per_cu", "gpu_util_pct", "occupancy_pct",
        "api_overhead_pct", "avg_kernel_duration_us", "total_kernel_calls",
        "no_dominant_bottleneck",
    }
    undocumented = referenced - producible
    assert not undocumented, (
        f"bottleneck_types references metrics that no tool produces: {undocumented}"
    )


def test_amdahl_thresholds_high_greater_than_low():
    t = load_yaml("amdahl_thresholds")
    assert t["high_threshold"] > t["low_threshold"]


def test_optimization_techniques_categories_valid():
    """Each technique's category must be compute/memory/latency."""
    techs = load_yaml("optimization_techniques")
    valid = {"compute", "memory", "latency"}
    for t in techs:
        assert t["category"] in valid, (
            f"technique {t['id']!r} has invalid category {t['category']!r}"
        )


def test_sol_metrics_have_sensible_thresholds():
    """high_threshold > medium_threshold for every SOL metric."""
    metrics = load_yaml("sol_metrics")
    for m in metrics:
        assert m["high_threshold"] > m["medium_threshold"], (
            f"{m['name']}: high_threshold ({m['high_threshold']}) "
            f"must exceed medium_threshold ({m['medium_threshold']})"
        )
