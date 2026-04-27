"""Tests for perfxpert.tools.metrics — HW counter arithmetic primitives."""

import pytest

from perfxpert.tools import metrics
from perfxpert.tools._class import ToolClass


# -- compute_gpu_utilization -----------------------------------------------

def test_gpu_utilization_normal():
    """GPU busy 800M cycles out of 1G total = 0.80 utilization."""
    r = metrics.compute_gpu_utilization(grbm_gui_active=800_000_000, grbm_count=1_000_000_000)
    assert r == pytest.approx(0.80)


def test_gpu_utilization_zero_count_raises():
    """Divide-by-zero must raise ValueError."""
    with pytest.raises(ValueError):
        metrics.compute_gpu_utilization(grbm_gui_active=0, grbm_count=0)


# -- compute_l1_miss_rate --------------------------------------------------

def test_l1_miss_rate_normal():
    """200 misses out of 1000 total reads = 0.20 miss rate."""
    r = metrics.compute_l1_miss_rate(tcp_tcc_read_req=200, tcp_total_read=1000)
    assert r == pytest.approx(0.20)


def test_l1_miss_rate_zero_reads_raises():
    with pytest.raises(ValueError):
        metrics.compute_l1_miss_rate(tcp_tcc_read_req=0, tcp_total_read=0)


# -- compute_l2_hit_rate ---------------------------------------------------

def test_l2_hit_rate_normal():
    """900 hits out of 1000 total = 0.90 hit rate."""
    r = metrics.compute_l2_hit_rate(tcc_hit=900, tcc_miss=100)
    assert r == pytest.approx(0.90)


def test_l2_hit_rate_zero_accesses_raises():
    with pytest.raises(ValueError):
        metrics.compute_l2_hit_rate(tcc_hit=0, tcc_miss=0)


# -- compute_hbm_bandwidth -------------------------------------------------

def test_hbm_bandwidth_normal():
    """1 GiB read + 1 GiB write in 1 second = 2 GiB/s ≈ 2.147 GB/s."""
    r = metrics.compute_hbm_bandwidth(
        fetch_kib=1024 * 1024,      # 1 GiB in KiB
        write_kib=1024 * 1024,      # 1 GiB in KiB
        duration_ns=1_000_000_000,  # 1 second
    )
    assert "gb_per_s" in r
    assert "gib_per_s" in r
    assert r["gib_per_s"] == pytest.approx(2.0)
    # 2 GiB = 2 * 2^30 bytes = 2_147_483_648 bytes / 1 s = 2.1475 GB/s (decimal)
    assert r["gb_per_s"] == pytest.approx(2.147483648, rel=1e-6)


def test_hbm_bandwidth_zero_duration_raises():
    with pytest.raises(ValueError):
        metrics.compute_hbm_bandwidth(fetch_kib=1024, write_kib=1024, duration_ns=0)


# -- compute_latency -------------------------------------------------------

def test_latency_normal():
    """VMEM latency = accum_prev_hires / insts_vmem; LDS latency = .../insts_lds;
       busy_ratio = busy_cycles / (vmem+lds+busy) approximation placeholder."""
    r = metrics.compute_latency(
        accum_prev_hires=10000,
        insts_vmem=100,
        insts_lds=50,
        busy_cycles=8000,
    )
    assert "vmem_latency_cycles" in r
    assert "lds_latency_cycles" in r
    assert "busy_ratio" in r
    assert r["vmem_latency_cycles"] == pytest.approx(100.0)   # 10000/100
    assert r["lds_latency_cycles"] == pytest.approx(200.0)    # 10000/50


# -- READ_ONLY class annotation --------------------------------------------

def test_all_functions_are_read_only():
    assert metrics.compute_gpu_utilization.__tool_class__ == ToolClass.READ_ONLY
    assert metrics.compute_l1_miss_rate.__tool_class__ == ToolClass.READ_ONLY
    assert metrics.compute_l2_hit_rate.__tool_class__ == ToolClass.READ_ONLY
    assert metrics.compute_hbm_bandwidth.__tool_class__ == ToolClass.READ_ONLY
    assert metrics.compute_latency.__tool_class__ == ToolClass.READ_ONLY
