# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for roofline/benchmark: the kernel-source contract and GPU locking.

Each benchmark in Bench_base compiles an architecture's kernel source with
hiprtc and then asks Program.get_kernel() for a specific kernel by name. A
source that does not define that name still compiles, so the mismatch only
surfaces as a HIP error on real hardware. These tests resolve every kernel
name each architecture would request and assert its source defines it.

No GPU is required; the sources are plain strings built in the constructor.
"""

import csv
import fcntl
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import roofline.benchmark.benchmark_base as benchmark_base
import utils.utils_profile as utils_profile
from tests.common import check_roofline_csv

try:
    from roofline.benchmark.gfx9.benchmark_gfx90a import Bench_gfx90a
    from roofline.benchmark.gfx9.benchmark_gfx942 import Bench_gfx942
    from roofline.benchmark.gfx9.benchmark_gfx950 import Bench_gfx950
    from roofline.benchmark.gfx11.benchmark_gfx1150 import Bench_gfx1150
    from roofline.benchmark.gfx11.benchmark_gfx1151 import Bench_gfx1151
    from roofline.benchmark.gfx11.benchmark_gfx1152 import Bench_gfx1152
    from roofline.benchmark.gfx11.benchmark_gfx1153 import Bench_gfx1153
    from roofline.benchmark.gfx12.benchmark_gfx1250 import Bench_gfx1250
except OSError as hip_load_error:
    # hip_interface and hiprtc_interface dlopen the ROCm libraries at import.
    pytest.skip(
        f"ROCm runtime libraries unavailable: {hip_load_error}",
        allow_module_level=True,
    )

# =============================================================================
# Test data
# =============================================================================

BENCH_CLASSES = [
    Bench_gfx90a,
    Bench_gfx942,
    Bench_gfx950,
    Bench_gfx1150,
    Bench_gfx1151,
    Bench_gfx1152,
    Bench_gfx1153,
    Bench_gfx1250,
]

# set_cache_kernel_selector() derives a kernel name per level present here, so
# every level is populated to exercise the whole selector.
CACHE_SIZES = {
    "L0": 32 * 1024,
    "L1": 16 * 1024,
    "L2": 4 * 1024 * 1024,
    "MALL": 256 * 1024 * 1024,
}

# Mirrors the source selection in Bench_base.matrix_bench().
MATRIX_SOURCE_ATTRS = {
    "F16": "matrix_f16_src",
    "F32": "matrix_f32_src",
    "F64": "matrix_f64_src",
    "F8": "matrix_f8_src",
    "BF16": "matrix_bf16_src",
    "I8": "matrix_i8_src",
}
PACKED_MATRIX_SOURCE_ATTR = "matrix_f8f6f4_src"

# flops_kernel_selector keys against the names used in Bench.tests.
FLOPS_TEST_KEYS = {
    "FP16": "F16",
    "BF16": "BF16",
    "FP32": "F32",
    "FP64": "F64",
    "INT8": "I8",
    "INT32": "I32",
    "INT64": "I64",
}

# Benchmarks whose launcher hard-codes the kernel name it resolves.
LITERAL_KERNEL_NAMES = [
    ("HBM", "hbm_bw_src", "HBM_bw"),
    ("LDS", "lds_benchmark_src", "LDS_bw"),
]


# =============================================================================
# Helpers
# =============================================================================


def assert_source_defines(kernel_source, kernel_name, context):
    """Assert the source defines the symbol Program.get_kernel() resolves.

    get_kernel() routes any name containing "<" through hiprtcGetLoweredName,
    which needs a matching template in the source. Every other name is looked
    up verbatim in the code object, which only succeeds when the kernel is
    declared extern "C" and so escapes C++ name mangling.
    """
    assert kernel_source.strip(), f"{context}: kernel source is empty"

    if "<" in kernel_name:
        template_name = kernel_name.split("<", 1)[0]
        assert "template" in kernel_source, (
            f"{context}: {kernel_name} is a template instantiation but the "
            f"source declares no template"
        )
        assert f"__global__ void {template_name}(" in kernel_source, (
            f"{context}: source does not define __global__ {template_name}"
        )
        return

    assert f'extern "C" __global__ void {kernel_name}(' in kernel_source, (
        f'{context}: source does not define extern "C" __global__ {kernel_name}'
    )


def runs_benchmark(bench, test_key):
    """Report whether run_benchmark() would launch this benchmark."""
    return test_key in bench.tests and test_key not in bench.unsupported_data_types


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize("bench_class", BENCH_CLASSES, ids=lambda cls: cls.__name__)
def test_literal_kernel_sources_define_their_symbol(bench_class):
    """HBM and LDS sources must define the names their launchers hard-code."""
    bench = bench_class(0, dict(CACHE_SIZES))

    for test_key, source_attr, kernel_name in LITERAL_KERNEL_NAMES:
        if not runs_benchmark(bench, test_key):
            continue
        assert_source_defines(
            getattr(bench, source_attr),
            kernel_name,
            f"{bench_class.__name__} {test_key}",
        )


@pytest.mark.parametrize("bench_class", BENCH_CLASSES, ids=lambda cls: cls.__name__)
def test_cache_kernel_sources_define_selected_symbol(bench_class):
    """Every cache level selects a Cache_bw instantiation from one source."""
    bench = bench_class(0, dict(CACHE_SIZES))
    assert bench.cache_kernel_selector, "no cache kernels selected"

    for cache_type, kernel_name in bench.cache_kernel_selector.items():
        if not runs_benchmark(bench, cache_type):
            continue
        assert_source_defines(
            bench.cache_bw_src, kernel_name, f"{bench_class.__name__} {cache_type}"
        )


@pytest.mark.parametrize("bench_class", BENCH_CLASSES, ids=lambda cls: cls.__name__)
def test_flops_kernel_sources_define_selected_symbol(bench_class):
    """VALU benchmarks read their kernel name from flops_kernel_selector."""
    bench = bench_class(0, dict(CACHE_SIZES))

    for flops_type, selection in bench.flops_kernel_selector.items():
        test_key = FLOPS_TEST_KEYS[flops_type]
        if not runs_benchmark(bench, test_key):
            continue
        source_attr = (
            "bf16_flops_benchmark_src"
            if flops_type == "BF16"
            else "flops_benchmark_src"
        )
        assert_source_defines(
            getattr(bench, source_attr),
            selection[0],
            f"{bench_class.__name__} {test_key}",
        )


@pytest.mark.parametrize("bench_class", BENCH_CLASSES, ids=lambda cls: cls.__name__)
def test_matrix_kernel_sources_define_selected_symbol(bench_class):
    """Matrix benchmarks read their kernel name from matrix_kernel_selector."""
    bench = bench_class(0, dict(CACHE_SIZES))

    for matrix_type, kernel_name in bench.matrix_kernel_selector.items():
        test_key = f"{bench.MATRIX_OPS_TYPE}-{matrix_type}"
        if not runs_benchmark(bench, test_key):
            continue
        source_attr = MATRIX_SOURCE_ATTRS.get(matrix_type, PACKED_MATRIX_SOURCE_ATTR)
        assert_source_defines(
            getattr(bench, source_attr),
            kernel_name,
            f"{bench_class.__name__} {test_key}",
        )


# =============================================================================
# GPU Benchmark Locking Tests
# =============================================================================


@pytest.mark.misc
def test_gpu_benchmark_locking(tmp_path, monkeypatch, capsys):
    """Test GPU benchmark locking functions."""

    # --- Setup: redirect lock directory to temp path ---
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()

    # Mock GPU UUID
    monkeypatch.setattr(
        benchmark_base.hip,
        "hipGetDeviceProperties",
        lambda d: mock.Mock(uuid=mock.Mock(uuid=bytes([0x01, 0x02, 0x03, 0x04]))),
    )

    # Mock Path to use our temp directory
    original_path = Path

    def mock_path(p):
        if p == "/tmp/rocprof-compute-benchmark":
            return lock_dir
        return original_path(p)

    monkeypatch.setattr(benchmark_base, "Path", mock_path)

    deviceID = 0
    cache_sizes = {}
    # Create Bench_base object in order to call gpu benchmark lock method
    # Device ID list arg doesn't matter since we are just using the base class
    # cache_sizes can be empty for this test since we do not need it to test locking
    testClass = benchmark_base.Bench_base(deviceID, cache_sizes)

    # --- Test lock acquisition and lock file creation ---
    with testClass.gpu_benchmark_lock(deviceID):
        lock_file = lock_dir / "rocprof-compute-benchmark-01020304.lock"
        assert lock_file.exists()

    # --- Test no message when lock acquired immediately ---
    capsys.readouterr()  # Clear previous output
    with testClass.gpu_benchmark_lock(deviceID):
        pass
    output = capsys.readouterr().out
    assert "Waiting" not in output

    # --- Test waiting/acquired messages when lock is contended ---
    call_count = {"count": 0}

    def mock_flock(fd, op):
        call_count["count"] += 1
        if call_count["count"] == 1 and (op & fcntl.LOCK_NB):
            raise BlockingIOError("Lock held by another process")

    monkeypatch.setattr(utils_profile.fcntl, "flock", mock_flock)

    with testClass.gpu_benchmark_lock(deviceID):
        pass

    output = capsys.readouterr().out
    assert "Waiting for GPU 0" in output
    assert "another rocprof-compute benchmark is in progress" in output
    assert "Acquired lock for GPU 0" in output


@pytest.mark.parametrize("bench_class", BENCH_CLASSES, ids=lambda cls: cls.__name__)
def test_run_benchmark_skips_unsupported_callbacks(bench_class, monkeypatch, tmp_path):
    """Unsupported benchmark callbacks never run and export zero triplets."""
    bench = bench_class(0, dict(CACHE_SIZES))
    calls = []
    supported_metrics = benchmark_base.PerfMetrics(20.0, 19.0, 21.0)

    def callback(name):
        def run(device):
            assert device == 0
            assert name not in bench.unsupported_data_types
            calls.append(name)
            return supported_metrics

        return run

    bench.tests = {name: callback(name) for name in bench.tests}
    monkeypatch.setattr(bench, "gpu_benchmark_lock", lambda device: nullcontext())
    monkeypatch.setattr(bench, "get_gfx_arch", lambda device: "test-arch")
    monkeypatch.setattr(
        benchmark_base.hip,
        "hipGetDeviceProperties",
        lambda device: SimpleNamespace(multiProcessorCount=120),
    )

    metrics = bench.run_benchmark(0)
    expected_calls = [
        name for name in bench.tests if name not in bench.unsupported_data_types
    ]
    assert calls == expected_calls
    assert set(metrics) == set(bench.tests)
    for name, values in metrics.items():
        expected = (
            benchmark_base.PerfMetrics(0, 0, 0)
            if name in bench.unsupported_data_types
            else supported_metrics
        )
        assert values == expected

    csv_path = tmp_path / "roofline.csv"
    bench.dump_csv(metrics, str(csv_path))
    allowed_zero = {
        bench.csv_cols_map[name]
        for name in bench.unsupported_data_types
        if name in bench.csv_cols_map
    }
    frame = check_roofline_csv(csv_path, allowed_zero=allowed_zero)
    assert frame["device"].tolist() == [0]
    if bench_class is Bench_gfx942:
        assert allowed_zero == {
            "MFMAF4Flops",
            "MFMAF6Flops",
            "MFMAF6F4Flops",
            "MFMAMXF8Flops",
        }
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 2
    for name in bench.unsupported_data_types:
        if name in bench.csv_cols_map:
            index = rows[0].index(bench.csv_cols_map[name])
            assert rows[1][index : index + 3] == ["0", "0", "0"]


def test_dump_csv_preserves_device_and_ordered_metric_triplets(tmp_path):
    """CSV mean/Low/High columns retain map order and their matching values."""
    bench = benchmark_base.Bench_base(0, {})
    bench.csv_cols_map = {"F32": "FP32Flops", "HBM": "HBMBw", "I8": "I8Ops"}
    metrics = {
        "HBM": benchmark_base.PerfMetrics(200, 190, 210),
        "I8": benchmark_base.PerfMetrics(300, 290, 310),
        "F32": benchmark_base.PerfMetrics(100, 90, 110),
    }
    csv_path = tmp_path / "roofline.csv"

    bench.dump_csv(metrics, str(csv_path))

    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [
            [
                "device",
                "FP32Flops",
                "FP32FlopsLow",
                "FP32FlopsHigh",
                "HBMBw",
                "HBMBwLow",
                "HBMBwHigh",
                "I8Ops",
                "I8OpsLow",
                "I8OpsHigh",
            ],
            ["0", "100", "90", "110", "200", "190", "210", "300", "290", "310"],
        ]
