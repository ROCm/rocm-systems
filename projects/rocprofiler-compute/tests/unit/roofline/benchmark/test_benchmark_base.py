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

import fcntl
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import roofline.benchmark.benchmark_base as benchmark_base
import utils.utils_profile as utils_profile
from roofline import run_benchmark
from roofline.benchmark.gfx12.benchmark_gfx1201 import (
    INFINITY_CACHE_BYTES,
    L2_ROOF,
    MALL_WORKING_SET_DIVISOR,
    percent_of_ceiling,
)
from utils.mi_gpu_spec import mi_gpu_specs

try:
    # The class of every arch the roofline benchmark runs on, imported as load_bench()
    # imports it.
    ROOFLINE_ARCHS = mi_gpu_specs.get_roofline_archs()
    BENCH_CLASSES = {arch: run_benchmark.bench_class(arch) for arch in ROOFLINE_ARCHS}
except OSError as hip_load_error:
    # hip_interface and hiprtc_interface dlopen the ROCm libraries at import.
    pytest.skip(
        f"ROCm runtime libraries unavailable: {hip_load_error}",
        allow_module_level=True,
    )

# =============================================================================
# Test data
# =============================================================================

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
    "BF8": "matrix_bf8_src",
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
    ("HBM", "device_memory_bw_src", "HBM_bw"),
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


@pytest.mark.parametrize("gpu_arch", ROOFLINE_ARCHS)
def test_spec_matches_the_benchmark_class(gpu_arch):
    """Every ceiling mi_gpu_spec.yaml lists is a column the class writes, and the
    class's matrix tests use the spec's matrix type; otherwise a ceiling is never
    measured."""
    bench = BENCH_CLASSES[gpu_arch](0, dict(CACHE_SIZES), gpu_arch)
    columns = mi_gpu_specs.get_roofline_columns(gpu_arch)
    assert columns <= set(bench.csv_cols_map.values())
    matrix_types = {test.split("-")[0] for test in bench.tests if "-" in test}
    assert matrix_types == {mi_gpu_specs.get_roofline_matrix_type(gpu_arch)}


@pytest.mark.parametrize("gpu_arch", ROOFLINE_ARCHS)
def test_literal_kernel_sources_define_their_symbol(gpu_arch):
    """Device-memory and LDS sources must define the names their launchers resolve."""
    bench = BENCH_CLASSES[gpu_arch](0, dict(CACHE_SIZES), gpu_arch)

    for test_key, source_attr, kernel_name in LITERAL_KERNEL_NAMES:
        if not runs_benchmark(bench, test_key):
            continue
        assert_source_defines(
            getattr(bench, source_attr),
            kernel_name,
            f"{gpu_arch} {test_key}",
        )


@pytest.mark.parametrize("gpu_arch", ROOFLINE_ARCHS)
def test_cache_kernel_sources_define_selected_symbol(gpu_arch):
    """Every cache level selects a Cache_bw instantiation from one source."""
    bench = BENCH_CLASSES[gpu_arch](0, dict(CACHE_SIZES), gpu_arch)
    assert bench.cache_kernel_selector, "no cache kernels selected"

    for cache_type, kernel_name in bench.cache_kernel_selector.items():
        if not runs_benchmark(bench, cache_type):
            continue
        assert_source_defines(
            bench.cache_bw_src, kernel_name, f"{gpu_arch} {cache_type}"
        )


@pytest.mark.parametrize("gpu_arch", ROOFLINE_ARCHS)
def test_flops_kernel_sources_define_selected_symbol(gpu_arch):
    """VALU benchmarks read their kernel name from flops_kernel_selector."""
    bench = BENCH_CLASSES[gpu_arch](0, dict(CACHE_SIZES), gpu_arch)

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
            f"{gpu_arch} {test_key}",
        )


@pytest.mark.parametrize("gpu_arch", ROOFLINE_ARCHS)
def test_matrix_kernel_sources_define_selected_symbol(gpu_arch):
    """Matrix benchmarks read their kernel name from matrix_kernel_selector."""
    bench = BENCH_CLASSES[gpu_arch](0, dict(CACHE_SIZES), gpu_arch)

    for matrix_type, kernel_name in bench.matrix_kernel_selector.items():
        test_key = f"{bench.MATRIX_OPS_TYPE}-{matrix_type}"
        if not runs_benchmark(bench, test_key):
            continue
        source_attr = MATRIX_SOURCE_ATTRS.get(matrix_type, PACKED_MATRIX_SOURCE_ATTR)
        assert_source_defines(
            getattr(bench, source_attr),
            kernel_name,
            f"{gpu_arch} {test_key}",
        )


def test_gfx1201_enabled_matrix_tests_have_operations_and_kernels():
    """Every WMMA test gfx1201 runs has a FLOP count and a kernel name."""
    bench = BENCH_CLASSES["gfx1201"](0, dict(CACHE_SIZES), "gfx1201")
    prefix = f"{bench.MATRIX_OPS_TYPE}-"
    enabled = [
        test_key[len(prefix) :]
        for test_key in bench.tests
        if test_key.startswith(prefix) and runs_benchmark(bench, test_key)
    ]

    assert sorted(enabled) == ["BF16", "BF8", "F16", "F8"]
    for matrix_type in enabled:
        assert bench.matrix_ops[matrix_type] > 0
        assert matrix_type in bench.matrix_kernel_selector


def test_gfx1201_measures_vram_in_its_own_column():
    """gfx1201 launches DRAM_bw and writes the VRAMBw column."""
    bench = BENCH_CLASSES["gfx1201"](0, dict(CACHE_SIZES), "gfx1201")

    assert not runs_benchmark(bench, "HBM")
    assert runs_benchmark(bench, "VRAM")
    vram = bench.tests["VRAM"]
    assert vram.func == bench.device_memory_bw_benchmark
    assert vram.keywords == {"kernel_name": "DRAM_bw", "label": "VRAM"}
    assert list(bench.tests)[0] == "VRAM"
    assert bench.csv_cols_map["VRAM"] == "VRAMBw"
    assert "HBM" not in bench.csv_cols_map
    assert_source_defines(bench.device_memory_bw_src, "DRAM_bw", "gfx1201 DRAM")
    # GL1 is a buffer, not a cache, so the L1 test is skipped and L1Bw is 0.
    assert "L1" in bench.unsupported_data_types
    assert bench.csv_cols_map["L1"] == "L1Bw"


def test_gfx1201_measures_infinity_cache_with_its_own_kernel():
    """gfx1201 runs the MALL test through its grid-stride MALL_bw kernel."""
    bench = BENCH_CLASSES["gfx1201"](0, dict(CACHE_SIZES), "gfx1201")

    assert runs_benchmark(bench, "MALL")
    assert bench.tests["MALL"].func == bench.cache_stream_bw_bench
    assert bench.tests["MALL"].keywords == {
        "level": "MALL",
        "default_capacity_bytes": INFINITY_CACHE_BYTES,
        "working_set_divisor": MALL_WORKING_SET_DIVISOR,
    }
    assert bench.csv_cols_map["MALL"] == "MALLBw"
    assert_source_defines(bench.mall_bw_src, "MALL_bw", "gfx1201 MALL")


def test_percent_of_ceiling_reports_an_unusable_ceiling_as_unavailable():
    """The ceiling is a diagnostic print. A zero clock reading, which
    shader_clock_hz returns for an empty wall-clock span, must not stop the
    benchmark that prints it."""
    assert percent_of_ceiling(50.0, 200.0) == "25.0%"
    assert percent_of_ceiling(50.0, 0.0) == "n/a"
    assert percent_of_ceiling(50.0, -1.0) == "n/a"


def test_gfx1201_vram_test_runs_the_device_memory_benchmark_under_the_vram_name(
    monkeypatch,
):
    """The VRAM test launches the DRAM_bw kernel, labels it VRAM, and returns the
    measured peak."""
    calls = []

    def fake_device_memory_benchmark(self, device, kernel_name="HBM_bw", label="HBM"):
        calls.append((device, kernel_name, label))
        return 480.0

    # The VRAM test binds the method when the benchmark is created.
    monkeypatch.setattr(
        BENCH_CLASSES["gfx1201"],
        "device_memory_bw_benchmark",
        fake_device_memory_benchmark,
    )
    bench = BENCH_CLASSES["gfx1201"](0, dict(CACHE_SIZES), "gfx1201")

    assert bench.tests["VRAM"](0) == 480.0
    assert calls == [(0, "DRAM_bw", "VRAM")]


def test_device_memory_benchmark_defaults_to_the_hbm_kernel_and_label(
    monkeypatch, capsys
):
    """device_memory_bw_benchmark gained a kernel name and label for gfx1201. A
    call without them keeps every other architecture on the HBM_bw kernel and HBM
    label, and a caller can ask for another kernel and label."""
    kernels = []
    grids = []

    class FakeProgram:
        def __init__(self, source):
            pass

        def get_kernel(self, name):
            kernels.append(name)
            return object()

    fake_hip = SimpleNamespace(
        hipSetDevice=lambda device: None,
        hipGetDeviceProperties=lambda device: SimpleNamespace(multiProcessorCount=28),
        hipMalloc=lambda size: object(),
        hipDeviceSynchronize=lambda: None,
    )
    monkeypatch.setattr(benchmark_base, "hip", fake_hip)
    bench = BENCH_CLASSES["gfx90a"](0, dict(CACHE_SIZES), "gfx90a")
    monkeypatch.setattr(bench, "Program", FakeProgram)
    monkeypatch.setattr(
        bench, "launch_kernel", lambda func, grid, *args: grids.append(grid)
    )
    monkeypatch.setattr(bench, "run_get_samples", lambda *args: [100.0, 250.0])

    assert benchmark_base.Bench_base.device_memory_bw_benchmark(bench, 0) == 250.0
    assert kernels == ["HBM_bw"]
    # 128 workgroups per multiprocessor, as HIP reports them: 28 x 128.
    assert grids[0] == [3584, 1, 1]
    assert "HBM BW, GPU ID: 0" in capsys.readouterr().out

    benchmark_base.Bench_base.device_memory_bw_benchmark(bench, 0, "DRAM_bw", "VRAM")
    assert kernels == ["HBM_bw", "DRAM_bw"]
    assert "VRAM BW, GPU ID: 0" in capsys.readouterr().out


@pytest.mark.parametrize(
    "working_set_bytes",
    [3 * 1024, 16 * 1000, 16 * 8],
    ids=["not-a-power-of-two", "element-count-not-a-power-of-two", "too-small"],
)
def test_gfx1201_cache_roof_rejects_a_working_set_it_cannot_stream(
    working_set_bytes,
):
    """The resident-set kernel masks its index with count - 1 and issues
    workgroup_size * loads_in_flight loads per step, so the set has to be a
    power-of-two element count of at least that many loads."""
    bench = BENCH_CLASSES["gfx1201"](0, dict(CACHE_SIZES), "gfx1201")

    roof = L2_ROOF._replace(
        working_set_bytes=working_set_bytes,
        instances=1,
        bytes_per_clock=1,
        pipe_name="test pipe",
    )

    with pytest.raises(ValueError, match="cache working set"):
        bench.cache_resident_bw(0, "test", roof)


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
