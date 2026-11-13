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
import roofline.benchmark.gfx12.benchmark_gfx1201 as benchmark_gfx1201
import utils.utils_profile as utils_profile
from roofline import run_benchmark
from roofline.benchmark.gfx12.benchmark_gfx1201 import (
    CACHE_ELEMENT_BYTES,
    CACHE_LOADS_IN_FLIGHT,
    CACHE_WORKGROUPS_PER_CU,
    DRAM_BUS_BITS,
    DRAM_GBPS_PER_PIN,
    GL0_CAPACITY_BYTES,
    GL0_PER_MULTIPROCESSOR,
    GL0_READ_BYTES_PER_CLOCK,
    GL2_BANKS,
    GL2_HIT_RETURN_BYTES_PER_CLOCK,
    INFINITY_CACHE_BYTES,
    L0_LOADS_IN_FLIGHT,
    L0_ROOF,
    L0_WORKGROUPS_PER_CU,
    L0_WORKING_SET_BYTES,
    L2_ROOF,
    L2_WORKING_SET_BYTES,
    MALL_WORKING_SET_DIVISOR,
    ResidentReadRoof,
    dram_pin_bandwidth_gbps,
    percent_of_ceiling,
    pipe_bandwidth_gbps,
    resident_stream_plan,
    shader_clock_hz,
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
    assert vram.func == bench.dram_pin_rated_bw_benchmark
    assert vram.keywords == {
        "kernel_name": "DRAM_bw",
        "label": "VRAM",
        "bus_bits": DRAM_BUS_BITS,
        "gbps_per_pin": DRAM_GBPS_PER_PIN,
    }
    assert list(bench.tests)[0] == "VRAM"
    assert bench.csv_cols_map["VRAM"] == "VRAMBw"
    assert "HBM" not in bench.csv_cols_map
    assert_source_defines(bench.hbm_bw_src, "DRAM_bw", "gfx1201 DRAM")


def test_gfx1201_cache_roofs_stay_inside_the_cache_that_serves_them():
    """L0 fits in one GL0. L2 misses GL0 and still fits in GL2.

    GL1 is a buffer, not a cache, so there is no L1 roof. The L1 test is
    skipped and L1Bw is written as 0.
    """
    bench = BENCH_CLASSES["gfx1201"](0, dict(CACHE_SIZES), "gfx1201")

    for level, roof in (("L0", L0_ROOF), ("L2", L2_ROOF)):
        assert bench.tests[level].func == bench.cache_resident_bw
        assert bench.tests[level].keywords == {"label": level, "roof": roof}
    assert "L1" in bench.unsupported_data_types
    assert bench.csv_cols_map["L1"] == "L1Bw"
    assert L0_ROOF == ResidentReadRoof(
        L0_WORKING_SET_BYTES,
        GL0_PER_MULTIPROCESSOR,
        GL0_READ_BYTES_PER_CLOCK,
        "GL0 read return",
        L0_WORKGROUPS_PER_CU,
        L0_LOADS_IN_FLIGHT,
        "cache_l0_bw",
        instances_per_multiprocessor=True,
    )
    assert L2_ROOF == ResidentReadRoof(
        L2_WORKING_SET_BYTES,
        GL2_BANKS,
        GL2_HIT_RETURN_BYTES_PER_CLOCK,
        "GL2 hit return",
        CACHE_WORKGROUPS_PER_CU,
        CACHE_LOADS_IN_FLIGHT,
        "cache_resident_bw",
    )
    assert_source_defines(
        bench.cache_resident_src, "cache_resident_bw", "gfx1201 cache"
    )
    assert_source_defines(bench.cache_resident_src, "cache_l0_bw", "gfx1201 L0")
    workgroup_size = benchmark_base.DEFAULT_WORKGROUP_SIZE
    assert (
        L0_WORKING_SET_BYTES // CACHE_ELEMENT_BYTES
        >= workgroup_size * L0_LOADS_IN_FLIGHT
    )

    assert L0_WORKING_SET_BYTES <= GL0_CAPACITY_BYTES
    assert GL0_CAPACITY_BYTES < L2_WORKING_SET_BYTES
    # 6 MB is the smallest Navi 48 GL2.
    assert L2_WORKING_SET_BYTES < 6 * 1024 * 1024
    for working_set in (L0_WORKING_SET_BYTES, L2_WORKING_SET_BYTES):
        element_count = working_set // CACHE_ELEMENT_BYTES
        assert working_set % CACHE_ELEMENT_BYTES == 0
        assert element_count >= workgroup_size * CACHE_LOADS_IN_FLIGHT
        assert element_count & (element_count - 1) == 0

    # 2.97 GHz, the boost the cache spec comparison used. Decimal GB/s.
    # 32 multiprocessors are 64 CUs, each with one GL0.
    assert pipe_bandwidth_gbps(
        GL0_PER_MULTIPROCESSOR * 32, GL0_READ_BYTES_PER_CLOCK, 2.97e9
    ) == pytest.approx(24330.24)
    assert pipe_bandwidth_gbps(
        GL2_BANKS, GL2_HIT_RETURN_BYTES_PER_CLOCK, 2.97e9
    ) == pytest.approx(12165.12)
    # The GL1 port is the ceiling these resident reads must not be scored against.
    assert pipe_bandwidth_gbps(8, 128, 2.97e9) == pytest.approx(3041.28)
    assert dram_pin_bandwidth_gbps(DRAM_BUS_BITS, DRAM_GBPS_PER_PIN) == 640.0
    # 3e9 shader cycles over 1e8 ticks of the 100 MHz wall clock is 3 GHz.
    assert shader_clock_hz(3_000_000_000, 100_000_000) == pytest.approx(3e9)
    assert shader_clock_hz(1, 0) == 0.0


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
    monkeypatch, capsys
):
    """The VRAM test launches the DRAM_bw kernel, labels it VRAM, and returns the
    measured peak. The pin-rate reference it prints is a percentage of 640 GB/s."""
    bench = BENCH_CLASSES["gfx1201"](0, dict(CACHE_SIZES), "gfx1201")
    calls = []

    def fake_device_memory_benchmark(device, kernel_name="HBM_bw", label="HBM"):
        calls.append((device, kernel_name, label))
        return 480.0

    monkeypatch.setattr(bench, "hbm_bw_benchmark", fake_device_memory_benchmark)

    assert bench.tests["VRAM"](0) == 480.0
    assert calls == [(0, "DRAM_bw", "VRAM")]
    assert "(75.0%)" in capsys.readouterr().out


def test_device_memory_benchmark_defaults_to_the_hbm_kernel_and_label(
    monkeypatch, capsys
):
    """hbm_bw_benchmark gained a kernel name and label for gfx1201. A call without
    them keeps every other architecture on the HBM_bw kernel and HBM label, and
    a caller can ask for another kernel and label."""
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

    assert benchmark_base.Bench_base.hbm_bw_benchmark(bench, 0) == 250.0
    assert kernels == ["HBM_bw"]
    # 128 workgroups per multiprocessor, as HIP reports them: 28 x 128.
    assert grids[0] == [3584, 1, 1]
    assert "HBM BW, GPU ID: 0" in capsys.readouterr().out

    benchmark_base.Bench_base.hbm_bw_benchmark(bench, 0, "DRAM_bw", "VRAM")
    assert kernels == ["HBM_bw", "DRAM_bw"]
    assert "VRAM BW, GPU ID: 0" in capsys.readouterr().out


def test_resident_stream_plan_sizes_whole_grid_steps_and_the_pass_count():
    """With 28 multiprocessors, one grid step is 112 workgroups x 256 threads x
    16 unrolled 16-byte loads, 7,340,032 bytes. A 32 MiB set holds four steps,
    29,360,128 bytes, which takes 1,170 passes to reach 34,351,349,760 bytes."""
    step = 7_340_032

    workgroups, steps, buffer_bytes, passes, total = resident_stream_plan(
        28, 256, 32 * 1024 * 1024
    )

    assert (workgroups, steps, buffer_bytes) == (112, 4, 29_360_128)
    assert buffer_bytes == 4 * step
    assert (passes, total) == (1170, 34_351_349_760)

    # A set smaller than one step still streams one whole step.
    _, steps, buffer_bytes, passes, _ = resident_stream_plan(28, 256, 1024)
    assert (steps, buffer_bytes) == (1, step)
    assert passes >= 1


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


@pytest.fixture
def fake_gfx12_runtime(monkeypatch):
    """A HIP runtime for the gfx12 launchers with 32 multiprocessors. Event
    timings cycle through 4, 2, and 8 ms, and every in-kernel clock reading is
    2.97e9 shader cycles over 1e8 wall-clock ticks, 2.97 GHz. Returns a
    function that routes a benchmark's kernels through this runtime and returns
    the list of (kernel, grid, args) launches."""
    elapsed_ms = iter([4.0, 2.0, 8.0] * 10)
    launches = []

    class FakeMemory:
        def __init__(self, size):
            self.size = size

    def copy_clock_spans(spans, clocks, size):
        spans[0] = 2_970_000_000
        spans[1] = 100_000_000

    fake_hip = SimpleNamespace(
        hipSetDevice=lambda device: None,
        hipGetDeviceProperties=lambda device: SimpleNamespace(multiProcessorCount=32),
        hipMalloc=FakeMemory,
        hipDeviceSynchronize=lambda: None,
        hipEventCreate=lambda: None,
        hipEventRecord=lambda event: None,
        hipEventElapsedTime=lambda start, stop: next(elapsed_ms),
        hipMemcpyHtoD=lambda clocks, zeros, size: None,
        hipMemcpyDtoH=copy_clock_spans,
        HIPDeviceMemory=FakeMemory,
    )
    monkeypatch.setattr(benchmark_base, "hip", fake_hip)
    monkeypatch.setattr(benchmark_gfx1201, "hip", fake_hip)

    def record_launch(func, grid, block, shared_mem_size, stream, args):
        launches.append((
            func,
            grid,
            [
                arg.size if isinstance(arg, FakeMemory) else getattr(arg, "value", arg)
                for arg in args
            ],
        ))

    def install(bench):
        program = SimpleNamespace(get_kernel=lambda name: name)
        monkeypatch.setattr(bench, "Program", lambda source: program)
        monkeypatch.setattr(bench, "launch_kernel", record_launch)
        return launches

    return install


def test_gfx1201_cache_roof_reports_the_fastest_launch_at_its_measured_clock(
    fake_gfx12_runtime, capsys
):
    """The L2 roof reads a 4 MiB set from 8 x 32 workgroups, 64 passes to
    reach 64 GiB. The 2 ms launch is the peak, and the GL2 hit return at the
    measured 2.97 GHz is 12,165.12 GB/s."""
    bench = BENCH_CLASSES["gfx1201"](0, dict(CACHE_SIZES), "gfx1201")
    launches = fake_gfx12_runtime(bench)

    peak = bench.tests["L2"](0)

    assert peak == pytest.approx(68_719_476_736 / 2.0 / 1e6)
    assert len(launches) == 1 + benchmark_base.DEFAULT_NUM_EXPERIMENTS
    kernel, grid, args = launches[0]
    assert kernel == "cache_resident_bw"
    assert grid == [256, 1, 1]
    assert args == [L2_WORKING_SET_BYTES, 262_144, 64, CACHE_ELEMENT_BYTES, 16]
    log = capsys.readouterr().out
    assert "L2 BW, GPU ID: 0" in log
    assert "GL2 hit return ceiling:12165.1 GB/sec (282.4% at 2970 MHz)" in log


# TODO: Fold this into the cache-roof ceiling test above, parametrized by cache
# level and multiprocessor count, instead of keeping a separate function.
def test_gfx1201_l0_ceiling_follows_the_reported_multiprocessor_count(
    fake_gfx12_runtime, monkeypatch, capsys
):
    """Navi 48 boards differ in CU count. With 24 multiprocessors, 48 CUs, the
    GL0 read return at 2.97 GHz is 48 x 128 B x 2.97e9, 18,247.68 GB/s."""
    bench = BENCH_CLASSES["gfx1201"](0, dict(CACHE_SIZES), "gfx1201")
    launches = fake_gfx12_runtime(bench)
    monkeypatch.setattr(
        benchmark_gfx1201.hip,
        "hipGetDeviceProperties",
        lambda device: SimpleNamespace(multiProcessorCount=24),
    )

    bench.tests["L0"](0)

    assert launches[0][1] == [L0_WORKGROUPS_PER_CU * 24, 1, 1]
    assert "GL0 read return ceiling:18247.7 GB/sec" in capsys.readouterr().out


@pytest.mark.parametrize(
    "reported_bytes, buffer_bytes, steps, passes",
    [
        (None, 33_554_432, 4, 1024),
        (48 * 1024 * 1024, 25_165_824, 3, 1365),
    ],
    ids=["no-reported-infinity-cache", "reported-48-MiB"],
)
def test_gfx1201_infinity_cache_streams_half_its_capacity(
    fake_gfx12_runtime, capsys, reported_bytes, buffer_bytes, steps, passes
):
    """The MALL test streams half the Infinity Cache that AMD-SMI reports, or
    half of 64 MB when it reports none, in whole 8 MiB grid steps."""
    cache_sizes = {
        level: size for level, size in CACHE_SIZES.items() if level != "MALL"
    }
    if reported_bytes is not None:
        cache_sizes["MALL"] = reported_bytes
    bench = BENCH_CLASSES["gfx1201"](0, cache_sizes, "gfx1201")
    launches = fake_gfx12_runtime(bench)

    peak = bench.tests["MALL"](0)

    total_bytes = passes * buffer_bytes
    assert peak == pytest.approx(total_bytes / 2.0 / 1e6)
    kernel, grid, args = launches[0]
    assert kernel == "MALL_bw"
    assert grid == [128, 1, 1]
    assert args == [buffer_bytes, steps, passes, CACHE_ELEMENT_BYTES]
    assert (
        f"working set:{buffer_bytes} bytes, passes:{passes}" in capsys.readouterr().out
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
