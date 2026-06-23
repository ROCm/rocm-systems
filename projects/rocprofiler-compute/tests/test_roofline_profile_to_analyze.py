# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""End-to-end roofline coverage: profile then analyze.

These tests run ``rocprof-compute profile`` against the multi-precision
``fma_throughput`` sample on real GPU hardware and then ``analyze`` the freshly
produced workload, exercising the full roofline pipeline (built-in microbenchmark
-> ``roofline.csv`` -> AI calc -> per-datatype HTML) across the precisions the
detected SoC supports.
"""

from pathlib import Path

import common
import pytest

from roofline.roofline_main import ROOFLINE_SUPPORTED
from utils.roofline_calc import SUPPORTED_DATATYPES

config = {}
config["cleanup"] = True
config["app_fma_throughput"] = ["./tests/fma_throughput"]

gpu_arch, soc = common.gpu_soc()


def _skip_if_no_roofline_soc():
    """Skip unless a roofline-capable GPU is present.

    RDNA (gfx11xx) archs are roofline-capable via WMMA; MI200/MI300/MI350 via
    MFMA. Anything outside ROOFLINE_SUPPORTED (e.g. MI100) is skipped.
    """
    if not gpu_arch or not soc:
        pytest.skip("No supported GPU detected")
    if soc == "MI100":
        pytest.skip("Roofline not supported on MI100")
    if gpu_arch not in ROOFLINE_SUPPORTED:
        pytest.skip(f"Roofline not supported on {soc} ({gpu_arch})")


def _supported_dtypes():
    """Datatypes the detected SoC can generate roofline data for."""
    return SUPPORTED_DATATYPES.get(gpu_arch, [])


def _matrix_op():
    """Matrix-op legend label for the detected SoC.

    Mirrors ``utils.utils_analysis.get_matrix_ops_type``: MI200/MI300/MI350
    (CDNA2/3/4) use MFMA; all other supported archs (RDNA gfx11xx) use WMMA.
    """
    return "MFMA" if soc in ("MI200", "MI300", "MI350") else "WMMA"


# =============================================================================
# Profile -> roofline.csv
# =============================================================================


def test_profile_roofline_dtypes_generates_csv(
    binary_handler_profile_rocprof_compute,
):
    """profile --roof-only on the sample produces roofline.csv + sysinfo.csv."""
    _skip_if_no_roofline_soc()

    options = ["--device", "0", "--roof-only"]
    workload_dir = common.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        roof=True,
        app_name="app_fma_throughput",
    )
    assert returncode == 0

    assert (Path(workload_dir) / "roofline.csv").exists()
    assert (Path(workload_dir) / "sysinfo.csv").exists()

    common.clean_output_dir(config["cleanup"], workload_dir)


# =============================================================================
# Profile (with roofline) -> analyze -> per-datatype HTML
# =============================================================================


def test_profile_then_analyze_roofline_e2e(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    """Full profile with roofline, then analyze over all supported datatypes."""
    _skip_if_no_roofline_soc()

    dtypes = _supported_dtypes()
    assert dtypes, f"No supported datatypes for {gpu_arch}"

    options = ["--device", "0"]
    workload_dir = common.get_output_dir()
    returncode = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        roof=True,
        app_name="app_fma_throughput",
    )
    assert returncode == 0
    assert (Path(workload_dir) / "roofline.csv").exists()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        *dtypes,
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


# Datatypes whose VALU/matrix legend branch we assert on the produced HTML.
# FP64 is dual-path (VALU + matrix); BF16 is matrix-only (no VALU roof).
_LEGEND_DTYPES = ["FP64", "BF16"]


@pytest.mark.parametrize("dtype", _LEGEND_DTYPES)
def test_analyze_roofline_per_datatype_e2e(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    dtype,
):
    """Per-datatype analyze on a freshly profiled workload embeds the right legend.

    The matrix-op roof is labelled WMMA on gfx1151 and MFMA on the MFMA SoCs.
    """
    _skip_if_no_roofline_soc()

    if dtype not in _supported_dtypes():
        pytest.skip(f"{dtype} not supported on {soc} ({gpu_arch})")

    options = ["--device", "0"]
    workload_dir = common.get_output_dir(param_id=dtype)
    returncode = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        roof=True,
        app_name="app_fma_throughput",
    )
    assert returncode == 0
    assert (Path(workload_dir) / "roofline.csv").exists()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        dtype,
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob(f"empirRoof_*{dtype}*.html"))
    assert len(html_files) > 0, f"Analyze should generate a {dtype} roofline HTML"

    html_text = html_files[0].read_text(encoding="utf-8")
    matrix_op = _matrix_op()
    assert f"Peak {matrix_op}-{dtype}" in html_text, (
        f"{dtype} HTML should contain 'Peak {matrix_op}-{dtype}'"
    )
    if dtype == "BF16":
        assert f"Peak VALU-{dtype}" not in html_text, (
            f"{dtype} is matrix-only; HTML should not contain a VALU roof"
        )

    common.clean_output_dir(config["cleanup"], workload_dir)
