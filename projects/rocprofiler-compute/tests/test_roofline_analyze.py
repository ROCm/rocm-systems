# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Roofline coverage for the ``analyze`` roofline path.

MI200 (gfx90a) tests run through the ``analyze`` CLI and assert that roofline
HTML is generated, including the per-datatype VALU/MFMA legend.
"""

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import common
import pytest

config = {}
config["cleanup"] = True

roofline_dir = "tests/workloads/mem_levels_HBM/MI200"


# =============================================================================
# Roofline HTML generation
# =============================================================================


def test_analyze_generates_roofline_html(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
) -> None:
    """
    Analyze generates roofline HTML from existing workload data.
    Uses MI200 workload with roofline.csv.
    """
    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "FP32",
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_roofline_datatype_independently(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
) -> None:
    """
    Analyze with multiple data types.
    Verifies each datatype can be requested independently.
    """
    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    for dtype in ["FP32", "FP64", "BF16"]:
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--roofline-data-type",
            dtype,
        ])
        assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_roofline_multiple_datatypes_single_invocation(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
) -> None:
    """
    Analyze with multiple data types in a single invocation.
    Verifies the multi-datatype request path works end to end.
    """
    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "FP32",
        "FP64",
        "BF16",
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_missing_roofline_csv_graceful(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
) -> None:
    """
    Analyze without roofline.csv should not crash.
    Uses a workload directory that has sysinfo.csv but no roofline.csv.
    """
    workload_dir = common.setup_workload_dir(roofline_dir)
    roofline_csv = Path(workload_dir) / "roofline.csv"
    if roofline_csv.exists():
        roofline_csv.unlink()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_roofline_idempotent(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
) -> None:
    """
    Running analyze twice on the same profiling output should produce
    consistent results without errors.
    """
    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    analyze_args = [
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "FP32",
    ]

    code1 = binary_handler_analyze_rocprof_compute(analyze_args)
    assert code1 == 0

    code2 = binary_handler_analyze_rocprof_compute(analyze_args)
    assert code2 == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_corrupted_roofline_csv_graceful(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
) -> None:
    """
    Analyze with a corrupted roofline.csv should handle gracefully.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        workload_dir = Path(temp_dir) / "corrupted_workload"
        shutil.copytree(roofline_dir, workload_dir)

        roofline_csv = workload_dir / "roofline.csv"
        roofline_csv.write_text("this,is,bad,csv")

        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "-b",
            "4",
            "--path",
            str(workload_dir),
        ])
        assert code == 0


def test_roof_invalid_data_type(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid --roofline-data-type should be rejected by the analyze argparser."""
    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "INVALID_TYPE",
    ])

    err = capsys.readouterr().err
    assert "--roofline-data-type" in err
    assert "invalid choice" in err

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_roof_invalid_mem_level(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid --mem-level should be rejected by the analyze argparser."""
    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--mem-level",
        "INVALID_LEVEL",
    ])

    err = capsys.readouterr().err
    assert "--mem-level" in err
    assert "invalid choice" in err

    common.clean_output_dir(config["cleanup"], workload_dir)


roofline_mem_level_dirs = {
    "vL1D": "tests/workloads/mem_levels_vL1D/MI200",
    "LDS": "tests/workloads/mem_levels_LDS/MI200",
}


@pytest.mark.parametrize(
    "mem_level",
    ["vL1D", "LDS"],
    ids=["vL1D", "LDS"],
)
def test_roof_mem_levels(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
    mem_level: str,
) -> None:
    """Analyze with --mem-level generates roofline HTML output."""
    workload_src = roofline_mem_level_dirs[mem_level]
    if not Path(workload_src).exists():
        pytest.skip(f"Workload directory {workload_src} not found")

    workload_dir = common.setup_workload_dir(workload_src, param_id=mem_level)

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--mem-level",
        mem_level,
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


# =============================================================================
# Per-datatype HTML output validation
# =============================================================================

# Each datatype must take the correct roofline branch:
# FP64 is dual-path (VALU + MFMA), BF16 is MFMA-only.
DATATYPE_LEGEND_CASES = {
    "FP64": {"present": ["Peak VALU-FP64", "Peak MFMA-FP64"], "absent": []},
    "BF16": {"present": ["Peak MFMA-BF16"], "absent": ["Peak VALU-BF16"]},
}


@pytest.mark.parametrize("dtype", list(DATATYPE_LEGEND_CASES))
def test_analyze_roofline_datatype_html_legend(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
    dtype: str,
) -> None:
    """Per-datatype roofline HTML embeds the expected VALU/MFMA legend.

    FP8/FP4/FP6 are intentionally excluded: they are unsupported on the
    available gfx90a (MI200) test data and are covered by the unit tests.
    """
    workload_dir = common.setup_workload_dir(roofline_dir, param_id=dtype)

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
    for legend in DATATYPE_LEGEND_CASES[dtype]["present"]:
        assert legend in html_text, f"{dtype} HTML should contain '{legend}'"
    for legend in DATATYPE_LEGEND_CASES[dtype]["absent"]:
        assert legend not in html_text, f"{dtype} HTML should not contain '{legend}'"

    common.clean_output_dir(config["cleanup"], workload_dir)
