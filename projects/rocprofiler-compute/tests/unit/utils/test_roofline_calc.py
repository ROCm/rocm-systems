# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT


import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pytest

from utils import schema
from utils.roofline_calc import (
    XMAX_DEFAULT,
    GraphPoints,
    calc_ai_analyze,
    calc_ceilings,
    construct_roof,
    device_ceilings,
    machine_ceilings,
    roofline_mem_levels,
    sanitize_ai_value,
)


def assert_graph_points_positive_finite(
    graph_points: Dict[str, List[Union[List[float], float, None]]],
) -> None:
    """Assert every populated graph point coordinate and scalar is positive finite."""
    for values in graph_points.values():
        for value in values:
            if isinstance(value, list):
                assert all(
                    coordinate > 0 and math.isfinite(coordinate) for coordinate in value
                )
            elif value is not None:
                assert value > 0 and math.isfinite(value)


def run_calc_ai_analyze_with_values(
    monkeypatch: pytest.MonkeyPatch,
    metric_values: dict[str, object],
    top_stats: dict[str, object] | None = None,
) -> dict:
    """
    Build mocks and invoke calc_ai_analyze with controlled metric values.

    metric_values is a dict with keys ai_hbm, ai_l2, ai_l1,
    ai_lds, performance whose values are injected into the table-402
    DataFrame that eval_metric would normally populate.

    top_stats adds columns to the top-kernels table, which is where the
    per-kernel dispatch count, aggregate time, and percent runtime are joined
    from. Omit it to exercise the missing-stats path.

    Returns the plot-points dict produced by calc_ai_analyze.

    Note: this mock simulates MI350 AI metric values based on the architecture's cache
    levels available on the hardware. Cache levels will vary for other architectures.
    """
    kernel_name = "test_kernel"
    kernel_id = 0

    top_columns: dict[str, list[object]] = {"Kernel_Name": [kernel_name]}
    for column, value in (top_stats or {}).items():
        top_columns[column] = [value]

    workload = schema.Workload()
    workload.dfs = {1: pd.DataFrame(top_columns, index=[kernel_id])}
    workload.sys_info = pd.DataFrame([{"gpu_arch": "gfx90a"}])
    workload.roofline_peaks = pd.DataFrame()
    workload.filter_kernel_ids = []
    workload.path = "/mock/path"

    arch_config = schema.ArchConfig()
    arch_config.dfs = {
        401: pd.DataFrame(),
        402: pd.DataFrame({
            "Metric": pd.Series(dtype="str"),
            "Value": pd.Series(dtype="object"),
        }),
    }
    arch_config.dfs_type = {401: "metric_table", 402: "metric_table"}

    pmc_df = pd.DataFrame({"Kernel_Name": [kernel_name]})

    def mock_eval_metric(
        dfs: dict,
        dfs_type: dict,
        dfs_expressions: object,
        sys_info_row: object,
        roofline_peaks: object,
        pmc_data: object,
        debug: object,
    ) -> None:
        dfs[402] = pd.DataFrame({
            "Metric": [
                "AI HBM",
                "AI L2",
                "AI L1",
                "AI LDS",
                "Performance (GFLOPs)",
            ],
            "Value": pd.array(
                [
                    metric_values["ai_hbm"],
                    metric_values["ai_l2"],
                    metric_values["ai_l1"],
                    metric_values["ai_lds"],
                    metric_values["performance"],
                ],
                dtype=object,
            ),
        })

    monkeypatch.setattr("utils.roofline_calc.eval_metric", mock_eval_metric)

    monkeypatch.setattr("utils.roofline_calc.console_debug", lambda *a, **kw: None)
    monkeypatch.setattr("utils.roofline_calc.console_warning", lambda *a, **kw: None)

    return calc_ai_analyze(
        workload=workload,
        pmc_df=pmc_df,
        arch_config=arch_config,
    )


def ceilings_by_roof(
    ceilings: List,
) -> Dict[Tuple[str, object, object, object], Tuple[str, float]]:
    """Index ceilings by (kind, memory level, data type, pipe) for comparison."""
    return {
        (c.kind, c.memory_level, c.data_type, c.pipe): (c.unit, c.value)
        for c in ceilings
    }


def test_calc_ai_analyze_replaces_inf_with_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """np.inf / -np.inf metric values are replaced with 0."""
    result = run_calc_ai_analyze_with_values(
        monkeypatch,
        {
            "ai_hbm": np.inf,
            "ai_l2": -np.inf,
            "ai_l1": 1.5,
            "ai_lds": np.inf,
            "performance": 100.0,
        },
    )

    assert result["kernelNames"] == ["test_kernel"]
    assert result["ai_hbm"][0] == [0], "np.inf should be replaced with 0"
    assert result["ai_hbm"][1] == [100.0]
    assert result["ai_l2"][0] == [0], "-np.inf should be replaced with 0"
    assert result["ai_l2"][1] == [100.0]
    assert result["ai_l1"][0] == [1.5], "valid float should pass through"
    assert result["ai_l1"][1] == [100.0]
    assert result["ai_lds"][0] == [0], "np.inf should be replaced with 0"
    assert result["ai_lds"][1] == [100.0]


@pytest.mark.parametrize(
    "unusable",
    [None, "N/A", "", np.nan, np.inf, -np.inf],
    ids=["none", "na", "empty", "nan", "inf", "-inf"],
)
def test_calc_ai_analyze_reports_an_unusable_ai_as_zero(
    monkeypatch: pytest.MonkeyPatch, unusable: object
) -> None:
    """An AI that cannot be plotted is reported as 0 rather than dropped, so
    every level's AI and performance list stays parallel with kernelNames. The
    result also has to stay JSON-safe: the browser's JSON.parse rejects a bare
    NaN outright, taking the whole page with it.
    """
    result = run_calc_ai_analyze_with_values(
        monkeypatch,
        {
            "ai_hbm": unusable,
            "ai_l2": unusable,
            "ai_l1": 2.0,
            "ai_lds": unusable,
            "performance": 100.0,
        },
    )

    assert result["kernelNames"] == ["test_kernel"]
    for level in ("ai_hbm", "ai_l2", "ai_lds"):
        assert result[level][0] == [0], f"{level} should report an unusable AI as 0"
        assert result[level][1] == [100.0], f"{level} performance list desynced"
    assert result["ai_l1"][0] == [2.0], "a usable AI alongside must pass through"
    json.dumps(result, allow_nan=False)


def test_calc_ai_analyze_valid_values_pass_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal positive floats pass through unchanged."""
    result = run_calc_ai_analyze_with_values(
        monkeypatch,
        {
            "ai_hbm": 2.5,
            "ai_l2": 3.0,
            "ai_l1": 1.5,
            "ai_lds": 4.0,
            "performance": 100.0,
        },
    )

    assert result["kernelNames"] == ["test_kernel"]
    assert result["ai_hbm"][0] == [2.5]
    assert result["ai_hbm"][1] == [100.0]
    assert result["ai_l2"][0] == [3.0]
    assert result["ai_l2"][1] == [100.0]
    assert result["ai_l1"][0] == [1.5]
    assert result["ai_l1"][1] == [100.0]
    assert result["ai_lds"][0] == [4.0]
    assert result["ai_lds"][1] == [100.0]


def test_calc_ai_analyze_joins_per_kernel_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch count, aggregate time, and percent runtime join onto each kernel
    when the top-kernels table carries them, and read None when it does not,
    which the tooltip renders as N/A."""
    ai_values: dict[str, object] = {
        "ai_hbm": 2.0,
        "ai_l2": 2.0,
        "ai_l1": 2.0,
        "ai_lds": 2.0,
        "performance": 100.0,
    }

    joined = run_calc_ai_analyze_with_values(
        monkeypatch,
        ai_values,
        top_stats={"Count": 128, "Sum(ns)": 154000.0, "Percent": 12.4},
    )
    assert joined["counts"] == [128]
    assert joined["kernelTime"] == [154000.0]
    assert joined["kernelPctRuntime"] == [12.4]
    assert joined["timeUnit"] == "ns", "the unit is only recoverable from the column"

    missing = run_calc_ai_analyze_with_values(monkeypatch, ai_values)
    assert missing["counts"] == [None]
    assert missing["kernelTime"] == [None]
    assert missing["kernelPctRuntime"] == [None]
    assert missing["timeUnit"] == ""


def test_sanitize_ai_value_replaces_invalid_values_with_zero() -> None:
    """Invalid values are replaced with 0."""
    assert sanitize_ai_value(np.inf) == 0
    assert sanitize_ai_value(-np.inf) == 0
    assert sanitize_ai_value(np.nan) == 0
    assert sanitize_ai_value("N/A") == 0
    assert sanitize_ai_value("") == 0
    assert sanitize_ai_value(None) == 0
    assert sanitize_ai_value(1.5) == 1.5


##############################################################################
# roofline_mem_levels Tests
##############################################################################


def test_roofline_mem_levels_are_measured_levels_in_hierarchy_order() -> None:
    """MALL (measured, not plotted) and levels without a measured ceiling (gfx1250's
    L1) are left out."""
    assert roofline_mem_levels("gfx90a", "mi210") == ["LDS", "L1", "L2", "HBM"]
    assert roofline_mem_levels("gfx1151", "rdna35_halo") == ["LDS", "L0", "L1", "L2"]
    assert roofline_mem_levels("gfx1250", "gfx1250") == ["LDS", "L0", "L2", "HBM"]


##############################################################################
# calc_ceilings Tests
##############################################################################

# gfx942-class model whose memory levels (LDS/L1/L2/HBM) match BW_COLUMNS.
MFMA_GPU_MODEL = "mi300x_a1"
MFMA_GPU_ARCH = "gfx942"
# gfx1151-class model; memory levels resolve to LDS/L0/L1/L2 (MALL skipped).
WMMA_GPU_MODEL = "rdna35_halo"
WMMA_GPU_ARCH = "gfx1151"

# Union of BW columns needed by both models (mi300x_a1 reads HBM, rdna35_halo reads
# L0); calc_ceilings only consumes the levels its model supports.
BW_COLUMNS = ["HBMBw", "L2Bw", "L1Bw", "L0Bw", "LDSBw"]
BW_VALUE = 500.0

PEAK_VALUES = {
    "FP16Flops": 1000.0,
    "BF16Flops": 1500.0,
    "FP32Flops": 2000.0,
    "FP64Flops": 3000.0,
    "I8Ops": 4000.0,
    "I32Ops": 5000.0,
    "I64Ops": 6000.0,
    "MFMAF8Flops": 9000.0,
    "MFMAF16Flops": 10000.0,
    "MFMABF16Flops": 10500.0,
    "MFMAF32Flops": 11000.0,
    "MFMAF64Flops": 12000.0,
    "MFMAI8Ops": 13000.0,
    "WMMAF16Flops": 20000.0,
    "WMMABF16Flops": 21000.0,
    "WMMAF32Flops": 22000.0,
    "WMMAF64Flops": 23000.0,
    "WMMAI8Ops": 24000.0,
}

# MFMA (CDNA3) supports the full datatype set expect F4 and F6.
MFMA_CASES = [
    ("FP32", "FP32Flops", "MFMAF32Flops"),
    ("FP16", "FP16Flops", "MFMAF16Flops"),
    ("FP64", "FP64Flops", "MFMAF64Flops"),
    ("I8", "I8Ops", "MFMAI8Ops"),
    ("I32", "I32Ops", None),
    ("I64", "I64Ops", None),
    ("BF16", None, "MFMABF16Flops"),
    ("FP8", None, "MFMAF8Flops"),
]

# FP4/FP6/FP8 are not supported on gfx1151.
WMMA_CASES = [
    ("FP32", "FP32Flops", None),
    ("FP16", "FP16Flops", None),
    ("FP64", "FP64Flops", None),
    ("I8", "I8Ops", None),
    ("I32", "I32Ops", None),
    ("I64", "I64Ops", None),
    ("BF16", "BF16Flops", None),
]

# (matrix_ops_type, gpu_model, gpu_arch, cases) tuples driving the parametrized tests.
MATRIX_FAMILIES = [
    ("MFMA", MFMA_GPU_MODEL, MFMA_GPU_ARCH, MFMA_CASES),
    ("WMMA", WMMA_GPU_MODEL, WMMA_GPU_ARCH, WMMA_CASES),
]

# Flattened (matrix_ops_type, gpu_model, gpu_arch, dtype, valu_col, matrix_col) rows.
ROOFLINE_DATATYPE_CASES = [
    (prefix, model, arch, dtype, valu_col, matrix_col)
    for prefix, model, arch, cases in MATRIX_FAMILIES
    for (dtype, valu_col, matrix_col) in cases
]


class MockMspec:
    """Minimal stand-in for MachineSpecs; calc_ceilings reads gpu_model and gpu_arch."""

    def __init__(
        self, gpu_model: str = MFMA_GPU_MODEL, gpu_arch: str = MFMA_GPU_ARCH
    ) -> None:
        self.gpu_model = gpu_model
        self.gpu_arch = gpu_arch


def roofline_parameters(
    matrix_ops_type: str = "MFMA", gpu_arch: str = MFMA_GPU_ARCH
) -> dict[str, object]:
    # matrix_ops_type is "MFMA" for CDNA (MI-series) and "WMMA" for RDNA.
    return {
        "device_id": 0,
        "workload_dir": "/tmp",
        "matrix_ops_type": matrix_ops_type,
        "gpu_arch": gpu_arch,
    }


def full_benchmark_data() -> dict[str, list[str]]:
    """Benchmark dict with every BW, PEAK_OPS and matrix column populated."""
    data = {col: [str(BW_VALUE)] for col in BW_COLUMNS}
    for col, value in PEAK_VALUES.items():
        data[col] = [str(value)]
    return data


@pytest.mark.parametrize(
    ("matrix_ops_type", "gpu_model", "gpu_arch", "dtype", "valu_col", "matrix_col"),
    ROOFLINE_DATATYPE_CASES,
    ids=[f"{row[0]}-{row[3]}" for row in ROOFLINE_DATATYPE_CASES],
)
def test_calc_ceilings_roofline_datatype(
    matrix_ops_type: str,
    gpu_model: str,
    gpu_arch: str,
    dtype: str,
    valu_col: str | None,
    matrix_col: str | None,
) -> None:
    """Each datatype populates exactly its expected VALU and/or matrix roof."""
    result = calc_ceilings(
        roofline_parameters(matrix_ops_type, gpu_arch),
        dtype,
        full_benchmark_data(),
        MockMspec(gpu_model, gpu_arch),
        0,
    )

    if valu_col is None:
        assert result["valu"] == [], (
            f"{dtype} should not produce a VALU (PEAK_OPS) roof"
        )
    else:
        assert result["valu"], f"{dtype} should produce a VALU (PEAK_OPS) roof"
        assert result["valu"][2] == PEAK_VALUES[valu_col], (
            f"{dtype} VALU peak should be read from {valu_col}"
        )

    if matrix_col is None:
        assert result["matrix_ops"] == [], (
            f"{dtype} should not produce a {matrix_ops_type} (matrix) roof"
        )
    else:
        assert result["matrix_ops"], (
            f"{dtype} should produce a {matrix_ops_type} (matrix) roof"
        )
        assert result["matrix_ops"][2] == PEAK_VALUES[matrix_col], (
            f"{dtype} {matrix_ops_type} peak should be read from {matrix_col}"
        )


def test_fp8_special_mfma_only() -> None:
    """FP8 is matrix-only and reads its peak from the dedicated MFMAF8 column.

    FP8 is a CDNA/MFMA datatype only.
    """
    result = calc_ceilings(
        roofline_parameters("MFMA", MFMA_GPU_ARCH),
        "FP8",
        full_benchmark_data(),
        MockMspec(),
        0,
    )

    assert result["valu"] == [], "FP8 is not a PEAK_OPS datatype; no VALU roof"
    assert result["matrix_ops"][2] == PEAK_VALUES["MFMAF8Flops"]


def test_missing_peak_ops_column_returns_empty() -> None:
    """A PEAK_OPS datatype missing its Flops column returns empty ceilings."""
    benchmark_data = full_benchmark_data()
    del benchmark_data["FP64Flops"]

    result = calc_ceilings(
        roofline_parameters(gpu_arch=MFMA_GPU_ARCH),
        "FP64",
        benchmark_data,
        MockMspec(),
        0,
    )

    assert result == GraphPoints.empty().__dict__


@pytest.mark.parametrize(
    ("matrix_ops_type", "gpu_model", "gpu_arch", "matrix_col"),
    [("MFMA", MFMA_GPU_MODEL, MFMA_GPU_ARCH, "MFMABF16Flops")],
    ids=["MFMA"],
)
def test_missing_matrix_column_skips_matrix_roof(
    matrix_ops_type: str, gpu_model: str, gpu_arch: str, matrix_col: str
) -> None:
    """A matrix datatype missing its matrix column emits no matrix roof.

    BF16 is matrix-only.
    """
    benchmark_data = full_benchmark_data()
    del benchmark_data[matrix_col]

    result = calc_ceilings(
        roofline_parameters(matrix_ops_type, gpu_arch),
        "BF16",
        benchmark_data,
        MockMspec(gpu_model, gpu_arch),
        0,
    )

    assert result["valu"] == [], "BF16 has no VALU roof regardless of matrix data"
    assert result["matrix_ops"] == [], (
        f"missing {matrix_ops_type} column should skip the roof"
    )


##############################################################################
# machine_ceilings Tests
##############################################################################


def write_roofline_csv(path: Path, header: str, rows: List[str]) -> None:
    """Write a roofline.csv with device-id column and data rows."""
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def capture_roofline_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> List[Tuple[object, ...]]:
    """Patch console_warning and return captured messages."""
    warnings: List[Tuple[object, ...]] = []

    def record_warning(*args: object) -> None:
        warnings.append(args)

    monkeypatch.setattr("utils.roofline_calc.console_warning", record_warning)
    return warnings


def test_machine_ceilings_reads_base_values_for_device(tmp_path: Path) -> None:
    """Device 0 selects base Bw/Flops/Ops columns, not Low/High variants."""
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,HBMBwLow,L2Bw,FP32Flops,FP32FlopsHigh,I8Ops",
        [
            "0,5300,5290,10000,81000,81100,40000",
            "1,6300,6290,20000,91000,91100,50000",
        ],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)

    assert machine_ceilings(parameters, MockMspec()) == (
        [5300.0, 10000.0],
        [81000.0, 40000.0],
    )

    parameters["device_id"] = 1
    assert machine_ceilings(parameters, MockMspec()) == (
        [6300.0, 20000.0],
        [91000.0, 50000.0],
    )


def test_machine_ceilings_drops_invalid_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N/A, nan, inf, and negative cells are dropped, yielding empty lists."""
    warnings = capture_roofline_warnings(monkeypatch)
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,L2Bw,FP32Flops,I8Ops",
        ["0,N/A,nan,inf,-1"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)

    assert machine_ceilings(parameters, MockMspec()) == ([], [])
    assert len(warnings) == 1
    assert "no usable" in str(warnings[0]).lower()


def test_machine_ceilings_missing_file_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing roofline.csv returns empty lists and logs a warning."""
    warnings = capture_roofline_warnings(monkeypatch)
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)

    assert machine_ceilings(parameters, MockMspec()) == ([], [])
    assert len(warnings) == 1
    assert "roofline.csv" in str(warnings[0])


def test_machine_ceilings_warns_when_workload_path_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent workload path returns empty lists and logs a warning."""
    warnings = capture_roofline_warnings(monkeypatch)
    parameters = roofline_parameters()
    parameters["workload_dir"] = None

    assert machine_ceilings(parameters, MockMspec()) == ([], [])
    assert len(warnings) == 1
    assert "workload path is absent" in str(warnings[0])


def test_machine_ceilings_negative_device_id(tmp_path: Path) -> None:
    """Negative device IDs return empty lists."""
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,FP32Flops",
        ["0,5300,81000"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)
    parameters["device_id"] = -1

    assert machine_ceilings(parameters, MockMspec()) == ([], [])


def test_machine_ceilings_out_of_range_device_id(tmp_path: Path) -> None:
    """Device IDs missing from the CSV return empty lists."""
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,FP32Flops",
        ["0,5300,81000"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)
    parameters["device_id"] = 5

    assert machine_ceilings(parameters, MockMspec()) == ([], [])


def test_machine_ceilings_matches_sparse_device_id(tmp_path: Path) -> None:
    """Device id matches the CSV first-column value, not the row offset."""
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,FP32Flops",
        ["2,5300,81000"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)
    parameters["device_id"] = 2

    assert machine_ceilings(parameters, MockMspec()) == ([5300.0], [81000.0])


def test_construct_roof_matches_sparse_semantic_device_id(tmp_path: Path) -> None:
    """A sparse device id selects its sole CSV row for every drawn ceiling."""
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,L2Bw,L1Bw,LDSBw,FP32Flops,MFMAF32Flops",
        ["2,500,600,700,800,2000,3000"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)
    parameters["device_id"] = 2

    graph_points = construct_roof(parameters, "FP32", MockMspec())

    assert machine_ceilings(parameters, MockMspec()) == (
        [500.0, 600.0, 700.0, 800.0],
        [2000.0, 3000.0],
    )
    assert graph_points["hbm"][2] == 500.0
    assert graph_points["valu"][2] == 2000.0
    assert graph_points["matrix_ops"][2] == 3000.0


def test_construct_roof_matches_reordered_semantic_device_id(tmp_path: Path) -> None:
    """Requested device 0 selects its semantic row when CSV rows are reordered."""
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,L2Bw,L1Bw,LDSBw,FP32Flops,MFMAF32Flops",
        [
            "2,2500,2600,2700,2800,12000,13000",
            "0,500,600,700,800,2000,3000",
        ],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)
    parameters["device_id"] = 0

    graph_points = construct_roof(parameters, "FP32", MockMspec())

    assert machine_ceilings(parameters, MockMspec()) == (
        [500.0, 600.0, 700.0, 800.0],
        [2000.0, 3000.0],
    )
    assert graph_points["hbm"][2] == 500.0
    assert graph_points["valu"][2] == 2000.0
    assert graph_points["matrix_ops"][2] == 3000.0


def test_construct_roof_invalid_device_id_returns_empty_graph(tmp_path: Path) -> None:
    """A requested semantic ID absent from the CSV fails without row indexing."""
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,L2Bw,L1Bw,LDSBw,FP32Flops,MFMAF32Flops",
        ["2,500,600,700,800,2000,3000"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)
    parameters["device_id"] = 0

    assert construct_roof(parameters, "FP32", MockMspec()) == (
        GraphPoints.empty().__dict__
    )


@pytest.mark.parametrize("invalid_value", ["N/A", "nan", "inf"])
def test_construct_roof_invalid_valu_returns_empty_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_value: str,
) -> None:
    """An unusable VALU peak returns an empty graph without nonfinite values."""
    warnings = capture_roofline_warnings(monkeypatch)
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,L2Bw,L1Bw,LDSBw,FP32Flops,MFMAF32Flops",
        [f"0,500,600,700,800,{invalid_value},3000"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)

    graph_points = construct_roof(parameters, "FP32", MockMspec())

    assert graph_points == GraphPoints.empty().__dict__
    assert len(warnings) == 1
    assert "invalid peak operations" in str(warnings[0]).lower()
    assert_graph_points_positive_finite(graph_points)


@pytest.mark.parametrize("invalid_value", ["N/A", "nan", "inf"])
def test_construct_roof_invalid_bandwidth_skips_level(
    tmp_path: Path,
    invalid_value: str,
) -> None:
    """An unusable bandwidth skips its level and preserves finite valid roofs."""
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,L2Bw,L1Bw,LDSBw,FP32Flops,MFMAF32Flops",
        [f"0,{invalid_value},600,700,800,2000,3000"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)

    graph_points = construct_roof(parameters, "FP32", MockMspec())

    assert graph_points["hbm"] == []
    assert graph_points["l2"][2] == 600.0
    assert graph_points["valu"][2] == 2000.0
    assert graph_points["matrix_ops"][2] == 3000.0
    assert_graph_points_positive_finite(graph_points)


@pytest.mark.parametrize("invalid_value", ["N/A", "nan", "inf"])
def test_construct_roof_invalid_matrix_skips_matrix_geometry(
    tmp_path: Path,
    invalid_value: str,
) -> None:
    """An unusable matrix peak is skipped while valid geometry remains finite."""
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,L2Bw,L1Bw,LDSBw,FP32Flops,MFMAF32Flops",
        [f"0,500,600,700,800,2000,{invalid_value}"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)

    graph_points = construct_roof(parameters, "FP32", MockMspec())

    assert graph_points["matrix_ops"] == []
    assert graph_points["hbm"][2] == 500.0
    assert graph_points["valu"][2] == 2000.0
    assert_graph_points_positive_finite(graph_points)


@pytest.mark.parametrize("invalid_value", ["N/A", "nan", "inf"])
def test_construct_roof_invalid_matrix_only_peak_returns_empty_geometry(
    tmp_path: Path,
    invalid_value: str,
) -> None:
    """Matrix-only BF16 emits no geometry without a valid matrix peak."""
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,L2Bw,L1Bw,LDSBw,MFMAF16Flops",
        [f"0,500,600,700,800,{invalid_value}"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)

    graph_points = construct_roof(parameters, "BF16", MockMspec())

    assert graph_points == {
        "hbm": [],
        "l2": [],
        "l1": [],
        "l0": [],
        "lds": [],
        "valu": [],
        "matrix_ops": [],
    }
    assert_graph_points_positive_finite(graph_points)


@pytest.mark.parametrize("device_id", [True, False], ids=["true", "false"])
def test_machine_ceilings_rejects_bool_device_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_id: bool
) -> None:
    """Boolean device_id values are rejected."""
    warnings = capture_roofline_warnings(monkeypatch)
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,FP32Flops",
        ["0,5300,81000"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)
    parameters["device_id"] = device_id

    assert machine_ceilings(parameters, MockMspec()) == ([], [])
    assert "boolean" in str(warnings[0]).lower()


def test_machine_ceilings_rejects_fractional_device_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fractional device_id values are rejected."""
    warnings = capture_roofline_warnings(monkeypatch)
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,FP32Flops",
        ["0,5300,81000"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)
    parameters["device_id"] = 2.5

    assert machine_ceilings(parameters, MockMspec()) == ([], [])
    assert "integral" in str(warnings[0]).lower()


def test_machine_ceilings_rejects_duplicate_benchmark_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicate benchmark headers fail closed instead of corrupting row mapping."""
    warnings = capture_roofline_warnings(monkeypatch)
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,HBMBw,FP32Flops",
        ["0,5300,10000,81000"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)

    assert machine_ceilings(parameters, MockMspec()) == ([], [])
    assert "duplicate benchmark header" in str(warnings[0]).lower()


def test_machine_ceilings_rejects_duplicate_device_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicate device ids in roofline.csv are rejected."""
    warnings = capture_roofline_warnings(monkeypatch)
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,FP32Flops",
        ["0,5300,81000", "0,6300,91000"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)

    assert machine_ceilings(parameters, MockMspec()) == ([], [])
    assert "duplicate device id" in str(warnings[0]).lower()


def test_machine_ceilings_rejects_short_csv_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows with too few columns are rejected."""
    warnings = capture_roofline_warnings(monkeypatch)
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,FP32Flops",
        ["0,5300"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)

    assert machine_ceilings(parameters, MockMspec()) == ([], [])
    assert "expected 3" in str(warnings[0]).lower()


def test_machine_ceilings_rejects_over_wide_csv_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows with too many columns are rejected."""
    warnings = capture_roofline_warnings(monkeypatch)
    write_roofline_csv(
        tmp_path / "roofline.csv",
        "device,HBMBw,FP32Flops",
        ["0,5300,81000,extra"],
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)

    assert machine_ceilings(parameters, MockMspec()) == ([], [])
    assert "expected 3" in str(warnings[0]).lower()


def test_machine_ceilings_rejects_empty_csv_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty data rows are rejected."""
    warnings = capture_roofline_warnings(monkeypatch)
    (tmp_path / "roofline.csv").write_text(
        "device,HBMBw,FP32Flops\n0,5300,81000\n\n",
        encoding="utf-8",
    )
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)

    assert machine_ceilings(parameters, MockMspec()) == ([], [])
    assert "empty" in str(warnings[0]).lower()


@pytest.mark.parametrize(
    ("header", "row", "message"),
    [
        ("device,HBMBw,FP32Flops", "0,N/A,81000", "bandwidth"),
        ("device,HBMBw,FP32Flops", "0,5300,N/A", "compute"),
    ],
    ids=["no-bandwidth", "no-peaks"],
)
def test_machine_ceilings_warns_when_ceiling_group_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    header: str,
    row: str,
    message: str,
) -> None:
    """Valid CSV rows with no usable bandwidth or peaks warn and return empty."""
    warnings = capture_roofline_warnings(monkeypatch)
    write_roofline_csv(tmp_path / "roofline.csv", header, [row])
    parameters = roofline_parameters()
    parameters["workload_dir"] = str(tmp_path)

    assert machine_ceilings(parameters, MockMspec()) == ([], [])
    assert message in str(warnings[0]).lower()


def test_calc_ceilings_uses_xmax_default_for_compute_roofs() -> None:
    """Compute roof x-axis endpoints use XMAX_DEFAULT, not kernel AI extent."""
    result = calc_ceilings(
        roofline_parameters("MFMA", MFMA_GPU_ARCH),
        "FP32",
        full_benchmark_data(),
        MockMspec(),
        0,
    )

    assert result["valu"][0][1] == XMAX_DEFAULT
    assert result["matrix_ops"][0][1] == XMAX_DEFAULT


# -------------------------------------------------------------------------------------
#                     device_ceilings: ceilings stored in the database
# -------------------------------------------------------------------------------------
def test_device_ceilings_stores_every_positive_mean_of_device_zero(
    tmp_path: Path,
) -> None:
    """Each positive device-0 mean is one ceiling in base units, MALL included;
    the device-1 row is not read."""
    csv_path = tmp_path / "roofline.csv"
    write_roofline_csv(
        csv_path,
        "device,HBMBw,MALLBw,L2Bw,L1Bw,LDSBw,FP32Flops,MFMAF16Flops,MFMAI8Ops",
        [
            "0,5300,9000,10000,20000,30000,81000,500000,700000",
            "1,6300,9100,20100,21000,31000,91000,600000,800000",
        ],
    )

    assert ceilings_by_roof(device_ceilings(csv_path, "gfx942")) == {
        ("bandwidth", "HBM", None, None): ("Bytes/s", 5.3e12),
        ("bandwidth", "MALL", None, None): ("Bytes/s", 9e12),
        ("bandwidth", "L2", None, None): ("Bytes/s", 1e13),
        ("bandwidth", "L1", None, None): ("Bytes/s", 2e13),
        ("bandwidth", "LDS", None, None): ("Bytes/s", 3e13),
        ("compute", None, "FP32", "VALU"): ("FLOP/s", 8.1e13),
        ("compute", None, "FP16", "MFMA"): ("FLOP/s", 5e14),
        ("compute", None, "I8", "MFMA"): ("IOP/s", 7e14),
    }


def test_device_ceilings_skips_non_positive_and_invalid_means(
    tmp_path: Path,
) -> None:
    """Zero, negative, N/A, nan and inf means are not ceilings, and neither are the
    Low and High bounds of a mean."""
    csv_path = tmp_path / "roofline.csv"
    write_roofline_csv(
        csv_path,
        "device,HBMBw,HBMBwLow,HBMBwHigh,L2Bw,L1Bw,FP32Flops,MFMAF16Flops",
        ["0,0,4000,4100,10000,-1,nan,N/A"],
    )

    assert ceilings_by_roof(device_ceilings(csv_path, "gfx942")) == {
        ("bandwidth", "L2", None, None): ("Bytes/s", 1e13),
    }


def test_device_ceilings_ignores_roofs_the_arch_does_not_measure(
    tmp_path: Path,
) -> None:
    """A column is a ceiling only when the arch's roofline benchmark measures it."""
    csv_path = tmp_path / "roofline.csv"
    write_roofline_csv(csv_path, "device,MALLBw,L2Bw", ["0,9000,10000"])

    assert ceilings_by_roof(device_ceilings(csv_path, "gfx90a")) == {
        ("bandwidth", "L2", None, None): ("Bytes/s", 1e13),
    }


def test_device_ceilings_names_wmma_pipe_and_mixed_precision_types(
    tmp_path: Path,
) -> None:
    """WMMA ceilings carry the WMMA pipe; mixed-precision types such as F6F4 keep
    their roofline.csv name and are stored too."""
    csv_path = tmp_path / "roofline.csv"
    write_roofline_csv(
        csv_path,
        "device,L0Bw,WMMAF16Flops,WMMAF6F4Flops",
        ["0,8000,300000,250000"],
    )

    assert ceilings_by_roof(device_ceilings(csv_path, "gfx1250")) == {
        ("bandwidth", "L0", None, None): ("Bytes/s", 8e12),
        ("compute", None, "FP16", "WMMA"): ("FLOP/s", 3e14),
        ("compute", None, "F6F4", "WMMA"): ("FLOP/s", 2.5e14),
    }


def test_device_ceilings_without_device_zero_raises(tmp_path: Path) -> None:
    """A roofline.csv with no device-0 row has no ceilings to store."""
    csv_path = tmp_path / "roofline.csv"
    write_roofline_csv(csv_path, "device,HBMBw", ["1,5300"])

    with pytest.raises(ValueError, match="device id 0 occurs 0 times"):
        device_ceilings(csv_path, "gfx942")


def test_device_ceilings_reads_the_mi200_fixture() -> None:
    """The MI200 mem_levels fixture gives its device-0 row, with the MI200 mean
    bandwidths and compute rates."""
    fixture = (
        Path(__file__).resolve().parents[2]
        / "workloads/mem_levels_HBM/MI200/roofline.csv"
    )

    ceilings = ceilings_by_roof(device_ceilings(fixture, "gfx90a"))

    assert len(ceilings) == 11
    unit, hbm = ceilings[("bandwidth", "HBM", None, None)]
    assert unit == "Bytes/s"
    assert hbm == pytest.approx(1388.7961e9)
    assert ceilings[("compute", None, "FP32", "VALU")] == (
        "FLOP/s",
        pytest.approx(20948.875e9),
    )
