# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT


import numpy as np
import pandas as pd
import pytest

from utils import schema
from utils.mi_gpu_spec import mi_gpu_specs
from utils.roofline_calc import (
    GraphPoints,
    calc_ai_analyze,
    calc_ceilings,
    sanitize_ai_value,
    sanitize_mem_level,
)


def run_calc_ai_analyze_with_values(
    monkeypatch: pytest.MonkeyPatch, metric_values: dict[str, object]
) -> dict:
    """
    Build mocks and invoke calc_ai_analyze with controlled metric values.

    ``metric_values`` is a dict with keys ``ai_hbm``, ``ai_l2``, ``ai_l1``,
    ``ai_lds``, ``performance`` whose values are injected into the table-402
    DataFrame that ``eval_metric`` would normally populate.

    Returns the plot-points dict produced by ``calc_ai_analyze``.

    Note: this mock simulates MI350 AI metric values based on the architecture's cache
    levels available on the hardware. Cache levels will vary for other architectures.
    """
    kernel_name = "test_kernel"
    kernel_id = 0

    workload = schema.Workload()
    workload.dfs = {1: pd.DataFrame({"Kernel_Name": [kernel_name]}, index=[kernel_id])}
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


def test_calc_ai_analyze_replaces_none_with_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None metric values are replaced with 0 and still included in plot points."""
    result = run_calc_ai_analyze_with_values(
        monkeypatch,
        {
            "ai_hbm": None,
            "ai_l2": None,
            "ai_l1": None,
            "ai_lds": None,
            "performance": 50.0,
        },
    )

    assert result["kernelNames"] == ["test_kernel"]
    assert result["ai_hbm"][0] == [0], "None should be replaced with 0"
    assert result["ai_l2"][0] == [0], "None should be replaced with 0"
    assert result["ai_l1"][0] == [0], "None should be replaced with 0"
    assert result["ai_lds"][0] == [0], "None should be replaced with 0"


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


def test_calc_ai_analyze_na_and_empty_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sentinel values 'N/A' and '' are replaced with 0."""
    result = run_calc_ai_analyze_with_values(
        monkeypatch,
        {
            "ai_hbm": "N/A",
            "ai_l2": "",
            "ai_l1": "N/A",
            "ai_lds": "",
            "performance": 75.0,
        },
    )

    assert result["kernelNames"] == ["test_kernel"]
    assert result["ai_hbm"][0] == [0], "'N/A' should be replaced with 0"
    assert result["ai_l2"][0] == [0], "'' should be replaced with 0"
    assert result["ai_l1"][0] == [0], "'N/A' should be replaced with 0"
    assert result["ai_lds"][0] == [0], "'' should be replaced with 0"


def test_sanitize_ai_value_replaces_invalid_values_with_zero() -> None:
    """Invalid values are replaced with 0."""
    assert sanitize_ai_value(np.inf) == 0
    assert sanitize_ai_value(-np.inf) == 0
    assert sanitize_ai_value("N/A") == 0
    assert sanitize_ai_value("") == 0
    assert sanitize_ai_value(None) == 0
    assert sanitize_ai_value(1.5) == 1.5


##############################################################################
# sanitize_mem_level Tests
##############################################################################


def test_sanitize_mem_level_all_falls_back_to_hierarchy() -> None:
    """'ALL' results in full memory levels for the model, minus MALL."""
    result = sanitize_mem_level("ALL", "mi210")
    # mi210 has no MALL, so result equals memory_levels directly
    assert result == mi_gpu_specs.get_memory_levels("mi210")


def test_sanitize_mem_level_all_list_falls_back_to_hierarchy() -> None:
    """['ALL'] behaves the same as 'ALL'."""
    result = sanitize_mem_level(["ALL"], "mi210")
    assert result == mi_gpu_specs.get_memory_levels("mi210")


def test_sanitize_mem_level_supported_string() -> None:
    """A supported single string level is returned as a single-item list."""
    result = sanitize_mem_level("HBM", "mi210")
    assert result == ["HBM"]


def test_sanitize_mem_level_supported_list() -> None:
    """A list of supported levels is returned unchanged."""
    result = sanitize_mem_level(["HBM", "L2"], "mi210")
    assert result == ["HBM", "L2"]


def test_sanitize_mem_level_unsupported_falls_back_to_hierarchy() -> None:
    """Fully unsupported input falls back to full memory levels, minus MALL."""
    result = sanitize_mem_level("HBM", "rdna35_halo")
    # rdna35_halo memory_levels includes MALL, which is stripped by sanitize_mem_level
    expected = [m for m in mi_gpu_specs.get_memory_levels("rdna35_halo") if m != "MALL"]
    assert result == expected


def test_sanitize_mem_level_mixed_filters_unsupported() -> None:
    """Unsupported levels in a mixed list are filtered out."""
    # rdna35_halo supports L0, L1, L2, MALL, LDS — not HBM; MALL is also stripped
    result = sanitize_mem_level(["HBM", "L2"], "rdna35_halo")
    assert result == ["L2"]


def test_sanitize_mem_level_mall_is_stripped() -> None:
    """MALL is always removed from results even when explicitly requested."""
    result = sanitize_mem_level("MALL", "rdna35_halo")
    assert "MALL" not in result


def test_sanitize_mem_level_vl1d_normalised() -> None:
    """'vL1D' is normalised to 'L1' before filtering."""
    result = sanitize_mem_level("vL1D", "mi210")
    assert result == ["L1"]


##############################################################################
# calc_ceilings Tests
##############################################################################

# gfx90a-class model whose memory levels (LDS/L1/L2/HBM) match BW_COLUMNS.
MFMA_GPU_MODEL = "mi210"
# gfx1151-class model; memory levels resolve to LDS/L0/L1/L2 (MALL skipped).
WMMA_GPU_MODEL = "rdna35_halo"

# Union of BW columns needed by both models (mi210 reads HBM, rdna35_halo reads
# L0); calc_ceilings only consumes the levels its model supports.
BW_COLUMNS = ["HBMBw", "L2Bw", "L1Bw", "L0Bw", "LDSBw"]
BW_VALUE = 500.0

PEAK_VALUES = {
    "FP16Flops": 1000.0,
    "FP32Flops": 2000.0,
    "FP64Flops": 3000.0,
    "I8Ops": 4000.0,
    "I32Ops": 5000.0,
    "I64Ops": 6000.0,
    "MFMAF4Flops": 7000.0,
    "MFMAF6Flops": 8000.0,
    "MFMAF8Flops": 9000.0,
    "MFMAF16Flops": 10000.0,
    "MFMAF32Flops": 11000.0,
    "MFMAF64Flops": 12000.0,
    "MFMAI8Ops": 13000.0,
    "WMMAF16Flops": 20000.0,
    "WMMABF16Flops": 21000.0,
    "WMMAF32Flops": 22000.0,
    "WMMAF64Flops": 23000.0,
    "WMMAI8Ops": 24000.0,
}

# MFMA (CDNA) supports the full datatype set.
MFMA_CASES = [
    ("FP32", "FP32Flops", "MFMAF32Flops"),
    ("FP16", "FP16Flops", "MFMAF16Flops"),
    ("FP64", "FP64Flops", "MFMAF64Flops"),
    ("I8", "I8Ops", "MFMAI8Ops"),
    ("I32", "I32Ops", None),
    ("I64", "I64Ops", None),
    ("BF16", None, "MFMAF16Flops"),
    ("FP8", None, "MFMAF8Flops"),
    ("FP4", None, "MFMAF4Flops"),
    ("FP6", None, "MFMAF6Flops"),
]

# FP4/FP6/FP8 are not supported on gfx1151.
WMMA_CASES = [
    ("FP32", "FP32Flops", "WMMAF32Flops"),
    ("FP16", "FP16Flops", "WMMAF16Flops"),
    ("FP64", "FP64Flops", "WMMAF64Flops"),
    ("I8", "I8Ops", "WMMAI8Ops"),
    ("I32", "I32Ops", None),
    ("I64", "I64Ops", None),
    ("BF16", None, "WMMAF16Flops"),
]

# (matrix_ops_type, gpu_model, cases) tuples driving the parametrized tests.
MATRIX_FAMILIES = [
    ("MFMA", MFMA_GPU_MODEL, MFMA_CASES),
    ("WMMA", WMMA_GPU_MODEL, WMMA_CASES),
]

# Flattened (matrix_ops_type, gpu_model, dtype, valu_col, matrix_col) rows.
ROOFLINE_DATATYPE_CASES = [
    (prefix, model, dtype, valu_col, matrix_col)
    for prefix, model, cases in MATRIX_FAMILIES
    for (dtype, valu_col, matrix_col) in cases
]


class MockMspec:
    """Minimal stand-in for MachineSpecs; calc_ceilings only reads gpu_model."""

    def __init__(self, gpu_model: str = MFMA_GPU_MODEL) -> None:
        self.gpu_model = gpu_model


def roofline_parameters(matrix_ops_type: str = "MFMA") -> dict[str, object]:
    # matrix_ops_type is "MFMA" for CDNA (MI-series) and "WMMA" for RDNA.
    return {
        "device_id": 0,
        "mem_level": "ALL",
        "workload_dir": "/tmp",
        "matrix_ops_type": matrix_ops_type,
    }


def full_benchmark_data() -> dict[str, list[str]]:
    """Benchmark dict with every BW, PEAK_OPS and matrix column populated."""
    data = {col: [str(BW_VALUE)] for col in BW_COLUMNS}
    for col, value in PEAK_VALUES.items():
        data[col] = [str(value)]
    return data


@pytest.mark.parametrize(
    ("matrix_ops_type", "gpu_model", "dtype", "valu_col", "matrix_col"),
    ROOFLINE_DATATYPE_CASES,
    ids=[f"{row[0]}-{row[2]}" for row in ROOFLINE_DATATYPE_CASES],
)
def test_calc_ceilings_roofline_datatype(
    matrix_ops_type: str,
    gpu_model: str,
    dtype: str,
    valu_col: str | None,
    matrix_col: str | None,
) -> None:
    """Each datatype populates exactly its expected VALU and/or matrix roof."""
    result = calc_ceilings(
        roofline_parameters(matrix_ops_type),
        dtype,
        full_benchmark_data(),
        MockMspec(gpu_model),
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


@pytest.mark.parametrize(
    ("matrix_ops_type", "gpu_model", "matrix_col"),
    [
        ("MFMA", MFMA_GPU_MODEL, "MFMAF16Flops"),
        ("WMMA", WMMA_GPU_MODEL, "WMMAF16Flops"),
    ],
    ids=["MFMA", "WMMA"],
)
def test_bf16_uses_f16_matrix_column(
    matrix_ops_type: str, gpu_model: str, matrix_col: str
) -> None:
    """BF16 has no VALU roof and reads its matrix peak from the F16 column.

    Proves the ``f"F{dtype[2:]}"`` remap: BF16 -> F16 -> {prefix}F16Flops.
    """
    result = calc_ceilings(
        roofline_parameters(matrix_ops_type),
        "BF16",
        full_benchmark_data(),
        MockMspec(gpu_model),
    )

    assert result["valu"] == [], "BF16 is matrix-only; no VALU roof expected"
    assert result["matrix_ops"][2] == PEAK_VALUES[matrix_col]


def test_fp8_special_mfma_only() -> None:
    """FP8 is matrix-only and reads its peak from the dedicated MFMAF8 column.

    FP8 is a CDNA/MFMA datatype only.
    """
    result = calc_ceilings(
        roofline_parameters("MFMA"), "FP8", full_benchmark_data(), MockMspec()
    )

    assert result["valu"] == [], "FP8 is not a PEAK_OPS datatype; no VALU roof"
    assert result["matrix_ops"][2] == PEAK_VALUES["MFMAF8Flops"]


def test_missing_peak_ops_column_returns_empty() -> None:
    """A PEAK_OPS datatype missing its Flops column returns empty ceilings."""
    benchmark_data = full_benchmark_data()
    del benchmark_data["FP64Flops"]

    result = calc_ceilings(roofline_parameters(), "FP64", benchmark_data, MockMspec())

    assert result == GraphPoints.empty().__dict__


@pytest.mark.parametrize(
    ("matrix_ops_type", "gpu_model", "matrix_col"),
    [
        ("MFMA", MFMA_GPU_MODEL, "MFMAF16Flops"),
        ("WMMA", WMMA_GPU_MODEL, "WMMAF16Flops"),
    ],
    ids=["MFMA", "WMMA"],
)
def test_missing_matrix_column_skips_matrix_roof(
    matrix_ops_type: str, gpu_model: str, matrix_col: str
) -> None:
    """A matrix datatype missing its matrix column emits no matrix roof.

    BF16 is matrix-only.
    """
    benchmark_data = full_benchmark_data()
    del benchmark_data[matrix_col]

    result = calc_ceilings(
        roofline_parameters(matrix_ops_type),
        "BF16",
        benchmark_data,
        MockMspec(gpu_model),
    )

    assert result["valu"] == [], "BF16 has no VALU roof regardless of matrix data"
    assert result["matrix_ops"] == [], (
        f"missing {matrix_ops_type} column should skip the roof"
    )
