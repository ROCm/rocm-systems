# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Round-trip coverage for shared roofline computation and queried rendering."""

from types import SimpleNamespace

import pandas as pd
import pytest

from roofline.roofline_analysis import load_roofline_view, persist_roofline
from utils import analysis_orm as orm
from utils.roofline_calc import RooflineBenchmark


def make_arch_config():
    rates = pd.DataFrame(
        {
            "Metric": ["VALU rate", "Unmapped peak"],
            "Value": ["rate", "rate"],
            "Peak (Empirical)": [
                "FP32Flops_empirical_peak",
                "MFMAF6F4Flops_empirical_peak",
            ],
            "Percent of Peak": [True, False],
            "Unit": ["GFLOP/s", "GFLOP/s"],
        },
        index=["4.1.0", "4.1.1"],
    )
    points = pd.DataFrame(
        {
            "Metric": ["AI HBM", "AI L2", "Performance (GFLOPs)"],
            "Value": ["intensity", "missing", "performance"],
            "Unit": ["FLOPs/Byte", "FLOPs/Byte", "GFLOP/s"],
        },
        index=["4.2.0", "4.2.1", "4.2.2"],
    )
    return SimpleNamespace(dfs={401: rates, 402: points})


def make_frames():
    return pd.DataFrame({
        "Kernel_Name": ["long", "frequent", "frequent", "zero"],
        "Start_Timestamp": [0, 1000, 2000, 3000],
        "End_Timestamp": [900, 1600, 2600, 3100],
        "performance": [10.0, 20.0, 20.0, None],
        "intensity": [2.0, 4.0, 4.0, None],
        "rate": [5.0, 10.0, 10.0, None],
    })


def make_benchmark():
    return RooflineBenchmark(
        2,
        {
            "HBMBw": 100.0,
            "L2Bw": 200.0,
            "MALLBw": 300.0,
            "FP32Flops": 50.0,
            "MFMAF32Flops": 100.0,
            "MFMAF6F4Flops": 150.0,
            "I8Ops": 25.0,
        },
        "MFMA",
    )


def evaluate(name, expression, pmc_df, context):
    if expression.endswith("_empirical_peak"):
        return context[expression]
    if expression == "missing":
        return None
    return pmc_df[expression].iloc[0]


def compute(pmc_df=None, stats_df=None, arch_config=None):
    from roofline.roofline_analysis import compute_roofline

    return compute_roofline(
        sys_info={"gpu_arch": "gfx90a", "gpu_model": "MI210"},
        arch_config=arch_config or make_arch_config(),
        benchmark=make_benchmark(),
        pmc_df=make_frames() if pmc_df is None else pmc_df,
        stats_df=make_frames() if stats_df is None else stats_df,
        evaluate=evaluate,
    )


def test_compute_retains_kernels_and_uses_unfiltered_stats():
    frame = make_frames()
    result = compute(frame[frame.Kernel_Name != "long"], frame)
    assert [kernel.kernel_name for kernel in result.kernels] == ["frequent", "zero"]
    frequent, zero = result.kernels
    assert frequent.kernel_rank == 0
    assert frequent.dispatch_count == 2
    assert frequent.total_duration_ns == 1200
    assert frequent.percent_runtime == pytest.approx(1200 / 2200 * 100)
    assert zero.kernel_rank == 2
    assert zero.performance == 0
    assert zero.level_ai == {"HBM": 0, "L2": 0}
    assert zero.metrics[-1].value is None
    assert frequent.metrics[0].peak == 50
    assert frequent.metrics[0].percent_of_peak == 20
    assert frequent.metrics[1].peak == 150
    assert set(frequent.envelopes) == {"FP16", "BF16", "FP32", "FP64", "MEMORY"}
    assert frequent.envelopes["FP32"].compute_ceiling == 100
    assert frequent.envelopes["FP32"].compute_ceiling_label == "FP32 MFMA"
    assert frequent.envelopes["FP32"].bounds["HBM"] == (100, 20)


@pytest.mark.parametrize("points", [None, pd.DataFrame()])
def test_missing_points_table_returns_ceilings_without_kernels(points):
    arch = make_arch_config()
    if points is None:
        del arch.dfs[402]
    else:
        arch.dfs[402] = points
    result = compute(arch_config=arch)
    assert result.kernels == []
    assert result.benchmark.peaks == make_benchmark().peaks


def test_compute_persist_view_round_trip_keeps_nulls_and_render_stats(db_session):
    result = compute()
    assert [kernel.kernel_name for kernel in result.kernels] == [
        "long",
        "frequent",
        "zero",
    ]
    workload = orm.Workload(
        name="workload", sub_name="run", sys_info_extdata={"gpu_arch": "gfx90a"}
    )
    kernels = {
        kernel.kernel_name: orm.Kernel(
            kernel_name=kernel.kernel_name, workload=workload
        )
        for kernel in result.kernels
    }
    db_session.add(workload)
    persist_roofline(result, workload, kernels)
    orm.Database.commit()
    view = load_roofline_view(workload.workload_id, 2, "us")
    assert view.benchmark_peaks == make_benchmark().peaks
    assert view.plot_points["kernelNames"] == ["frequent", "long"]
    assert view.plot_points["counts"] == [2, 1]
    assert view.plot_points["totalTime"] == [1.2, 0.9]
    assert view.plot_points["timeUnit"] == "us"
    assert view.plot_points["ai_hbm"] == [[4, 2], [20, 10]]
    assert view.plot_points["ai_l0"] == [[0, 0], [20, 10]]
    assert db_session.query(orm.KernelRooflineMetric).count() == 15
    assert db_session.query(orm.KernelRooflineLimiter).count() == 15
    assert (
        db_session
        .query(orm.RooflineComputeCeiling)
        .filter_by(benchmark_column="MFMAF6F4Flops")
        .one()
        .datatype
        is None
    )
    tables = view.tty_tables(make_arch_config(), [])
    assert list(tables) == [0, 1, 2]
    assert tables[2]["calc_table"].loc["4.2.2", "Value"] == "N/A"
    assert tables[2]["ai_table"].loc["4.1.0", "Percent of Peak"] == ""
    assert list(view.tty_tables(make_arch_config(), [2, 0])) == [2, 0]
    assert not set(db_session.query(orm.KernelRooflinePoint.mem_level).all()) & {
        ("MALL",)
    }


def test_positive_kernel_with_no_mapped_levels_survives_round_trip(db_session):
    arch = make_arch_config()
    arch.dfs[402] = arch.dfs[402].loc[["4.2.2"]]
    result = compute(arch_config=arch)
    workload = orm.Workload(
        name="workload", sub_name="run", sys_info_extdata={"gpu_arch": "gfx90a"}
    )
    kernels = {
        kernel.kernel_name: orm.Kernel(
            kernel_name=kernel.kernel_name, workload=workload
        )
        for kernel in result.kernels
    }
    db_session.add(workload)
    persist_roofline(result, workload, kernels)
    orm.Database.commit()
    view = load_roofline_view(workload.workload_id, 2, "ns")
    assert view.plot_points["kernelNames"] == ["frequent", "long"]
    assert view.plot_points["ai_hbm"] == [[0, 0], [20, 10]]


@pytest.mark.parametrize("raw", [None, "N/A", float("nan"), float("inf"), -5.0])
def test_invalid_points_are_cleaned_without_dropping_the_kernel(raw):
    frame = make_frames().iloc[[0]].copy()
    frame["performance"] = raw
    frame["intensity"] = raw
    result = compute(frame, frame)
    assert len(result.kernels) == 1
    expected = -5.0 if raw == -5.0 else 0.0
    assert result.kernels[0].performance == expected
    assert result.kernels[0].level_ai["HBM"] == expected


def test_builtin_context_isolated_per_kernel_and_all_empirical_vars(monkeypatch):
    from roofline import roofline_analysis

    contexts = []

    def inject_builtin(frame, context, expressions):
        assert "kernel_builtin" not in context
        assert context["MFMAF6F4Flops_empirical_peak"] == 150
        context["kernel_builtin"] = str(frame.Kernel_Name.iloc[0])
        contexts.append(context)

    monkeypatch.setattr(roofline_analysis, "calc_builtin_vars", inject_builtin)
    result = compute()
    assert len(contexts) == len(result.kernels) == 3
    assert len({id(context) for context in contexts}) == 3


def test_persist_missing_kernel_warns_and_keeps_ceilings(db_session, caplog):
    from roofline.roofline_analysis import persist_roofline

    workload = orm.Workload(
        name="workload", sub_name="run", sys_info_extdata={"gpu_arch": "gfx90a"}
    )
    db_session.add(workload)
    persist_roofline(compute(), workload, {})
    orm.Database.commit()
    assert db_session.query(orm.RooflineComputeCeiling).count() == 4
    assert db_session.query(orm.KernelRooflineData).count() == 0
    assert "from roofline data not found in dispatch data" in caplog.text


def test_view_loads_only_query_rows_and_keeps_templates(monkeypatch):
    from roofline.roofline_analysis import load_roofline_view

    monkeypatch.setattr(
        orm.Database,
        "get_roofline_ceilings",
        lambda *args: [{"benchmark_column": "HBMBw", "value": 123.0}],
    )
    monkeypatch.setattr(
        orm.Database,
        "get_kernel_roofline_rows",
        lambda *args: [
            {
                "kernel_uuid": 9,
                "kernel_name": "kernel",
                "kernel_rank": 4,
                "dispatch_count": 3,
                "total_duration_ns": 1500,
                "percent_runtime": 75,
                "envelope": None,
                "total_flops": 42,
                "hbm_cache_data": 2,
            }
        ],
    )
    monkeypatch.setattr(
        orm.Database,
        "get_kernel_roofline_metrics",
        lambda *args: [
            {
                "kernel_uuid": 9,
                "metric_id": "4.1.0",
                "value": None,
                "peak": 50,
                "percent_of_peak": None,
            }
        ],
    )
    view = load_roofline_view(1, 2, "us")
    assert view.plot_points["ai_hbm"] == [[2], [42]]
    assert view.benchmark_peaks == {"HBMBw": 123}
    arch = make_arch_config()
    originals = {key: frame.copy() for key, frame in arch.dfs.items()}
    assert view.tty_tables(arch, [4])[4]["ai_table"].loc["4.1.0", "Value"] == "N/A"
    for key, frame in originals.items():
        pd.testing.assert_frame_equal(arch.dfs[key], frame)


@pytest.mark.parametrize("performance_row_present", [False, True])
def test_persist_distinguishes_absent_performance_row_from_invalid_value(
    db_session, performance_row_present
):
    """Absent performance stores NULL; a present invalid evaluation stores zero."""
    arch = make_arch_config()
    metric_ids = ["4.2.0", "4.2.2"] if performance_row_present else ["4.2.0"]
    arch.dfs[402] = arch.dfs[402].loc[metric_ids]
    frame = make_frames().iloc[[0]].copy()
    frame["performance"] = None
    result = compute(frame, frame, arch)
    assert result.kernels[0].performance == 0
    workload = orm.Workload(
        name="workload", sub_name="run", sys_info_extdata={"gpu_arch": "gfx90a"}
    )
    kernels = {
        kernel.kernel_name: orm.Kernel(
            kernel_name=kernel.kernel_name, workload=workload
        )
        for kernel in result.kernels
    }
    db_session.add(workload)
    persist_roofline(result, workload, kernels)
    orm.Database.commit()
    stored = db_session.query(orm.KernelRooflineData).one()
    if performance_row_present:
        assert stored.total_flops == 0
    else:
        assert stored.total_flops is None
    assert stored.hbm_cache_data == 2
    assert stored.l2_cache_data is None
    assert (
        load_roofline_view(workload.workload_id, 2, "ns").plot_points["kernelNames"]
        == []
    )
