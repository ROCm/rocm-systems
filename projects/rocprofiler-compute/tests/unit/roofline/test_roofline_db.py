# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit coverage for the roofline HTML rendered from the analysis database rows."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from roofline.roofline_db import (
    PLOT_POINTS_SUB_TABLE,
    KernelPlotRow,
    plot_points_from_rows,
    write_roofline_html,
)
from roofline.roofline_main import Roofline
from utils import analysis_orm as orm
from utils.roofline_calc import device_ceilings

FIXTURE_DIR = (
    Path(__file__).resolve().parents[2] / "workloads" / "mem_levels_HBM" / "MI200"
)
GFLOP = 1e9


class MI200Mspec:
    """The MI210 (gfx90a) the mem_levels_HBM MI200 fixture was profiled on."""

    gpu_model = "MI210"
    gpu_series = "mi200"
    gpu_arch = "gfx90a"


def embedded_model(document: str) -> dict:
    """Parse the JSON model from a standalone roofline document."""
    match = re.search(
        r'<script id="roofline-model" type="application/json">(.*?)</script>',
        document,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def kernel_hovers(document: str) -> Dict[str, Dict[str, Any]]:
    """Each kernel's limiting peak and the tooltip cells of its first point."""
    return {
        kernel["name"]: {
            "limitingPeak": kernel["limitingPeak"],
            "roof": kernel["points"][0]["hoverCells"][0],
            "percent": kernel["points"][0]["hoverCells"][1],
        }
        for kernel in embedded_model(document)["kernels"]
    }


# Kernel points in GFLOP/s. kernelB is compute bound: it sits above the HBM roof,
# under the FP64 VALU peak. The MI200 fixture plots no kernels from real data (its
# only kernel has zero performance), so these points are hand-set.
CLI_KERNELS = {
    "kernelA": {
        "ai": {"HBM": 1.0, "L2": 0.5, "L1": 0.3, "LDS": 2.0},
        "performance": 500.0,
    },
    "kernelB": {
        "ai": {"HBM": 20.0, "L2": 10.0, "L1": 8.0, "LDS": 40.0},
        "performance": 15000.0,
    },
}


def cli_ai_data() -> Dict[str, Any]:
    """The AI dataset the CLI roofline builds for CLI_KERNELS."""
    names = list(CLI_KERNELS)
    data: Dict[str, Any] = {
        "kernelNames": names,
        "counts": [4.0, 6.0],
        "kernelTime": [4.0e6, 6.0e6],
        "kernelPctRuntime": [40.0, 60.0],
        "totalAppTime": 1.0e7,
        "totalAppDispatches": 10.0,
        "timeUnit": "ns",
    }
    levels = [("L1", "ai_l1"), ("L2", "ai_l2"), ("HBM", "ai_hbm"), ("LDS", "ai_lds")]
    for level, field in levels:
        data[field] = [
            [CLI_KERNELS[name]["ai"][level] for name in names],
            [CLI_KERNELS[name]["performance"] for name in names],
        ]
    return data


def kernel_plot_row(
    name: str,
    performance: Optional[float],
    ai: Dict[str, Optional[float]],
    duration_ns: int = 1000,
    dispatch_count: int = 1,
) -> KernelPlotRow:
    """One kernel's stored rows. Values are in base units: FLOP/s and FLOPs/Byte."""
    metrics: Dict[str, Any] = {
        "Performance (GFLOPs)": None if performance is None else performance * GFLOP
    }
    for level, value in ai.items():
        metrics[f"AI {level}"] = value
    return KernelPlotRow(
        kernel_name=name,
        metrics=metrics,
        dispatch_count=dispatch_count,
        duration_ns=duration_ns,
    )


def add_ceilings(session, workload) -> None:
    """Store the MI200 fixture's device-0 ceiling rows against the workload."""
    for ceiling in device_ceilings(FIXTURE_DIR / "roofline.csv", "gfx90a"):
        session.add(
            orm.RooflineCeiling(
                workload=workload,
                kind=ceiling.kind,
                memory_level=ceiling.memory_level,
                data_type=ceiling.data_type,
                pipe=ceiling.pipe,
                unit=ceiling.unit,
                value=ceiling.value,
            )
        )


def add_kernel_rows(session, workload, kernels: Dict[str, Dict[str, Any]]) -> None:
    """Store each kernel with its dispatches and Roofline Plot Points metric rows.

    Only the metrics the kernels use are defined, as the analysis defines them.
    """
    used_levels = sorted({
        level for values in kernels.values() for level in values["ai"]
    })
    metric_names = ["Performance (GFLOPs)", *(f"AI {level}" for level in used_levels)]
    metric_objs = {
        name: orm.MetricDefinition(
            workload=workload,
            name=name,
            metric_id=f"4.2.{index}",
            table_name="Roofline",
            sub_table_name=PLOT_POINTS_SUB_TABLE,
        )
        for index, name in enumerate(metric_names)
    }
    for name, values in kernels.items():
        kernel = orm.Kernel(kernel_name=name, workload=workload)
        session.add(kernel)
        session.add(
            orm.Dispatch(
                dispatch_id=1,
                gpu_id=0,
                start_timestamp=0,
                end_timestamp=1000,
                kernel=kernel,
            )
        )
        performance = values["performance"]
        stored = {
            "Performance (GFLOPs)": None if performance is None else performance * GFLOP
        }
        for level, ai_value in values["ai"].items():
            stored[f"AI {level}"] = ai_value
        for metric_name, value in stored.items():
            session.add(
                orm.KernelMetricValue(
                    metric=metric_objs[metric_name],
                    kernel=kernel,
                    value_name="Value",
                    value=value,
                )
            )


@pytest.fixture
def mi200_workload(db_session):
    """A workload row holding the MI200 ceilings and the CLI kernels in the DB."""
    workload = orm.Workload(
        name="mem_levels_HBM",
        sub_name="MI200",
        total_duration_ns=10_000_000,
    )
    db_session.add(workload)
    add_ceilings(db_session, workload)
    add_kernel_rows(db_session, workload, CLI_KERNELS)
    db_session.commit()
    return workload


def test_plot_points_skip_a_kernel_with_null_performance_and_name_it() -> None:
    plot_points, notes = plot_points_from_rows(
        [
            kernel_plot_row("kernelA", None, {"HBM": 1.0}),
            kernel_plot_row("kernelB", 500.0, {"HBM": 1.0}),
        ],
        total_duration_ns=2000,
    )

    assert plot_points["kernelNames"] == ["kernelB"]
    assert notes == [
        "Kernel kernelA is left off the plot because its performance is N/A."
    ]


def test_plot_points_null_intensity_omits_that_point_and_names_the_kernel() -> None:
    plot_points, notes = plot_points_from_rows(
        [kernel_plot_row("kernelA", 500.0, {"HBM": 1.0, "L2": None})],
        total_duration_ns=1000,
    )

    assert plot_points["kernelNames"] == ["kernelA"]
    assert plot_points["ai_l2"][0] == [0.0]
    assert plot_points["ai_hbm"][0] == [1.0]
    assert notes == [
        "Kernel kernelA has N/A arithmetic intensity at L2, so those "
        "points are left off the plot."
    ]


def test_plot_points_unmeasured_level_is_zero_without_a_note() -> None:
    """A level the workload has no metric for is not a NULL, so it adds no note."""
    row = kernel_plot_row("kernelA", 500.0, {"HBM": 1.0})

    plot_points, notes = plot_points_from_rows([row], total_duration_ns=1000)

    assert plot_points["ai_l2"] == [[0.0], [500.0]]
    assert notes == []


def test_plot_points_convert_performance_and_order_by_duration() -> None:
    plot_points, _ = plot_points_from_rows(
        [
            kernel_plot_row("short", 100.0, {"HBM": 2.0}, duration_ns=200),
            kernel_plot_row("long", 300.0, {"HBM": 4.0}, duration_ns=800),
        ],
        total_duration_ns=1000,
    )

    assert plot_points["kernelNames"] == ["long", "short"]
    assert plot_points["ai_hbm"] == [[4.0, 2.0], [300.0, 100.0]]
    assert plot_points["kernelPctRuntime"] == [80.0, 20.0]
    assert plot_points["totalAppTime"] == 1000.0
    assert plot_points["totalAppDispatches"] == 2.0


def test_db_roofline_limiter_and_percent_match_the_cli_builder_on_the_same_points(
    mi200_workload, tmp_path: Path
):
    """The DB rows, through the Plotly builder, give the limiter and percent of
    roofline that the CLI builder gives for the same kernel points on the MI200
    ceilings. The hover values are anchored to hand-worked numbers too: kernelA sits
    under the HBM roof, and kernelB under the FP64 VALU peak.
    """
    cli_roofline = Roofline(
        argparse.Namespace(),
        MI200Mspec(),
        {
            "workload_dir": str(FIXTURE_DIR),
            "device_id": 0,
            "matrix_ops_type": "MFMA",
        },
    )
    cli_path = cli_roofline.save_html_files(
        *cli_roofline.construct_plotly_figures(cli_ai_data()),
        output_path=tmp_path / "cli.html",
    )
    db_path = write_roofline_html(
        orm.Database.get_session(),
        mi200_workload.workload_id,
        MI200Mspec(),
        tmp_path / "db.html",
    )

    assert db_path == tmp_path / "db.html"
    cli_document = Path(cli_path).read_text(encoding="utf-8")
    db_document = db_path.read_text(encoding="utf-8")
    assert kernel_hovers(db_document) == kernel_hovers(cli_document)
    assert "Limited by Memory: HBM" in db_document
    assert "Limited by Compute: FP64 VALU" in db_document
    assert kernel_hovers(db_document)["kernelA"]["percent"] == "36.00"


def test_db_roofline_draws_the_same_datatypes_and_levels_as_the_cli(
    mi200_workload, tmp_path: Path
) -> None:
    """Every memory level with a ceiling and every compute datatype is drawn, as the
    CLI draws them. The browser filters them, so none is dropped here."""
    cli_roofline = Roofline(
        argparse.Namespace(),
        MI200Mspec(),
        {
            "workload_dir": str(FIXTURE_DIR),
            "device_id": 0,
            "matrix_ops_type": "MFMA",
        },
    )
    cli_path = cli_roofline.save_html_files(
        *cli_roofline.construct_plotly_figures(cli_ai_data()),
        output_path=tmp_path / "cli.html",
    )
    db_path = write_roofline_html(
        orm.Database.get_session(),
        mi200_workload.workload_id,
        MI200Mspec(),
        tmp_path / "db.html",
    )

    cli_model = embedded_model(Path(cli_path).read_text(encoding="utf-8"))
    db_model = embedded_model(db_path.read_text(encoding="utf-8"))
    assert db_model["precisions"] == cli_model["precisions"] == ["BF16", "FP32", "FP64"]
    assert db_model["computeTraces"] == cli_model["computeTraces"]
    assert db_model["rooflineTraces"] == cli_model["rooflineTraces"]
    assert len(db_model["rooflineTraces"]) == 4  # HBM, L2, L1 and LDS


def test_null_points_in_db_are_left_off_with_a_note(
    db_session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload = orm.Workload(name="w", sub_name="MI200", total_duration_ns=1000)
    db_session.add(workload)
    add_ceilings(db_session, workload)
    add_kernel_rows(
        db_session,
        workload,
        {
            "kernelA": {"ai": {"HBM": None, "L2": 0.5}, "performance": 500.0},
            "kernelB": {"ai": {"HBM": 1.0}, "performance": None},
        },
    )
    db_session.commit()
    warnings: List[str] = []
    monkeypatch.setattr(
        "roofline.roofline_db.console_warning",
        lambda *args: warnings.append(args[-1]),
    )

    output = write_roofline_html(
        db_session, workload.workload_id, MI200Mspec(), tmp_path / "roofline.html"
    )

    assert output is not None
    model = embedded_model(output.read_text(encoding="utf-8"))
    assert [kernel["name"] for kernel in model["kernels"]] == ["kernelA"]
    assert [point["peak"] for point in model["kernels"][0]["points"]] == ["L2"]
    assert warnings == [
        "Kernel kernelA has N/A arithmetic intensity at HBM, so those "
        "points are left off the plot.",
        "Kernel kernelB is left off the plot because its performance is N/A.",
    ]


def test_write_roofline_html_writes_only_the_given_output_path(
    mi200_workload, tmp_path: Path
) -> None:
    output_path = tmp_path / "run" / "roofline-gpu-0.html"
    output_path.parent.mkdir()

    written = write_roofline_html(
        orm.Database.get_session(),
        mi200_workload.workload_id,
        MI200Mspec(),
        output_path,
    )

    assert written == output_path
    assert sorted(path.name for path in tmp_path.rglob("*") if path.is_file()) == [
        "roofline-gpu-0.html"
    ]


def test_write_roofline_html_without_ceilings_writes_nothing(
    db_session, tmp_path: Path
) -> None:
    workload = orm.Workload(name="w", sub_name="no-roof", total_duration_ns=1000)
    db_session.add(workload)
    db_session.commit()
    output_path = tmp_path / "roofline-gpu-0.html"

    assert (
        write_roofline_html(db_session, workload.workload_id, MI200Mspec(), output_path)
        is None
    )
    assert not output_path.exists()
