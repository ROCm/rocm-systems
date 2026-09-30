# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Roofline coverage for the ``analyze`` roofline path.

MI200 (gfx90a) tests run through the ``analyze`` CLI and assert that roofline
HTML is generated, including the per-datatype VALU/MFMA legend.
"""

import json
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Callable
from pathlib import Path

import common
import pandas as pd
import pytest

from roofline.roofline_frame import canonical_frame
from tests.integration import common as integration_common

config = {}
config["cleanup"] = True

roofline_dir = "tests/workloads/mem_levels_HBM/MI200"


def embedded_roofline_model(document: str) -> dict:
    """Parse the JSON model from a standalone roofline document."""
    match = re.search(
        r'<script id="roofline-model" type="application/json">(.*?)</script>',
        document,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def setup_db_roofline_workload(tmp_path: Path, positive: bool) -> Path:
    """Create two coherent kernels, optionally making one execute FP32 FMAs."""
    workload_dir = tmp_path / "profiles" / "run"
    shutil.copytree(roofline_dir, workload_dir)
    for csv_path in workload_dir.glob("results_*.csv.gz"):
        frame = pd.read_csv(csv_path)
        duplicate = frame.copy()
        duplicate["Kernel_Name"] = "zeroKernel(double*, double*, double*, int, int)"
        duplicate["Kernel_ID"] = 1
        for column in ["Dispatch_ID", "Correlation_Id"]:
            duplicate[column] += 3
        if positive:
            frame.loc[
                frame["Counter_Name"] == "SQ_INSTS_VALU_FMA_F32", "Counter_Value"
            ] = 10000
        pd.concat([frame, duplicate], ignore_index=True).to_csv(csv_path, index=False)
    return workload_dir


def sqlite_rows(connection: sqlite3.Connection, query: str) -> list[dict]:
    """Read named exported database columns for comparisons with HTML."""
    return [dict(row) for row in connection.execute(query)]


# Roofline HTML generation


def test_analyze_generates_roofline_html(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
    tmp_path: Path,
) -> None:
    """
    Analyze generates roofline HTML from existing workload data.
    Uses MI200 workload with roofline.csv.
    """
    workload_dir = integration_common.setup_workload_dir(roofline_dir)
    try:
        assert (Path(workload_dir) / "roofline.csv").exists()

        before = common.read_binary_file_tree(Path(workload_dir))
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--roofline-data-type",
            "FP32",
        ])
        assert code == 0
        assert common.read_binary_file_tree(Path(workload_dir)) == before

        html_files = list((tmp_path / "analysis").glob("empirRoof_*.html"))
        assert [html_file.name for html_file in html_files] == ["empirRoof_gpu-0.html"]

        html_text = html_files[0].read_text(encoding="utf-8")
        assert 'id="roofline-precision-btn"' in html_text
        model = embedded_roofline_model(html_text)
        assert "FP32" in model["precisions"]
        assert model["frame"] == {
            "x": [0.01, 1000.0],
            "y": [10.0, 1000000.0],
        }
    finally:
        common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_roofline_datatype_independently(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
    tmp_path: Path,
) -> None:
    """
    Analyze with multiple data types.
    Verifies each datatype can be requested independently.
    """
    workload_dir = integration_common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    for dtype in ["FP32", "FP64", "BF16"]:
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--roofline-data-type",
            dtype,
            "--overwrite",
        ])
        assert code == 0

    html_files = list((tmp_path / "analysis").glob("empirRoof_*.html"))
    assert [html_file.name for html_file in html_files] == ["empirRoof_gpu-0.html"]

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_roofline_multiple_datatypes_single_invocation(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
    tmp_path: Path,
) -> None:
    """
    Analyze with multiple data types in a single invocation.
    Verifies the multi-datatype request path works end to end.
    """
    workload_dir = integration_common.setup_workload_dir(roofline_dir)

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

    html_files = list((tmp_path / "analysis").glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_missing_roofline_csv_graceful(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
    tmp_path: Path,
) -> None:
    """
    Analyze without roofline.csv should not crash.
    Uses a workload directory that has sysinfo.csv but no roofline.csv.
    """
    workload_dir = integration_common.setup_workload_dir(roofline_dir)
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


def test_analyze_roofline_rerun_requires_overwrite(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    An existing analysis directory requires explicit overwrite on reruns.
    """
    workload_dir = integration_common.setup_workload_dir(roofline_dir)

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

    html_path = tmp_path / "analysis" / "empirRoof_gpu-0.html"
    first_html = html_path.read_bytes()
    sentinel = tmp_path / "analysis" / "stale.txt"
    sentinel.write_text("old result", encoding="utf-8")
    capsys.readouterr()
    code2 = binary_handler_analyze_rocprof_compute(analyze_args)
    assert code2 != 0
    captured = capsys.readouterr()
    assert "--overwrite" in captured.out + captured.err
    assert html_path.read_bytes() == first_html
    assert sentinel.is_file()

    code3 = binary_handler_analyze_rocprof_compute([*analyze_args, "--overwrite"])
    assert code3 == 0
    assert not sentinel.exists()
    assert html_path.read_bytes() == first_html

    html_files = list((tmp_path / "analysis").glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_corrupted_roofline_csv_graceful(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
    tmp_path: Path,
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
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid --roofline-data-type should be rejected by the analyze argparser."""
    workload_dir = integration_common.setup_workload_dir(roofline_dir)

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
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid --mem-level should be rejected by the analyze argparser."""
    workload_dir = integration_common.setup_workload_dir(roofline_dir)

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
    tmp_path: Path,
    mem_level: str,
) -> None:
    """Analyze with --mem-level generates roofline HTML output."""
    workload_src = roofline_mem_level_dirs[mem_level]
    if not Path(workload_src).exists():
        pytest.skip(f"Workload directory {workload_src} not found")

    workload_dir = integration_common.setup_workload_dir(
        workload_src, param_id=mem_level
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--mem-level",
        mem_level,
    ])
    assert code == 0

    html_files = list((tmp_path / "analysis").glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


# Per-datatype HTML output validation

# Each datatype must take the correct roofline branch:
DATATYPE_LEGEND_CASES = {
    "FP64": {"present": ["Peak VALU-FP64", "Peak MFMA-FP64"], "absent": []},
    "BF16": {"present": ["Peak MFMA-BF16"], "absent": ["Peak VALU-BF16"]},
}


@pytest.mark.parametrize("dtype", list(DATATYPE_LEGEND_CASES))
def test_analyze_roofline_datatype_html_legend(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
    tmp_path: Path,
    dtype: str,
) -> None:
    """Per-datatype roofline HTML embeds the expected VALU/MFMA legend.

    FP8/FP4/FP6 are intentionally excluded: they are unsupported on the
    available gfx90a (MI200) test data and are covered by the unit tests.
    """
    workload_dir = integration_common.setup_workload_dir(roofline_dir, param_id=dtype)

    assert (Path(workload_dir) / "roofline.csv").exists()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        dtype,
    ])
    assert code == 0

    html_path = tmp_path / "analysis" / "empirRoof_gpu-0.html"
    assert html_path.is_file(), f"Analyze should generate a {dtype} roofline HTML"

    html_text = html_path.read_text(encoding="utf-8")
    for legend in DATATYPE_LEGEND_CASES[dtype]["present"]:
        assert legend in html_text, f"{dtype} HTML should contain '{legend}'"
    for legend in DATATYPE_LEGEND_CASES[dtype]["absent"]:
        assert legend not in html_text, f"{dtype} HTML should not contain '{legend}'"

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.parametrize(
    "output_kind", ["workload", "ancestor", "descendant", "symlink"]
)
def test_analyze_output_directory_protects_workload(
    binary_handler_analyze_rocprof_compute, tmp_path, output_kind
):
    """Overwrite cannot remove a workload through containment or a symlink."""
    workload_dir = tmp_path / "profiles" / "run"
    shutil.copytree(roofline_dir, workload_dir)
    before = common.read_binary_file_tree(workload_dir)
    output_dir = {
        "workload": workload_dir,
        "ancestor": workload_dir.parent,
        "descendant": workload_dir / "analysis",
        "symlink": tmp_path / "workload_link",
    }[output_kind]
    if output_kind == "symlink":
        output_dir.symlink_to(workload_dir, target_is_directory=True)

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        str(workload_dir),
        "--output-directory",
        str(output_dir),
        "--overwrite",
    ])

    assert code != 0
    assert common.read_binary_file_tree(workload_dir) == before


def test_analyze_overwrite_unlinks_children_without_following_symlinks(
    binary_handler_analyze_rocprof_compute, tmp_path
):
    """Clearing an output directory leaves symlink targets intact."""
    workload_dir = tmp_path / "profiles" / "run"
    shutil.copytree(roofline_dir, workload_dir)
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    sentinel = external_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    (output_dir / "linked_dir").symlink_to(external_dir, target_is_directory=True)
    (output_dir / "linked_file").symlink_to(sentinel)
    (output_dir / "dangling").symlink_to(tmp_path / "missing")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        str(workload_dir),
        "--output-directory",
        str(output_dir),
        "--overwrite",
    ])

    assert code == 0
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert {path.name for path in output_dir.iterdir()} == {"empirRoof_gpu-0.html"}


@pytest.mark.parametrize("output_format", ["txt", "db", "csv"])
def test_analyze_artifacts_use_explicit_output_directory(
    binary_handler_analyze_rocprof_compute, tmp_path, output_format
):
    """Each output format writes only into the requested analysis directory."""
    workload_dir = tmp_path / "profiles" / "run"
    shutil.copytree(roofline_dir, workload_dir)
    before = common.read_binary_file_tree(workload_dir)
    output_dir = tmp_path / "reports"

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        str(workload_dir),
        "--output-directory",
        str(output_dir),
        "--output-format",
        output_format,
        "--output-name",
        "result",
    ])

    assert code == 0
    assert common.read_binary_file_tree(workload_dir) == before
    if output_format == "csv":
        assert (output_dir / "result" / "roofline_ceiling.csv").is_file()
        assert not (output_dir / "result.db").exists()
    else:
        assert (output_dir / f"result.{output_format}").is_file()


@pytest.mark.parametrize(
    "positive", [False, True], ids=["zero-performance", "positive-performance"]
)
def test_analyze_db_roofline_retains_kernels_and_matches_html(
    binary_handler_analyze_rocprof_compute, tmp_path, positive
):
    """DB output retains zero kernels while HTML renders positive points only."""
    workload_dir = setup_db_roofline_workload(tmp_path, positive)
    before = common.read_binary_file_tree(workload_dir)
    output_dir = tmp_path / "reports"
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        str(workload_dir),
        "--block",
        "4",
        "--output-directory",
        str(output_dir),
        "--output-format",
        "db",
        "--output-name",
        "result",
        "--roofline-data-type",
        "FP32",
    ])
    assert code == 0
    html_path = output_dir / "empirRoof_gpu-0.html"
    assert html_path.is_file()
    model = embedded_roofline_model(html_path.read_text(encoding="utf-8"))
    assert common.read_binary_file_tree(workload_dir) == before
    connection = sqlite3.connect(output_dir / "result.db")
    connection.row_factory = sqlite3.Row
    try:
        stats = sqlite_rows(
            connection,
            "SELECT * FROM compute_kernel_roofline_data ORDER BY kernel_rank",
        )
        assert len(stats) == 2
        assert {row["kernel_rank"] for row in stats} == {0, 1}
        assert {row["dispatch_count"] for row in stats} == {3}
        assert all(row["total_duration_ns"] > 0 for row in stats)
        assert sum(row["percent_runtime"] for row in stats) == pytest.approx(100)
        assert all(row["l0_cache_data"] is None for row in stats)
        assert sum(row["total_flops"] > 0 for row in stats) == int(positive)
        metrics = sqlite_rows(
            connection, "SELECT * FROM compute_kernel_roofline_metric_view"
        )
        assert {row["table_id"] for row in metrics} == {401, 402}
        assert len(metrics) == 34
        assert any(row["value"] is None for row in metrics)
        # The ordinary metric export retains the same evaluated roofline values.
        generic_metrics = sqlite_rows(
            connection,
            "SELECT kernel_uuid, metric_id, value FROM compute_kernel_metric_view "
            "WHERE value_name = 'Value' AND "
            "(metric_id LIKE '4.1.%' OR metric_id LIKE '4.2.%')",
        )
        generic_values = {
            (row["kernel_uuid"], row["metric_id"]): row["value"]
            for row in generic_metrics
        }
        assert generic_values
        for row in metrics:
            key = (row["kernel_uuid"], row["metric_id"])
            if key in generic_values:
                assert row["value"] == pytest.approx(generic_values[key])
        ceiling_rows = sqlite_rows(
            connection, "SELECT * FROM compute_roofline_ceiling_view"
        )
        assert ceiling_rows
        bandwidths = {
            row["mem_level"]: row["value"]
            for row in ceiling_rows
            if row["ceiling_kind"] == "bandwidth"
        }
        compute_peaks = [
            row["value"] for row in ceiling_rows if row["ceiling_kind"] == "compute"
        ]
        frame = canonical_frame(list(bandwidths.values()), compute_peaks)
        assert model["frame"] == {"x": list(frame[:2]), "y": list(frame[2:])}
        roof_rows = sqlite_rows(
            connection,
            "SELECT * FROM compute_roofline_roof_view WHERE datatype = 'FP32'",
        )
        for trace in model["rooflineTraces"]:
            roof = next(row for row in roof_rows if row["mem_level"] == trace["level"])
            assert trace["bandwidth"] == pytest.approx(roof["bandwidth"])
            assert trace["kneeAi"] == pytest.approx(roof["knee_ai"])
            assert trace["kneePerf"] == pytest.approx(roof["roof_peak"])
        point_rows = sqlite_rows(
            connection,
            "SELECT * FROM compute_kernel_roofline_view WHERE envelope = 'FP32'",
        )
        assert len(point_rows) == 8
        assert len(model["kernels"]) == int(positive)
        if positive:
            rendered_kernel = model["kernels"][0]
            rows = [
                row
                for row in point_rows
                if row["kernel_name"] == rendered_kernel["name"]
            ]
            assert rows
            assert rendered_kernel["pctRuntime"] == pytest.approx(
                rows[0]["percent_runtime"]
            )
            assert rendered_kernel["points"]
            candidates = [
                (
                    row["arithmetic_intensity"] * bandwidths[row["mem_level"]],
                    row["mem_level"],
                )
                for row in rows
                if row["arithmetic_intensity"] > 0
            ]
            candidates.append((
                rows[0]["compute_ceiling"],
                rows[0]["compute_ceiling_label"],
            ))
            expected_limiter = min(candidates, key=lambda candidate: candidate[0])[1]
            assert rows[0]["limiter"] == expected_limiter
            for point in rendered_kernel["points"]:
                row = next(row for row in rows if row["mem_level"] == point["peak"])
                assert point["ai"] == pytest.approx(row["arithmetic_intensity"])
                assert point["perf"] == pytest.approx(row["performance"])
                expected_roof = min(
                    row["arithmetic_intensity"] * bandwidths[row["mem_level"]],
                    row["compute_ceiling"],
                )
                assert row["roof_performance"] == pytest.approx(expected_roof)
                assert row["percent_of_roof"] == pytest.approx(
                    row["performance"] / expected_roof * 100
                )
                assert point["hoverCells"] == [
                    f"{row['roof_performance']:,.3f}",
                    f"{row['percent_of_roof']:.4f}",
                ]
            # Plotly stores the kernel limiter in its hover template.
            document = html_path.read_text(encoding="utf-8")
            assert f"Performance limiter: {expected_limiter}" in document
            assert "Total dispatches: 3" in document
            assert (
                f"Aggregate time in kernel: {rows[0]['total_duration_ns']:,.2f} ns"
                in document
            )
    finally:
        connection.close()


def test_analyze_roofline_csv_is_populated_before_session_close(
    binary_handler_analyze_rocprof_compute, tmp_path
):
    """CSV export contains every persisted roofline row and the DB-mode HTML."""
    workload_dir = setup_db_roofline_workload(tmp_path, positive=True)
    output_dir = tmp_path / "reports"
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        str(workload_dir),
        "--block",
        "4",
        "--output-directory",
        str(output_dir),
        "--output-format",
        "csv",
        "--output-name",
        "result",
    ])
    assert code == 0
    assert (output_dir / "empirRoof_gpu-0.html").is_file()
    csv_dir = output_dir / "result"
    points = pd.read_csv(csv_dir / "kernel_roofline.csv")
    metrics = pd.read_csv(csv_dir / "kernel_roofline_metric.csv")
    ceilings = pd.read_csv(csv_dir / "roofline_ceiling.csv")
    roofs = pd.read_csv(csv_dir / "roofline_roof.csv")
    assert points["kernel_uuid"].nunique() == 2
    assert points["envelope"].nunique() > 1
    assert points["performance"].gt(0).any()
    assert points["performance"].eq(0).any()
    assert len(metrics) == 34
    assert set(ceilings["ceiling_kind"]) == {"bandwidth", "compute"}
    assert roofs["knee_ai"].gt(0).all()
