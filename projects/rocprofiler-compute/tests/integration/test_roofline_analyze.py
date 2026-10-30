# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Roofline coverage for the ``analyze`` roofline path.

MI200 (gfx90a) tests run through the ``analyze`` CLI and assert that roofline
HTML is generated, including the per-datatype VALU/MFMA legend.
"""

import json
import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import common
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from roofline.roofline_db import write_roofline_html
from tests.integration import common as integration_common
from utils import analysis_orm as orm

config = {}
config["cleanup"] = True

roofline_dir = "tests/workloads/mem_levels_HBM/MI200"
# The MI210 (gfx90a) the MI200 mem_levels fixtures were profiled on.
FIXTURE_MSPEC = SimpleNamespace(
    gpu_model="MI210", gpu_series="mi200", gpu_arch="gfx90a"
)


def embedded_roofline_model(document: str) -> dict:
    """Parse the JSON model from a standalone roofline document."""
    match = re.search(
        r'<script id="roofline-model" type="application/json">(.*?)</script>',
        document,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def analyze_into_run_folder(
    analyze: Callable[[list[str]], int], workload_dir: str, run_folder: Path
) -> int:
    """Run analyze on workload_dir with run_folder as its run folder."""
    return analyze([
        "analyze",
        "--path",
        workload_dir,
        "--output-directory",
        str(run_folder.parent),
        "--output-name",
        run_folder.name,
    ])


def roofline_html_path(run_folder: Path, workload_dir: str) -> Path:
    """Return where analyze writes the roofline HTML of the single -p workload."""
    return run_folder / f"1-{Path(workload_dir).name}" / "roofline-gpu-0.html"


# Roofline HTML generation


def test_analyze_generates_roofline_html(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Analyze generates roofline HTML from existing workload data.
    Uses MI200 workload with roofline.csv.
    """
    workload_dir = integration_common.setup_workload_dir(roofline_dir)
    try:
        assert (Path(workload_dir) / "roofline.csv").exists()

        run_folder = tmp_path / "run"
        code = analyze_into_run_folder(
            binary_handler_analyze_rocprof_compute, workload_dir, run_folder
        )
        assert code == 0
        # The terminal lists the roofline HTML among the artifacts written.
        terminal = capsys.readouterr().out
        assert "roofline-gpu-0.html" in terminal.split("Artifacts written", 1)[1]

        html_path = roofline_html_path(run_folder, workload_dir)
        assert html_path.is_file()
        assert list(Path(workload_dir).glob("*.html")) == []

        html_text = html_path.read_text(encoding="utf-8")
        assert 'id="roofline-precision-btn"' in html_text
        model = embedded_roofline_model(html_text)
        assert "FP32" in model["precisions"]
        assert model["frame"] == {
            "x": [0.01, 1000.0],
            "y": [10.0, 1000000.0],
        }
    finally:
        common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_missing_roofline_csv_graceful(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
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


def test_analyze_roofline_rerun_writes_a_new_run_folder(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
    tmp_path: Path,
) -> None:
    """
    Running analyze twice on the same profiling output gives two run folders
    with the same roofline HTML.
    """
    workload_dir = integration_common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    html_documents = []
    for run_name in ("first", "second"):
        run_folder = tmp_path / run_name
        code = analyze_into_run_folder(
            binary_handler_analyze_rocprof_compute, workload_dir, run_folder
        )
        assert code == 0
        html_path = roofline_html_path(run_folder, workload_dir)
        html_documents.append(embedded_roofline_model(html_path.read_text("utf-8")))

    assert html_documents[0] == html_documents[1]

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


# Per-datatype HTML output validation

# Each datatype must take the correct roofline branch:
DATATYPE_LEGEND_CASES = {
    "FP64": {"present": ["Peak VALU-FP64", "Peak MFMA-FP64"], "absent": []},
    "BF16": {"present": ["Peak MFMA-BF16"], "absent": ["Peak VALU-BF16"]},
}


@pytest.mark.parametrize("dtype", list(DATATYPE_LEGEND_CASES))
def test_analyze_roofline_datatype_html_legend(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
    dtype: str,
    tmp_path: Path,
) -> None:
    """The roofline HTML embeds the expected VALU/MFMA legend for each datatype.

    FP8/FP4/FP6 are intentionally excluded: they are unsupported on the
    available gfx90a (MI200) test data and are covered by the unit tests.
    """
    workload_dir = integration_common.setup_workload_dir(roofline_dir, param_id=dtype)

    assert (Path(workload_dir) / "roofline.csv").exists()

    run_folder = tmp_path / "run"
    code = analyze_into_run_folder(
        binary_handler_analyze_rocprof_compute, workload_dir, run_folder
    )
    assert code == 0

    html_path = roofline_html_path(run_folder, workload_dir)
    assert html_path.is_file(), "Analyze should generate the roofline HTML"

    html_text = html_path.read_text(encoding="utf-8")
    for legend in DATATYPE_LEGEND_CASES[dtype]["present"]:
        assert legend in html_text, f"{dtype} HTML should contain '{legend}'"
    for legend in DATATYPE_LEGEND_CASES[dtype]["absent"]:
        assert legend not in html_text, f"{dtype} HTML should not contain '{legend}'"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_roofline_html_matches_the_roofline_from_the_run_database(
    binary_handler_analyze_rocprof_compute: Callable[[list[str]], int],
    tmp_path: Path,
) -> None:
    """The roofline HTML analyze writes equals the one rendered from the run's DB.

    The MI200 fixture plots no kernels: its only kernel has zero performance. So this
    compares the ceilings, frame, precisions and roofs from real data, and the
    limiter and percent of roofline are checked by the unit tests.
    """
    workload_dir = integration_common.setup_workload_dir(roofline_dir)
    try:
        run_folder = tmp_path / "run"
        code = analyze_into_run_folder(
            binary_handler_analyze_rocprof_compute, workload_dir, run_folder
        )
        assert code == 0
        analyze_model = embedded_roofline_model(
            roofline_html_path(run_folder, workload_dir).read_text(encoding="utf-8")
        )

        engine = create_engine(f"sqlite:///{run_folder / 'analysis.db'}")
        session = sessionmaker(bind=engine)()
        try:
            workload = session.query(orm.Workload).one()
            db_path = write_roofline_html(
                session,
                workload.workload_id,
                FIXTURE_MSPEC,
                tmp_path / "roofline-db.html",
            )
        finally:
            session.close()
            engine.dispose()

        assert db_path is not None
        db_model = embedded_roofline_model(db_path.read_text(encoding="utf-8"))
        assert analyze_model["kernels"] == []
        assert db_model == analyze_model
    finally:
        common.clean_output_dir(config["cleanup"], workload_dir)
