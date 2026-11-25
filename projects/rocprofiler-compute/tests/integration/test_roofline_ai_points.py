# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration test: roofline AI plot points for mat_mul_max.

Requires a GPU: profiles mat_mul_max with --roofline (live counter collection
and benchmark), then analyzes block 4 and checks that the generated roofline
HTML embeds positive AI values for every memory level the architecture's
roofline config defines.
"""

from pathlib import Path

import common
import pytest
import yaml

import config as app_config
from tests.integration.common import config, skip_unsupported_roofline_soc
from tests.integration.test_roofline_analyze import embedded_roofline_model
from utils.mi_gpu_spec import mi_gpu_specs
from utils.roofline_calc import _METRIC_TO_AI_FIELD
from utils.utils_common import canonical_config_arch

_ROOFLINE_ARCHS = mi_gpu_specs.get_roofline_archs()


def _table_402_ai_metrics(gpu_arch: str) -> list[str]:
    """Return the AI metric names table 402 defines for gpu_arch."""
    config_arch = canonical_config_arch(gpu_arch) or gpu_arch
    roofline_yaml = app_config.analysis_configs_dir / config_arch / "0400_roofline.yaml"
    doc = yaml.safe_load(roofline_yaml.read_text(encoding="utf-8"))
    for source in doc["Panel Config"]["data source"]:
        for table_config in source.values():
            if table_config.get("id") == 402:
                return [
                    name for name in table_config["metric"] if name.startswith("AI ")
                ]
    pytest.fail(f"No table 402 in {roofline_yaml}")


def _plotted_bandwidth_levels(gpu_arch: str) -> list[str]:
    """Return the bandwidth levels the roofline plots, excluding MALL."""
    return [
        level
        for level in mi_gpu_specs.get_roofline_bandwidth(gpu_arch)
        if level != "MALL"
    ]


@pytest.mark.parametrize("gpu_arch", _ROOFLINE_ARCHS)
def test_roofline_config_covers_all_memory_levels(gpu_arch):
    """Table 402 must have an AI row for every plotted memory level.

    No GPU required: reads the roofline YAML config and mi_gpu_spec.yaml only.
    """
    spec_levels = _plotted_bandwidth_levels(gpu_arch)
    assert spec_levels, f"No roofline memory levels for {gpu_arch}"

    ai_metrics = _table_402_ai_metrics(gpu_arch)

    # Each spec level (HBM, L2, L0, LDS, ...) must appear in an AI metric name.
    # The metric name may use a GL-prefixed alias (AI GL0 for L0, AI GL2 for L2).
    for level in spec_levels:
        matching = [
            m for m in ai_metrics if level in m or level.replace("L", "GL") in m
        ]
        assert matching, (
            f"Memory level '{level}' from mi_gpu_spec.yaml has no matching "
            f"AI row in table 402 for {gpu_arch}. "
            f"AI metrics found: {ai_metrics}"
        )


@pytest.mark.roofline_validation
def test_roofline_ai_points_mat_mul_max(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    gpu_arch,
    soc,
):
    """Profile mat_mul_max and check that every AI level has a positive value."""
    skip_unsupported_roofline_soc()
    if not gpu_arch:
        pytest.skip("No GPU detected")

    ai_metrics = _table_402_ai_metrics(gpu_arch)
    expected_fields = [
        _METRIC_TO_AI_FIELD[m] for m in ai_metrics if m in _METRIC_TO_AI_FIELD
    ]
    assert expected_fields, f"No AI metrics in roofline config for {gpu_arch}"

    workload_dir = common.get_output_dir()
    try:
        profile_code = binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options=["--roofline"],
            check_success=True,
            roof=True,
            app_name="app_mat_mul_max",
        )
        assert profile_code == 0

        analyze_code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "-b",
            "4",
        ])
        assert analyze_code == 0

        # Find the generated roofline HTML
        html_files = list(Path(workload_dir).rglob("empirRoof_*.html"))
        assert html_files, f"No roofline HTML generated in {workload_dir}"

        model = embedded_roofline_model(html_files[0].read_text(encoding="utf-8"))

        # The model embeds kernel data under each figure class (FLOP/OP).
        # Find kernels from either figure class.
        kernels = []
        for fig_class in ("flops", "ops"):
            fig = model.get(fig_class)
            if fig and "kernels" in fig:
                kernels.extend(fig["kernels"])
        assert kernels, "No kernels in the roofline HTML model"

        # Each kernel has 'points', each point has an 'ai' (x-axis) and
        # 'perf' (y-axis) value, plus a 'peak' label naming the memory level.
        kernel = kernels[0]
        points = kernel.get("points", [])
        assert points, f"Kernel '{kernel.get('name', '?')}' has no plot points"

        # Check performance (y-axis) is positive
        perf = points[0].get("perf", 0)
        assert perf > 0, (
            f"Kernel '{kernel.get('name', '?')}' has performance {perf}, "
            "expected > 0 for mat_mul_max"
        )

        # Check that each point has a positive AI value
        for point in points:
            ai = point.get("ai", 0)
            peak = point.get("peak", "?")
            assert ai > 0, (
                f"AI for level '{peak}' is {ai} on kernel "
                f"'{kernel.get('name', '?')}', expected > 0"
            )

        # Check we have the expected number of memory level points
        point_peaks = {point.get("peak") for point in points}
        assert len(points) >= len(expected_fields), (
            f"Expected at least {len(expected_fields)} AI points "
            f"(one per memory level), got {len(points)}: {point_peaks}"
        )

    finally:
        common.clean_output_dir(config["cleanup"], workload_dir)
