# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_tui/widgets/charts.py."""

from typing import Any

import pytest

from rocprof_compute_tui.widgets import charts

pytestmark = pytest.mark.tui


@pytest.mark.parametrize(
    ("gpu_arch", "renderer_name"),
    [
        pytest.param("gfx942", "plot_mem_chart_gfx9", id="gfx9"),
        pytest.param("gfx1151", "plot_mem_chart_gfx11", id="gfx115x"),
        pytest.param("gfx1250", "plot_mem_chart_gfx1250", id="gfx1250"),
    ],
)
def test_render_memory_chart_dispatches_by_architecture(
    monkeypatch: pytest.MonkeyPatch,
    gpu_arch: str,
    renderer_name: str,
) -> None:
    calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def renderer(metric_dict: dict[str, Any], **kwargs: Any) -> str:  # noqa: ANN401
        calls.append((metric_dict, kwargs))
        return "rendered memory chart"

    monkeypatch.setattr(charts, renderer_name, renderer)

    metric_dict = {"Metric A": 1}
    result = charts._render_memory_chart(metric_dict, gpu_arch)

    assert result == "rendered memory chart"
    assert calls[0][0] == metric_dict
    assert calls[0][1]["chart_title"] == ("3. Memory Chart (Normalization: per_kernel)")
    if gpu_arch.startswith("gfx9"):
        assert calls[0][1]["gpu_arch"] == gpu_arch
    else:
        assert "gpu_arch" not in calls[0][1]


def test_render_memory_chart_rejects_unsupported_architecture() -> None:
    with pytest.raises(
        ValueError,
        match="Memory chart not supported by this architecture",
    ):
        charts._render_memory_chart({}, "gfx1201")
