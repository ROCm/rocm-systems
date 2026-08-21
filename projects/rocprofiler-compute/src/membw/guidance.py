# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Render guidance templates with actual metric and threshold values."""

import re
from typing import Optional

_METRIC_THRESHOLD_RE = re.compile(r"\{(threshold|metric):([^}]+)\}")


def render_guidance_blocks(
    guidance_ids: list[str],
    guidance_templates: dict[str, str],
    thresholds: dict[str, float],
    metric_values: dict[str, Optional[float]],
    max_blocks: int = 5,
) -> tuple[str, ...]:
    """Render guidance blocks for active leaf bottleneck nodes."""
    blocks: list[str] = []
    for guidance_id in guidance_ids[:max_blocks]:
        template = guidance_templates.get(guidance_id)
        if template is None:
            continue
        try:
            rendered = _render_template(template, thresholds, metric_values)
            blocks.append(rendered.rstrip())
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            blocks.append(f"[{guidance_id}] template error: {err}")

    overflow = sum(1 for gid in guidance_ids[max_blocks:] if gid in guidance_templates)
    if overflow > 0:
        blocks.append(f"...and {overflow} more (see block 30 for full detail)")

    return tuple(blocks)


def _render_template(
    template: str,
    thresholds: dict[str, float],
    metric_values: dict[str, Optional[float]],
) -> str:
    """Replace {threshold:...} and {metric:...} placeholders."""

    def _replace_match(match: re.Match) -> str:  # type: ignore[type-arg]
        kind = match.group(1)
        key = match.group(2)
        if kind == "threshold":
            value = thresholds.get(key)
            if value is None:
                return "N/A"
            return _format_threshold_value(value)
        return _format_metric_value(metric_values.get(key))

    return _METRIC_THRESHOLD_RE.sub(_replace_match, template)


def _format_threshold_value(value: float) -> str:
    """Format a threshold value, stripping trailing zeros."""
    formatted = f"{value:.1f}"
    return formatted.rstrip("0").rstrip(".")


def _format_metric_value(value: Optional[float]) -> str:
    """Format a metric value for guidance display."""
    if value is None:
        return "N/A"
    return f"{value:.1f}"
