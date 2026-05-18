# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Helpers for selective-region config file tests (flat text, XML, preset JSON)."""

from __future__ import annotations

import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from rocprofsys.config import RocprofsysConfig

_SELECTIVE_REGION_SETTINGS: dict[str, str] = {
    "ROCPROFSYS_TRACE": "true",
    "ROCPROFSYS_PROFILE": "true",
    "ROCPROFSYS_USE_SAMPLING": "false",
    "ROCPROFSYS_USE_PROCESS_SAMPLING": "false",
    "ROCPROFSYS_ROCM_DOMAINS": (
        "hip_runtime_api,marker_api,kernel_dispatch,marker_core_range_api"
    ),
    "ROCPROFSYS_SELECTED_REGIONS": "Region1",
}


def bundled_preset_json_path() -> Path:
    """Bundled preset JSON with tracing.region."""
    return Path(__file__).resolve().parents[1] / "config" / "rocprof-sys-selected-region1.cfg"


def find_preset_json_path(rocprof_config: RocprofsysConfig) -> Path | None:
    """Resolve preset JSON from install tree or pytest bundle."""
    examples = rocprof_config.rocprofsys_examples_dir
    candidates = (
        examples / "roctx" / "config" / "rocprof-sys-selected-region1.cfg",
        examples / "config" / "rocprof-sys-selected-region1.cfg",
        bundled_preset_json_path(),
        Path(__file__).resolve().parents[3]
        / "examples"
        / "roctx"
        / "config"
        / "rocprof-sys-selected-region1.cfg",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def flat_config_content(
    *,
    selected_regions: str = "Region1",
    extra: dict[str, str] | None = None,
) -> str:
    """Plain-text rocprof-sys.cfg body with selective-region settings."""
    settings = {**_SELECTIVE_REGION_SETTINGS, "ROCPROFSYS_SELECTED_REGIONS": selected_regions}
    if extra:
        settings.update(extra)
    lines = ["# auto-generated for selective-region pytest", ""]
    for key, value in settings.items():
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def write_flat_config(path: Path, **kwargs) -> Path:
    path.write_text(flat_config_content(**kwargs))
    return path


def _set_xml_value(settings: ET.Element, tag: str, value: str) -> None:
    element = settings.find(tag)
    if element is None:
        raise KeyError(f"Missing <{tag}> in XML settings")
    value_node = element.find("value")
    if value_node is None:
        raise KeyError(f"Missing <value> under <{tag}>")
    value_node.text = value


def patch_selective_region_xml(path: Path, *, selected_regions: str = "Region1") -> Path:
    """Apply selective-region overrides to an rocprof-sys-avail XML file."""
    tree = ET.parse(path)
    settings = tree.getroot().find(".//settings")
    if settings is None:
        raise ValueError(f"No <settings> block in {path}")

    overrides = {**_SELECTIVE_REGION_SETTINGS, "ROCPROFSYS_SELECTED_REGIONS": selected_regions}
    for tag, value in overrides.items():
        _set_xml_value(settings, tag, value)

    if hasattr(ET, "indent"):
        ET.indent(tree, space="\t")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def generate_selective_region_xml(
    rocprof_config: RocprofsysConfig,
    output_path: Path,
    *,
    selected_regions: str = "Region1",
) -> Path:
    """Generate XML via rocprof-sys-avail and patch selective-region settings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(rocprof_config.rocprofsys_avail),
        "-G",
        str(output_path),
        "-F",
        "xml",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"rocprof-sys-avail failed ({result.returncode}): "
            f"{result.stdout}\n{result.stderr}"
        )
    if not output_path.exists():
        raise FileNotFoundError(f"Expected XML config at {output_path}")
    return patch_selective_region_xml(output_path, selected_regions=selected_regions)


def combined_output_text(result) -> str:
    """Stdout/stderr plus optional extra output from TestResult."""
    parts = [result.test_output or ""]
    if result.extra_output:
        parts.append(result.extra_output)
    return "\n".join(parts)


def output_indicates_successful_trace(combined: str) -> bool:
    """True when run log shows finalize + Perfetto generation."""
    return bool(
        re.search(r"Finalizing rocprof-sys", combined)
        and re.search(r"perfetto.*Done|Output Summary", combined, re.IGNORECASE)
    )


def output_has_region_filter_log(combined: str, region: str = "Region1") -> bool:
    """True when trace_control logged an active region filter."""
    return bool(
        re.search(
            rf"region filter active for regions:.*{re.escape(region)}",
            combined,
        )
    )
