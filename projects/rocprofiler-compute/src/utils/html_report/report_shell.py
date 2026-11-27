# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Build self-contained HTML documents from report assets and JSON models."""

import html
import json
import math
from pathlib import Path
from string import Template
from typing import Any, Dict, Final, List, Optional

DARK_THEME_CLASS: Final[str] = "report-theme-dark"
KERNEL_NAME_FONT_FAMILY: Final[str] = (
    "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
)
_ASSETS_DIR = Path(__file__).parent / "assets"


def read_asset(assets_dir: Path, name: str) -> str:
    """Read and cache an asset from the given page's assets directory."""
    return _AssetCache.read(assets_dir / name)


def truncate_kernel_name(name: str) -> str:
    """Cap a kernel name at 200 characters so a pathological name can't blow
    up a kernel list or tooltip."""
    max_length = 200
    suffix = "..."
    if len(name) <= max_length:
        return name
    return name[: max_length - len(suffix)] + suffix


def report_file_name(
    stem: str,
    device_id: int,
    gpu_kernel: Optional[List[Any]],
) -> str:
    """File name of an analyze HTML report, such as empirRoof_gpu-0_1_3.html.

    The -k kernel ids, sorted as text, end the name, so each kernel filter writes
    its own report.
    """
    kernels = [
        str(kernel)
        for group in gpu_kernel or []
        for kernel in (group if isinstance(group, list) else [group])
    ]
    return f"{stem}_gpu-{device_id}{''.join('_' + k for k in sorted(kernels))}.html"


def write_report(
    workload_dir: str,
    stem: str,
    device_id: int,
    gpu_kernel: Optional[List[Any]],
    document: str,
) -> str:
    """Write a report next to the workload, named by report_file_name, and return
    its path."""
    path = f"{workload_dir}/{report_file_name(stem, device_id, gpu_kernel)}"
    Path(path).write_text(document, encoding="utf-8")
    return path


def build_page(
    assets_dir: Path,
    stem: str,
    *,
    title: str,
    model_id: str,
    model: object,
    body_html: Optional[str] = None,
) -> str:
    """The document of a page whose assets are <stem>.html, .css and .js.

    A page that fills in its body template passes the filled body as body_html.
    """
    return build_document(
        title=title,
        body_html=read_asset(assets_dir, f"{stem}.html")
        if body_html is None
        else body_html,
        page_css=read_asset(assets_dir, f"{stem}.css"),
        page_js=read_asset(assets_dir, f"{stem}.js"),
        model_id=model_id,
        model=model,
    )


def build_document(
    *,
    title: str,
    body_html: str,
    page_css: str,
    page_js: str,
    model_id: str,
    model: object,
) -> str:
    """Insert a page's trusted fragments into the shared HTML shell."""
    shell = Template(read_asset(_ASSETS_DIR, "report_shell.html"))
    return shell.substitute(
        TITLE=html.escape(title),
        KERNEL_FONT=KERNEL_NAME_FONT_FAMILY,
        BASE_CSS=read_asset(_ASSETS_DIR, "report_base.css"),
        PAGE_CSS=page_css,
        DARK_THEME_CLASS=DARK_THEME_CLASS,
        BODY_HTML=body_html,
        MODEL_ID=model_id,
        MODEL_JSON=_embed_json(model),
        BASE_JS=read_asset(_ASSETS_DIR, "report_base.js"),
        PAGE_JS=page_js,
    )


def _json_safe(value: object) -> object:
    """Copy JSON containers while converting non-finite Python floats."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _embed_json(payload: object) -> str:
    """Serialize a model safely inside an HTML script element."""
    return json.dumps(_json_safe(payload), allow_nan=False).replace("</", "<\\/")


class _AssetCache:
    """Keep UTF-8 report assets loaded once per interpreter."""

    _contents: Dict[Path, str] = {}

    @classmethod
    def read(cls, path: Path) -> str:
        if path not in cls._contents:
            cls._contents[path] = path.read_text(encoding="utf-8")
        return cls._contents[path]
