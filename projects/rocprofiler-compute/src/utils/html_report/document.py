# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Build self-contained HTML documents from report assets and JSON models."""

import html
import json
import math
import re
from pathlib import Path
from string import Template
from typing import Dict, Final

import numpy as np

DARK_THEME_CLASS: Final[str] = "report-theme-dark"
_ASSETS_DIR = Path(__file__).parent / "assets"
_MODEL_ID_PATTERN = re.compile(r"[a-z][a-z0-9-]*\Z")


class _AssetCache:
    """Keep UTF-8 report assets loaded once per interpreter."""

    _contents: Dict[Path, str] = {}

    @classmethod
    def read(cls, path: Path) -> str:
        if path not in cls._contents:
            cls._contents[path] = path.read_text(encoding="utf-8")
        return cls._contents[path]


def read_asset(assets_dir: Path, name: str) -> str:
    """Read and cache an asset from the given page's assets directory."""
    return _AssetCache.read(assets_dir / name)


def json_safe(value: object) -> object:
    """Copy JSON containers while converting NumPy and non-finite numbers."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _embed_json(payload: object) -> str:
    """Serialize a model safely inside an HTML script element."""
    serialized = json.dumps(json_safe(payload), allow_nan=False)
    return (
        serialized
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
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
    if _MODEL_ID_PATTERN.fullmatch(model_id) is None:
        raise ValueError(f"invalid model id: {model_id!r}")

    shell = Template(read_asset(_ASSETS_DIR, "report_shell.html"))
    return shell.substitute(
        TITLE=html.escape(title),
        BASE_CSS=read_asset(_ASSETS_DIR, "report_base.css"),
        PAGE_CSS=page_css,
        DARK_THEME_CLASS=DARK_THEME_CLASS,
        BODY_HTML=body_html,
        MODEL_ID=model_id,
        MODEL_JSON=_embed_json(model),
        BASE_JS=read_asset(_ASSETS_DIR, "report_base.js"),
        PAGE_JS=page_js,
    )
