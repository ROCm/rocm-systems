from __future__ import annotations

import json
import re
from pathlib import Path

from perfxpert import analyze as analyze_mod
from perfxpert import output_config
from perfxpert.connection import PerfxpertConnection as RocpdImportData


_EXT_BY_FORMAT = {
    "json": ".json",
    "markdown": ".md",
    "text": ".txt",
    "webview": ".html",
}


def _render_all_formats(db_path: Path, tmp_path: Path) -> dict[str, str]:
    rendered: dict[str, str] = {}
    for fmt, ext in _EXT_BY_FORMAT.items():
        out_dir = tmp_path / fmt
        out_dir.mkdir()
        cfg = output_config.output_config(output_file="report", output_path=str(out_dir))
        conn = RocpdImportData([str(db_path)])
        analyze_mod._execute_agentic(conn, config=cfg, output_format=fmt)
        rendered[fmt] = (out_dir / f"report{ext}").read_text()
    return rendered


def _extract_primary_bottleneck(fmt: str, rendered: str) -> str:
    if fmt == "json":
        payload = json.loads(rendered)
        top_level = payload.get("primary_bottleneck")
        summary_value = (payload.get("summary") or {}).get("primary_bottleneck")
        assert top_level == summary_value
        assert top_level
        return str(top_level)

    patterns = {
        "markdown": r"\*\*Primary bottleneck:\*\*\s*([^\n]+)",
        "text": r"^Primary bottleneck:\s*([^\n]+)$",
        "webview": r"Primary bottleneck:</strong>\s*([^<]+)",
    }
    match = re.search(patterns[fmt], rendered, re.IGNORECASE | re.MULTILINE)
    assert match, f"missing primary bottleneck marker in {fmt} output"
    return match.group(1).strip()


def test_db_backed_compute_reports_preserve_analysis_bottleneck(
    compute_bound_db: Path,
    tmp_path: Path,
) -> None:
    rendered = _render_all_formats(compute_bound_db, tmp_path)
    verdicts = {
        fmt: _extract_primary_bottleneck(fmt, text)
        for fmt, text in rendered.items()
    }
    assert set(verdicts.values()) == {"compute"}, verdicts


def test_trace_only_reports_preserve_data_insufficient_bottleneck(
    trace_only_elementwise_db: Path,
    tmp_path: Path,
) -> None:
    rendered = _render_all_formats(trace_only_elementwise_db, tmp_path)
    verdicts = {
        fmt: _extract_primary_bottleneck(fmt, text)
        for fmt, text in rendered.items()
    }
    assert set(verdicts.values()) == {"data_insufficient"}, verdicts
