"""Tests for perfxpert.formatters._roofline_svg (Phase 10 Live Roofline)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from perfxpert.formatters._roofline_svg import (
    render_roofline_svg,
    sanitize_rec_anchor,
)


def _fake_payload(num_kernels: int = 3) -> dict:
    return {
        "schema_version": "0.3.x",
        "arch": "gfx942",
        "arch_peaks": {
            "fp32": 163.4e12,
            "fp16": 1307e12,
            "bf16": 1307e12,
            "fp64": 81.7e12,
            "fp8":  2614e12,
            "int8": 2614e12,
        },
        "hbm_bandwidth_bytes_per_s": 5.3e12,
        "dtype": "bf16",
        "dtype_confidence": "from_kernel_name",
        "kernels": [
            {
                "name": f"gemm_bf16_kernel_{i}",
                "ai": 10.0 * (i + 1),
                "achieved_flops_per_s": 1e12 * (i + 1),
                "flops": 1e10,
                "bytes": 1e9,
                "duration_ns": 1_000_000,
                "duration_pct": 100.0 / num_kernels,
                "fp_type": "bf16",
                "bottleneck_class": "compute" if i == 0 else "memory",
                "confidence": "high",
            }
            for i in range(num_kernels)
        ],
        "ridge_point": {"ai": 30.8, "flops_per_s": 1.634e14},
    }


def test_render_roofline_svg_produces_single_section() -> None:
    rf = _fake_payload(num_kernels=3)
    html = render_roofline_svg(rf)

    assert html.count('<section class="scard">') == 1
    assert html.count("</section>") == 1
    assert "<h2>Live Roofline</h2>" in html


def test_render_roofline_svg_has_log_log_axes() -> None:
    html = render_roofline_svg(_fake_payload(num_kernels=1))
    for label in ("0.01", "0.1", "1", "10", "100", "1000"):
        assert f">{label}</text>" in html, f"missing X label {label}"
    assert "10.0 GF/s" in html
    assert "100.0 GF/s" in html
    assert "1.0 TF/s" in html
    assert "10.0 TF/s" in html
    assert "100.0 TF/s" in html


def test_render_roofline_svg_dots_count_equals_top_k() -> None:
    rf = _fake_payload(num_kernels=5)
    html = render_roofline_svg(rf)
    m = re.search(r'<g id="rf-dots">(.*?)</g>', html, re.DOTALL)
    assert m is not None
    assert m.group(1).count("<circle") == 5


def test_render_roofline_svg_ridge_annotation_present() -> None:
    rf = _fake_payload(num_kernels=1)
    html = render_roofline_svg(rf)

    assert "gfx942" in html
    assert "ridge @" in html
    assert "FLOPs/B" in html
    assert "HBM" in html
    assert "TB/s" in html


def test_render_roofline_svg_onclick_uses_rec_anchor() -> None:
    rf = _fake_payload(num_kernels=1)
    html = render_roofline_svg(rf)

    assert "data-k=" in html
    assert "document.getElementById('rec-'+this.dataset.k)" in html
    assert 'data-k="gemm_bf16_kernel_0"' in html


def test_render_roofline_svg_empty_payload_is_empty_string() -> None:
    assert render_roofline_svg(None) == ""
    assert render_roofline_svg({}) == ""
    assert render_roofline_svg({"kernels": []}) == ""


def test_sanitize_rec_anchor_strips_args_and_templates() -> None:
    assert sanitize_rec_anchor(
        "heavy_valu_kernel(float const*, float*, int, int)"
    ) == "heavy_valu_kernel"
    assert sanitize_rec_anchor("ns::foo<int>::bar(float)") == "bar"
    assert sanitize_rec_anchor("gemm_bf16_kernel") == "gemm_bf16_kernel"


_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "compute_bound.db"
)


def test_webview_has_single_doctype_after_roofline_section(tmp_path, monkeypatch) -> None:
    """After the roofline scard is spliced into ``webview.html``, the
    rendered document still has exactly one ``<!DOCTYPE html>`` (no
    accidental nested scaffolding) and exactly one ``<h2>Live Roofline</h2>``.
    """
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")

    from perfxpert import analyze as analyze_mod
    from perfxpert import output_config
    from perfxpert.connection import PerfxpertConnection

    out_dir = tmp_path / "out_webview"
    out_dir.mkdir()
    cfg = output_config.output_config(
        output_file="report", output_path=str(out_dir)
    )
    conn = PerfxpertConnection([str(_FIXTURE)])
    analyze_mod._execute_agentic(
        conn, config=cfg, output_format="webview", source_dir=None,
    )
    out = (out_dir / "report.html").read_text()

    assert out.count("<!DOCTYPE html>") == 1
    assert out.count("</body>") == 1
    assert out.count("</html>") == 1
    assert out.count('class="card"') == 0

    assert out.count("<h2>Live Roofline</h2>") == 1
    i_hw = out.find("<h2>Hardware Counters</h2>")
    i_rl = out.find("<h2>Live Roofline</h2>")
    i_recs = out.find("<h2>Optimization Recommendations</h2>")
    assert i_hw != -1 and i_rl != -1 and i_recs != -1
    assert i_hw < i_rl < i_recs
    assert "<circle cx=" in out
