#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

"""
AI-powered performance analysis for GPU traces.

This module analyzes rocpd database files and provides human-readable insights,
bottleneck identification, and optimization recommendations.

Pure analysis logic lives in the ``analysis/`` sub-package; this file is the
thin orchestration and CLI layer.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

try:
    from importlib.metadata import version as _pkg_version

    _PERFXPERT_VERSION = _pkg_version("perfxpert")
except Exception:
    _PERFXPERT_VERSION = "0.2.0"  # fallback if metadata not available (common in dev / ROCm system installs)

from .connection import PerfxpertConnection as RocpdImportData, execute_statement
from .tracelens_port import (
    compute_interval_timeline,
    analyze_kernels_by_category,
    analyze_short_kernels,
)
from . import output_config

# ---------------------------------------------------------------------------
# Re-export analysis functions from the analysis/ sub-package so that
# ``from perfxpert.analyze import compute_time_breakdown`` (etc.) keeps
# working for all existing callers.
# ---------------------------------------------------------------------------
from .analysis import (  # noqa: F401 -- re-exports for backward compat
    identify_hotspots,
    analyze_memory_copies,
    analyze_hardware_counters,
    detect_warmup_issues,
    analyze_kernel_resources,
    analyze_api_overhead,
    analyze_thread_trace,
    generate_recommendations,
    _split_pmc_into_passes,
    _detect_already_collected,
    _filter_rec_commands,
    _is_code_change_rec,
    _ATT_STALL_CATEGORY_MAP,
    _ATT_MIN_HITCOUNT,
    _att_stall_category,
    _SYS_TRACE_IMPLIED,
    _OUTPUT_ONLY_ARGS,
    _PMC_BLOCK_LIMIT_DEFAULT,
    _PMC_BLOCK_LIMITS,
    _TCC_DERIVED_COUNTERS,
    _pmc_block,
    _pmc_block_limit,
    _INIT_OVERHEAD_MAX_KERNEL_PCT,
    _INIT_OVERHEAD_MAX_RUNTIME_NS,
)
from .analysis import core as _analysis_core


def compute_time_breakdown(connection: RocpdImportData) -> Dict[str, Any]:
    """Backward-compat shim — delegates to ``analysis.core`` but uses
    this module's ``execute_statement`` so that
    ``mock.patch("perfxpert.analyze.execute_statement")`` keeps working."""
    import perfxpert.analysis.core as _m

    _saved = _m.execute_statement
    try:
        _m.execute_statement = execute_statement  # pick up any mock on this module
        return _analysis_core.compute_time_breakdown(connection)
    finally:
        _m.execute_statement = _saved

__all__ = [
    "compute_time_breakdown",
    "identify_hotspots",
    "analyze_memory_copies",
    "analyze_hardware_counters",
    "generate_recommendations",
    "format_analysis_output",
    "add_args",
    "execute",
    "main",
]


# ---------------------------------------------------------------------------
# Output formatting functions (extracted to formatters.py)
# ---------------------------------------------------------------------------
from .formatters import (  # noqa: F401 -- re-exports for backward compat
    _format_as_json,
    _build_summary,
    _build_hw_counters_json,
    _build_recommendations_json,
    _build_warnings_json,
    _format_as_markdown,
    _format_as_webview,
    _tier0_recommendations_text,
    _format_tier0_text,
    _tier0_to_dict,
    _format_tier0_json,
    _format_tier0_markdown,
    _format_tier0_webview,
    format_analysis_output,
    _CATEGORY_IDS,
)




def add_args(parser: argparse.ArgumentParser):
    """
    Add command-line arguments for AI analysis.

    Args:
        parser: Argument parser to add arguments to

    Returns:
        Function to process parsed arguments
    """
    analysis_options = parser.add_argument_group("Analysis options")

    analysis_options.add_argument(
        "--source-dir",
        type=str,
        default=None,
        dest="source_dir",
        help=(
            "Path to GPU application source directory for Tier 0 static analysis. "
            "Scans .hip/.cpp/.cu files and generates a profiling plan. "
            "Can be used alone (no -i required) or alongside -i for combined analysis."
        ),
    )

    analysis_options.add_argument(
        "--baseline",
        type=str,
        default=None,
        dest="baseline_db",
        metavar="DB",
        help=(
            "Optional baseline rocpd database to compare the current analysis "
            "against. When supplied, the analyze report gains a ``trace_diff`` "
            "section (JSON) / 'Changed vs baseline' section (text/markdown/"
            "webview) summarizing per-kernel deltas. Uses the same engine as "
            "``perfxpert diff``."
        ),
    )

    analysis_options.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom analysis prompt/question to guide analysis (e.g., 'Why is my matmul kernel slow?')",
    )

    analysis_options.add_argument(
        "--top-kernels",
        type=int,
        default=10,
        help="Number of top kernels to analyze (default: 10)",
    )

    analysis_options.add_argument(
        "--format",
        type=str,
        dest="output_format",
        choices=["text", "json", "markdown", "webview"],
        default="text",
        help="Output format: text, json, markdown, or webview (default: text). "
        "File extension is set automatically: .txt, .json, .md, .html. "
        "Non-text formats write a report file by default.",
    )

    analysis_options.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Minimum kernel duration threshold in microseconds (filter out short kernels)",
    )

    analysis_options.add_argument(
        "--advanced",
        action="store_true",
        default=False,
        dest="advanced",
        help=(
            "Include advanced specialist recommendations (currently: LLVM "
            "loop-hint pragma advice). Default is OFF — pragma recs are "
            "filtered out of the report. Can also be enabled via the "
            "PERFXPERT_ADVANCED_RECS=1 environment variable. Each "
            "pragma rec carries a `[advanced]` badge and requires a "
            "Tier-0 source anchor."
        ),
    )

    # LLM Enhancement Options
    llm_options = parser.add_argument_group(
        "LLM enhancement options (optional)",
        "Enable natural language explanations via one of five LLM providers: "
        "anthropic, openai, ollama (local), private (self-hosted OpenAI-compatible), "
        "or opencode (bundled CLI). Requires API key or local endpoint - see "
        "https://console.anthropic.com/ , https://platform.openai.com/api-keys , "
        "or the matching PERFXPERT_LLM_* environment variable.",
    )

    llm_options.add_argument(
        "--llm",
        type=str,
        dest="llm_provider",
        choices=["anthropic", "openai", "ollama", "private", "opencode", "claude-code"],
        default=None,
        help=(
            "Enable LLM-powered analysis enhancement. Choose one of: "
            "'anthropic' (ANTHROPIC_API_KEY), 'openai' (OPENAI_API_KEY), "
            "'ollama' (local daemon, PERFXPERT_LLM_LOCAL_URL), "
            "'private' (self-hosted OpenAI-compatible endpoint, PERFXPERT_LLM_PRIVATE_URL + _API_KEY), "
            "'opencode' (bundled opencode CLI), "
            "'claude-code' (compatibility alias for opencode). "
            "Local analysis always runs first; LLM provides additional natural language insights."
        ),
    )

    llm_options.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="API key for LLM provider. Alternatively, set the matching environment "
        "variable: ANTHROPIC_API_KEY (anthropic), OPENAI_API_KEY (openai), "
        "PERFXPERT_LLM_PRIVATE_API_KEY (private). ollama/opencode typically do not "
        "require a key. Example: --llm anthropic --llm-api-key sk-ant-... "
        "Or: export ANTHROPIC_API_KEY='sk-ant-...' && perfxpert analyze --llm anthropic",
    )

    llm_options.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override the LLM model name. Defaults to claude-sonnet-4-5 for Anthropic "
        "and gpt-4o-mini for OpenAI. Can also be set via (in priority order): "
        "PERFXPERT_AGENTS_MODEL_<PROVIDER>, PERFXPERT_<PROVIDER>_MODEL (e.g. "
        "PERFXPERT_ANTHROPIC_MODEL, PERFXPERT_OPENAI_MODEL), or PERFXPERT_LLM_MODEL. "
        "`--llm-model` takes precedence over every env var. "
        "Examples: --llm-model claude-opus-4-6, --llm-model gpt-4o",
    )

    llm_options.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging (shows LLM API calls, reference guide loading, etc.)",
    )

    analysis_options.add_argument(
        "--att-dir",
        type=str,
        default=None,
        dest="att_dir",
        help=(
            "Path to directory containing ATT stats_*.csv files from rocprofv3 --att. "
            "Enables Tier 3 Advanced Thread Trace analysis: per-instruction stall ratios "
            "and bottleneck classification (VMEM latency, LDS bank conflict, dependency chains, "
            "branch divergence). Requires rocprof-trace-decoder to be installed. "
            "Example: --att-dir ./att_output"
        ),
    )


    llm_options.add_argument(
        "--llm-thinking",
        metavar="TOKENS",
        type=int,
        default=None,
        dest="llm_thinking",
        help=(
            "Enable extended thinking for deeper LLM analysis. Specify the thinking "
            "budget in tokens (e.g. --llm-thinking 8000). Only available with the "
            "Anthropic provider and compatible models (claude-opus-4, "
            "claude-sonnet-4-5, claude-3-7-sonnet). Adds latency but improves "
            "analysis quality for complex traces with multiple interacting "
            "bottlenecks. Requires --llm anthropic. Also configurable via the "
            "PERFXPERT_LLM_THINKING environment variable (set to token count)."
        ),
    )

    llm_options.add_argument(
        "--llm-local",
        type=str,
        choices=["ollama"],
        default=None,
        dest="llm_local",
        help=(
            "Local LLM provider for Stage 1 source summarization (before online LLM). "
            "Choices: 'ollama'. Requires Ollama running at localhost:11434. "
            "Set PERFXPERT_LLM_LOCAL_URL to override endpoint."
        ),
    )

    llm_options.add_argument(
        "--llm-local-model",
        type=str,
        default=None,
        dest="llm_local_model",
        help=(
            "Model name for local LLM (default: codellama:13b). "
            "Can also be set via PERFXPERT_LLM_LOCAL_MODEL environment variable."
        ),
    )

    llm_options.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        dest="no_progress",
        help=(
            "Disable live progress feedback during LLM analysis. Useful "
            "for CI and log-capture contexts where spinner escape codes "
            "or repeated status lines would pollute output. Progress is "
            "only emitted when --llm is set; this flag is a no-op under "
            "airgap."
        ),
    )

    def process_args(input: RocpdImportData, args: argparse.Namespace):
        """Process and return valid arguments as dictionary.

        Arg names are chosen to match `_execute_agentic`'s kwarg
        expectations directly (review E2E bug 1): ``output_format`` and
        ``llm_provider`` are wired via argparse ``dest=`` overrides on the
        `--format` / `--llm` flags. ``enable_llm`` is derived from
        ``llm_provider`` being truthy so the agentic path activates the
        live LLM session without a separate boolean flag.
        """
        valid_args = [
            "source_dir",
            "att_dir",
            "prompt",
            "top_kernels",
            "output_format",
            "min_duration",
            "llm_provider",
            "llm_api_key",
            "llm_model",
            "llm_thinking",
            "verbose",
            "llm_local",
            "llm_local_model",
            "no_progress",
            "baseline_db",
            "advanced",
        ]
        # Argparse defaults argparse emits for flags not passed by the
        # user; we skip these so kwargs do not carry noise that the
        # downstream agentic runtime has to special-case. Example: the
        # `--verbose` store_true flag defaults to False, and the
        # `--top-kernels` integer flag defaults to 10; passing them
        # unconditionally would mask "user did not set this" from
        # `_execute_agentic`.
        _cli_defaults = {
            "verbose": False,
            "top_kernels": 10,
            "min_duration": 0.0,
            "no_progress": False,
            "advanced": False,
        }
        ret = {}
        for itr in valid_args:
            if hasattr(args, itr):
                val = getattr(args, itr)
                if val is None:
                    continue
                # Drop pure-default values so kwargs reflect what the user
                # actually set on the CLI.
                if itr in _cli_defaults and val == _cli_defaults[itr]:
                    continue
                ret[itr] = val
        # Convert min_duration from microseconds to nanoseconds
        if "min_duration" in ret:
            ret["min_duration"] = ret["min_duration"] * 1000
        # Derive enable_llm: non-None llm_provider means the user asked for LLM
        if ret.get("llm_provider"):
            ret["enable_llm"] = True
        return ret

    return process_args


def execute(
    input: Optional[RocpdImportData],
    config: Optional[output_config.output_config] = None,
    **kwargs: Any,
) -> Optional[RocpdImportData]:
    """
    Public CLI entry point — delegates to the agentic implementation.

    Args:
        input: RocpdImportData object with database connection, or None for source-only mode
        config: Optional output configuration
        **kwargs: Analysis parameters (may include source_dir for Tier 0)

    Returns:
        The input RocpdImportData object (for chaining), or None in source-only mode
    """
    return _execute_agentic(input, config=config, **kwargs)


# ---------------------------------------------------------------------------
# Webview splice helpers
# ---------------------------------------------------------------------------
#
# The agentic pipeline builds the narrative/primary_bottleneck/warnings block
# independently from the deterministic webview template. These helpers keep
# the splice logic symmetrical with the other formats:
#
#  * ``_build_summary_scard`` renders the Summary as a standard
#    ``.scard/.shdr/.sbody`` card (no ad-hoc ``.card`` class).
#  * ``_splice_after_overview`` inserts it as the FIRST section inside
#    ``<div class="wrap">`` (right after the Overview card).
#  * ``_build_tier0_wrapper_scard`` takes the full tier-0 HTML (a complete
#    ``<!DOCTYPE html>…</html>`` document) and extracts only the inner
#    ``<section class="scard">`` siblings, wrapping them in one parent
#    ``.scard`` labelled "Tier-0 Source Scan".
#  * ``_splice_before_wrap_end`` inserts the wrapper just before
#    ``</div><footer>`` so tier-0 always renders as the LAST top-level
#    card in the main report.


def _splice_baseline_diff(
    report_text: str,
    *,
    baseline_db: str,
    new_db: str,
    output_format: str,
    top_kernels: int,
) -> str:
    """Append / splice a trace-diff section into ``report_text``.

    Confluence row #7 end-of-session recap. The caller passes the fully
    formatted analyze report plus the baseline DB path; this helper
    computes the diff using the shared ``trace_diff.diff_runs`` engine
    and appends a "Changed vs baseline" section appropriate for the
    chosen format.

    * ``json``    — add a top-level ``trace_diff`` key (parse, merge, reserialize).
    * ``markdown``— append a ``## Changed vs baseline`` section.
    * ``text``    — append a trace-diff text block.
    * ``webview`` — append a ``.scard`` inside the existing wrap, before
                    the closing ``</div></body></html>`` tags.
    """
    try:
        from perfxpert.tools.trace_diff import diff_runs
        from perfxpert.cli.diff_cmd import render_diff
    except Exception:
        return report_text  # best-effort; never crash analyze on diff failure

    try:
        diff_result = diff_runs(baseline_db, new_db, top_kernels=int(top_kernels or 20))
    except Exception:
        return report_text

    if output_format == "json":
        import json as _json_mod

        try:
            parsed = _json_mod.loads(report_text)
        except Exception:
            return report_text
        # Splice as a top-level key so schema-aware consumers see it.
        parsed["trace_diff"] = diff_result
        # Keep schema_version coherent with trace_diff.
        if parsed.get("schema_version", "0.1.0") < "0.3.1":
            parsed["schema_version"] = "0.3.1"
            if "metadata" in parsed:
                parsed["metadata"]["analysis_version"] = "0.3.1"
        return _json_mod.dumps(parsed, indent=2)

    if output_format == "markdown":
        md_block = render_diff(diff_result, "markdown")
        # Re-title the top-level heading so it reads as a section, not a
        # standalone report.
        md_block = md_block.replace(
            "# PerfXpert — Trace diff",
            "## Changed vs baseline",
            1,
        )
        return report_text.rstrip() + "\n\n" + md_block + "\n"

    if output_format == "webview":
        # Render a minimal ``.scard`` with the diff summary + per-kernel
        # rows; splice it just before ``</body>``.
        html_block = _render_diff_scard_html(diff_result)
        if "</body>" in report_text:
            return report_text.replace("</body>", html_block + "</body>", 1)
        return report_text + html_block

    # text format
    text_block = render_diff(diff_result, "text")
    # Retitle the top banner so it reads as a section.
    text_block = text_block.replace(
        " PerfXpert trace diff",
        " Changed vs baseline",
        1,
    )
    return report_text.rstrip() + "\n\n" + text_block + "\n"


def _render_diff_scard_html(diff_result: Dict[str, Any]) -> str:
    """Render a diff result as a single ``.scard`` for webview splicing.

    The full ``_format_diff_webview`` function is intended for standalone
    diff pages; the analyze-report splice uses a more compact form that
    fits inside the existing ``.wrap`` container.
    """
    import html as _h_mod

    def _h(v: Any) -> str:
        return _h_mod.escape(str(v), quote=True)

    wall_pct = float(diff_result.get("wall_delta_pct", 0.0))
    wall_ns = int(diff_result.get("wall_delta_ns", 0))
    regs = diff_result.get("primary_regressions", []) or []
    imps = diff_result.get("primary_improvements", []) or []
    color = (
        "#e84040"
        if wall_pct > 5.0
        else ("#3acc66" if wall_pct < -5.0 else "#4d8ef2")
    )
    rows = []
    for k in (diff_result.get("per_kernel") or []):
        dp = float(k.get("delta_pct", 0.0))
        rc = "#e84040" if dp > 3 else ("#3acc66" if dp < -3 else "var(--sub)")
        rows.append(
            "<tr>"
            f'<td><code>{_h(k.get("name", "?"))}</code></td>'
            f'<td>{int(k.get("baseline_ns", 0)):,}</td>'
            f'<td>{int(k.get("new_ns", 0)):,}</td>'
            f'<td style="color:{rc}">{dp:+.2f}%</td>'
            "</tr>"
        )
    return (
        '<section class="scard" id="trace-diff">'
        '<div class="shdr">'
        '<span class="shdr-icon">&#128200;</span>'
        '<h2>Changed vs baseline</h2>'
        '</div>'
        '<div class="sbody">'
        f'<p style="margin-bottom:.75rem">Baseline: <code>{_h(diff_result.get("baseline_db"))}</code> '
        f'&rarr; New: <code>{_h(diff_result.get("new_db"))}</code></p>'
        f'<p style="margin-bottom:.75rem;color:{color};font-weight:600">'
        f'Wall-time delta: {wall_pct:+.2f}% ({wall_ns:+,} ns) — '
        f'{len(regs)} regression(s), {len(imps)} improvement(s).</p>'
        '<div class="tbl-wrap"><table class="dtable">'
        '<thead><tr><th>Kernel</th><th>Baseline (ns)</th>'
        '<th>New (ns)</th><th>Δ %</th></tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody></table></div>'
        '</div></section>'
    )


def _build_summary_scard(
    *, narrative: str, primary_bottleneck: str, warnings: list
) -> str:
    """Render the agent Summary as a standard ``.scard``.

    The narrative goes in a ``<p>`` inside ``.sbody`` (preserving
    paragraph breaks), the primary bottleneck surfaces as a
    ``shdr-badge sbadge-info`` pill, and warnings render as a
    bulleted list below the narrative with a subtle warn styling.
    """
    import html as _html_mod

    narrative_s = (narrative or "").strip()
    # Split on blank lines so multi-paragraph LLM output renders cleanly.
    paragraphs = [p.strip() for p in narrative_s.split("\n\n") if p.strip()]
    if not paragraphs and narrative_s:
        paragraphs = [narrative_s]
    body_parts = []
    if primary_bottleneck:
        body_parts.append(
            f'<p class="assess"><strong>Primary bottleneck:</strong> '
            f'{_html_mod.escape(str(primary_bottleneck))}</p>'
        )
    for para in paragraphs:
        body_parts.append(
            f'<p style="margin-bottom:.8rem;line-height:1.65">'
            f'{_html_mod.escape(para)}</p>'
        )
    if warnings:
        w_items = "".join(
            f"<li>{_html_mod.escape(str(w))}</li>" for w in warnings
        )
        body_parts.append(
            '<p style="margin-top:.85rem;color:var(--sub);font-size:.9rem">'
            "Warnings:</p>"
            f'<ul class="findings">{w_items}</ul>'
        )
    bn_badge = ""
    if primary_bottleneck:
        bn_badge = (
            f'<span class="shdr-badge sbadge-info">'
            f'{_html_mod.escape(str(primary_bottleneck))}</span>'
        )
    return (
        '\n<section class="scard" id="agentic-narrative">'
        '\n<div class="shdr">'
        '\n<span class="shdr-icon">&#128220;</span>'
        '\n<h2>Summary</h2>'
        f"\n{bn_badge}"
        '\n</div>'
        '\n<div class="sbody">'
        f'\n{"".join(body_parts)}'
        '\n</div>'
        '\n</section>\n'
    )


def _splice_after_overview(html: str, section_html: str) -> str:
    """Insert ``section_html`` right after the Overview ``.scard``.

    The main webview template opens with an Overview ``.scard`` inside
    ``<div class="wrap">``. We anchor on that card's ``<h2>Overview</h2>``
    and walk to its closing ``</section>``; the new section goes just
    after that closing tag.

    Fallback: if the anchor cannot be located (unexpected template
    drift), splice right after the opening ``<div class="wrap">`` instead
    — the Summary still lands at the TOP of the report, just above the
    Overview.
    """
    anchor = "<h2>Overview</h2>"
    idx = html.find(anchor)
    if idx != -1:
        close = html.find("</section>", idx)
        if close != -1:
            insert_at = close + len("</section>")
            return html[:insert_at] + section_html + html[insert_at:]
    wrap_open = '<div class="wrap">'
    idx = html.find(wrap_open)
    if idx != -1:
        insert_at = idx + len(wrap_open)
        return html[:insert_at] + section_html + html[insert_at:]
    # Last resort — append before </body> so we never lose the content.
    if "</body>" in html:
        return html.replace("</body>", section_html + "</body>", 1)
    return html + section_html


def _splice_before_wrap_end(html: str, section_html: str) -> str:
    """Insert ``section_html`` just before ``</div>`` that closes
    ``<div class="wrap">``.

    We locate the opening tag and walk the string to find the matching
    closing ``</div>`` that sits immediately before ``<footer>`` (the
    template always renders ``</div>\n\n<footer>``). This keeps tier-0
    inside the main wrap, not floating after the footer.
    """
    anchor = "</div>\n\n<footer>"
    idx = html.find(anchor)
    if idx != -1:
        return html[:idx] + section_html + html[idx:]
    # More tolerant fallback: any </div> followed by <footer.
    import re as _re

    m = _re.search(r"</div>\s*<footer", html)
    if m is not None:
        insert_at = m.start()
        return html[:insert_at] + section_html + html[insert_at:]
    if "</body>" in html:
        return html.replace("</body>", section_html + "</body>", 1)
    return html + section_html


def _build_tier0_wrapper_scard(tier0_full_html: str) -> str:
    """Extract inner ``<section class="scard">`` blocks from a full Tier-0
    webview document and wrap them in a single top-level ``.scard`` named
    "Tier-0 Source Scan".

    The Tier-0 webview is a complete, self-contained
    ``<!DOCTYPE html>…</html>`` document — splicing it verbatim produces
    nested ``<head>`` / ``<body>`` and duplicate ``<style>`` blocks.
    This helper pulls out ONLY the ``.scard`` children from the tier-0
    ``<div class="wrap">`` and composes them as siblings inside our
    wrapper card.
    """
    import re as _re

    # Greedy sibling match: ``<section class="scard"`` through the next
    # ``</section>``. The tier-0 renderer does not nest scards, so a
    # non-greedy closure is safe.
    inner_sections = _re.findall(
        r'<section class="scard".*?</section>',
        tier0_full_html,
        flags=_re.DOTALL,
    )
    if not inner_sections:
        # Defensive: fall back to the raw string so users still see
        # something (just without the extra wrapping). A comment marks
        # this as a known degraded path for debugging.
        return (
            '\n<section class="scard" id="tier0-scan">'
            '\n<div class="shdr">'
            '\n<span class="shdr-icon">&#128187;</span>'
            '\n<h2>Tier-0 Source Scan</h2>'
            '\n</div>'
            '\n<div class="sbody">'
            '\n<!-- tier-0 scards not found; raw content omitted -->'
            '\n</div>'
            '\n</section>\n'
        )
    inner = "\n".join(inner_sections)
    return (
        '\n<section class="scard" id="tier0-scan">'
        '\n<div class="shdr">'
        '\n<span class="shdr-icon">&#128187;</span>'
        '\n<h2>Tier-0 Source Scan</h2>'
        '\n<span class="shdr-badge sbadge-info">Static source analysis</span>'
        '\n</div>'
        '\n<div class="sbody">'
        f'\n{inner}'
        '\n</div>'
        '\n</section>\n'
    )


def _resolve_advanced_flag(advanced_arg: Optional[bool]) -> bool:
    """Resolve the effective ``--advanced`` gate.

    Precedence (highest first):

    1. Explicit CLI ``--advanced`` (``advanced_arg=True``).
    2. ``PERFXPERT_ADVANCED_RECS`` env var — truthy values ``1``,
       ``true``, ``yes``, ``on`` (case-insensitive).
    3. Default OFF.

    Returns ``True`` when advanced recommendations (currently: LLVM
    loop-hint pragma advice emitted via ``subtype: "pragma"``) should
    appear in the report.
    """
    if advanced_arg is True:
        return True
    env_val = os.environ.get("PERFXPERT_ADVANCED_RECS", "")
    return env_val.strip().lower() in {"1", "true", "yes", "on"}


def _filter_advanced_recs(
    recs: List[Dict[str, Any]],
    *,
    advanced: bool,
) -> List[Dict[str, Any]]:
    """Strip ``subtype="pragma"`` entries when the advanced gate is OFF.

    The Compute Specialist may emit pragma recs regardless of the
    gate state; the gate decides whether they *render*. This keeps
    the agentic pipeline stateless and lets the same payload drive
    both a default-mode report and an ``--advanced`` report without
    re-running the LLM.
    """
    if advanced:
        return list(recs)
    return [r for r in recs if r.get("subtype") != "pragma"]


# Phase 10 — Change-Impact Prediction final-pass.
# Maps deterministic rec categories / titles onto the change_type ids in
# ``knowledge/change_impact_models.yaml`` so the Predicted line appears on
# the rec cards even when the specialist catalog modules are absent. The
# specialist-driven path (agents._predict_attach) runs FIRST and already
# handles the LLM / airgap-with-catalog case; this is the safety net for
# deterministic rec cards surfaced by the static classifier.
_CATEGORY_TO_CHANGE_TYPE: Dict[str, str] = {
    "compute-bound kernel": "vgpr_reduction",
    "launch overhead": "hip_stream_overlap",
    "memory transfer": "hip_stream_overlap",
    "memory bandwidth": "lds_tiling",
    "api overhead": "hip_stream_overlap",
    "memory-bound kernel": "lds_tiling",
    "mixed bottleneck kernel": "mfma_enablement",
    "low occupancy": "vgpr_reduction",
}


def _attach_predictions_by_category(
    recs: List[Dict[str, Any]],
    *,
    hotspots: List[Dict[str, Any]],
    primary_bottleneck: str,
    counter_data_available: bool,
) -> List[Dict[str, Any]]:
    """Attach predicted_impact_range / confidence / source_citation to recs
    whose category matches a known technique in the change-impact catalog.

    Hard rules (spec §6) are all enforced by
    ``predict_impact.predict_change_impact`` itself — so when a gate
    fires the rec is returned unchanged.
    """
    if not recs or not hotspots:
        return list(recs or [])
    from perfxpert.tools import predict_impact

    top = hotspots[0]
    kernel_name = str(top.get("name") or "")
    if not kernel_name:
        return list(recs)
    kernel_time_pct = top.get("percent_of_total")
    if isinstance(kernel_time_pct, (int, float)) and kernel_time_pct > 1.0:
        kernel_time_pct = float(kernel_time_pct) / 100.0

    # Prefer the rec matching the workload's primary bottleneck — predicting
    # API-overlap speedup on a compute-bound report misrepresents gain.
    def _rec_primary_rank(rec: Dict[str, Any]) -> int:
        cat = str(rec.get("category") or "").strip().lower()
        primary = (primary_bottleneck or "").lower()
        if primary == "compute" and cat in (
            "compute-bound kernel",
            "mixed bottleneck kernel",
            "low occupancy",
        ):
            return 0
        if primary in ("memory", "memory_transfer") and cat in (
            "memory-bound kernel",
            "memory transfer",
            "memory bandwidth",
        ):
            return 0
        if primary in ("latency", "api_overhead") and cat in (
            "launch overhead",
            "api overhead",
        ):
            return 0
        return 1

    ordered_indices = sorted(range(len(recs)), key=lambda i: _rec_primary_rank(recs[i]))
    target_idx: Optional[int] = None
    target_change_type: Optional[str] = None
    for i in ordered_indices:
        cat = str(recs[i].get("category") or "").strip().lower()
        ct = _CATEGORY_TO_CHANGE_TYPE.get(cat)
        if ct is not None and recs[i].get("predicted_impact_range") is None:
            target_idx = i
            target_change_type = ct
            break

    out: List[Dict[str, Any]] = []
    for idx, rec in enumerate(recs):
        if idx != target_idx:
            out.append(dict(rec))
            continue
        try:
            prediction = predict_impact.predict_change_impact(
                baseline_db="",
                kernel_name=kernel_name,
                change_type=target_change_type,
                change_params={
                    "kernel_time_pct": kernel_time_pct,
                    "counter_data_available": counter_data_available,
                },
            )
        except Exception:
            out.append(dict(rec))
            continue
        rng = prediction.get("predicted_speedup_range")
        if rng is None or prediction.get("confidence", 0.0) <= 0.0:
            out.append(dict(rec))
            continue
        copy = dict(rec)
        copy["predicted_impact_range"] = list(rng)
        copy["predicted_confidence"] = float(prediction["confidence"])
        copy["predicted_rationale"] = prediction.get("rationale", "")
        copy["source_citation"] = prediction.get("source_citation", "")
        copy["roofline_delta"] = prediction.get("roofline_delta") or {}
        copy["prediction_id"] = prediction.get("prediction_id", "")
        out.append(copy)
    return out


def _format_agentic_output(
    root_output: Any,
    output_format: str,
    *,
    database_path: str = "",
    analysis_payload: Optional[Dict[str, Any]] = None,
    advanced: bool = False,
) -> str:
    """Render an agentic RootOutput + deterministic analysis payload.

    The agentic pipeline supplies the brain (narrative, primary_bottleneck,
    recommendations, warnings); ``analysis_payload`` supplies the
    deterministic dataset (time_breakdown, hotspots, memory_analysis,
    hardware_counters, kernel_resources, api_overhead, thread_trace,
    tier0_findings). Both are merged so every format renders the full
    contract — tables + stats + narrative + recommendations — regardless of
    whether an LLM is in the loop.

    ``root_output`` may be either:
      - a :class:`perfxpert.agents.schemas.RootOutput` (Pydantic model),
        in which case attributes are read via ``getattr``; or
      - a plain ``dict`` (e.g. the return value of
        :func:`perfxpert.api.agent_root`), in which case the same five
        keys are read via ``dict.get``.

    When ``analysis_payload`` only carries a ``tier0_findings`` section and
    no database-backed sections (i.e. ``--source-dir`` without ``-i``), the
    tier-0-specific formatters are used for a source-only report.
    """
    from .analysis.payload import merge_recommendations, tier0_dict_to_ns

    def _read(name: str, default: Any) -> Any:
        if isinstance(root_output, dict):
            return root_output.get(name, default)
        return getattr(root_output, name, default)

    narrative = _read("narrative", "") or ""
    llm_recs = list(_read("recommendations", []) or [])
    primary_bottleneck = _read("primary_bottleneck", "mixed") or "mixed"
    warnings = list(_read("warnings", []) or [])
    metadata = dict(_read("metadata", {}) or {})

    payload = analysis_payload or {}

    time_breakdown = payload.get("time_breakdown") or {}
    hotspots = payload.get("hotspots") or []
    memory_analysis = payload.get("memory_analysis") or {}
    hardware_counters = payload.get("hardware_counters") or {}
    kernel_resources = payload.get("kernel_resources") or {}
    api_overhead = payload.get("api_overhead") or {}
    thread_trace = payload.get("thread_trace")
    tier0_findings = payload.get("tier0_findings")
    roofline_points = payload.get("roofline")
    communication = payload.get("communication")  # Phase 10 RCCL / NIC
    det_recs = list(payload.get("recommendations_deterministic") or [])

    # Merge LLM + deterministic recommendations (dedupe by (type,target) or
    # (category,issue)). LLM recs keep their verdict; deterministic citations
    # / code snippets are carried across when they disambiguate the same rec.
    merged_recs = merge_recommendations(llm_recs, det_recs)

    # Phase 10: advanced-gate filter. When --advanced (or
    # PERFXPERT_ADVANCED_RECS=1) is OFF, pragma recs never render —
    # keeping default output clean while the pipeline itself stays
    # stateless. See docs/guides/getting-started.md §4.1.
    merged_recs = _filter_advanced_recs(merged_recs, advanced=advanced)

    # Phase 10 — Change-Impact Prediction final-pass. Attach predicted
    # speedup + confidence + source citation to the rec card whose
    # category matches a named optimization technique in
    # ``knowledge/change_impact_models.yaml``. See
    # docs/guides/getting-started.md §4 "Predicted impact on
    # recommendations".
    merged_recs = _attach_predictions_by_category(
        merged_recs,
        hotspots=hotspots,
        primary_bottleneck=primary_bottleneck,
        counter_data_available=bool(
            (hardware_counters or {}).get("has_counters")
        ),
    )

    # Source-only path: dispatch entirely to the tier-0 formatters when
    # there is no DB-side data at all.
    tier0_only = (
        tier0_findings is not None
        and not time_breakdown
        and not hotspots
        and not memory_analysis
    )
    if tier0_only:
        # Attach narrative + merged recs onto the tier-0 result so the
        # tier-0 formatters carry the agent brain through the output.
        tier0_ns = tier0_dict_to_ns(tier0_findings)
        if narrative and not getattr(tier0_ns, "llm_explanation", None):
            tier0_ns.llm_explanation = narrative
        if merged_recs:
            existing = list(getattr(tier0_ns, "recommendations", None) or [])
            tier0_ns.recommendations = merged_recs + existing

        if output_format == "json":
            # Tier-0 JSON is already schema-shaped, but top-level schema
            # callers expect the agentic keys too (narrative / primary_bottleneck /
            # warnings) so we merge them in post-hoc.
            import json as _json
            base = _format_tier0_json(tier0_ns)
            try:
                doc = _json.loads(base)
            except Exception:
                doc = {}
            doc["narrative"] = narrative
            doc["primary_bottleneck"] = primary_bottleneck
            doc["warnings"] = warnings
            # Flat convenience keys so `jq '.time_breakdown'` returns {} not null.
            doc.setdefault("time_breakdown", {})
            doc.setdefault("memory_analysis", {})
            doc.setdefault("hardware_counters", doc.get("hardware_counters") or {"has_counters": False})
            doc.setdefault("tier0_findings", tier0_findings)
            # Bump schema to 0.3.0 to reflect the agentic + tier-0 contract.
            doc["schema_version"] = "0.3.0"
            doc.setdefault("metadata", {})["analysis_version"] = "0.3.0"
            if narrative and not doc.get("llm_enhanced_explanation"):
                doc["llm_enhanced_explanation"] = narrative
            return _json.dumps(doc, indent=2)
        if output_format == "markdown":
            md = _format_tier0_markdown(tier0_ns)
            summary_parts: list = []
            if primary_bottleneck:
                summary_parts.append(
                    f"**Primary bottleneck:** {primary_bottleneck}"
                )
                summary_parts.append("")
            if narrative:
                summary_parts.append(narrative.rstrip())
                summary_parts.append("")
            if warnings:
                for w in warnings:
                    summary_parts.append(f"- \u26a0 {w}")
                summary_parts.append("")
            if summary_parts:
                md = (
                    "## Summary\n\n"
                    + "\n".join(summary_parts).rstrip()
                    + "\n\n---\n\n"
                    + md
                )
            return md
        if output_format == "webview":
            html = _format_tier0_webview(tier0_ns)
            if narrative or primary_bottleneck or warnings:
                summary_section = _build_summary_scard(
                    narrative=narrative,
                    primary_bottleneck=primary_bottleneck,
                    warnings=warnings,
                )
                html = _splice_after_overview(html, summary_section)
            return html
        # text: prepend "PERFXPERT ANALYSIS" banner so tests that assert
        # the title remain green even on the tier-0-only path.
        tier0_text = _format_tier0_text(tier0_ns)
        width = 80
        banner = (
            "=" * width + "\n"
            + "PERFXPERT ANALYSIS".center(width) + "\n"
            + "=" * width + "\n"
        )
        if "PERFXPERT ANALYSIS" not in tier0_text:
            tier0_text = banner + tier0_text
        if narrative:
            summary = (
                "\n" + ("\u2501" * width) + "\n"
                + "SUMMARY".center(width) + "\n"
                + ("\u2501" * width) + "\n"
                + narrative.rstrip() + "\n"
            )
            tier0_text = summary + tier0_text
        return tier0_text

    if output_format == "json":
        _json_detected_kernels = None
        if tier0_findings is not None:
            _json_detected_kernels = tier0_findings.get("detected_kernels") or []
        base_json = _format_as_json(
            time_breakdown=time_breakdown,
            hotspots=hotspots,
            memory_analysis=memory_analysis,
            recommendations=merged_recs,
            hardware_counters=hardware_counters,
            database_path=database_path,
            att_analysis=thread_trace,
            kernel_resources=kernel_resources,
            api_overhead=api_overhead,
            detected_kernels=_json_detected_kernels,
            communication=communication,
            roofline=roofline_points,
        )
        import json as _json
        try:
            doc = _json.loads(base_json)
        except Exception:
            doc = {}
        # Merge the agentic brain on top of the deterministic doc so the
        # JSON contract has EVERY section: narrative + primary_bottleneck +
        # warnings + tier0_findings, on top of time_breakdown / hotspots /
        # memory_analysis / hardware_counters.
        doc["narrative"] = narrative
        doc["primary_bottleneck"] = primary_bottleneck
        doc["warnings"] = warnings
        doc.setdefault("summary", {})["primary_bottleneck"] = primary_bottleneck
        # Preserve a flat ``recommendations`` key in addition to the
        # structured ``recommendations`` inside the schema so callers that
        # do ``jq '.recommendations | length'`` continue to work.
        doc["recommendations"] = merged_recs
        doc["metadata"] = {**doc.get("metadata", {}), **metadata}
        if tier0_findings is not None:
            # When a profiling database IS present, strip instrumentation-advice
            # fields (suggested_counters / profiling_plan / profiling_plan_actions
            # / suggested_first_command) from the Tier-0 payload — they only
            # make sense in the source-only (no-DB) path.
            if database_path:
                _strip_keys = {
                    "suggested_counters",
                    "profiling_plan",
                    "profiling_plan_actions",
                    "suggested_first_command",
                }
                doc["tier0_findings"] = {
                    k: v for k, v in tier0_findings.items() if k not in _strip_keys
                }
            else:
                doc["tier0_findings"] = tier0_findings
        # Schema bump: the agentic pipeline adds
        # ``narrative + primary_bottleneck + summary + tier0_findings`` on
        # top of the legacy deterministic schema. This is the contract the
        # four formatters render against, so bump to 0.3.0 unless a later
        # schema (0.4.0 for ATT) is already set by ``_format_as_json``.
        _existing = doc.get("schema_version", "0.1.0")
        if _existing in ("0.1.0", "0.2.0"):
            doc["schema_version"] = "0.3.0"
            doc.setdefault("metadata", {})["analysis_version"] = "0.3.0"
        # When hotspots carry source_locations (Confluence row #5 — Source
        # Code Line numbers), bump to 0.3.1.
        if any(h.get("source_locations") for h in doc.get("hotspots", [])):
            if _existing not in ("0.4.0",):
                doc["schema_version"] = "0.3.1"
                doc.setdefault("metadata", {})["analysis_version"] = "0.3.1"
        # ``narrative`` is the canonical agent-brain field; mirror it into
        # the legacy ``llm_enhanced_explanation`` alias so consumers that
        # still read that key do not regress.
        if narrative and not doc.get("llm_enhanced_explanation"):
            doc["llm_enhanced_explanation"] = narrative
        # Flat convenience keys for the test contract.
        doc.setdefault("time_breakdown", time_breakdown)
        doc.setdefault("hotspots", hotspots)
        doc.setdefault("memory_analysis", memory_analysis)
        doc.setdefault("hardware_counters", hardware_counters)
        doc.setdefault("kernel_resources", kernel_resources)
        doc.setdefault("api_overhead", api_overhead)
        return _json.dumps(doc, indent=2)

    if output_format == "markdown":
        _md_detected_kernels = None
        if tier0_findings is not None:
            _md_detected_kernels = tier0_findings.get("detected_kernels") or []
        md = _format_as_markdown(
            time_breakdown=time_breakdown,
            hotspots=hotspots,
            memory_analysis=memory_analysis,
            recommendations=merged_recs,
            hardware_counters=hardware_counters,
            database_path=database_path,
            detected_kernels=_md_detected_kernels,
            communication=communication,
            roofline=roofline_points,
        )
        # Splice the agent Summary in between the H1 and the metadata
        # block (``**Database:**`` / ``**Analysis Date:**`` /
        # ``**Analysis Tier:**``) with a horizontal rule between the
        # narrative and the metadata. Format:
        #
        #   # PerfXpert AI Performance Analysis
        #
        #   ## Summary
        #
        #   **Primary bottleneck:** compute
        #
        #   <narrative>
        #
        #   ⚠ warning 1
        #
        #   ---
        #
        #   **Database:** `…`
        #   **Analysis Date:** …
        #   **Analysis Tier:** …
        #
        #   ## Time Breakdown
        summary_parts: list = []
        if primary_bottleneck:
            summary_parts.append(
                f"**Primary bottleneck:** {primary_bottleneck}"
            )
            summary_parts.append("")
        if narrative:
            summary_parts.append(narrative.rstrip())
            summary_parts.append("")
        if warnings:
            for w in warnings:
                summary_parts.append(f"- \u26a0 {w}")
            summary_parts.append("")
        summary_block = ""
        if summary_parts:
            summary_block = (
                "## Summary\n\n"
                + "\n".join(summary_parts).rstrip()
                + "\n\n---\n\n"
            )
        if summary_block:
            anchor = "**Database:**"
            idx = md.find(anchor)
            if idx != -1:
                # Rewind to the start of the line that contains the anchor
                # so we can insert the Summary BEFORE the metadata lines.
                line_start = md.rfind("\n", 0, idx)
                if line_start < 0:
                    line_start = 0
                md = md[:line_start + 1] + summary_block + md[line_start + 1:]
            else:
                # No metadata block — splice Summary right under the H1.
                h1_end = md.find("\n", md.find("# "))
                if h1_end != -1:
                    md = (
                        md[: h1_end + 1]
                        + "\n"
                        + summary_block
                        + md[h1_end + 1:]
                    )
                else:
                    md = summary_block + md
        if tier0_findings is not None:
            tier0_ns = tier0_dict_to_ns(tier0_findings)
            md += "\n\n---\n\n## Tier 0 — Source Scan\n\n"
            md += _format_tier0_markdown(tier0_ns, has_profiling=True)
        return md

    if output_format == "webview":
        # Pass Tier-0 detected_kernels into the webview so the Top Kernel
        # Hotspots table can correlate each row with its source definition
        # + launch site (expandable per-row panel).
        _detected_kernels = None
        if tier0_findings is not None:
            _detected_kernels = tier0_findings.get("detected_kernels") or []
        html = _format_as_webview(
            time_breakdown=time_breakdown,
            hotspots=hotspots,
            memory_analysis=memory_analysis,
            recommendations=merged_recs,
            hardware_counters=hardware_counters,
            database_path=database_path,
            att_analysis=thread_trace,
            detected_kernels=_detected_kernels,
            roofline=roofline_points,
            communication=communication,
        )
        # Splice the narrative as a standard `.scard` Summary section at the
        # TOP of the report (right after the Overview card inside
        # `<div class="wrap">`). Uses the shared `.scard/.shdr/.sbody`
        # template so it matches every other section visually.
        if narrative or primary_bottleneck or warnings:
            summary_section = _build_summary_scard(
                narrative=narrative,
                primary_bottleneck=primary_bottleneck,
                warnings=warnings,
            )
            html = _splice_after_overview(html, summary_section)
        # When tier-0 findings are present in the combined path
        # (-i + --source-dir), splice a SINGLE dedicated "Tier-0 Source Scan"
        # `.scard` before the closing `</div>` of `<div class="wrap">`.
        # Inner `<section class="scard">` elements from the tier-0 renderer
        # are extracted and inlined — the nested `<!DOCTYPE html>…</html>`
        # scaffolding is stripped entirely.
        if tier0_findings is not None:
            tier0_ns = tier0_dict_to_ns(tier0_findings)
            # Pass hotspots so the Detected GPU Kernels rows are colored
            # by the matched Tier-1 % Total bucket. Source-only callers
            # upstream (formatters/__init__.py source_only path) do not
            # have hotspots and pass None → no severity coloring.
            tier0_full = _format_tier0_webview(
                tier0_ns, has_profiling=True, hotspots=hotspots
            )
            tier0_wrapper = _build_tier0_wrapper_scard(tier0_full)
            html = _splice_before_wrap_end(html, tier0_wrapper)
        return html

    # Default: structured text — dispatch to the legacy text formatter for
    # the deterministic skeleton, then splice narrative + primary_bottleneck
    # + warnings on top.
    base_text = format_analysis_output(
        time_breakdown=time_breakdown,
        hotspots=hotspots,
        memory_analysis=memory_analysis,
        recommendations=merged_recs,
        hardware_counters=hardware_counters,
        database_path=database_path,
        att_analysis=thread_trace,
        kernel_resources=kernel_resources,
        api_overhead=api_overhead,
        tier0_result=tier0_dict_to_ns(tier0_findings) if tier0_findings else None,
        source_only=False,
        output_format="text",
        communication=communication,
        roofline=roofline_points,
    )

    # Prepend a Summary block with narrative + primary_bottleneck so the
    # user sees the agent brain before the tables.
    head_lines = []
    width = 80
    head_lines.append("")
    head_lines.append("\u2501" * width)
    head_lines.append("SUMMARY".center(width))
    head_lines.append("\u2501" * width)
    head_lines.append(f"Primary bottleneck: {primary_bottleneck}")
    head_lines.append("")
    if narrative:
        head_lines.append(narrative.rstrip())
        head_lines.append("")
    if warnings:
        head_lines.append("Warnings:")
        for w in warnings:
            head_lines.append(f"  \u26a0  {w}")
        head_lines.append("")

    # Splice head_lines BEFORE the existing "TIME BREAKDOWN" section so the
    # summary sits at the top of the report under the title.
    split_token = "TIME BREAKDOWN"
    idx = base_text.find(split_token)
    if idx != -1:
        # Find start of the line holding TIME BREAKDOWN (go back to preceding \n)
        line_start = base_text.rfind("\n", 0, idx)
        # Also back up one more line for the box rule above "TIME BREAKDOWN"
        line_start = base_text.rfind("\n", 0, line_start) if line_start > 0 else line_start
        if line_start < 0:
            line_start = 0
        return base_text[:line_start] + "\n" + "\n".join(head_lines) + base_text[line_start:]
    return "\n".join(head_lines) + "\n" + base_text


# ---------------------------------------------------------------------------
# Live-progress plumbing — spinner on TTY, plain lines on non-TTY, silent
# when --no-progress or no --llm. The callback is fed to
# ``perfxpert.api.agent_root`` which forwards it through the agents
# runtime (``AnalysisSession`` + ``_cascade``) so every phase transition
# is surfaced to the user without tight coupling to Rich or any UI lib.
# ---------------------------------------------------------------------------


def _progress_context(*, enable_llm: bool, no_progress: bool, verbose: bool):
    """Return ``(progress_cb, contextmanager)`` — the callback fed to
    ``api.agent_root`` plus the CM that owns the spinner / log-line UI.

    The CM is always-safe: pushing ``None`` as the callback means zero
    overhead on the agents hot path (cascade + phase emit short-circuit
    when no callback is set).

    Rules (per the Phase 8 design):

    * No ``--llm`` → silent (airgap path, nothing to surface).
    * ``--no-progress`` → silent even with ``--llm``.
    * ``--verbose`` → silent (verbose logging already narrates).
    * stderr is a TTY → Rich ``Live`` spinner on stderr, transient=True.
    * stderr is not a TTY → plain ``[perfxpert] <phase>`` on stderr.

    If Rich is not installed the TTY branch falls back to plain lines
    (same as non-TTY) so core install still works.
    """
    import contextlib

    # Silent modes — callback is None, zero overhead.
    if not enable_llm or no_progress or verbose:
        @contextlib.contextmanager
        def _silent():
            yield
        return None, _silent()

    # stderr decides whether we draw a spinner or plain lines. The
    # progress stream belongs on stderr so piping stdout (JSON / HTML)
    # remains clean.
    stderr_is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    def _plain_callback(msg: str) -> None:
        print(f"[perfxpert] {msg}", file=sys.stderr, flush=True)

    if not stderr_is_tty:
        @contextlib.contextmanager
        def _plain():
            yield
        return _plain_callback, _plain()

    # TTY path — try to build a Rich spinner; fall back to plain lines
    # if Rich isn't installed. We deliberately DON'T require rich for
    # core install.
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.spinner import Spinner
        from rich.text import Text
    except ImportError:
        @contextlib.contextmanager
        def _plain_tty():
            yield
        return _plain_callback, _plain_tty()

    console = Console(stderr=True)
    current_phase = Text("waiting on agent", style="cyan")
    spinner = Spinner("dots", text=current_phase)

    def _rich_callback(msg: str) -> None:
        current_phase.plain = msg
        current_phase.style = "cyan"

    @contextlib.contextmanager
    def _rich_ctx():
        with Live(spinner, console=console, refresh_per_second=8, transient=True):
            yield

    return _rich_callback, _rich_ctx()


# -- credential helpers (Bug 1 + Bug 3) ------------------------------------
#
# These helpers live alongside ``_execute_agentic`` so the CLI auth
# story is visible in one place: pre-flight check first, then an env-vs-
# flag mismatch warning, then the actual call. The env var map mirrors
# ``perfxpert.agents.runtime._PROVIDER_CANONICAL_ENV``; we duplicate it
# here so the CLI layer can emit per-provider help text without pulling
# runtime into import time.


# Per-provider credential source-of-truth. Each entry lists the env vars
# that already unlock the provider (used for pre-flight existence) and a
# one-line hint that names the minimum required flag or env var. Keep
# in lock-step with providers/*.py and agents/runtime.py.
_PROVIDER_CREDENTIALS = {
    "anthropic": {
        "env_vars": ("ANTHROPIC_API_KEY", "PERFXPERT_LLM_ANTHROPIC_KEY"),
        "hint": (
            "no API key — pass --llm-api-key sk-ant-… or export ANTHROPIC_API_KEY"
        ),
    },
    "openai": {
        "env_vars": ("OPENAI_API_KEY", "PERFXPERT_LLM_OPENAI_KEY"),
        "hint": (
            "no API key — pass --llm-api-key sk-… or export OPENAI_API_KEY"
        ),
    },
    "private": {
        "env_vars": ("PERFXPERT_LLM_PRIVATE_API_KEY",),
        "hint": (
            "private provider requires PERFXPERT_LLM_PRIVATE_URL "
            "(or PRIVATE_LLM_ENDPOINT) plus PERFXPERT_LLM_PRIVATE_API_KEY "
            "(or --llm-api-key <key>)"
        ),
        "required_env_any": (("PERFXPERT_LLM_PRIVATE_URL", "PRIVATE_LLM_ENDPOINT"),),
    },
    "ollama": {
        "env_vars": (),  # Ollama needs no key; URL is the credential.
        "hint": (
            "ollama provider requires a running daemon reachable via "
            "PERFXPERT_LLM_LOCAL_URL (default http://localhost:11434)"
        ),
        "required_env": (),  # URL has an in-provider default.
    },
    "opencode": {
        "env_vars": (),
        "hint": (
            "opencode provider requires the bundled patched CLI produced by "
            "the perfxpert install/build hook"
        ),
        "required_env": (),
    },
}


def _preflight_provider_auth(provider: str, flag_api_key: Optional[str]) -> None:
    """Raise :class:`AuthError` when the selected provider lacks a credential.

    Runs before ``agent_root`` so a misconfiguration surfaces as a clean
    one-line error on stderr with no half-written output files. Each
    branch names the exact flag / env var the user needs to set.
    """
    from perfxpert.providers._exceptions import AuthError

    info = _PROVIDER_CREDENTIALS.get(provider)
    if info is None:
        # Unknown provider name — ``build_session`` already validates
        # against the registry. Skip pre-flight so we don't mask the
        # clearer ValueError downstream.
        return

    # A flag-supplied key satisfies providers that take a key.
    if flag_api_key and info.get("env_vars"):
        _require_additional_env(provider, info)
        return

    # An existing env var satisfies key-bearing providers.
    for var in info.get("env_vars", ()):
        if os.environ.get(var):
            _require_additional_env(provider, info)
            return

    # Providers that don't need a key (ollama, opencode) still need
    # their connection info.
    if not info.get("env_vars"):
        _require_additional_env(provider, info)
        return

    raise AuthError(provider, info["hint"])


def _require_additional_env(provider: str, info: Dict[str, Any]) -> None:
    """Verify any additional env requirements for ``provider`` are set."""
    from perfxpert.providers._exceptions import AuthError

    for var in info.get("required_env", ()) or ():
        if not os.environ.get(var):
            raise AuthError(provider, info["hint"])
    for choices in info.get("required_env_any", ()) or ():
        if not any(os.environ.get(var) for var in choices):
            raise AuthError(provider, info["hint"])


def _warn_if_flag_overrides_env(provider: str, flag_api_key: str) -> None:
    """Emit a stderr WARNING when ``--llm-api-key`` disagrees with env.

    The CLI flag always wins; this warning tells the user which key is
    active so a mismatched ``ANTHROPIC_API_KEY`` doesn't silently affect
    unrelated runs. No-op when the env var is unset or identical.
    """
    info = _PROVIDER_CREDENTIALS.get(provider)
    if not info:
        return
    for var in info.get("env_vars", ()) or ():
        env_val = os.environ.get(var)
        if env_val and env_val != flag_api_key:
            print(
                f"⚠ --llm-api-key overrides {var} (env value ignored for "
                f"this run)",
                file=sys.stderr,
            )
            return


# -- known kwargs accepted by `_execute_agentic` ---------------------------
# Any kwarg not in this set that is forwarded from `execute()` will emit a
# WARNING so future argparse additions cannot silently drop through the
# agentic pipeline (cycle-2 I-1 regression guard).
_KNOWN_EXECUTE_KWARGS = frozenset({
    # Output routing
    "format",
    "output_format",
    "output_file",
    "output_path",
    # LLM provider wiring
    "enable_llm",
    "llm_provider",
    "llm_api_key",
    "llm_model",
    "llm_thinking",
    "llm_local",
    "llm_local_model",
    # Analysis options forwarded through RootInput.analysis_options
    "source_dir",
    "att_dir",
    "prompt",
    "custom_prompt",  # historical alias for prompt
    "top_kernels",
    "min_duration",
    # Execution flags
    "verbose",
    "no_progress",
    # ``perfxpert analyze --baseline <db>`` splice (Confluence row #7).
    # The baseline DB is diffed via ``trace_diff.diff_runs`` after the
    # agentic Root emits its report — the path here is plumbed through
    # so ``_execute_agentic`` does not emit a RuntimeWarning about
    # unused kwargs.
    "baseline_db",
    # ``--advanced`` (Phase 10): include advanced specialist recs
    # (currently pragma loop-hint advice) in the report. When False
    # pragma recs are filtered out before formatting. Also honours
    # the ``PERFXPERT_ADVANCED_RECS=1`` env var.
    "advanced",
})


def _execute_agentic(
    input: Optional[RocpdImportData],
    config: Optional[output_config.output_config] = None,
    **kwargs: Any,
) -> Optional[RocpdImportData]:
    """Agentic path: delegates to :func:`perfxpert.api.agent_root`.

    The CLI routes through the public Python API (which in turn wraps
    the MCP-exposed Root tool) so batch CLI, library API, and MCP
    server share a single entry point. The airgap + provider +
    fallback-chain semantics are preserved because ``agent_root``
    defers to ``agents.runtime.build_session``.
    """
    try:
        from perfxpert import api as perfxpert_api  # 1:1 mirror of agent MCP tools
    except ImportError as e:
        raise RuntimeError(
            "perfxpert.api is not available. "
            "perfxpert.tools.agents must be importable for the agentic path."
        ) from e

    # Guard rail against silent kwarg drop — any new CLI flag that isn't
    # wired here surfaces a WARNING instead of being ignored (I-1).
    _unused = set(kwargs) - _KNOWN_EXECUTE_KWARGS
    if _unused:
        import warnings
        warnings.warn(
            f"perfxpert.analyze: unused kwargs ignored by agentic runtime: "
            f"{sorted(_unused)}. Wire them in _execute_agentic or drop the "
            f"corresponding --flag.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Update config if provided
    if config is not None:
        config = config.update(**kwargs)
    else:
        config = output_config.output_config(**kwargs)

    # Get database path for display
    database_path = ""
    if input is not None and hasattr(input, "_paths") and input._paths:
        database_path = str(
            input._paths[0] if isinstance(input._paths, list) else input._paths
        )

    # Get source_dir if provided (for Tier 0 analysis)
    source_dir = kwargs.get("source_dir")

    # Get custom prompt if provided. CLI emits `prompt` (argparse dest);
    # accept `custom_prompt` as a back-compat alias for library callers.
    custom_prompt = kwargs.get("prompt") or kwargs.get("custom_prompt")

    requested_llm = bool(kwargs.get("enable_llm", False))
    llm_provider = kwargs.get("llm_provider")
    llm_api_key = kwargs.get("llm_api_key")
    no_progress = bool(kwargs.get("no_progress", False))
    verbose = bool(kwargs.get("verbose", False))
    att_dir = kwargs.get("att_dir")
    top_kernels = int(kwargs.get("top_kernels") or 10)
    min_duration = float(kwargs.get("min_duration") or 0.0)
    analysis_options = {
        key: value
        for key, value in {
            "top_kernels": top_kernels,
            "att_dir": att_dir,
            "min_duration": min_duration,
            "llm_model": kwargs.get("llm_model"),
            "llm_thinking": kwargs.get("llm_thinking"),
            "llm_local": kwargs.get("llm_local"),
            "llm_local_model": kwargs.get("llm_local_model"),
            "verbose": verbose if verbose else None,
        }.items()
        if value is not None
    }
    env_forces_airgap = os.environ.get("PERFXPERT_AIRGAP", "0") == "1"
    effective_airgap = env_forces_airgap or (not requested_llm)
    llm_active = requested_llm and (not effective_airgap)
    effective_provider = None if effective_airgap else llm_provider
    if effective_provider == "claude-code":
        effective_provider = "opencode"
    effective_api_key = None if effective_airgap else llm_api_key

    # Bug 3 — pre-flight auth check. Surface a clean ``AuthError`` BEFORE
    # building the session + making any network call when the selected
    # provider has no usable credential. Airgap + disabled LLM skip this
    # check so deterministic runs remain credential-free.
    if effective_provider:
        _preflight_provider_auth(effective_provider, effective_api_key)

    # Bug 1 — if BOTH ``--llm-api-key`` and the provider's canonical env
    # var are set and differ, the CLI flag wins (the session env override
    # in ``build_session`` handles the actual injection). Emit a one-line
    # WARNING on stderr so the user knows which credential is active.
    if effective_provider and effective_api_key:
        _warn_if_flag_overrides_env(effective_provider, effective_api_key)

    # Build progress feedback (spinner / plain lines / silent) from the
    # effective LLM state, not the raw CLI request. ``PERFXPERT_AIRGAP=1``
    # must fully disable LLM-only UX even when the user also passed
    # ``--llm``.
    progress_cb, progress_cm = _progress_context(
        enable_llm=llm_active,
        no_progress=no_progress,
        verbose=verbose,
    )

    # Tier-0 (source-only) path doesn't run through the agentic Root
    # today — emit one status line before the scan and one after when
    # it's the only work happening. agent_root below still runs for the
    # combined -i + --source-dir path.
    tier0_only = input is None and source_dir and progress_cb is not None
    if tier0_only:
        import time as _time
        _t0 = _time.monotonic()
    else:
        _t0 = None

    # Route the agentic path through the public Python API — same
    # function the MCP server wraps as ``agent_root``.
    #
    # Let typed ProviderError subclasses (QuotaExceeded, AuthError,
    # RateLimitError, TransientError, FatalError) propagate unchanged so
    # the outermost CLI boundary in ``main()`` can render a one-line
    # user-facing message instead of a 30-line traceback. Only non-taxonomy
    # exceptions (schema / wiring bugs, our own code) get wrapped as
    # RuntimeError with the "Agentic root analysis failed" diagnostic.
    from perfxpert.providers._exceptions import ProviderError
    try:
        with progress_cm:
            root_output = perfxpert_api.agent_root(
                user_query=custom_prompt or "Analyze this GPU performance trace.",
                database_path=database_path if input else None,
                source_dir=source_dir,
                provider=effective_provider,
                airgap=effective_airgap,
                analysis_options=analysis_options,
                progress_callback=progress_cb,
                api_key=effective_api_key,
            )
    except ProviderError:
        raise  # let __main__.main render clean one-liner
    except Exception as e:
        raise RuntimeError(f"Agentic root analysis failed: {e}") from e

    # Tier-0 timing emit — only if the scan was the primary work AND it
    # took > 500 ms (per the phase-8 design).
    if tier0_only and progress_cb is not None and _t0 is not None:
        import time as _time
        if (_time.monotonic() - _t0) > 0.5:
            progress_cb("Source scan done")

    # Deterministic analysis pass — runs in parallel with the LLM brain so
    # every format populates the full contract (time_breakdown / hotspots /
    # memory_analysis / hardware_counters / tier-0 / …) regardless of
    # whether an LLM is in the loop. The airgap path takes this path too;
    # only the LLM narrative is skipped under airgap.
    from perfxpert.analysis.payload import build_analysis_payload as _build_payload
    try:
        analysis_payload = _build_payload(
            input,
            source_dir=source_dir,
            att_dir=att_dir,
            top_kernels=top_kernels,
            min_duration=min_duration,
            progress_callback=progress_cb,
        )
    except Exception as _payload_exc:
        # Never fail analysis for a deterministic-pass hiccup — degrade
        # gracefully by emitting an empty payload the formatters can
        # still render a skeleton from.
        import warnings as _warnings
        _warnings.warn(
            f"perfxpert.analyze: deterministic pass failed ({_payload_exc}); "
            "formats will render the LLM-only skeleton.",
            RuntimeWarning,
            stacklevel=2,
        )
        analysis_payload = {
            "database_path": database_path,
            "time_breakdown": {},
            "hotspots": [],
            "memory_analysis": {},
            "hardware_counters": {"has_counters": False, "metrics": {}, "counters": {}},
            "kernel_resources": {},
            "api_overhead": {},
            "warmup_issues": {},
            "thread_trace": None,
            "tier0_findings": None,
            "recommendations_deterministic": [],
            "metadata": {},
        }

    # Format output according to requested format.
    #
    # The legacy formatters (_format_as_markdown / _format_as_webview)
    # produce AMD-themed HTML and structured Markdown with headings.
    # Wire them to the agentic RootOutput so `--format markdown` emits
    # real Markdown (not raw narrative prose) and `--format webview`
    # emits a real HTML report (not a plaintext narrative).
    output_format = kwargs.get("output_format") or kwargs.get("format", "text")
    advanced_effective = _resolve_advanced_flag(kwargs.get("advanced"))
    output = _format_agentic_output(
        root_output,
        output_format,
        database_path=database_path,
        analysis_payload=analysis_payload,
        advanced=advanced_effective,
    )

    # ``--baseline <db>`` splice (Confluence row #7 end-of-session recap).
    # When the caller supplies a baseline, append / splice a "Changed vs
    # baseline" section so every analyze report can report back the
    # effect of changes without a second command.
    baseline_db = kwargs.get("baseline_db")
    if baseline_db and database_path and os.path.exists(baseline_db):
        output = _splice_baseline_diff(
            output,
            baseline_db=baseline_db,
            new_db=database_path,
            output_format=output_format,
            top_kernels=top_kernels,
        )

    # Handle output writing
    _ext_map = {"json": ".json", "markdown": ".md", "webview": ".html", "text": ".txt"}
    _ext = _ext_map.get(output_format, ".txt")

    if config and config.output_file == "-":
        print(output)
        return input

    if config and output_format != "text":
        if not config.output_path:
            config.output_path = "."
        if not config.output_file:
            if database_path:
                config.output_file = os.path.splitext(os.path.basename(database_path))[0]
            else:
                config.output_file = "analysis"
    elif config and config.output_file and not config.output_path:
        config.output_path = "."
    elif config and config.output_path and not config.output_file:
        if database_path:
            config.output_file = os.path.splitext(os.path.basename(database_path))[0]
        else:
            config.output_file = "analysis"

    if config and config.output_file and config.output_path:
        base = config.output_file
        if not base.endswith(_ext):
            base = base + _ext
        output_file = os.path.join(config.output_path, base)
        os.makedirs(config.output_path, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Analysis written to: {output_file}", file=sys.stderr)
        if output_format == "text":
            print(
                "Tip: use --format webview for an interactive HTML report, "
                "--format json for machine-readable output, "
                "or --format markdown for Markdown.",
                file=sys.stderr,
            )
    else:
        print(output)

    return input


def main(argv=None) -> int:
    """
    Main entry point for standalone execution.

    Args:
        argv: Command-line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        prog="rocpd.analyze",
        description="AI-powered performance analysis for GPU traces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=str,
        required=True,
        help="Input rocpd database file(s)",
    )

    # Add output config args
    output_config.add_args(parser)

    # Add analysis args
    process_args = add_args(parser)

    # Parse arguments
    args = parser.parse_args(argv)

    try:
        # Create database connection
        input_data = RocpdImportData(args.input)

        # Process arguments
        analysis_args = process_args(input_data, args)

        # Execute analysis
        execute(input_data, **analysis_args)

        return 0

    except Exception as e:
        # Bug 2/3 — ProviderError (AuthError / FatalError / …) is the
        # "user misconfigured something" path. Delete any half-written
        # output file the formatter may have touched, and return rc=2
        # to distinguish from unrelated failures (rc=1).
        from perfxpert.providers._exceptions import ProviderError
        if isinstance(e, ProviderError):
            _cleanup_empty_output(args)
            _render_cli_error(e)
            return 2
        return _render_cli_error(e)


def _cleanup_empty_output(args: argparse.Namespace) -> None:
    """Delete any zero-byte output file that may have been created mid-flow.

    When the agentic pipeline raises a ProviderError after the
    formatter has opened the output file but before writing, the file
    lives on disk as an empty HTML / markdown blob. That was the
    original bug symptom. Clean up defensively so the user never finds
    an empty report alongside a clean stderr error message.

    Scans the output directory for files matching the selected format
    extension and removes the ones that are zero bytes. The exact
    filename depends on the input db stem (e.g. ``890189_results.html``)
    which the CLI computes deep inside ``_execute_agentic`` — pruning
    by extension + size avoids threading that detail back out through
    the exception handler.
    """
    output_path = getattr(args, "output_path", None) or getattr(args, "output_dir", None)
    if not output_path or not os.path.isdir(output_path):
        return
    ext_map = {"json": ".json", "markdown": ".md", "webview": ".html", "text": ".txt"}
    ext = ext_map.get(getattr(args, "output_format", "text"), ".txt")
    try:
        for entry in os.listdir(output_path):
            if not entry.endswith(ext):
                continue
            full = os.path.join(output_path, entry)
            try:
                if os.path.isfile(full) and os.path.getsize(full) == 0:
                    os.remove(full)
            except OSError:
                pass  # best-effort cleanup; don't mask the original error
    except OSError:
        pass


def _render_cli_error(exc: BaseException) -> int:
    """Render a top-level CLI error as a one-line user message.

    Typed :class:`ProviderError` subclasses get a concise, actionable
    message. Anything else falls back to the short ``Error: ...`` line.
    The full traceback is only printed when ``PERFXPERT_DEBUG=1`` is set
    so interactive users don't see 30 lines of stack trace.
    """
    from perfxpert.providers._exceptions import (
        AuthError,
        FatalError,
        ProviderChainExhausted,
        QuotaExceededError,
        RateLimitError,
        TimeoutError,
        TransientError,
    )

    debug = os.environ.get("PERFXPERT_DEBUG", "0") == "1"

    if isinstance(exc, QuotaExceededError):
        prov = exc.provider
        model = exc.model or "<default>"
        raw = getattr(exc, "raw_message", "") or ""
        print(
            f"⚠ LLM quota exhausted on {prov} ({model}). "
            f"Top up the account or switch provider: "
            f"PERFXPERT_AIRGAP=1 OR --llm <other>. "
            f"Raw SDK message: {raw}",
            file=sys.stderr,
        )
    elif isinstance(exc, AuthError):
        prov = getattr(exc, "provider", "<unknown>")
        env_var = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "ollama": "PERFXPERT_LLM_LOCAL_URL",
            "private": (
                "PERFXPERT_LLM_PRIVATE_URL or PRIVATE_LLM_ENDPOINT, plus "
                "PERFXPERT_LLM_PRIVATE_API_KEY or --llm-api-key"
            ),
            "opencode": "the bundled patched opencode build",
            "claude-code": "the bundled patched opencode build",
        }.get(prov, f"{prov.upper()}_API_KEY")
        print(
            f"⚠ LLM auth failed for {prov}. "
            f"Check {env_var} is set correctly.",
            file=sys.stderr,
        )
    elif isinstance(exc, RateLimitError):
        prov = getattr(exc, "provider", "<unknown>")
        print(
            f"⚠ LLM rate-limited on {prov}; retry in a minute, or set "
            f"PERFXPERT_LLM_FALLBACK_CHAIN to cascade providers.",
            file=sys.stderr,
        )
    elif isinstance(exc, TimeoutError):
        prov = getattr(exc, "provider", "<unknown>")
        timeout_seconds = getattr(exc, "timeout_seconds", 0.0)
        timeout_detail = f" after {timeout_seconds}s" if timeout_seconds else ""
        print(
            f"⚠ LLM provider {prov} timed out{timeout_detail}; "
            f"retry, switch provider, or PERFXPERT_AIRGAP=1.",
            file=sys.stderr,
        )
    elif isinstance(exc, TransientError):
        prov = getattr(exc, "provider", "<unknown>")
        kind = getattr(exc, "kind", "transient") or "transient"
        print(
            f"⚠ LLM provider {prov} returned a transient error ({kind}); "
            f"retry, or PERFXPERT_AIRGAP=1 for deterministic fallback.",
            file=sys.stderr,
        )
    elif isinstance(exc, FatalError):
        prov = getattr(exc, "provider", "<unknown>")
        raw = getattr(exc, "raw_message", "") or str(exc)
        print(f"⚠ LLM provider {prov} failed: {raw}", file=sys.stderr)
    elif isinstance(exc, ProviderChainExhausted):
        attempted = " -> ".join(getattr(exc, "providers", ()) or ("<none>",))
        last_error = getattr(exc, "last_error", None)
        suffix = f" Last error: {last_error}" if last_error else ""
        print(
            f"⚠ LLM fallback chain exhausted after {attempted}.{suffix}",
            file=sys.stderr,
        )
    else:
        print(f"Error: {exc}", file=sys.stderr)

    if debug:
        import traceback
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)

    return 1


if __name__ == "__main__":
    sys.exit(main())
