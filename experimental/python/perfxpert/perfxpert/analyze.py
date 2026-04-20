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
from typing import Any, Dict, Optional

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
        "File extension is set automatically: .txt, .json, .md, .html",
    )

    analysis_options.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Minimum kernel duration threshold in microseconds (filter out short kernels)",
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
        choices=["anthropic", "openai", "ollama", "private", "opencode"],
        default=None,
        help=(
            "Enable LLM-powered analysis enhancement. Choose one of: "
            "'anthropic' (ANTHROPIC_API_KEY), 'openai' (OPENAI_API_KEY), "
            "'ollama' (local daemon, PERFXPERT_LLM_LOCAL_URL), "
            "'private' (self-hosted OpenAI-compatible endpoint, PERFXPERT_LLM_PRIVATE_URL + _API_KEY), "
            "'opencode' (bundled opencode CLI, PERFXPERT_OPENCODE_PATH). "
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


def _format_agentic_output(
    root_output: Any,
    output_format: str,
    *,
    database_path: str = "",
) -> str:
    """Render a :class:`RootOutput` (or its ``model_dump`` dict) via the
    shared formatters.

    The legacy formatters expect a rich dict of analysis metrics
    (``time_breakdown``, ``hotspots``, ``memory_analysis``, …). The
    agentic pipeline today only passes the narrative + recommendations
    up through Root, so we construct a *minimal* analysis dict: it has
    all the keys the formatters check for (so they don't KeyError) but
    the kernel / counter / memcpy sections are intentionally empty,
    letting the formatters short-circuit those sections cleanly.

    Downstream work will lift ``time_breakdown`` + ``hotspots`` from the
    Analysis agent's output up through Root's metadata so the report is
    richer; until then we render the narrative inside a proper Markdown /
    HTML skeleton with a recommendations table.

    ``root_output`` may be either:
      - a :class:`perfxpert.agents.schemas.RootOutput` (Pydantic model),
        in which case attributes are read via ``getattr``; or
      - a plain ``dict`` (e.g. the return value of
        :func:`perfxpert.api.agent_root`), in which case the same five
        keys are read via ``dict.get``.
    """
    def _read(name: str, default: Any) -> Any:
        if isinstance(root_output, dict):
            return root_output.get(name, default)
        return getattr(root_output, name, default)

    narrative = _read("narrative", "") or ""
    recommendations = list(_read("recommendations", []) or [])
    primary_bottleneck = _read("primary_bottleneck", "mixed") or "mixed"
    warnings = list(_read("warnings", []) or [])
    metadata = dict(_read("metadata", {}) or {})

    if output_format == "json":
        import json as _json
        return _json.dumps(
            {
                "narrative": narrative,
                "recommendations": recommendations,
                "primary_bottleneck": primary_bottleneck,
                "warnings": warnings,
                "metadata": metadata,
            },
            indent=2,
        )

    # The Analysis agent's time_breakdown could flow up through Root.metadata
    # if the pipeline is wired to do so. Default to an empty-but-shaped dict
    # so the formatters emit their skeleton without failing.
    time_breakdown = metadata.get("time_breakdown") or {}
    hotspots = metadata.get("hotspots") or []
    memory_analysis = metadata.get("memory_analysis") or {}
    hardware_counters = metadata.get("hardware_counters") or {}

    # Normalise recommendations to the shape the formatters expect.
    # Agentic recs carry ``type`` / ``target`` / ``summary`` — map them
    # onto the legacy ``category`` / ``issue`` / ``suggestion`` keys so
    # the rendered report reads sensibly.
    normalised_recs: list = []
    for rec in recommendations:
        if not isinstance(rec, dict):
            continue
        r = dict(rec)
        r.setdefault("priority", "INFO")
        r.setdefault("category", r.get("type", "analysis"))
        r.setdefault("issue", r.get("summary", ""))
        r.setdefault("suggestion", r.get("summary", ""))
        normalised_recs.append(r)

    if output_format == "markdown":
        md = _format_as_markdown(
            time_breakdown=time_breakdown,
            hotspots=hotspots,
            memory_analysis=memory_analysis,
            recommendations=normalised_recs,
            hardware_counters=hardware_counters,
            database_path=database_path,
        )
        # Inject the LLM / airgap narrative between the title and the
        # metric sections so the report reads as a cohesive document.
        if narrative:
            narrative_block = "\n## Summary\n\n" + narrative.rstrip() + "\n"
            # Splice after the first blank line following the H1 title.
            parts = md.split("\n", 3)
            if len(parts) >= 2 and parts[0].startswith("# "):
                md = parts[0] + "\n" + (parts[1] + "\n" if len(parts) > 1 else "") + \
                     narrative_block + "\n".join(parts[2:])
            else:
                md = narrative_block + "\n" + md
        if primary_bottleneck:
            md += f"\n\n*Primary bottleneck:* **{primary_bottleneck}**\n"
        if warnings:
            md += "\n\n## Warnings\n\n"
            for w in warnings:
                md += f"- {w}\n"
        return md

    if output_format == "webview":
        html = _format_as_webview(
            time_breakdown=time_breakdown,
            hotspots=hotspots,
            memory_analysis=memory_analysis,
            recommendations=normalised_recs,
            hardware_counters=hardware_counters,
            database_path=database_path,
        )
        # The template places a narrative only if summary / findings land
        # in _build_summary; for a pure agentic RootOutput we splice the
        # narrative into a prose panel so the HTML isn't empty of text.
        if narrative:
            import html as _html_mod
            narrative_panel = (
                '<section class="card" id="agentic-narrative">'
                '<h2>Summary</h2>'
                f'<pre style="white-space:pre-wrap;">{_html_mod.escape(narrative)}</pre>'
                '</section>'
            )
            # Inject right before </main> if present, else before </body>.
            if "</main>" in html:
                html = html.replace("</main>", narrative_panel + "</main>", 1)
            elif "</body>" in html:
                html = html.replace("</body>", narrative_panel + "</body>", 1)
            else:
                html = html + "\n" + narrative_panel
        return html

    # text: structured plaintext with section separators — NOT raw narrative.
    width = 80
    lines = []
    lines.append("=" * width)
    lines.append("PERFXPERT ANALYSIS".center(width))
    lines.append("=" * width)
    if database_path:
        lines.append(f"Database: {database_path}")
    lines.append(f"Primary bottleneck: {primary_bottleneck}")
    lines.append("")
    lines.append("== Summary ==")
    lines.append("")
    lines.append(narrative.rstrip() or "(no narrative — airgap / empty response)")
    lines.append("")
    if normalised_recs:
        lines.append("== Recommendations ==")
        lines.append("")
        for i, rec in enumerate(normalised_recs, 1):
            prio = rec.get("priority", "INFO")
            cat = rec.get("category", "")
            issue = rec.get("issue", "") or rec.get("summary", "")
            lines.append(f"  {i}. [{prio}] {cat}")
            if issue:
                lines.append(f"     * {issue}")
            suggestion = rec.get("suggestion", "")
            if suggestion and suggestion != issue:
                lines.append(f"     * suggestion: {suggestion}")
        lines.append("")
    if warnings:
        lines.append("== Warnings ==")
        lines.append("")
        for w in warnings:
            lines.append(f"  * {w}")
        lines.append("")
    lines.append("=" * width)
    lines.append("Analysis complete.".center(width))
    lines.append("=" * width)
    return "\n".join(lines)


# -- known kwargs accepted by `_execute_agentic` ---------------------------
# Any kwarg not in this set that is forwarded from `execute()` will emit a
# WARNING so future argparse additions cannot silently drop through the
# agentic pipeline (cycle-2 I-1 regression guard).
_KNOWN_EXECUTE_KWARGS = frozenset({
    # Output routing
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

    enable_llm = kwargs.get("enable_llm", False)
    llm_provider = kwargs.get("llm_provider")

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
        root_output = perfxpert_api.agent_root(
            user_query=custom_prompt or "Analyze this GPU performance trace.",
            database_path=database_path if input else None,
            source_dir=source_dir,
            provider=llm_provider if enable_llm else None,
            airgap=(not enable_llm),
        )
    except ProviderError:
        raise  # let __main__.main render clean one-liner
    except Exception as e:
        raise RuntimeError(f"Agentic root analysis failed: {e}") from e

    # Format output according to requested format.
    #
    # The legacy formatters (_format_as_markdown / _format_as_webview)
    # produce AMD-themed HTML and structured Markdown with headings.
    # Wire them to the agentic RootOutput so `--format markdown` emits
    # real Markdown (not raw narrative prose) and `--format webview`
    # emits a real HTML report (not a plaintext narrative).
    output_format = kwargs.get("output_format", "text")
    output = _format_agentic_output(root_output, output_format, database_path=database_path)

    # Handle output writing
    _ext_map = {"json": ".json", "markdown": ".md", "webview": ".html", "text": ".txt"}
    _ext = _ext_map.get(output_format, ".txt")

    if config and config.output_path and not config.output_file:
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
        with open(output_file, "w") as f:
            f.write(output)
        print(f"Analysis written to: {output_file}")
        if output_format == "text":
            print(
                "Tip: use --format webview for an interactive HTML report, "
                "--format json for machine-readable output, "
                "or --format markdown for Markdown."
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
        return _render_cli_error(e)


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
        QuotaExceededError,
        RateLimitError,
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
            "private": "PERFXPERT_LLM_PRIVATE_API_KEY",
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
    else:
        print(f"Error: {exc}", file=sys.stderr)

    if debug:
        import traceback
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)

    return 1


if __name__ == "__main__":
    sys.exit(main())
