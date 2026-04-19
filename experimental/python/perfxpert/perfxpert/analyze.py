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
    _PERFXPERT_VERSION = "0.1.0"  # fallback if metadata not available (common in dev / ROCm system installs)

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
        "Enable natural language explanations via Anthropic Claude or OpenAI GPT. "
        "Requires API key - see https://console.anthropic.com/ or https://platform.openai.com/api-keys",
    )

    llm_options.add_argument(
        "--llm",
        type=str,
        dest="llm_provider",
        choices=["anthropic", "openai", "claude-code"],
        default=None,
        help=(
            "Enable LLM-powered analysis enhancement. "
            "'anthropic' uses the Anthropic API (requires ANTHROPIC_API_KEY). "
            "'openai' uses the OpenAI API (requires OPENAI_API_KEY). "
            "'claude-code' uses the Claude Code CLI installed on this machine — "
            "no API key needed, uses existing Claude Code credentials. "
            "Local analysis always runs first; LLM provides additional natural language insights."
        ),
    )

    llm_options.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="API key for LLM provider. Alternatively, set environment variable: "
        "ANTHROPIC_API_KEY for Anthropic Claude, or OPENAI_API_KEY for OpenAI GPT. "
        "Example: --llm anthropic --llm-api-key sk-ant-... "
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
    """Agentic path: delegates to the agents session API.

    Builds an AnalysisSession, invokes run_root with a RootInput,
    and formats the output according to the requested format.
    """
    try:
        from perfxpert.agents import runtime, schemas
    except ImportError as e:
        raise RuntimeError(
            "Agent runtime is not available. "
            "perfxpert.agents must be importable for the agentic path."
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

    # Build session
    enable_llm = kwargs.get("enable_llm", False)
    llm_provider = kwargs.get("llm_provider")
    session = runtime.build_session(
        provider=llm_provider if enable_llm else None,
        airgap=(not enable_llm),
    )

    # Collect downstream analysis options as a side-channel dict on
    # RootInput so specialised agents (Analysis et al.) can read them
    # without exploding the schema field count.
    analysis_options: Dict[str, Any] = {}
    for key in ("top_kernels", "att_dir", "min_duration", "llm_api_key",
                "llm_model", "llm_thinking", "llm_local", "llm_local_model",
                "verbose"):
        val = kwargs.get(key)
        if val is not None:
            analysis_options[key] = val

    # Build RootInput payload
    try:
        root_input = schemas.RootInput(
            user_query=custom_prompt or "Analyze this GPU performance trace.",
            database_path=database_path if input else None,
            source_dir=source_dir,
            provider=llm_provider if enable_llm else None,
            airgap=(not enable_llm),
            session_id=session.session_id,
            analysis_options=analysis_options,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to build RootInput: {e}") from e

    # Run root analysis via the agents session API
    try:
        root_output = session.run_root(root_input)
    except Exception as e:
        raise RuntimeError(f"Agentic root analysis failed: {e}") from e

    # Format output according to requested format
    output_format = kwargs.get("output_format", "text")
    if output_format == "json":
        import json
        output = json.dumps(
            {
                "narrative": root_output.narrative,
                "recommendations": root_output.recommendations,
                "primary_bottleneck": root_output.primary_bottleneck,
                "warnings": root_output.warnings,
                "metadata": root_output.metadata,
            },
            indent=2,
        )
    else:
        # text, markdown, webview — for now, just use the narrative
        output = root_output.narrative

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
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
