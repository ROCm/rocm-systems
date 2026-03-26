#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
Standalone CLI entry point for rocm-ai-analysis.

Provides the ``rocm-ai-analyze`` command that wraps the public API in
:mod:`rocm_ai_analysis.api`.

Usage:
    rocm-ai-analyze output.db
    rocm-ai-analyze output.db --format webview --llm anthropic
    rocm-ai-analyze --source-dir ./my_app/src
    rocm-ai-analyze output.db -I "./my_gpu_app --batch 64" --llm anthropic
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with all supported options."""

    parser = argparse.ArgumentParser(
        prog="rocm-ai-analyze",
        description="AI-powered performance analysis for AMD GPU traces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Positional arguments ------------------------------------------------

    parser.add_argument(
        "database",
        nargs="*",
        type=str,
        help=(
            "Input rocpd database file(s). "
            "Not required when using --source-dir for source-only analysis."
        ),
    )

    # -- Analysis options ----------------------------------------------------

    analysis = parser.add_argument_group("Analysis options")

    analysis.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "markdown", "webview"],
        default="text",
        dest="format",
        help=(
            "Output format: text, json, markdown, or webview (self-contained HTML). "
            "File extension is set automatically when --output is used: "
            ".txt, .json, .md, .html"
        ),
    )

    analysis.add_argument(
        "--top-kernels",
        type=int,
        default=10,
        dest="top_kernels",
        help="Number of top kernels to analyze",
    )

    analysis.add_argument(
        "--source-dir",
        type=str,
        default=None,
        dest="source_dir",
        help=(
            "Path to GPU application source directory for Tier 0 static analysis. "
            "Scans .hip/.cpp/.cu files and generates a profiling plan. "
            "Can be used alone (no database required) or alongside a database "
            "for combined analysis."
        ),
    )

    analysis.add_argument(
        "--att-dir",
        type=str,
        default=None,
        dest="att_dir",
        help=(
            "Path to directory containing ATT stats_*.csv files from "
            "rocprofv3 --att. Enables Tier 3 Advanced Thread Trace analysis: "
            "per-instruction stall ratios and bottleneck classification."
        ),
    )

    analysis.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Custom analysis prompt/question to guide analysis "
            "(e.g., 'Why is my matmul kernel slow?')"
        ),
    )

    analysis.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        dest="min_duration",
        help="Minimum kernel duration threshold in microseconds (filter out short kernels)",
    )

    analysis.add_argument(
        "--interactive",
        "-I",
        metavar="RUN_COMMAND",
        type=str,
        default=None,
        dest="interactive",
        help=(
            "Launch the 7-phase interactive profiling + optimization workflow. "
            "RUN_COMMAND is the full command used to run your GPU application. "
            'Example: --interactive "./my_gpu_app --batch-size 64".'
        ),
    )

    analysis.add_argument(
        "--resume-session",
        type=str,
        default=None,
        dest="resume_session",
        help=(
            "Resume a previous interactive session by session ID or file path. "
            "Example: --resume-session 2026-03-10_14-23-01_myapp"
        ),
    )

    # -- Output options ------------------------------------------------------

    output_group = parser.add_argument_group("Output options")

    output_group.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write output to this file path (default: print to stdout)",
    )

    output_group.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging (shows LLM API calls, reference guide loading, etc.)",
    )

    # -- LLM enhancement options --------------------------------------------

    llm = parser.add_argument_group(
        "LLM enhancement options (optional)",
        "Enable natural language explanations via Anthropic Claude or OpenAI GPT. "
        "Requires API key.",
    )

    llm.add_argument(
        "--llm",
        type=str,
        choices=["anthropic", "openai", "local", "private"],
        default=None,
        help=(
            "Enable LLM-powered analysis enhancement. "
            "'anthropic' (Claude), 'openai' (GPT), 'local' (Ollama), "
            "or 'private' (any OpenAI-compatible server). "
            "Local analysis always runs first; LLM provides additional insights."
        ),
    )

    llm.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        dest="llm_api_key",
        help=(
            "API key for LLM provider. Alternatively set ANTHROPIC_API_KEY "
            "or OPENAI_API_KEY environment variable."
        ),
    )

    llm.add_argument(
        "--llm-model",
        type=str,
        default=None,
        dest="llm_model",
        help=(
            "Override the LLM model name. "
            "Defaults to claude-sonnet-4-20250514 for Anthropic, "
            "gpt-4-turbo-preview for OpenAI. "
            "Can also be set via ROCPD_LLM_MODEL."
        ),
    )

    llm.add_argument(
        "--llm-thinking",
        metavar="TOKENS",
        type=int,
        default=None,
        dest="llm_thinking",
        help=(
            "Enable extended thinking for deeper LLM analysis. Specify the "
            "thinking budget in tokens (e.g. --llm-thinking 8000). "
            "Only available with the Anthropic provider."
        ),
    )

    return parser


def _format_ext(fmt: str) -> str:
    """Return the file extension for a given output format string."""
    return {
        "json": ".json",
        "markdown": ".md",
        "webview": ".html",
        "text": ".txt",
    }.get(fmt, ".txt")


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code: 0 for success, non-zero for error.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate: need at least a database or --source-dir
    if not args.database and not args.source_dir and not args.interactive:
        parser.error(
            "Please provide at least one database file, --source-dir, "
            "or --interactive."
        )

    try:
        from .api import (
            analyze_database,
            analyze_source,
            OutputFormat,
        )
        from .db import AnalysisConnection
    except ImportError as exc:
        print(f"Import error: {exc}", file=sys.stderr)
        print(
            "Make sure rocm-ai-analysis is installed correctly.",
            file=sys.stderr,
        )
        return 1

    # Map CLI format string to OutputFormat enum
    _fmt_map = {
        "text": OutputFormat.TEXT,
        "json": OutputFormat.JSON,
        "markdown": OutputFormat.MARKDOWN,
        "webview": OutputFormat.WEBVIEW,
    }
    output_format = _fmt_map.get(args.format, OutputFormat.TEXT)

    verbose = args.verbose
    outputs: List[str] = []

    try:
        # -- Source-only analysis (no database required) ---------------------
        if args.source_dir and not args.database:
            result = analyze_source(
                source_dir=Path(args.source_dir),
                custom_prompt=args.prompt,
                enable_llm=args.llm is not None,
                llm_provider=args.llm,
                llm_api_key=args.llm_api_key,
                verbose=verbose,
            )
            # Format output
            if args.format == "json":
                from dataclasses import asdict

                outputs.append(json.dumps(asdict(result), indent=2, default=str))
            else:
                outputs.append(str(result))

        # -- Database analysis -----------------------------------------------
        for db_path in args.database or []:
            path = Path(db_path)
            if not path.exists():
                print(f"Error: database file not found: {db_path}", file=sys.stderr)
                return 1

            result = analyze_database(
                database_path=path,
                custom_prompt=args.prompt,
                enable_llm=args.llm is not None,
                llm_provider=args.llm,
                llm_api_key=args.llm_api_key,
                llm_thinking_tokens=args.llm_thinking,
                output_format=output_format,
                verbose=verbose,
                top_kernels=args.top_kernels,
                att_dir=args.att_dir,
            )

            # Format output based on requested format
            if args.format == "json":
                from dataclasses import asdict

                outputs.append(json.dumps(asdict(result), indent=2, default=str))
            else:
                outputs.append(str(result))

    except KeyboardInterrupt:
        print("\nAnalysis interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if verbose:
            traceback.print_exc()
        return 1

    # -- Write or print output -----------------------------------------------
    combined = "\n".join(outputs)

    if args.output:
        # Auto-append format extension if not already present
        output_path = args.output
        ext = _format_ext(args.format)
        if not output_path.endswith(ext):
            output_path += ext

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(combined)
        print(f"Analysis written to: {output_path}")
        if args.format == "text":
            print(
                "Tip: use --format webview for an interactive HTML report, "
                "--format json for machine-readable output, "
                "or --format markdown for Markdown."
            )
    else:
        print(combined)

    return 0


if __name__ == "__main__":
    sys.exit(main())
