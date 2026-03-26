#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################
"""
rocinsight — standalone GPU trace analysis tool.

Entry point for both ``python -m rocinsight`` and the ``rocinsight`` script.

Usage
-----
    rocinsight analyze -i trace.db
    rocinsight analyze -i trace.db --format json -d ./out -o report
    rocinsight analyze --source-dir ./my_app
    rocinsight analyze -i trace.db --llm anthropic
    rocinsight analyze -i trace.db --interactive
"""

from __future__ import absolute_import

__author__ = "Advanced Micro Devices, Inc."
__copyright__ = "Copyright 2025, Advanced Micro Devices, Inc."
__license__ = "MIT"


def main(argv=None):
    """Main entry point for the rocinsight command-line tool."""
    import argparse
    import sys

    from . import analyze
    from . import output_config
    from .connection import RocinsightConnection

    parser = argparse.ArgumentParser(
        prog="rocinsight",
        description=(
            "rocinsight -- AI-powered GPU trace analysis\n\n"
            "Reads rocprofiler-sdk trace databases (.db) and provides\n"
            "performance insights, bottleneck detection, and optimization\n"
            "recommendations. Optionally enhances output with LLM analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + _get_version(),
    )

    subparsers = parser.add_subparsers(dest="subcommand", title="subcommands")

    # ------------------------------------------------------------------
    # analyze subcommand
    # ------------------------------------------------------------------
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a rocprofiler-sdk trace database for GPU performance issues",
        description=(
            "Analyze one or more rocprofiler-sdk trace databases and produce\n"
            "a performance report with bottleneck detection, hotspot ranking,\n"
            "and actionable optimization recommendations.\n\n"
            "Tiers:\n"
            "  Tier 0 -- static source code scan (--source-dir, no .db required)\n"
            "  Tier 1 -- trace analysis (kernel hotspots, memory copies, idle time)\n"
            "  Tier 2 -- hardware counter analysis (--pmc data required)\n"
            "  Tier 3 -- ATT instruction-level stall analysis (--att-dir)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Wire analyze subcommand using the same add_args/execute pattern as rocpd
    analyze.add_args(analyze_parser)
    output_config.add_args(parser)

    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_help()
        sys.exit(0)

    if args.subcommand == "analyze":
        input_data = None
        if getattr(args, "input", None):
            input_data = RocinsightConnection(args.input)
        try:
            analyze.execute(input_data, args)
        finally:
            if input_data is not None:
                input_data.close()
    else:
        parser.print_help()
        sys.exit(1)


def _get_version():
    try:
        from importlib.metadata import version
        return version("rocinsight")
    except Exception:
        return "0.1.0"


if __name__ == "__main__":
    main()
