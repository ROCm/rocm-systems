#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################
"""
perfxpert — standalone GPU trace analysis tool.

Entry point for both ``python -m perfxpert`` and the ``perfxpert`` script.

Usage
-----
    perfxpert analyze -i trace.db
    perfxpert analyze -i trace.db --format json -d ./out -o report
    perfxpert analyze --source-dir ./my_app
    perfxpert analyze -i trace.db --llm anthropic
    perfxpert analyze -i trace.db --interactive
"""

from __future__ import absolute_import

__author__ = "Advanced Micro Devices, Inc."
__copyright__ = "Copyright 2025, Advanced Micro Devices, Inc."
__license__ = "MIT"


def main(argv=None):
    """Main entry point for the perfxpert command-line tool."""
    import argparse
    import sys

    from . import analyze
    from . import output_config
    from .connection import PerfxpertConnection

    parser = argparse.ArgumentParser(
        prog="perfxpert",
        description=(
            "PerfXpert -- AI-powered GPU trace analysis\n\n"
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

    # Wire analyze subcommand using the same add_args/execute pattern as rocpd.
    # add_args() registers all analysis flags AND returns a process_args() callback.
    process_args = analyze.add_args(analyze_parser)

    # Register -i / --input on the analyze subparser (not the top-level parser)
    analyze_parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=str,
        default=None,
        metavar="DB",
        help="Input rocprofiler-sdk trace database file(s) (.db). "
        "Required unless --source-dir is used for Tier 0 source analysis.",
    )

    # Add -o / -d output flags to both the top-level parser (for help display)
    # and the analyze subparser (so they can appear after the subcommand name).
    output_config.add_args(analyze_parser)
    output_config.add_args(parser)

    # ------------------------------------------------------------------
    # config subcommand
    # ------------------------------------------------------------------
    config_parser = subparsers.add_parser(
        "config",
        help="Show or set perfxpert configuration (~/.config/perfxpert/config.yaml)",
    )
    config_sub = config_parser.add_subparsers(dest="config_action", required=True)
    config_sub.add_parser("show", help="Print current effective config as YAML")
    set_p = config_sub.add_parser("set", help="Set a field and persist to config.yaml")
    set_p.add_argument("key", help="Field name (e.g. provider, airgap, max_tokens)")
    set_p.add_argument("value", help="New value")

    # ------------------------------------------------------------------
    # providers subcommand
    # ------------------------------------------------------------------
    providers_parser = subparsers.add_parser(
        "providers",
        help="LLM provider management",
    )
    providers_sub = providers_parser.add_subparsers(dest="providers_action", required=True)
    providers_sub.add_parser("list", help="List available LLM providers + configuration status")

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
        # Validate: need at least -i or --source-dir
        has_input = bool(getattr(args, "input", None))
        has_source = bool(getattr(args, "source_dir", None))
        if not has_input and not has_source:
            analyze_parser.error(
                "at least one of -i/--input (trace database) or "
                "--source-dir (source code) is required"
            )

        input_data = None
        if has_input:
            input_data = PerfxpertConnection(args.input)

        # Build output config from -o / -d flags
        cfg = output_config.output_config(
            output_file=getattr(args, "output_file", None),
            output_path=getattr(args, "output_path", None),
        )

        # Collect analysis kwargs via the process_args callback from add_args()
        kwargs = process_args(input_data, args)

        try:
            analyze.execute(input_data, cfg, **kwargs)
        finally:
            if input_data is not None:
                input_data.close()
    elif args.subcommand == "config":
        from perfxpert.config._cli import run_config_show, run_config_set
        if args.config_action == "show":
            run_config_show()
            sys.exit(0)
        if args.config_action == "set":
            run_config_set(args.key, args.value)
            sys.exit(0)
    elif args.subcommand == "providers":
        if args.providers_action == "list":
            # Import providers eagerly so the registry is populated
            import perfxpert.providers.anthropic_provider  # noqa: F401
            import perfxpert.providers.openai_provider  # noqa: F401
            import perfxpert.providers.ollama_provider  # noqa: F401
            import perfxpert.providers.private_provider  # noqa: F401
            import perfxpert.providers.opencode_provider  # noqa: F401
            from perfxpert.cli.branding import get_provider_status_table
            from perfxpert.providers.registry import list_providers

            print(get_provider_status_table())
            print()
            print("Registered providers (name — description):")
            for name, desc in sorted(list_providers().items()):
                print(f"  {name}: {desc}")
            return 0
    else:
        parser.print_help()
        sys.exit(1)


def _get_version():
    try:
        from importlib.metadata import version
        return version("perfxpert")
    except (ImportError, ModuleNotFoundError):
        # importlib.metadata not available (Python < 3.8 edge case)
        return "0.1.0"
    except ValueError:
        # Package not installed / metadata lookup failed
        return "0.1.0"


if __name__ == "__main__":
    main()
