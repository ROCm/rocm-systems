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
    perfxpert init
    perfxpert analyze -i trace.db
    perfxpert analyze -i trace.db --format json -d ./out -o report
    perfxpert analyze --source-dir ./my_app
    perfxpert analyze -i trace.db --llm anthropic
    perfxpert diff baseline.db new.db --format text
    perfxpert ci baseline.db new.db --threshold 3.0
    perfxpert-code    (interactive TUI; launches the AMD-branded opencode session)
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

    # ------------------------------------------------------------------
    # doctor subcommand
    # ------------------------------------------------------------------
    subparsers.add_parser(
        "doctor",
        help="Health check: verify MCP server, LLM providers, and dependencies",
    )

    # ------------------------------------------------------------------
    # init subcommand
    # ------------------------------------------------------------------
    from perfxpert.cli import init_cmd as _init_cmd

    init_parser = subparsers.add_parser(
        "init",
        help=(
            "First-run wizard: detect GPU + framework, generate "
            "config, and print a suggested rocprofv3 command"
        ),
        description=(
            "Guided first-run wizard.\n"
            "  1. GPU detection via rocm-smi (fallback: rocminfo)\n"
            "  2. Framework detection from Tier-0 source scan + Python imports\n"
            "  3. Config generation at ~/.config/perfxpert/config.yaml\n"
            "  4. Suggested first rocprofv3 command\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _init_cmd.add_args(init_parser)

    # ------------------------------------------------------------------
    # diff subcommand
    # ------------------------------------------------------------------
    from perfxpert.cli import diff_cmd as _diff_cmd

    diff_parser = subparsers.add_parser(
        "diff",
        help="Compare two trace databases and emit a diff report",
        description=(
            "Compare a baseline trace database with a new one and emit\n"
            "a report with per-kernel deltas, wall-time delta, and a\n"
            "summary narrative. Always returns rc=0."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _diff_cmd.add_args(diff_parser)

    # ------------------------------------------------------------------
    # ci subcommand
    # ------------------------------------------------------------------
    from perfxpert.cli import ci_cmd as _ci_cmd

    ci_parser = subparsers.add_parser(
        "ci",
        help="CI wrapper: fail when runtime regresses above a threshold",
        description=(
            "Thin gating wrapper over `perfxpert diff`. Produces the same\n"
            "report shape but returns rc=1 when wall_delta_pct exceeds\n"
            "the configured threshold."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ci_cmd.add_args(ci_parser)

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
                "at least one of -i/--input (trace database) or " "--source-dir (source code) is required"
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
    elif args.subcommand == "doctor":
        sys.exit(_run_doctor())
    elif args.subcommand == "init":
        sys.exit(_init_cmd.run_init(args))
    elif args.subcommand == "diff":
        sys.exit(_diff_cmd.run_diff(args))
    elif args.subcommand == "ci":
        sys.exit(_ci_cmd.run_ci(args))
    else:
        parser.print_help()
        sys.exit(1)


def _check_version() -> tuple[bool, str]:
    """Check perfxpert version."""
    try:
        from importlib.metadata import version

        ver = version("perfxpert")
        return True, f"perfxpert {ver} installed"
    except Exception as e:
        return False, f"perfxpert version unknown: {e}"


def _check_python_version() -> tuple[bool, str]:
    """Check Python version >= 3.10."""
    import sys

    major, minor = sys.version_info[:2]
    ver_str = f"Python {major}.{minor}"
    if sys.version_info >= (3, 10):
        return True, f"{ver_str} (>= 3.10 required)"
    return False, f"{ver_str} (>= 3.10 required)"


def _check_openai_agents() -> tuple[bool, str]:
    """Check openai-agents SDK version."""
    try:
        from importlib.metadata import version

        ver = version("openai-agents")
        return True, f"openai-agents {ver}"
    except Exception as e:
        return False, f"openai-agents unavailable: {e}"


def _check_mcp_server() -> tuple[bool, str]:
    """Check that MCP server builds and tools are registered."""
    try:
        from mcp_server.server import build_server
        from mcp_server._registry import discover_read_only_tools

        server = build_server()  # noqa: F841
        n = len(discover_read_only_tools())
        return True, f"MCP server reachable"
    except Exception as e:
        return False, f"MCP server FAILED: {e}"


def _check_task_store() -> tuple[bool, str]:
    """Check Python task store readiness."""
    import os
    from pathlib import Path

    task_root_env = os.environ.get("PERFXPERT_TASK_ROOT")
    if task_root_env:
        task_store = Path(task_root_env)
    else:
        task_store = Path.home() / ".perfxpert"
    try:
        task_store.mkdir(exist_ok=True)
        if task_store.is_dir():
            msg = f"Python task store at {task_store} ready"
            if not task_root_env:
                msg += (
                    "\n  WARNING: PERFXPERT_TASK_ROOT not set — "
                    f"defaulting to {task_store}. "
                    "Set PERFXPERT_TASK_ROOT to use a custom location."
                )
            return True, msg
        return False, "task store not a directory"
    except Exception as e:
        return False, f"task store creation failed: {e}"


def _check_opencode_bundled() -> tuple[bool, str]:
    """Check that bundled opencode binary can be resolved with version."""
    from pathlib import Path
    from perfxpert.cli.opencode_launcher import resolve_opencode_binary

    try:
        p = resolve_opencode_binary()
        # Try to get version from the binary
        import subprocess

        try:
            result = subprocess.run([str(p), "--version"], capture_output=True, text=True, timeout=2)
            ver = result.stdout.strip().split()[-1] if result.returncode == 0 else "unknown"
        except Exception:
            ver = "unknown"
        opencode_path = str(p).replace(str(Path.home()), "~")
        return True, f"Bundled opencode {ver} detected at {opencode_path}"
    except FileNotFoundError as e:
        return False, str(e)


def _check_opencode_bundled_config() -> tuple[bool, str]:
    """Check that the bundled _bundled/opencode_config dir can be resolved.

    Dev builds that ship without the bundled config dir would otherwise
    fail at `perfxpert-code` startup (launcher calls `resolve_config_dir`
    eagerly). Previously the doctor did not surface this, so operators
    saw a confusing runtime failure instead of a doctor diagnostic
    (cycle-2 nitpick 3).
    """
    from perfxpert.cli.opencode_launcher import resolve_config_dir
    try:
        p = resolve_config_dir()
        return True, f"Bundled opencode config dir present at {p}"
    except FileNotFoundError as e:
        return False, str(e)


def _check_llm_providers() -> tuple[list[str], list[str]]:
    """Check which LLM providers are configured."""
    import os

    configured = []
    unconfigured = []

    # Import providers to trigger registration
    import perfxpert.providers.anthropic_provider  # noqa: F401
    import perfxpert.providers.openai_provider  # noqa: F401
    import perfxpert.providers.ollama_provider  # noqa: F401
    import perfxpert.providers.private_provider  # noqa: F401
    import perfxpert.providers.opencode_provider  # noqa: F401

    providers = {
        "anthropic": ("ANTHROPIC_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "ollama": ("PERFXPERT_LLM_LOCAL_URL", "OLLAMA_HOST"),
        "private": ("PERFXPERT_LLM_PRIVATE_URL", "PRIVATE_LLM_ENDPOINT"),
        "opencode": (),  # always available (bundled)
    }

    for name, env_vars in providers.items():
        if name == "opencode" or any(os.getenv(env_var) for env_var in env_vars):
            configured.append(name)
        else:
            unconfigured.append(name)

    return sorted(configured), sorted(unconfigured)


def _report_active_mode() -> str:
    """Return the active analysis mode string."""
    return "Mode: agentic"


def _status_symbol(kind: str, stream=None) -> str:
    """Return a status symbol representable by the target stream."""
    import sys

    stream = stream or sys.stdout
    glyphs = {"ok": "✓", "warn": "⚠", "fail": "✗"}
    fallback = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}
    glyph = glyphs[kind]
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        glyph.encode(encoding)
    except UnicodeEncodeError:
        return fallback[kind]
    return glyph


def _stream_safe_text(text: str, stream=None) -> str:
    """Return text that can be encoded by the target stream."""
    import sys

    stream = stream or sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        fallback = text.translate(str.maketrans({"—": "-", "–": "-"}))
        return fallback.encode(encoding, errors="replace").decode(encoding)


def _doctor_print(text: str = "", stream=None) -> None:
    """Print a doctor line without assuming UTF-8 stdout."""
    import sys

    stream = stream or sys.stdout
    print(_stream_safe_text(text, stream), file=stream)


def _run_doctor():
    """Run all health checks and print results in canonical format."""
    import sys

    checks = [
        ("perfxpert version", _check_version()),
        ("Python version", _check_python_version()),
        ("openai-agents", _check_openai_agents()),
        ("MCP server", _check_mcp_server()),
        ("task store", _check_task_store()),
        ("opencode binary", _check_opencode_bundled()),
        ("opencode config dir", _check_opencode_bundled_config()),
    ]

    all_ok = True
    for name, (ok, msg) in checks:
        kind = "ok" if ok else "warn" if "unknown" in msg.lower() else "fail"
        symbol = _status_symbol(kind, sys.stdout)
        if not ok and kind == "fail":
            all_ok = False
        _doctor_print(f"{symbol} {msg}", sys.stdout)

    # Check LLM providers
    configured, unconfigured = _check_llm_providers()
    all_ok = all_ok and len(configured) > 0  # at least one provider configured
    configured_str = ", ".join(configured) if configured else "(none)"
    unconfigured_str = ", ".join(unconfigured) if unconfigured else "(all configured)"
    _doctor_print(
        f"{_status_symbol('ok', sys.stdout)} "
        f"{len(configured)}/5 LLM providers configured ({configured_str})",
        sys.stdout,
    )
    if unconfigured:
        _doctor_print(
            f"  {len(unconfigured)}/5 providers unconfigured ({unconfigured_str}) — see README",
            sys.stdout,
        )

    # Report active mode
    _doctor_print(stream=sys.stdout)
    _doctor_print(_report_active_mode(), sys.stdout)

    # Final status
    _doctor_print(stream=sys.stdout)
    if all_ok:
        _doctor_print(f"{_status_symbol('ok', sys.stdout)} ALL CLEAN", sys.stdout)
        return 0
    else:
        _doctor_print(f"{_status_symbol('fail', sys.stdout)} Issues found — see above", sys.stdout)
        return 1


def _get_version():
    try:
        from perfxpert import __version__

        return __version__
    except ImportError:
        return "0+unknown"


if __name__ == "__main__":
    main()
