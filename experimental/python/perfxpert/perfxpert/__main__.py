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
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Health check: verify MCP server, LLM providers, and dependencies",
    )

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
        except Exception as e:
            # One-line user-facing message for known provider failures;
            # full traceback gated on PERFXPERT_DEBUG=1.
            rc = analyze._render_cli_error(e)
            sys.exit(rc)
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
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "ollama": "OLLAMA_HOST",
        "private": "PRIVATE_LLM_ENDPOINT",
        "opencode": None,  # always available (bundled)
    }

    for name, env_var in providers.items():
        if name == "opencode" or (env_var and os.getenv(env_var)):
            configured.append(name)
        else:
            unconfigured.append(name)

    return sorted(configured), sorted(unconfigured)


def _resolved_models() -> dict[str, str]:
    """Return {provider: resolved_model_id} for each LLM provider.

    Shared view across the agents-SDK path and Provider.complete path so
    `perfxpert doctor` output tells the user EXACTLY which model string
    a live `--llm <provider>` call would use.
    """
    from perfxpert.agents.framework import _resolve_model as _agents_resolve

    result: dict[str, str] = {}
    for prov in ("anthropic", "openai", "ollama", "private", "opencode"):
        try:
            result[prov] = _agents_resolve(prov)
        except Exception:  # pragma: no cover - defensive
            result[prov] = "<unresolved>"
    return result


def _report_active_mode() -> str:
    """Return the active analysis mode string."""
    return "Mode: agentic"


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
        symbol = "✓" if ok else "⚠" if "unknown" in msg.lower() else "✗"
        if not ok and symbol == "✗":
            all_ok = False
        print(f"{symbol} {msg}")

    # Check LLM providers
    configured, unconfigured = _check_llm_providers()
    all_ok = all_ok and len(configured) > 0  # at least one provider configured
    configured_str = ", ".join(configured) if configured else "(none)"
    unconfigured_str = ", ".join(unconfigured) if unconfigured else "(all configured)"
    total = len(configured) + len(unconfigured)
    print(f"✓ {len(configured)}/{total} LLM providers configured ({configured_str})")
    if unconfigured:
        print(
            f"  {len(unconfigured)}/{total} providers unconfigured "
            f"({unconfigured_str}) — see README"
        )

    # Cycle-4 B2: show the RESOLVED model id per provider so users can tell
    # at a glance what `--llm <provider>` would actually invoke. If the
    # default is stale (ancient hardcode), this is where it shows.
    print(f"  resolved models (override with --llm-model or PERFXPERT_<PROVIDER>_MODEL):")
    for prov, model_id in _resolved_models().items():
        marker = "✓" if prov in configured else "·"
        print(f"    {marker} {prov}: {model_id}")

    # Report active mode
    print()
    print(_report_active_mode())

    # Final status
    print()
    if all_ok:
        print("✓ ALL CLEAN")
        return 0
    else:
        print(f"✗ Issues found — see above")
        return 1


def _get_version():
    try:
        from importlib.metadata import version

        return version("perfxpert")
    except (ImportError, ModuleNotFoundError):
        # importlib.metadata not available (Python < 3.8 edge case)
        return "0.2.0"
    except ValueError:
        # Package not installed / metadata lookup failed
        return "0.2.0"


if __name__ == "__main__":
    main()
