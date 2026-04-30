"""CLI surface for TUI-only workflow adapter imports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from perfxpert.integrations.external_workflow import (
    ExternalWorkflowError,
    ExternalWorkflowRuntimeError,
    inspect_external_workflow,
)


def add_args(parser: argparse.ArgumentParser) -> None:
    """Register ``perfxpert workflow`` subcommands."""

    sub = parser.add_subparsers(dest="workflow_action", required=True)
    import_parser = sub.add_parser(
        "import",
        help="Inspect an external workflow adapter for the active TUI session",
        description=(
            "Inspect a local repo or user-approved HTTPS repo and emit an advisory "
            "adapter manifest for perfxpert-code. This command is intentionally "
            "TUI-interactive only; import does not install packages, run adapter "
            "scripts, or start MCP servers."
        ),
    )
    import_parser.add_argument("source", help="Local path or https:// repository URL to inspect")
    import_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Required consent marker; only perfxpert-code interactive sessions should pass it",
    )
    import_parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow cloning an https:// source after interactive consent",
    )
    import_parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Adapter cache root (defaults to ./.perfxpert/external-workflows)",
    )
    import_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full adapter manifest as JSON",
    )
    import_parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Inspect only; do not write the adapter manifest",
    )


def run_workflow(args: argparse.Namespace) -> int:
    """Run the selected workflow subcommand."""

    if args.workflow_action != "import":
        print(f"perfxpert workflow: unknown action {args.workflow_action!r}", file=sys.stderr)
        return 2

    if not args.interactive or not _in_perfxpert_tui_session():
        print(
            "perfxpert workflow import: external workflow adapters are TUI-interactive only; "
            "rerun inside perfxpert-code with --interactive.",
            file=sys.stderr,
        )
        return 2

    try:
        plan = inspect_external_workflow(
            args.source,
            interactive=True,
            allow_network=bool(args.allow_network),
            cache_root=args.cache_root,
            persist=not bool(args.no_persist),
        )
    except ExternalWorkflowError as exc:
        print(f"perfxpert workflow import: {exc}", file=sys.stderr)
        return 1 if isinstance(exc, ExternalWorkflowRuntimeError) else 2

    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        _print_summary(plan)
    return 0


def _print_summary(plan: dict[str, object]) -> None:
    capabilities = plan.get("capabilities") or []
    mcp_servers = plan.get("mcp_servers") or []
    knowledge_links = plan.get("knowledge_links") or []
    print(f"Imported external workflow adapter: {plan.get('adapter_id')}")
    if plan.get("manifest_path"):
        print(f"Manifest: {plan.get('manifest_path')}")
    print(f"Capabilities: {len(capabilities)}")
    print(f"MCP servers discovered: {len(mcp_servers)}")
    print(f"Knowledge links: {len(knowledge_links)}")
    print("Activation: advisory TUI context only; explicit consent required for execution or MCP registration")


def _in_perfxpert_tui_session() -> bool:
    return os.environ.get("PERFXPERT_TUI_INTERACTIVE") == "1"
