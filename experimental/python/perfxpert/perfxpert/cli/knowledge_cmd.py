"""Administrative CLI for retained performance observations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from perfxpert.retention.scope import build_retention_policy
from perfxpert.retention.store import ObservationStore, StoreUnavailableError


def add_args(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="knowledge_action", required=True)

    stats = actions.add_parser("stats", help="Show retained-observation counts")
    _add_scope(stats)

    query = actions.add_parser("query", help="Query exact retained-observation fields")
    _add_scope(query)
    query.add_argument("--kind", choices=("prediction", "trace_analysis", "run_comparison"))
    query.add_argument("--kernel-name")
    query.add_argument("--gfx-id")
    query.add_argument("--change-type")
    query.add_argument("--verdict")
    query.add_argument("--limit", type=int, default=50)

    clear = actions.add_parser("clear", help="Delete retained observations")
    _add_scope(clear)
    clear.add_argument("--yes", action="store_true", help="Confirm destructive deletion")
    clear.add_argument("--compact", action="store_true", help="VACUUM after deletion")

    prune = actions.add_parser("prune", help="Delete observations older than a number of days")
    _add_scope(prune)
    prune.add_argument("--older-than", type=int, required=True, metavar="DAYS")
    prune.add_argument("--yes", action="store_true", help="Confirm destructive deletion")
    prune.add_argument("--compact", action="store_true", help="VACUUM after deletion")


def run_knowledge(args: argparse.Namespace) -> int:
    policy = build_retention_policy()
    store = ObservationStore(policy)
    all_scopes = getattr(args, "scope", "current") == "all"

    if args.knowledge_action == "stats":
        try:
            result = store.stats(all_scopes=all_scopes)
        except StoreUnavailableError:
            result = _empty_stats(policy.scope.scope_id, all_scopes=all_scopes)
        _print_json(result)
        return 0

    if args.knowledge_action == "query":
        try:
            rows = store.query(
                kind=args.kind,
                kernel_name=args.kernel_name,
                gfx_id=args.gfx_id,
                change_type=args.change_type,
                verdict=args.verdict,
                limit=args.limit,
                all_scopes=all_scopes,
            )
        except StoreUnavailableError:
            rows = []
        _print_json(rows)
        return 0

    if args.knowledge_action == "clear":
        if not args.yes:
            print("error: knowledge clear requires --yes", file=sys.stderr)
            return 2
        count = store.clear(all_scopes=all_scopes, compact=args.compact)
        _print_json({"deleted": count, "scope": args.scope})
        return 0

    if args.knowledge_action == "prune":
        if not args.yes:
            print("error: knowledge prune requires --yes", file=sys.stderr)
            return 2
        if args.older_than < 0:
            print("error: --older-than must be non-negative", file=sys.stderr)
            return 2
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.older_than)
        count = store.prune(
            older_than=cutoff,
            all_scopes=all_scopes,
            compact=args.compact,
        )
        _print_json(
            {
                "deleted": count,
                "scope": args.scope,
                "older_than_days": args.older_than,
            }
        )
        return 0

    raise ValueError(f"unknown knowledge action: {args.knowledge_action!r}")


def _add_scope(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scope",
        choices=("current", "all"),
        default="current",
        help="Operate on the current project (default) or all projects",
    )


def _empty_stats(scope_id: str, *, all_scopes: bool) -> Dict[str, Any]:
    return {
        "scope_id": None if all_scopes else scope_id,
        "all_scopes": all_scopes,
        "records": 0,
        "observations": 0,
        "by_kind": {},
        "database_bytes": 0,
    }


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


__all__ = ["add_args", "run_knowledge"]
