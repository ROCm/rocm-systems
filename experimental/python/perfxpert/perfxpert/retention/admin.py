"""Explicit non-MCP administration API for retained observations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from perfxpert.retention.scope import build_retention_policy
from perfxpert.retention.store import ObservationStore, StoreUnavailableError


def query_observations(
    *,
    scope_id: Optional[str] = None,
    all_scopes: bool = False,
    kind: Optional[str] = None,
    kernel_name: Optional[str] = None,
    gfx_id: Optional[str] = None,
    change_type: Optional[str] = None,
    verdict: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    store = ObservationStore(build_retention_policy())
    try:
        return store.query(
            scope_id=scope_id,
            all_scopes=all_scopes,
            kind=kind,
            kernel_name=kernel_name,
            gfx_id=gfx_id,
            change_type=change_type,
            verdict=verdict,
            limit=limit,
        )
    except StoreUnavailableError:
        return []


def observation_stats(
    *,
    scope_id: Optional[str] = None,
    all_scopes: bool = False,
) -> Dict[str, Any]:
    policy = build_retention_policy()
    store = ObservationStore(policy)
    try:
        return store.stats(scope_id=scope_id, all_scopes=all_scopes)
    except StoreUnavailableError:
        return {
            "scope_id": None if all_scopes else (scope_id or policy.scope.scope_id),
            "all_scopes": all_scopes,
            "records": 0,
            "observations": 0,
            "by_kind": {},
            "database_bytes": 0,
        }


def clear_observations(
    *,
    scope_id: Optional[str] = None,
    all_scopes: bool = False,
    compact: bool = False,
) -> int:
    return ObservationStore(build_retention_policy()).clear(
        scope_id=scope_id,
        all_scopes=all_scopes,
        compact=compact,
    )


def prune_observations(
    *,
    older_than: datetime,
    scope_id: Optional[str] = None,
    all_scopes: bool = False,
    compact: bool = False,
) -> int:
    return ObservationStore(build_retention_policy()).prune(
        older_than=older_than,
        scope_id=scope_id,
        all_scopes=all_scopes,
        compact=compact,
    )


__all__ = [
    "clear_observations",
    "observation_stats",
    "prune_observations",
    "query_observations",
]
