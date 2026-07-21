"""Current-project READ_ONLY queries over retained performance observations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from perfxpert.retention.scope import build_retention_policy
from perfxpert.retention.store import (
    ObservationNotFoundError,
    ObservationStore,
    StoreUnavailableError,
)
from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def get_knowledge_observation(record_id: str) -> Dict[str, Any]:
    """Return one retained observation from the current project scope."""

    store = ObservationStore(build_retention_policy())
    try:
        return store.get_observation(record_id)
    except (ObservationNotFoundError, StoreUnavailableError) as exc:
        raise KeyError(f"knowledge observation {record_id!r} not found in current project scope") from exc


@tool_class(ToolClass.READ_ONLY)
def query_knowledge(
    kind: Optional[str] = None,
    kernel_name: Optional[str] = None,
    gfx_id: Optional[str] = None,
    change_type: Optional[str] = None,
    verdict: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Query current-project observations using exact structured filters."""

    store = ObservationStore(build_retention_policy())
    try:
        return store.query(
            kind=kind,
            kernel_name=kernel_name,
            gfx_id=gfx_id,
            change_type=change_type,
            verdict=verdict,
            limit=limit,
        )
    except StoreUnavailableError:
        return []


@tool_class(ToolClass.READ_ONLY)
def knowledge_stats() -> Dict[str, Any]:
    """Return retained-observation counts for the current project scope."""

    policy = build_retention_policy()
    store = ObservationStore(policy)
    try:
        return store.stats()
    except StoreUnavailableError:
        return {
            "scope_id": policy.scope.scope_id,
            "all_scopes": False,
            "records": 0,
            "observations": 0,
            "by_kind": {},
            "database_bytes": 0,
        }


__all__ = [
    "get_knowledge_observation",
    "knowledge_stats",
    "query_knowledge",
]
