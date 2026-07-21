"""Explicit knowledge-retention API.

Writers live here rather than under :mod:`perfxpert.tools`, so MCP discovery
cannot expose them accidentally.
"""

from perfxpert.retention.admin import (
    clear_observations,
    observation_stats,
    prune_observations,
    query_observations,
)
from perfxpert.retention.recorder import (
    capture_directory_sources,
    capture_sources,
    load_persisted_prediction,
    predict_change_impact_durable,
    record_analysis,
    record_comparison,
    record_prediction,
)
from perfxpert.retention.schemas import (
    PersistenceReceipt,
    RetentionPolicy,
    ScopeContext,
    SourceSnapshot,
)
from perfxpert.retention.scope import build_retention_policy, resolve_scope_context
from perfxpert.retention.store import (
    IdentityCollisionError,
    ObservationNotFoundError,
    ObservationStore,
    QuotaExceededError,
    RetentionError,
    ScopeMismatchError,
    StoreCorruptError,
    StoreUnavailableError,
    UnsupportedSchemaError,
    WrongStoreError,
)

__all__ = [
    "IdentityCollisionError",
    "ObservationNotFoundError",
    "ObservationStore",
    "PersistenceReceipt",
    "QuotaExceededError",
    "RetentionError",
    "RetentionPolicy",
    "ScopeContext",
    "ScopeMismatchError",
    "SourceSnapshot",
    "StoreCorruptError",
    "StoreUnavailableError",
    "UnsupportedSchemaError",
    "WrongStoreError",
    "build_retention_policy",
    "capture_directory_sources",
    "capture_sources",
    "clear_observations",
    "load_persisted_prediction",
    "observation_stats",
    "predict_change_impact_durable",
    "prune_observations",
    "query_observations",
    "record_analysis",
    "record_comparison",
    "record_prediction",
    "resolve_scope_context",
]
