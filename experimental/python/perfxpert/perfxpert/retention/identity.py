"""Canonical, domain-separated identities for retained observations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from perfxpert.retention.schemas import SourceSnapshot

PREDICTOR_SCHEMA_VERSION = "change-impact-v2"
OBSERVATION_SCHEMA_VERSION = 1

_PREDICTION_PROVENANCE_FIELDS = {
    "baseline_db",
    "baseline_source",
    "legacy_prediction_id",
    "prediction_id",
    "provenance_ambiguous",
    "provenance_redacted",
    "receipt",
    "source_descriptors",
}


def canonical_json(value: Any) -> str:
    """Serialize supported values with deterministic ordering and no NaNs."""

    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def domain_hash(domain: str, *parts: Any) -> str:
    """Hash framed canonical fields so concatenation cannot be ambiguous."""

    digest = hashlib.sha256()
    _update_frame(digest, domain.encode("utf-8"))
    for part in parts:
        encoded = canonical_json(part).encode("utf-8")
        _update_frame(digest, encoded)
    return digest.hexdigest()


def payload_hash(payload: Mapping[str, Any]) -> str:
    return domain_hash("perfxpert-payload-v1", payload)


def catalog_entry_hash(entry: Mapping[str, Any] | None) -> str:
    return domain_hash("perfxpert-prediction-catalog-v1", entry or {})


def semantic_prediction_projection(prediction: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip invocation/storage provenance from a public prediction result."""

    return {key: value for key, value in prediction.items() if key not in _PREDICTION_PROVENANCE_FIELDS}


def semantic_prediction_id(
    prediction: Mapping[str, Any],
    *,
    effective_params: Mapping[str, Any],
    catalog_entry: Mapping[str, Any] | None,
    producer_version: str,
    predictor_version: str = PREDICTOR_SCHEMA_VERSION,
) -> str:
    """Return a source/scope-independent prediction content identity."""

    return semantic_prediction_id_from_context(
        prediction,
        effective_params=effective_params,
        catalog_hash=catalog_entry_hash(catalog_entry),
        producer_version=producer_version,
        predictor_version=predictor_version,
    )


def semantic_prediction_id_from_context(
    prediction: Mapping[str, Any],
    *,
    effective_params: Mapping[str, Any],
    catalog_hash: str,
    producer_version: str,
    predictor_version: str = PREDICTOR_SCHEMA_VERSION,
) -> str:
    """Recompute a semantic ID from persisted identity metadata."""

    context = {
        "predictor_version": predictor_version,
        "producer_version": producer_version,
        "catalog_entry_hash": catalog_hash,
        "effective_params": dict(effective_params),
        "prediction": semantic_prediction_projection(prediction),
    }
    return domain_hash("perfxpert-prediction-v2", context)


def observation_key(
    *,
    scope_id: str,
    kind: str,
    sources: Sequence[SourceSnapshot],
    options: Mapping[str, Any],
    mode: str,
    producer_version: str,
    schema_version: int = OBSERVATION_SCHEMA_VERSION,
    external_id: str | None = None,
    extra_context: Mapping[str, Any] | None = None,
) -> str:
    context = {
        "scope_id": scope_id,
        "kind": kind,
        "sources": [source.identity_fields() for source in sources],
        "options": dict(options),
        "mode": mode,
        "producer_version": producer_version,
        "schema_version": schema_version,
        "external_id": external_id,
        "extra_context": dict(extra_context or {}),
    }
    return domain_hash("perfxpert-observation-key-v1", context)


def retained_record_id(
    *,
    scope_id: str,
    observation_key_value: str,
    payload_hash_value: str,
) -> str:
    return domain_hash(
        "perfxpert-record-v1",
        scope_id,
        observation_key_value,
        payload_hash_value,
    )


def dedupe_tags(tags: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    normalized = {
        (str(key).strip(), str(value).strip()) for key, value in tags if str(key).strip() and str(value).strip()
    }
    return tuple(sorted(normalized))


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize(item) for item in value]
        return sorted(normalized, key=canonical_json)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not valid retained knowledge")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported value for canonical JSON: {type(value).__name__}")


def _update_frame(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big", signed=False))
    digest.update(payload)


__all__ = [
    "OBSERVATION_SCHEMA_VERSION",
    "PREDICTOR_SCHEMA_VERSION",
    "canonical_json",
    "catalog_entry_hash",
    "dedupe_tags",
    "domain_hash",
    "observation_key",
    "payload_hash",
    "retained_record_id",
    "semantic_prediction_id",
    "semantic_prediction_id_from_context",
    "semantic_prediction_projection",
]
