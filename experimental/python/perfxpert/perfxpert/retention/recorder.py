"""Explicit persistence boundaries for predictions, analyses, and comparisons."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from perfxpert import __version__ as _PERFXPERT_VERSION
from perfxpert.retention.identity import (
    OBSERVATION_SCHEMA_VERSION,
    dedupe_tags,
    observation_key,
    payload_hash,
    retained_record_id,
    semantic_prediction_id_from_context,
)
from perfxpert.retention.schemas import (
    PersistenceReceipt,
    RetentionPolicy,
    SourceSnapshot,
)
from perfxpert.retention.scope import build_retention_policy
from perfxpert.retention.store import (
    ObservationNotFoundError,
    ObservationStore,
    QuotaExceededError,
    RetentionError,
    ScopeMismatchError,
)

_log = logging.getLogger(__name__)

_ANALYSIS_OPTION_KEYS = {
    "top_kernels",
    "min_duration",
    "att_enabled",
    "att_file_count",
    "provider",
    "airgap",
    "counter_options",
    "llm_model",
    "llm_thinking",
    "llm_local",
    "llm_local_model",
}
_ANALYSIS_PAYLOAD_KEYS = {
    "time_breakdown",
    "hotspots",
    "memory_analysis",
    "hardware_counters",
    "kernel_resources",
    "api_overhead",
    "warmup_issues",
    "thread_trace",
    "roofline",
    "communication",
}
_COMPARISON_KEYS = {
    "schema_version",
    "wall_delta_ns",
    "wall_delta_pct",
    "verdict",
    "verdict_threshold_pct",
    "per_kernel",
    "primary_regressions",
    "primary_improvements",
}


def predict_change_impact_durable(
    baseline_db: str,
    kernel_name: str,
    change_type: str,
    change_params: Optional[Dict[str, Any]] = None,
    *,
    project_root: Optional[str] = None,
    policy: Optional[RetentionPolicy] = None,
) -> Tuple[Dict[str, Any], PersistenceReceipt]:
    """Compute a pure prediction, then explicitly attempt durable retention."""

    from perfxpert.tools.predict_impact import predict_change_impact

    try:
        effective_policy = policy or build_retention_policy(project_root=project_root)
        snapshots = (
            [
                SourceSnapshot.capture(
                    baseline_db,
                    role="baseline",
                    ordinal=0,
                )
            ]
            if baseline_db
            else []
        )
    except Exception as exc:
        prediction = predict_change_impact(
            baseline_db=baseline_db,
            kernel_name=kernel_name,
            change_type=change_type,
            change_params=change_params,
        )
        return prediction, _configuration_error_receipt(
            str(exc),
            external_id=str(prediction.get("prediction_id") or "") or None,
        )
    prediction = predict_change_impact(
        baseline_db=baseline_db,
        kernel_name=kernel_name,
        change_type=change_type,
        change_params=change_params,
    )
    receipt = record_prediction(
        prediction,
        change_params=change_params,
        source_snapshots=snapshots,
        policy=effective_policy,
    )
    return prediction, receipt


def record_prediction(
    prediction: Mapping[str, Any],
    *,
    change_params: Optional[Mapping[str, Any]] = None,
    source_snapshots: Sequence[SourceSnapshot] = (),
    policy: Optional[RetentionPolicy] = None,
    project_root: Optional[str] = None,
    mode: str = "deterministic",
) -> PersistenceReceipt:
    """Retain an allowlisted prediction observation."""

    external_id = str(prediction.get("prediction_id") or "")
    try:
        effective_policy = policy or build_retention_policy(project_root=project_root)
    except Exception as exc:
        return _configuration_error_receipt(
            str(exc),
            external_id=external_id or None,
        )
    if not external_id:
        return _error_receipt(
            effective_policy,
            "prediction has no prediction_id",
        )
    if not effective_policy.enabled:
        return _disabled_receipt(effective_policy, external_id=external_id)

    snapshots = list(source_snapshots)
    if not snapshots:
        return _error_receipt(
            effective_policy,
            "record_prediction requires a pre-compute source snapshot",
            external_id=external_id,
        )
    if sum(snapshot.role == "baseline" for snapshot in snapshots) != 1:
        return _error_receipt(
            effective_policy,
            "prediction snapshots require exactly one baseline source",
            external_id=external_id,
        )
    baseline_snapshot = next(snapshot for snapshot in snapshots if snapshot.role == "baseline")
    baseline_db = str(prediction.get("baseline_db") or "")
    if not baseline_db or not _same_resolved_path(
        baseline_snapshot.resolved_path,
        baseline_db,
    ):
        return _error_receipt(
            effective_policy,
            "prediction baseline snapshot does not match baseline_db",
            external_id=external_id,
        )
    if any(not snapshot.exists for snapshot in snapshots):
        return _error_receipt(
            effective_policy,
            "prediction source snapshot does not reference an existing file",
            external_id=external_id,
        )

    from perfxpert.tools.predict_impact import (
        _prediction_identity_context_for_retention,
    )

    identity_context = _prediction_identity_context_for_retention(external_id)
    if identity_context is None:
        return _error_receipt(
            effective_policy,
            "prediction identity context is unavailable; record in the producing process",
            external_id=external_id,
        )
    try:
        recomputed_id = semantic_prediction_id_from_context(
            prediction,
            **identity_context,
        )
    except (TypeError, ValueError) as exc:
        return _error_receipt(
            effective_policy,
            f"prediction identity validation failed: {exc}",
            external_id=external_id,
        )
    if recomputed_id != external_id:
        return _error_receipt(
            effective_policy,
            "prediction payload does not match prediction_id",
            external_id=external_id,
        )

    changed = _changed_sources(snapshots)
    if changed:
        return _error_receipt(
            effective_policy,
            f"source metadata changed during prediction: {changed}",
            external_id=external_id,
        )

    stored_sources, provenance_redacted = _stored_sources(
        snapshots,
        policy=effective_policy,
    )
    stored_prediction = _stored_prediction_projection(
        prediction,
        stored_sources=stored_sources,
        provenance_redacted=provenance_redacted,
    )
    effective_params = dict(identity_context["effective_params"])
    payload = {
        "prediction": stored_prediction,
        "identity": identity_context,
        "sources": stored_sources,
    }
    tags = [
        ("kernel", str(prediction.get("kernel_name") or "")),
        ("change_type", str(prediction.get("change_type") or "")),
    ]
    return _record(
        kind="prediction",
        payload=payload,
        sources=snapshots,
        options=effective_params,
        mode=mode,
        external_id=external_id,
        deterministic=True,
        confidence=_optional_float(prediction.get("confidence")),
        tags=tags,
        policy=effective_policy,
        provenance_redacted=provenance_redacted,
        extra_context={"prediction_id": external_id},
    )


def record_analysis(
    analysis_payload: Mapping[str, Any],
    *,
    primary_bottleneck: str,
    source_snapshots: Sequence[SourceSnapshot],
    options: Optional[Mapping[str, Any]] = None,
    mode: str = "deterministic",
    model_derived: bool = False,
    policy: Optional[RetentionPolicy] = None,
    project_root: Optional[str] = None,
) -> PersistenceReceipt:
    """Retain one workload-level trace analysis without raw model prose."""

    try:
        effective_policy = policy or build_retention_policy(project_root=project_root)
    except Exception as exc:
        return _configuration_error_receipt(str(exc))
    if not effective_policy.enabled:
        return _disabled_receipt(effective_policy)
    snapshots = list(source_snapshots)
    if not snapshots:
        return _error_receipt(
            effective_policy,
            "trace_analysis requires at least one trace source",
        )
    if any(not snapshot.exists for snapshot in snapshots):
        return _error_receipt(
            effective_policy,
            "trace_analysis source snapshot does not reference an existing file",
        )
    if (options or {}).get("att_dir"):
        return _error_receipt(
            effective_policy,
            "raw att_dir is not accepted; pass ATT snapshots and att_enabled",
        )
    if (options or {}).get("att_enabled"):
        att_files = [snapshot for snapshot in snapshots if snapshot.role == "att_input"]
        if not any(snapshot.role == "att_manifest" for snapshot in snapshots):
            return _error_receipt(
                effective_policy,
                "ATT-backed analysis retention requires a directory manifest",
            )
        if len(att_files) != int((options or {}).get("att_file_count") or 0):
            return _error_receipt(
                effective_policy,
                "ATT file snapshots do not match att_file_count",
            )
    changed = _changed_sources(snapshots)
    if changed:
        return _error_receipt(
            effective_policy,
            f"source metadata changed during analysis: {changed}",
        )

    stored_sources, provenance_redacted = _stored_sources(
        snapshots,
        policy=effective_policy,
    )
    filtered_options = {
        key: _sanitize_retained_value(
            value,
            policy=effective_policy,
            key=key,
        )
        for key, value in dict(options or {}).items()
        if key in _ANALYSIS_OPTION_KEYS
    }
    deterministic_payload = {
        key: _sanitize_retained_value(
            analysis_payload.get(key),
            policy=effective_policy,
            key=key,
        )
        for key in _ANALYSIS_PAYLOAD_KEYS
        if key in analysis_payload and analysis_payload.get(key) is not None
    }
    payload = {
        "deterministic_evidence": deterministic_payload,
        "decision": {
            "primary_bottleneck": str(primary_bottleneck),
            "model_derived": bool(model_derived),
            "mode": str(mode),
        },
        "options": filtered_options,
        "sources": stored_sources,
    }
    tags = _analysis_tags(analysis_payload)
    return _record(
        kind="trace_analysis",
        payload=payload,
        sources=snapshots,
        options=filtered_options,
        mode=mode,
        external_id=None,
        deterministic=not model_derived,
        confidence=None,
        tags=tags,
        policy=effective_policy,
        provenance_redacted=provenance_redacted,
        extra_context={"primary_bottleneck": str(primary_bottleneck)},
    )


def record_comparison(
    diff_result: Mapping[str, Any],
    *,
    baseline_db: str,
    new_db: str,
    top_kernels: int = 20,
    source_snapshots: Sequence[SourceSnapshot] = (),
    policy: Optional[RetentionPolicy] = None,
    project_root: Optional[str] = None,
) -> PersistenceReceipt:
    """Retain a deterministic baseline/candidate comparison."""

    try:
        effective_policy = policy or build_retention_policy(project_root=project_root)
    except Exception as exc:
        return _configuration_error_receipt(str(exc))
    if not effective_policy.enabled:
        return _disabled_receipt(effective_policy)
    snapshots = list(source_snapshots)
    if not snapshots:
        return _error_receipt(
            effective_policy,
            "record_comparison requires pre-compute baseline and candidate snapshots",
        )
    roles = [(snapshot.role, snapshot.ordinal) for snapshot in snapshots]
    if len(snapshots) != 2 or roles.count(("baseline", 0)) != 1 or roles.count(("candidate", 0)) != 1:
        return _error_receipt(
            effective_policy,
            "comparison snapshots require exactly one baseline and one candidate",
        )
    baseline_snapshot = next(snapshot for snapshot in snapshots if snapshot.role == "baseline")
    candidate_snapshot = next(snapshot for snapshot in snapshots if snapshot.role == "candidate")
    if not _same_resolved_path(baseline_snapshot.resolved_path, baseline_db):
        return _error_receipt(
            effective_policy,
            "comparison baseline snapshot does not match baseline_db",
        )
    if not _same_resolved_path(candidate_snapshot.resolved_path, new_db):
        return _error_receipt(
            effective_policy,
            "comparison candidate snapshot does not match new_db",
        )
    if any(not snapshot.exists for snapshot in snapshots):
        return _error_receipt(
            effective_policy,
            "comparison source snapshot does not reference an existing file",
        )
    changed = _changed_sources(snapshots)
    if changed:
        return _error_receipt(
            effective_policy,
            f"source metadata changed during comparison: {changed}",
        )

    stored_sources, provenance_redacted = _stored_sources(
        snapshots,
        policy=effective_policy,
    )
    comparison = _comparison_projection(diff_result, policy=effective_policy)
    payload = {
        "comparison": comparison,
        "options": {"top_kernels": int(top_kernels)},
        "sources": stored_sources,
    }
    tags = [("verdict", str(diff_result.get("verdict") or ""))]
    for row in diff_result.get("per_kernel") or []:
        if isinstance(row, Mapping) and row.get("name"):
            tags.append(("kernel", str(row["name"])))
    return _record(
        kind="run_comparison",
        payload=payload,
        sources=snapshots,
        options={"top_kernels": int(top_kernels)},
        mode="deterministic",
        external_id=None,
        deterministic=True,
        confidence=None,
        tags=tags,
        policy=effective_policy,
        provenance_redacted=provenance_redacted,
        extra_context={"verdict": str(diff_result.get("verdict") or "")},
    )


def load_persisted_prediction(
    prediction_id: str,
    *,
    policy: Optional[RetentionPolicy] = None,
    project_root: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Read a durable prediction without creating or migrating a store."""

    effective_policy = policy or build_retention_policy(project_root=project_root)
    store = ObservationStore(effective_policy)
    try:
        observation = store.get_latest_by_external_id(
            kind="prediction",
            external_id=prediction_id,
        )
    except ObservationNotFoundError:
        if store.external_id_scopes(
            kind="prediction",
            external_id=prediction_id,
        ):
            raise ScopeMismatchError(f"prediction {prediction_id!r} exists outside the current project scope")
        return None
    payload = observation.get("payload") or {}
    prediction = payload.get("prediction")
    identity_context = payload.get("identity")
    if not isinstance(prediction, dict):
        raise RetentionError(f"prediction observation {observation['record_id']!r} " "has no prediction payload")
    if not isinstance(identity_context, dict):
        raise RetentionError(f"prediction observation {observation['record_id']!r} " "has no identity context")
    if prediction.get("prediction_id") != prediction_id:
        raise RetentionError(f"prediction observation {observation['record_id']!r} " "has a mismatched external id")
    try:
        recomputed_id = semantic_prediction_id_from_context(
            prediction,
            **identity_context,
        )
    except (TypeError, ValueError) as exc:
        raise RetentionError(
            f"prediction observation {observation['record_id']!r} " "has invalid identity metadata"
        ) from exc
    if recomputed_id != prediction_id:
        raise RetentionError(
            f"prediction observation {observation['record_id']!r} " "failed semantic identity validation"
        )
    return dict(prediction)


def capture_sources(
    paths: Iterable[str | Path],
    *,
    role: str = "input",
) -> list[SourceSnapshot]:
    return [SourceSnapshot.capture(path, role=role, ordinal=index) for index, path in enumerate(paths)]


def capture_directory_sources(
    directory: str | Path,
    *,
    role: str,
    max_files: int = 10_000,
) -> list[SourceSnapshot]:
    """Capture ordered directory manifests plus every regular file."""

    root = Path(directory).expanduser().resolve(strict=False)
    if not root.is_dir():
        return []
    directories = [
        root,
        *sorted(
            (path for path in root.rglob("*") if path.is_dir()),
            key=lambda path: str(path),
        ),
    ]
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: str(path),
    )
    if len(files) > max_files:
        raise ValueError(f"source directory contains {len(files)} files; limit is {max_files}")
    manifest_role = "att_manifest" if role == "att_input" else f"{role}_manifest"
    manifests = [
        SourceSnapshot.capture(path, role=manifest_role, ordinal=index) for index, path in enumerate(directories)
    ]
    snapshots = [SourceSnapshot.capture(path, role=role, ordinal=index) for index, path in enumerate(files)]
    return [*manifests, *snapshots]


def _record(
    *,
    kind: str,
    payload: Mapping[str, Any],
    sources: Sequence[SourceSnapshot],
    options: Mapping[str, Any],
    mode: str,
    external_id: Optional[str],
    deterministic: bool,
    confidence: Optional[float],
    tags: Iterable[tuple[str, str]],
    policy: RetentionPolicy,
    provenance_redacted: bool,
    extra_context: Mapping[str, Any],
) -> PersistenceReceipt:
    try:
        normalized_tags = dedupe_tags(tags)
        observation_key_value = observation_key(
            scope_id=policy.scope.scope_id,
            kind=kind,
            sources=sources,
            options=options,
            mode=mode,
            producer_version=_PERFXPERT_VERSION,
            schema_version=OBSERVATION_SCHEMA_VERSION,
            external_id=external_id,
            extra_context=extra_context,
        )
        payload_hash_value = payload_hash(
            {
                "payload": payload,
                "tags": normalized_tags,
            }
        )
        record_id = retained_record_id(
            scope_id=policy.scope.scope_id,
            observation_key_value=observation_key_value,
            payload_hash_value=payload_hash_value,
        )
        store = ObservationStore(policy)
        store.write_observation(
            record_id=record_id,
            observation_key=observation_key_value,
            payload_hash=payload_hash_value,
            kind=kind,
            schema_version=OBSERVATION_SCHEMA_VERSION,
            producer_version=_PERFXPERT_VERSION,
            external_id=external_id,
            deterministic=deterministic,
            confidence=confidence,
            payload=payload,
            tags=normalized_tags,
        )
    except QuotaExceededError as exc:
        return PersistenceReceipt(
            status="quota_exceeded",
            record_id=None,
            scope_id=policy.scope.scope_id,
            external_id=external_id,
            provenance_redacted=provenance_redacted,
            detail=str(exc),
        )
    except (RetentionError, OSError, TypeError, ValueError) as exc:
        _log.warning("PerfXpert knowledge retention failed: %s", exc)
        return _error_receipt(
            policy,
            str(exc),
            external_id=external_id,
            provenance_redacted=provenance_redacted,
        )
    return PersistenceReceipt(
        status="persisted",
        record_id=record_id,
        scope_id=policy.scope.scope_id,
        external_id=external_id,
        provenance_redacted=provenance_redacted,
    )


def _stored_prediction_projection(
    prediction: Mapping[str, Any],
    *,
    stored_sources: Sequence[Mapping[str, Any]],
    provenance_redacted: bool,
) -> Dict[str, Any]:
    allowed = {
        "predicted_speedup_range",
        "confidence",
        "rationale",
        "roofline_delta",
        "assumptions",
        "source_citation",
        "prediction_id",
        "change_type",
        "kernel_name",
        "baseline_db",
    }
    stored = {key: prediction.get(key) for key in allowed if key in prediction}
    baseline = next(
        (source for source in stored_sources if source.get("role") == "baseline"),
        None,
    )
    if baseline is not None:
        stored["baseline_db"] = str(baseline.get("display_path") or "")
    stored["provenance_redacted"] = bool(provenance_redacted)
    return stored


def _stored_sources(
    snapshots: Sequence[SourceSnapshot],
    *,
    policy: RetentionPolicy,
) -> tuple[list[Dict[str, Any]], bool]:
    stored = [
        snapshot.stored_fields(
            scope=policy.scope,
            store_paths=policy.store_paths,
        )
        for snapshot in snapshots
    ]
    redacted = any(bool(source.get("provenance_redacted")) for source in stored)
    return stored, redacted


def _changed_sources(snapshots: Sequence[SourceSnapshot]) -> Optional[str]:
    for before in snapshots:
        if not before.resolved_path:
            continue
        after = SourceSnapshot.capture(
            before.resolved_path,
            role=before.role,
            ordinal=before.ordinal,
        )
        if not before.same_revision(after):
            return before.resolved_path
    return None


def _same_resolved_path(first: str, second: str | Path) -> bool:
    if not first or not second:
        return False
    return Path(first) == Path(second).expanduser().resolve(strict=False)


def _comparison_projection(
    result: Mapping[str, Any],
    *,
    policy: RetentionPolicy,
) -> Dict[str, Any]:
    row_keys = {
        "name",
        "baseline_ns",
        "new_ns",
        "delta_ns",
        "delta_pct",
        "regressed",
        "was_hot",
    }
    projected: Dict[str, Any] = {
        key: _sanitize_retained_value(result.get(key), policy=policy, key=key)
        for key in _COMPARISON_KEYS
        if key not in {"per_kernel", "primary_regressions", "primary_improvements"} and key in result
    }
    for list_key in (
        "per_kernel",
        "primary_regressions",
        "primary_improvements",
    ):
        rows = []
        for row in result.get(list_key) or []:
            if not isinstance(row, Mapping):
                continue
            rows.append(
                {
                    key: _sanitize_retained_value(
                        row.get(key),
                        policy=policy,
                        key=key,
                    )
                    for key in row_keys
                    if key in row
                }
            )
        projected[list_key] = rows
    return projected


def _sanitize_retained_value(
    value: Any,
    *,
    policy: RetentionPolicy,
    key: str = "",
) -> Any:
    """Recursively remove model/source text and redact path-valued strings."""

    skipped_keys = {
        "api_key",
        "code",
        "code_snippet",
        "csv_file",
        "disassembly",
        "error",
        "instruction",
        "instructions",
        "llm_explanation",
        "model_response",
        "narrative",
        "prompt",
        "raw_response",
        "reason",
        "source",
        "source_code",
        "source_line",
        "top_stalling_instructions",
    }
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for child_key, child_value in value.items():
            normalized_key = str(child_key)
            if normalized_key.lower() in skipped_keys:
                continue
            result[normalized_key] = _sanitize_retained_value(
                child_value,
                policy=policy,
                key=normalized_key,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize_retained_value(item, policy=policy, key=key) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and not policy.store_paths:
        try:
            if Path(value).is_absolute():
                return Path(value).name
        except (OSError, ValueError):
            return "<redacted-path>"
        return re.sub(
            r"(^|[\s(\[=])(/[^\s)\],;]+)",
            lambda match: match.group(1) + Path(match.group(2)).name,
            value,
        )
    return value


def _analysis_tags(payload: Mapping[str, Any]) -> list[tuple[str, str]]:
    tags: list[tuple[str, str]] = []
    for hotspot in payload.get("hotspots") or []:
        if isinstance(hotspot, Mapping):
            name = hotspot.get("name") or hotspot.get("kernel_name")
            if name:
                tags.append(("kernel", str(name)))
    hardware = payload.get("hardware_counters") or {}
    if isinstance(hardware, Mapping):
        gfx_id = _find_first_key(hardware, {"gfx_id", "architecture"})
        if gfx_id:
            tags.append(("gfx_id", str(gfx_id)))
    if not any(key == "gfx_id" for key, _ in tags):
        metadata = payload.get("metadata") or {}
        if isinstance(metadata, Mapping) and metadata.get("gfx_id"):
            tags.append(("gfx_id", str(metadata["gfx_id"])))
    return tags


def _find_first_key(value: Any, names: set[str]) -> Any:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in names and item:
                return item
        for item in value.values():
            found = _find_first_key(item, names)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_first_key(item, names)
            if found:
                return found
    return None


def _optional_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _disabled_receipt(
    policy: RetentionPolicy,
    *,
    external_id: Optional[str] = None,
) -> PersistenceReceipt:
    return PersistenceReceipt(
        status="disabled",
        record_id=None,
        scope_id=policy.scope.scope_id,
        external_id=external_id,
        detail="knowledge retention is disabled",
    )


def _error_receipt(
    policy: RetentionPolicy,
    detail: str,
    *,
    external_id: Optional[str] = None,
    provenance_redacted: bool = False,
) -> PersistenceReceipt:
    return PersistenceReceipt(
        status="error",
        record_id=None,
        scope_id=policy.scope.scope_id,
        external_id=external_id,
        provenance_redacted=provenance_redacted,
        detail=detail,
    )


def _configuration_error_receipt(
    detail: str,
    *,
    external_id: Optional[str] = None,
) -> PersistenceReceipt:
    return PersistenceReceipt(
        status="error",
        record_id=None,
        scope_id="",
        external_id=external_id,
        detail=f"retention configuration failed: {detail}",
    )


__all__ = [
    "capture_directory_sources",
    "capture_sources",
    "load_persisted_prediction",
    "predict_change_impact_durable",
    "record_analysis",
    "record_comparison",
    "record_prediction",
]
