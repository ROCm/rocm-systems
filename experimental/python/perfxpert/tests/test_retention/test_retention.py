from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from perfxpert.config import PerfXpertConfig
from perfxpert.retention import (
    ObservationStore,
    SourceSnapshot,
    ScopeMismatchError,
    build_retention_policy,
    capture_directory_sources,
    predict_change_impact_durable,
    record_analysis,
    record_comparison,
    record_prediction,
)
from perfxpert.retention.store import (
    APPLICATION_ID,
    STORE_SCHEMA_VERSION,
    ObservationNotFoundError,
    QuotaExceededError,
    StoreCorruptError,
    UnsupportedSchemaError,
    WrongStoreError,
)
from perfxpert.retention.identity import payload_hash, retained_record_id
from perfxpert.tools import knowledge_history, predict_impact


def _policy(monkeypatch, tmp_path, *, enabled=True, store_paths=False, project=None):
    root = tmp_path / "knowledge-root"
    project_root = project or (tmp_path / "project")
    monkeypatch.setenv("PERFXPERT_KNOWLEDGE_ROOT", str(root))
    monkeypatch.setenv("PERFXPERT_PROJECT_ROOT", str(project_root))
    return build_retention_policy(
        config=PerfXpertConfig(
            knowledge_retention=enabled,
            knowledge_store_paths=store_paths,
        )
    )


def _prediction(path: str):
    return predict_impact.predict_change_impact(
        baseline_db=path,
        kernel_name="kernel_x",
        change_type="vgpr_reduction",
        change_params={
            "kernel_time_pct": 0.4,
            "counter_data_available": True,
        },
    )


def test_semantic_prediction_id_is_source_independent(tmp_path):
    first = _prediction(str(tmp_path / "first.db"))
    second = _prediction(str(tmp_path / "second.db"))

    assert first["prediction_id"] == second["prediction_id"]
    assert len(first["prediction_id"]) == 64
    assert first["baseline_db"].endswith("first.db")
    assert second["baseline_db"].endswith("second.db")


def test_ambiguous_in_process_provenance_is_not_misattributed(tmp_path):
    first = _prediction(str(tmp_path / "first.db"))
    second = _prediction(str(tmp_path / "second.db"))

    explained = predict_impact.explain_prediction(first["prediction_id"])

    assert first["prediction_id"] == second["prediction_id"]
    assert explained["baseline_db"] == ""
    assert explained["provenance_redacted"] is True
    assert explained["provenance_ambiguous"] is True


def test_prediction_cache_is_partitioned_by_project_scope(monkeypatch, tmp_path):
    monkeypatch.setenv("PERFXPERT_PROJECT_ROOT", str(tmp_path / "project-a"))
    prediction = _prediction(str(tmp_path / "secret.db"))
    monkeypatch.setenv("PERFXPERT_PROJECT_ROOT", str(tmp_path / "project-b"))

    with pytest.raises(KeyError):
        predict_impact.explain_prediction(prediction["prediction_id"])


def test_disabled_durable_prediction_creates_no_state(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path, enabled=False)
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")

    prediction, receipt = predict_change_impact_durable(
        str(trace),
        "kernel_x",
        "vgpr_reduction",
        {"kernel_time_pct": 0.4, "counter_data_available": True},
        policy=policy,
    )

    assert prediction["baseline_db"] == str(trace)
    assert receipt.status == "disabled"
    assert not Path(policy.root).exists()


def test_durable_prediction_survives_configuration_failure(monkeypatch, tmp_path):
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    monkeypatch.setattr(
        "perfxpert.retention.recorder.build_retention_policy",
        lambda **_: (_ for _ in ()).throw(ValueError("bad config")),
    )

    prediction, receipt = predict_change_impact_durable(
        str(trace),
        "kernel_x",
        "vgpr_reduction",
        {"kernel_time_pct": 0.4, "counter_data_available": True},
    )

    assert prediction["predicted_speedup_range"]
    assert receipt.status == "error"
    assert "configuration failed" in (receipt.detail or "")


def test_durable_prediction_rehydrates_with_redacted_provenance(
    monkeypatch,
    tmp_path,
):
    policy = _policy(monkeypatch, tmp_path)
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")

    prediction, receipt = predict_change_impact_durable(
        str(trace),
        "kernel_x",
        "vgpr_reduction",
        {"kernel_time_pct": 0.4, "counter_data_available": True},
        policy=policy,
    )
    assert receipt.status == "persisted"
    assert prediction["baseline_db"] == str(trace)

    predict_impact._reset_store_for_tests()
    retained = predict_impact.explain_prediction(prediction["prediction_id"])

    assert retained["prediction_id"] == prediction["prediction_id"]
    assert retained["predicted_speedup_range"] == prediction["predicted_speedup_range"]
    assert retained["baseline_db"] == trace.name
    assert retained["provenance_redacted"] is True


def test_identical_observation_increments_seen_count(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path)
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    prediction = _prediction(str(trace))
    snapshot = SourceSnapshot.capture(trace, role="baseline")

    first = record_prediction(
        prediction,
        change_params={"kernel_time_pct": 0.4, "counter_data_available": True},
        source_snapshots=[snapshot],
        policy=policy,
    )
    second = record_prediction(
        prediction,
        change_params={"kernel_time_pct": 0.4, "counter_data_available": True},
        source_snapshots=[snapshot],
        policy=policy,
    )

    assert first.record_id == second.record_id
    event = ObservationStore(policy).get_observation(first.record_id)
    assert event["seen_count"] == 2


def test_same_prediction_in_two_scopes_has_distinct_records(monkeypatch, tmp_path):
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    prediction = _prediction(str(trace))
    snapshot = SourceSnapshot.capture(trace, role="baseline")

    first = _policy(monkeypatch, tmp_path, project=tmp_path / "project-a")
    first_receipt = record_prediction(
        prediction,
        source_snapshots=[snapshot],
        policy=first,
    )
    second = _policy(monkeypatch, tmp_path, project=tmp_path / "project-b")
    second_receipt = record_prediction(
        prediction,
        source_snapshots=[snapshot],
        policy=second,
    )

    assert first_receipt.record_id != second_receipt.record_id
    assert first_receipt.external_id == second_receipt.external_id
    assert len(ObservationStore(first).query()) == 1
    assert len(ObservationStore(second).query()) == 1


def test_durable_explain_distinguishes_scope_mismatch(monkeypatch, tmp_path):
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    first = _policy(monkeypatch, tmp_path, project=tmp_path / "project-a")
    prediction, receipt = predict_change_impact_durable(
        str(trace),
        "kernel_x",
        "vgpr_reduction",
        {"kernel_time_pct": 0.4, "counter_data_available": True},
        policy=first,
    )
    assert receipt.persisted
    predict_impact._reset_store_for_tests()
    monkeypatch.setenv("PERFXPERT_PROJECT_ROOT", str(tmp_path / "project-b"))

    with pytest.raises(ScopeMismatchError):
        predict_impact.explain_prediction(prediction["prediction_id"])


def test_source_change_refuses_persistence(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path)
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"before")
    snapshot = SourceSnapshot.capture(trace, role="baseline")
    trace.write_bytes(b"after with different size")

    receipt = record_prediction(
        _prediction(str(trace)),
        source_snapshots=[snapshot],
        policy=policy,
    )

    assert receipt.status == "error"
    assert "source metadata changed" in (receipt.detail or "")
    assert not policy.db_path.exists()


def test_prediction_identity_is_revalidated_before_write(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path)
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    prediction = _prediction(str(trace))
    forged = {**prediction, "confidence": 0.123}

    receipt = record_prediction(
        forged,
        source_snapshots=[SourceSnapshot.capture(trace, role="baseline")],
        policy=policy,
    )

    assert receipt.status == "error"
    assert "does not match" in (receipt.detail or "")
    assert not policy.db_path.exists()


def test_explicit_recorders_require_precompute_snapshots(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path)
    prediction = _prediction(str(tmp_path / "trace.db"))

    receipt = record_prediction(prediction, policy=policy)

    assert receipt.status == "error"
    assert "pre-compute" in (receipt.detail or "")
    assert not policy.db_path.exists()

    comparison_receipt = record_comparison(
        {"verdict": "neutral"},
        baseline_db="before.db",
        new_db="after.db",
        policy=policy,
    )
    assert comparison_receipt.status == "error"
    assert "pre-compute" in (comparison_receipt.detail or "")


def test_snapshots_must_match_claimed_prediction_and_comparison_paths(
    monkeypatch,
    tmp_path,
):
    policy = _policy(monkeypatch, tmp_path)
    baseline = tmp_path / "baseline.db"
    candidate = tmp_path / "candidate.db"
    unrelated = tmp_path / "unrelated.db"
    for path in (baseline, candidate, unrelated):
        path.write_bytes(b"trace")
    prediction = _prediction(str(baseline))

    prediction_receipt = record_prediction(
        prediction,
        source_snapshots=[SourceSnapshot.capture(unrelated, role="baseline")],
        policy=policy,
    )
    assert prediction_receipt.status == "error"
    assert "does not match" in (prediction_receipt.detail or "")

    comparison_receipt = record_comparison(
        {"verdict": "neutral"},
        baseline_db=str(baseline),
        new_db=str(candidate),
        source_snapshots=[
            SourceSnapshot.capture(unrelated, role="baseline"),
            SourceSnapshot.capture(candidate, role="candidate"),
        ],
        policy=policy,
    )
    assert comparison_receipt.status == "error"
    assert "does not match" in (comparison_receipt.detail or "")


def test_analysis_projection_redacts_nested_paths(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path)
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    receipt = record_analysis(
        {
            "hotspots": [
                {
                    "name": "kernel",
                    "source_locations": [{"file": "/secret/src/kernel.cpp"}],
                }
            ],
            "hardware_counters": {"metrics": {"gfx_id": "gfx942"}},
        },
        primary_bottleneck="compute",
        source_snapshots=[SourceSnapshot.capture(trace, role="input")],
        options={"provider": "openai", "airgap": False},
        model_derived=True,
        policy=policy,
    )

    assert receipt.persisted
    payload = ObservationStore(policy).get_observation(receipt.record_id)["payload"]
    location = payload["deterministic_evidence"]["hotspots"][0]["source_locations"][0]
    assert location["file"] == "kernel.cpp"
    assert payload["options"]["provider"] == "openai"
    assert ObservationStore(policy).query(gfx_id="gfx942")[0]["record_id"] == receipt.record_id


def test_unserializable_analysis_returns_error_receipt(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path)
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")

    receipt = record_analysis(
        {"time_breakdown": {"kernel_percent": float("nan")}},
        primary_bottleneck="compute",
        source_snapshots=[SourceSnapshot.capture(trace, role="input")],
        policy=policy,
    )

    assert receipt.status == "error"
    assert "non-finite" in (receipt.detail or "")


def test_analysis_projection_drops_att_reason_and_source_text(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path)
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    receipt = record_analysis(
        {
            "thread_trace": {
                "has_att_data": False,
                "reason": "No files found in /secret/att.",
                "source_line": "password = 'secret'",
                "kernels": [
                    {
                        "name": "kernel",
                        "csv_file": "/secret/att/stats.csv",
                        "error": "Permission denied: '/secret/att/stats.csv'",
                        "top_stalling_instructions": [{"pc_offset": "0x10", "source_line": "secret"}],
                        "avg_stall_ratio": 0.5,
                    }
                ],
            }
        },
        primary_bottleneck="compute",
        source_snapshots=[SourceSnapshot.capture(trace, role="input")],
        policy=policy,
    )

    assert receipt.persisted
    payload = ObservationStore(policy).get_observation(receipt.record_id)["payload"]
    thread_trace = payload["deterministic_evidence"]["thread_trace"]
    assert thread_trace == {
        "has_att_data": False,
        "kernels": [{"name": "kernel", "avg_stall_ratio": 0.5}],
    }
    assert "/secret/att" not in str(payload)


def test_raw_att_path_is_not_accepted_by_recorder(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path)
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    receipt = record_analysis(
        {"hotspots": []},
        primary_bottleneck="compute",
        source_snapshots=[SourceSnapshot.capture(trace, role="input")],
        options={"att_dir": "/secret/att"},
        policy=policy,
    )

    assert receipt.status == "error"
    assert "raw att_dir" in (receipt.detail or "")
    assert not policy.db_path.exists()


def test_att_file_snapshots_participate_without_storing_directory_path(
    monkeypatch,
    tmp_path,
):
    policy = _policy(monkeypatch, tmp_path)
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    att_dir = tmp_path / "private-att"
    att_dir.mkdir()
    (att_dir / "stats.json").write_text('{"stall": 1}')
    snapshots = [SourceSnapshot.capture(trace, role="input")]
    snapshots.extend(capture_directory_sources(att_dir, role="att_input"))

    receipt = record_analysis(
        {"hotspots": []},
        primary_bottleneck="compute",
        source_snapshots=snapshots,
        options={"att_enabled": True, "att_file_count": 1},
        policy=policy,
    )

    assert receipt.persisted
    payload = ObservationStore(policy).get_observation(receipt.record_id)["payload"]
    assert payload["options"]["att_enabled"] is True
    assert "/private-att/" not in str(payload)


def test_att_manifest_detects_files_added_during_analysis(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path)
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    att_dir = tmp_path / "att"
    att_dir.mkdir()
    (att_dir / "before.json").write_text("{}")
    snapshots = [SourceSnapshot.capture(trace, role="input")]
    snapshots.extend(capture_directory_sources(att_dir, role="att_input"))
    (att_dir / "after.json").write_text("{}")

    receipt = record_analysis(
        {"thread_trace": {"has_att_data": True}},
        primary_bottleneck="compute",
        source_snapshots=snapshots,
        options={"att_enabled": True, "att_file_count": 1},
        policy=policy,
    )

    assert receipt.status == "error"
    assert "source metadata changed" in (receipt.detail or "")


def test_store_schema_permissions_and_rollback_mode(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path)
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    _, receipt = predict_change_impact_durable(
        str(trace),
        "kernel_x",
        "vgpr_reduction",
        {"kernel_time_pct": 0.4, "counter_data_available": True},
        policy=policy,
    )
    assert receipt.persisted

    with sqlite3.connect(policy.db_path) as conn:
        assert conn.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
        assert conn.execute("PRAGMA user_version").fetchone()[0] == STORE_SCHEMA_VERSION
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    if os.name == "posix":
        assert policy.db_path.stat().st_mode & 0o777 == 0o600
        assert Path(policy.root).stat().st_mode & 0o777 == 0o700


def test_reader_detects_payload_tampering(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path)
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    _, receipt = predict_change_impact_durable(
        str(trace),
        "kernel_x",
        "vgpr_reduction",
        {"kernel_time_pct": 0.4, "counter_data_available": True},
        policy=policy,
    )
    with sqlite3.connect(policy.db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM knowledge_observations WHERE record_id = ?",
            (receipt.record_id,),
        ).fetchone()
        conn.execute(
            "UPDATE knowledge_observations SET payload_json = ? WHERE record_id = ?",
            (row[0].replace('"confidence":0.7', '"confidence":0.1'), receipt.record_id),
        )

    with pytest.raises(StoreCorruptError):
        ObservationStore(policy).get_observation(receipt.record_id)


def test_read_tools_do_not_create_missing_store(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path)

    assert knowledge_history.query_knowledge() == []
    stats = knowledge_history.knowledge_stats()

    assert stats["records"] == 0
    assert not Path(policy.root).exists()


def test_concurrent_duplicate_writers_increment_atomically(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path)
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    prediction = _prediction(str(trace))
    snapshot = SourceSnapshot.capture(trace, role="baseline")

    def _record_once(_):
        return record_prediction(
            prediction,
            source_snapshots=[snapshot],
            policy=policy,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(_record_once, range(8)))

    assert all(receipt.persisted for receipt in receipts)
    assert len({receipt.record_id for receipt in receipts}) == 1
    event = ObservationStore(policy).get_observation(receipts[0].record_id)
    assert event["seen_count"] == 8


def test_reader_refuses_unrelated_and_newer_databases(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path)
    Path(policy.root).mkdir(parents=True)
    with sqlite3.connect(policy.db_path) as conn:
        conn.execute("CREATE TABLE unrelated (id INTEGER)")
    os.chmod(policy.db_path, 0o644)
    with pytest.raises(WrongStoreError):
        ObservationStore(policy).stats()
    with pytest.raises(WrongStoreError):
        ObservationStore(policy).write_observation(
            record_id="r" * 64,
            observation_key="o" * 64,
            payload_hash="p" * 64,
            kind="prediction",
            producer_version="test",
            external_id=None,
            deterministic=True,
            confidence=None,
            payload={},
        )
    assert policy.db_path.stat().st_mode & 0o777 == 0o644

    policy.db_path.unlink()
    with sqlite3.connect(policy.db_path) as conn:
        conn.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        conn.execute(f"PRAGMA user_version = {STORE_SCHEMA_VERSION + 1}")
    with pytest.raises(UnsupportedSchemaError):
        ObservationStore(policy).stats()


def test_writer_refuses_unrelated_view_before_schema_creation(monkeypatch, tmp_path):
    policy = _policy(monkeypatch, tmp_path)
    Path(policy.root).mkdir(parents=True)
    with sqlite3.connect(policy.db_path) as conn:
        conn.execute("CREATE VIEW unrelated AS SELECT 1 AS value")
        conn.execute(f"PRAGMA application_id = {APPLICATION_ID}")

    with pytest.raises(WrongStoreError):
        ObservationStore(policy).write_observation(
            record_id="r" * 64,
            observation_key="o" * 64,
            payload_hash="p" * 64,
            kind="prediction",
            producer_version="test",
            external_id=None,
            deterministic=True,
            confidence=None,
            payload={},
        )
    with sqlite3.connect(policy.db_path) as conn:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")}
    assert names == {"unrelated"}


def test_quota_and_lifecycle_operations(monkeypatch, tmp_path):
    base_policy = _policy(monkeypatch, tmp_path)
    tiny_policy = replace(base_policy, max_bytes=1)
    store = ObservationStore(tiny_policy)
    observation_key = "o" * 64
    payload = {"prediction": {"prediction_id": "e" * 64}}
    payload_hash_value = payload_hash({"payload": payload, "tags": ()})
    record_id = retained_record_id(
        scope_id=tiny_policy.scope.scope_id,
        observation_key_value=observation_key,
        payload_hash_value=payload_hash_value,
    )
    with pytest.raises(QuotaExceededError):
        store.write_observation(
            record_id=record_id,
            observation_key=observation_key,
            payload_hash=payload_hash_value,
            kind="prediction",
            producer_version="test",
            external_id="e" * 64,
            deterministic=True,
            confidence=0.5,
            payload=payload,
        )

    policy = replace(base_policy, max_bytes=1024 * 1024)
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    prediction = _prediction(str(trace))
    receipt = record_prediction(
        prediction,
        source_snapshots=[SourceSnapshot.capture(trace, role="baseline")],
        policy=policy,
    )
    assert receipt.persisted

    future = datetime.now(timezone.utc) + timedelta(days=1)
    assert ObservationStore(policy).prune(older_than=future) == 1
    with pytest.raises(ObservationNotFoundError):
        ObservationStore(policy).get_observation(receipt.record_id)
