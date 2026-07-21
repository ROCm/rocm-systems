from __future__ import annotations

import inspect

from perfxpert.retention import (
    SourceSnapshot,
    build_retention_policy,
    record_prediction,
)
from perfxpert.tools import knowledge_history, predict_impact
from perfxpert.tools._class import ToolClass


def test_knowledge_history_tools_are_current_scope_read_only():
    for fn in (
        knowledge_history.get_knowledge_observation,
        knowledge_history.query_knowledge,
        knowledge_history.knowledge_stats,
    ):
        assert fn.__tool_class__ == ToolClass.READ_ONLY
        assert "scope" not in inspect.signature(fn).parameters


def test_query_knowledge_filters_exact_tags(tmp_path):
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    prediction = predict_impact.predict_change_impact(
        str(trace),
        "kernel_exact",
        "vgpr_reduction",
        {"kernel_time_pct": 0.4, "counter_data_available": True},
    )
    policy = build_retention_policy()
    receipt = record_prediction(
        prediction,
        source_snapshots=[SourceSnapshot.capture(trace, role="baseline")],
        policy=policy,
    )

    assert receipt.persisted
    rows = knowledge_history.query_knowledge(
        kind="prediction",
        kernel_name="kernel_exact",
        change_type="vgpr_reduction",
    )
    assert [row["record_id"] for row in rows] == [receipt.record_id]
    assert knowledge_history.query_knowledge(kernel_name="other") == []
    assert knowledge_history.get_knowledge_observation(receipt.record_id)["record_id"] == receipt.record_id
