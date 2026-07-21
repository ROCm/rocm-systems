from __future__ import annotations

import argparse
import json

from perfxpert.cli import knowledge_cmd
from perfxpert.config import PerfXpertConfig
from perfxpert.retention import (
    SourceSnapshot,
    build_retention_policy,
    record_prediction,
)
from perfxpert.tools import predict_impact


def _args(**values):
    defaults = {
        "knowledge_action": "stats",
        "scope": "current",
    }
    defaults.update(values)
    return argparse.Namespace(**defaults)


def test_stats_on_missing_store_is_empty(capsys):
    assert knowledge_cmd.run_knowledge(_args()) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["records"] == 0


def test_query_and_clear_current_scope(capsys, tmp_path):
    trace = tmp_path / "trace.db"
    trace.write_bytes(b"trace")
    prediction = predict_impact.predict_change_impact(
        str(trace),
        "kernel_cli",
        "vgpr_reduction",
        {"kernel_time_pct": 0.4, "counter_data_available": True},
    )
    policy = build_retention_policy(config=PerfXpertConfig())
    record_prediction(
        prediction,
        source_snapshots=[SourceSnapshot.capture(trace, role="baseline")],
        policy=policy,
    )

    query_args = _args(
        knowledge_action="query",
        kind="prediction",
        kernel_name="kernel_cli",
        gfx_id=None,
        change_type=None,
        verdict=None,
        limit=50,
    )
    assert knowledge_cmd.run_knowledge(query_args) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1

    clear_args = _args(
        knowledge_action="clear",
        yes=True,
        compact=False,
    )
    assert knowledge_cmd.run_knowledge(clear_args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["deleted"] == 1


def test_clear_requires_confirmation(capsys):
    result = knowledge_cmd.run_knowledge(_args(knowledge_action="clear", yes=False, compact=False))
    assert result == 2
    assert "requires --yes" in capsys.readouterr().err
