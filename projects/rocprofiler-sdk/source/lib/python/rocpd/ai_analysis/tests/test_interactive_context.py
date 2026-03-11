# projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/tests/test_interactive_context.py
"""Tests for SessionContext and SessionData.context persistence."""
from __future__ import annotations

import dataclasses
import unittest

from rocpd.ai_analysis.interactive import (
    HistoryEntry,
    SessionContext,
    SessionData,
    SessionStore,
)


class TestSessionContext(unittest.TestCase):
    """SessionContext serialization and reconstruction."""

    def test_default_fields(self):
        ctx = SessionContext()
        self.assertEqual(ctx.iteration, 0)
        self.assertEqual(ctx.analyses, [])
        self.assertEqual(ctx.suggestions_given, [])
        self.assertEqual(ctx.commands_run, [])

    def test_round_trip_serialization(self):
        ctx = SessionContext(
            iteration=2,
            analyses=[{"db": "foo.db", "kernel_pct": 5.0, "top_issue": "IDLE", "top_priority": "HIGH", "memcpy_pct": 0.1, "idle_pct": 90.0}],
            suggestions_given=["Increase parallelism"],
            commands_run=[{"cmd": "rocprofv3 --sys-trace -- ./app", "exit_code": 0}],
        )
        d = dataclasses.asdict(ctx)
        restored = SessionContext(**d)
        self.assertEqual(restored.iteration, 2)
        self.assertEqual(restored.analyses[0]["db"], "foo.db")
        self.assertEqual(restored.suggestions_given[0], "Increase parallelism")
        self.assertEqual(restored.commands_run[0]["exit_code"], 0)

    def test_from_dict_missing_keys_backward_compat(self):
        # Old session file has no context key — context field is None (backward compatible)
        old_session_dict = {
            "session_id": "2026-01-01_my_app",
            "source_dir": "/src",
            "created_at": "2026-01-01T00:00:00+00:00",
            "last_updated": "2026-01-01T00:00:00+00:00",
        }
        sd = SessionData.from_dict(old_session_dict)
        self.assertIsNone(sd.context)

    def test_session_data_context_field_round_trip(self):
        ctx = SessionContext(iteration=1, analyses=[{"db": "x.db", "kernel_pct": 10.0, "memcpy_pct": 0.0, "idle_pct": 80.0, "top_issue": "GPU IDLE", "top_priority": "HIGH"}])
        sd = SessionData(
            session_id="test-id",
            source_dir="/src",
            created_at="2026-01-01T00:00:00+00:00",
            last_updated="2026-01-01T00:00:00+00:00",
            context=dataclasses.asdict(ctx),
        )
        d = sd.to_dict()
        self.assertIn("context", d)
        self.assertEqual(d["context"]["iteration"], 1)

        # Reconstruct
        sd2 = SessionData.from_dict(d)
        self.assertIsNotNone(sd2.context)
        self.assertEqual(sd2.context["iteration"], 1)
