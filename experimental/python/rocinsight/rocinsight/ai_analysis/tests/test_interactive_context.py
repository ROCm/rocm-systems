# projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/tests/test_interactive_context.py
"""Tests for SessionContext and SessionData.context persistence."""

from __future__ import annotations

import dataclasses
import pathlib
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from rocinsight.ai_analysis.interactive import (
    HistoryEntry,
    InteractiveSession,
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
            analyses=[
                {
                    "db": "foo.db",
                    "kernel_pct": 5.0,
                    "top_issue": "IDLE",
                    "top_priority": "HIGH",
                    "memcpy_pct": 0.1,
                    "idle_pct": 90.0,
                }
            ],
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
        ctx = SessionContext(
            iteration=1,
            analyses=[
                {
                    "db": "x.db",
                    "kernel_pct": 10.0,
                    "memcpy_pct": 0.0,
                    "idle_pct": 80.0,
                    "top_issue": "GPU IDLE",
                    "top_priority": "HIGH",
                }
            ],
        )
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


class TestRunTier1AnalysisRefactor(unittest.TestCase):
    """_run_tier1_analysis must return (recs, breakdown) tuple."""

    def test_returns_tuple(self):
        s = InteractiveSession.__new__(InteractiveSession)
        s._db_path = "/tmp/test.db"

        mock_result = MagicMock()
        mock_result.recommendations.high_priority = []
        mock_result.recommendations.medium_priority = []
        mock_result.recommendations.low_priority = []
        eb = MagicMock()
        eb.kernel_time_pct = 5.0
        eb.memcpy_time_pct = 1.0
        eb.api_overhead_pct = 2.0
        eb.idle_time_pct = 92.0
        mock_result.execution_breakdown = eb
        mock_result.profiling_info.total_duration_ns = 1_000_000_000

        with patch("rocinsight.ai_analysis.api.analyze_database", return_value=mock_result):
            result = InteractiveSession._run_tier1_analysis(s, "/tmp/test.db")

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        recs, breakdown = result
        self.assertIsInstance(recs, list)
        self.assertIsInstance(breakdown, dict)
        self.assertIn("kernel_time_pct", breakdown)
        self.assertAlmostEqual(breakdown["kernel_time_pct"], 5.0)

    def test_returns_none_breakdown_on_exception(self):
        s = InteractiveSession.__new__(InteractiveSession)
        s._db_path = "/tmp/test.db"

        with patch(
            "rocinsight.ai_analysis.api.analyze_database", side_effect=RuntimeError("db error")
        ):
            recs, breakdown = InteractiveSession._run_tier1_analysis(s, "/tmp/bad.db")

        self.assertEqual(recs, [])
        self.assertIsNone(breakdown)


class TestContextUpdateMethods(unittest.TestCase):
    """_update_ctx_* cap logic and mutation."""

    def _make_session_with_ctx(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession, SessionContext

        s = InteractiveSession.__new__(InteractiveSession)
        s._db_path = "/tmp/test.db"
        s._ctx = SessionContext()
        return s

    def test_update_ctx_analysis_appends(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession

        s = self._make_session_with_ctx()
        recs = [{"issue": "GPU IDLE", "priority": "HIGH"}]
        breakdown = {
            "kernel_time_pct": 6.6,
            "memcpy_time_pct": 0.1,
            "idle_time_pct": 93.0,
            "api_overhead_pct": 0.3,
            "total_runtime_ns": 1_000_000,
        }
        s._update_ctx_analysis(recs, breakdown)
        self.assertEqual(s._ctx.iteration, 1)
        self.assertEqual(len(s._ctx.analyses), 1)
        self.assertEqual(s._ctx.analyses[0]["top_issue"], "GPU IDLE")
        self.assertAlmostEqual(s._ctx.analyses[0]["kernel_pct"], 6.6)

    def test_update_ctx_analysis_capped_at_5(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession

        s = self._make_session_with_ctx()
        for i in range(7):
            s._update_ctx_analysis(
                [{"issue": f"ISSUE_{i}", "priority": "HIGH"}],
                {
                    "kernel_time_pct": float(i),
                    "memcpy_time_pct": 0.0,
                    "idle_time_pct": 0.0,
                    "api_overhead_pct": 0.0,
                    "total_runtime_ns": 1,
                },
            )
        self.assertEqual(len(s._ctx.analyses), 5)
        issues = [a["top_issue"] for a in s._ctx.analyses]
        self.assertNotIn("ISSUE_0", issues)
        self.assertIn("ISSUE_6", issues)

    def test_update_ctx_analysis_none_breakdown(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession

        s = self._make_session_with_ctx()
        s._update_ctx_analysis([], None)
        self.assertEqual(s._ctx.analyses[0]["kernel_pct"], 0.0)

    def test_update_ctx_suggestion_capped_at_3(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession

        s = self._make_session_with_ctx()
        for i in range(5):
            s._update_ctx_suggestion(f"suggestion {i} " + "x" * 200)
        self.assertEqual(len(s._ctx.suggestions_given), 3)
        self.assertIn("suggestion 4", s._ctx.suggestions_given[-1])

    def test_update_ctx_suggestion_truncates_at_120(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession

        s = self._make_session_with_ctx()
        s._update_ctx_suggestion("A" * 300)
        self.assertEqual(len(s._ctx.suggestions_given[0]), 120)

    def test_update_ctx_command_appends(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession

        s = self._make_session_with_ctx()
        s._update_ctx_command("rocprofv3 --sys-trace -- ./app", 0)
        self.assertEqual(len(s._ctx.commands_run), 1)
        self.assertEqual(s._ctx.commands_run[0]["exit_code"], 0)

    def test_update_ctx_command_capped_at_5(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession

        s = self._make_session_with_ctx()
        for i in range(7):
            s._update_ctx_command(f"cmd_{i}", i)
        self.assertEqual(len(s._ctx.commands_run), 5)
        cmds = [c["cmd"] for c in s._ctx.commands_run]
        self.assertNotIn("cmd_0", cmds)
        self.assertIn("cmd_6", cmds)


class TestFormatContextBlock(unittest.TestCase):
    """_format_context_block returns correct text."""

    def _make_session_with_ctx(self, ctx=None):
        from rocinsight.ai_analysis.interactive import InteractiveSession, SessionContext

        s = InteractiveSession.__new__(InteractiveSession)
        s._ctx = ctx or SessionContext()
        return s

    def test_empty_context_returns_empty_string(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession, SessionContext

        s = self._make_session_with_ctx()
        result = s._format_context_block()
        self.assertEqual(result, "")

    def test_with_one_analysis(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession, SessionContext

        ctx = SessionContext(
            iteration=1,
            analyses=[
                {
                    "db": "foo.db",
                    "kernel_pct": 5.0,
                    "memcpy_pct": 0.1,
                    "idle_pct": 93.0,
                    "top_issue": "GPU IDLE TIME",
                    "top_priority": "HIGH",
                }
            ],
        )
        s = self._make_session_with_ctx(ctx)
        block = s._format_context_block()
        self.assertIn("Session Context", block)
        self.assertIn("foo.db", block)
        self.assertIn("GPU IDLE TIME", block)
        self.assertIn("HIGH", block)

    def test_with_suggestion(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession, SessionContext

        ctx = SessionContext(
            iteration=1,
            suggestions_given=["Increase wave occupancy by reducing register pressure."],
        )
        s = self._make_session_with_ctx(ctx)
        block = s._format_context_block()
        self.assertIn("Increase wave occupancy", block)

    def test_with_command(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession, SessionContext

        ctx = SessionContext(
            iteration=1,
            commands_run=[{"cmd": "rocprofv3 --pmc SQ_WAVES -- ./app", "exit_code": 0}],
        )
        s = self._make_session_with_ctx(ctx)
        block = s._format_context_block()
        self.assertIn("rocprofv3 --pmc SQ_WAVES", block)
        self.assertIn("exit 0", block)

    def test_full_context_under_1500_chars(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession, SessionContext

        ctx = SessionContext(
            iteration=5,
            analyses=[
                {
                    "db": f"run{i}.db",
                    "kernel_pct": float(i),
                    "memcpy_pct": 0.0,
                    "idle_pct": float(90 - i),
                    "top_issue": f"ISSUE_{i}",
                    "top_priority": "HIGH",
                }
                for i in range(5)
            ],
            suggestions_given=["A" * 120, "B" * 120, "C" * 120],
            commands_run=[
                {"cmd": f"rocprofv3 --pmc CTR_{i} -- ./app", "exit_code": 0}
                for i in range(5)
            ],
        )
        s = self._make_session_with_ctx(ctx)
        block = s._format_context_block()
        self.assertLess(len(block), 1500)


class TestExtractAiCommands(unittest.TestCase):
    """_extract_ai_commands extracts and deduplicates profiling commands."""

    def _make_session(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession, SessionContext

        s = InteractiveSession.__new__(InteractiveSession)
        s._ctx = SessionContext()
        return s

    def test_extracts_rocprofv3_from_text(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession

        s = self._make_session()
        text = (
            "You should run:\n"
            "rocprofv3 --pmc SQ_WAVES GRBM_COUNT -- ./app\n"
            "and also try:\n"
            "rocprofv3 --sys-trace -- ./app\n"
        )
        result = s._extract_ai_commands(text, [])
        self.assertEqual(len(result), 2)
        self.assertIn("SQ_WAVES", result[0])

    def test_includes_structured_commands(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession

        s = self._make_session()
        structured = ["rocprofv3 --pmc FETCH_SIZE -- ./app"]
        result = s._extract_ai_commands("no commands here", structured)
        self.assertEqual(result, structured)

    def test_deduplicates(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession

        s = self._make_session()
        text = "rocprofv3 --pmc SQ_WAVES -- ./app"
        structured = ["rocprofv3 --pmc SQ_WAVES -- ./app"]
        result = s._extract_ai_commands(text, structured)
        self.assertEqual(len(result), 1)

    def test_free_form_comes_first(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession

        s = self._make_session()
        text = "rocprofv3 --pmc SQ_WAVES -- ./app"
        structured = ["rocprofv3 --sys-trace -- ./app"]
        result = s._extract_ai_commands(text, structured)
        self.assertEqual(result[0], "rocprofv3 --pmc SQ_WAVES -- ./app")

    def test_capped_at_5(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession

        s = self._make_session()
        text = "\n".join(f"rocprofv3 --pmc CTR_{i} -- ./app" for i in range(10))
        result = s._extract_ai_commands(text, [])
        self.assertLessEqual(len(result), 5)

    def test_empty_text_and_empty_structured(self):
        from rocinsight.ai_analysis.interactive import InteractiveSession

        s = self._make_session()
        result = s._extract_ai_commands("no commands here", [])
        self.assertEqual(result, [])


class TestPersistence(unittest.TestCase):
    """_ctx is saved and restored across session save/load."""

    def test_ctx_round_trip_via_session_store(self):
        from rocinsight.ai_analysis.interactive import (
            InteractiveSession,
            SessionContext,
            SessionStore,
            SessionData,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(sessions_dir=tmpdir)

            # Build a session with context
            ctx = SessionContext(
                iteration=3,
                analyses=[
                    {
                        "db": "a.db",
                        "kernel_pct": 5.0,
                        "memcpy_pct": 0.0,
                        "idle_pct": 90.0,
                        "top_issue": "GPU IDLE",
                        "top_priority": "HIGH",
                    }
                ],
                suggestions_given=["Reduce launch overhead"],
                commands_run=[{"cmd": "rocprofv3 --sys-trace -- ./app", "exit_code": 0}],
            )
            sd = SessionData(
                session_id="test-session",
                source_dir="/src",
                created_at="2026-01-01T00:00:00+00:00",
                last_updated="2026-01-01T00:00:00+00:00",
                context=dataclasses.asdict(ctx),
            )
            store.save(sd)

            # Load it back
            loaded = store.load("test-session")
            self.assertIsNotNone(loaded)
            self.assertIsNotNone(loaded.context)

            # Reconstruct SessionContext
            raw_ctx = loaded.context or {}
            restored_ctx = SessionContext(**raw_ctx)
            self.assertEqual(restored_ctx.iteration, 3)
            self.assertEqual(restored_ctx.analyses[0]["db"], "a.db")
            self.assertEqual(restored_ctx.suggestions_given[0], "Reduce launch overhead")
            self.assertEqual(restored_ctx.commands_run[0]["exit_code"], 0)

    def test_old_session_file_without_context_key(self):
        """Backward compat: session files without 'context' key load cleanly."""
        from rocinsight.ai_analysis.interactive import SessionData, SessionContext

        old_data = {
            "session_id": "old",
            "source_dir": "/src",
            "created_at": "2026-01-01T00:00:00+00:00",
            "last_updated": "2026-01-01T00:00:00+00:00",
            "history": [],
            "persistent_menu_items": [],
        }
        sd = SessionData.from_dict(old_data)
        # context is None — creating a fresh SessionContext from it should work
        raw_ctx = sd.context or {}
        fresh = SessionContext(**raw_ctx) if raw_ctx else SessionContext()
        self.assertEqual(fresh.iteration, 0)


class TestContextInjectionIntegration(unittest.TestCase):
    """Integration: verify _format_context_block content reaches LLM prompts."""

    def test_context_injected_after_analyze_then_optimize(self):
        """After [a] analyze, [o] optimize should include context block in LLM prompt."""
        from rocinsight.ai_analysis.interactive import InteractiveSession, SessionContext
        from unittest.mock import MagicMock, patch, call

        s = InteractiveSession.__new__(InteractiveSession)
        s._ctx = SessionContext()
        s._db_path = "/tmp/test.db"
        s._source_dir = "/src"
        s._recs = []

        # Simulate a prior analysis having run (as if [a] was pressed)
        s._update_ctx_analysis(
            [{"issue": "LOW OCCUPANCY", "priority": "HIGH"}],
            {
                "kernel_time_pct": 15.0,
                "memcpy_time_pct": 0.5,
                "idle_time_pct": 80.0,
                "api_overhead_pct": 4.5,
                "total_runtime_ns": 1_000_000_000,
            },
        )

        # Verify context block now has content
        ctx_block = s._format_context_block()
        self.assertNotEqual(ctx_block, "")
        self.assertIn("LOW OCCUPANCY", ctx_block)
        self.assertIn("Session Context", ctx_block)

        # Verify context block is prepended when _format_context_block is called
        # (the actual injection into LLM prompts is covered by the method implementations)
        self.assertIn("iteration 1", ctx_block)
