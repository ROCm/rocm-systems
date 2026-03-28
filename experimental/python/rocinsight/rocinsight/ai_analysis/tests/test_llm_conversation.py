# ai_analysis/tests/test_llm_conversation.py
"""Tests for LLMConversation persistent streaming session."""

from __future__ import annotations
import importlib.util
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rocinsight.ai_analysis.llm_conversation import LLMConversation

_has_anthropic = importlib.util.find_spec("anthropic") is not None
_has_openai = importlib.util.find_spec("openai") is not None

_skip_no_anthropic = unittest.skipUnless(
    _has_anthropic, "anthropic package not installed"
)
_skip_no_openai = unittest.skipUnless(_has_openai, "openai package not installed")

# ── Helpers ──────────────────────────────────────────────────────────────────


class _MockAnthropicStream:
    """Simulates anthropic.messages.stream() context manager."""

    def __init__(self, chunks):
        self._chunks = chunks

    @property
    def text_stream(self):
        return iter(self._chunks)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class _MockAnthropicMessages:
    def __init__(self, chunks):
        self._chunks = chunks

    def stream(self, **kwargs):
        return _MockAnthropicStream(self._chunks)

    def create(self, **kwargs):
        text = "".join(self._chunks)
        block = SimpleNamespace(text=text, type="text")
        return SimpleNamespace(content=[block])


class _MockAnthropicClient:
    def __init__(self, chunks):
        self.messages = _MockAnthropicMessages(chunks)


def _openai_chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class _MockOpenAICompletions:
    def __init__(self, chunks, raise_first=None):
        self._chunks = chunks
        self._raise_first = raise_first
        self._call_count = 0

    def create(self, **kwargs):
        if self._raise_first and self._call_count == 0:
            self._call_count += 1
            raise self._raise_first
        self._call_count += 1
        return iter([_openai_chunk(c) for c in self._chunks])


class _MockOpenAIChat:
    def __init__(self, chunks, raise_first=None):
        self.completions = _MockOpenAICompletions(chunks, raise_first)


class _MockOpenAIClient:
    def __init__(self, chunks, raise_first=None):
        self.chat = _MockOpenAIChat(chunks, raise_first)


# ── TestLLMConversation ───────────────────────────────────────────────────────


@_skip_no_anthropic
class TestLLMConversation(unittest.TestCase):
    """Core behavior: initialize, send, message growth, turn_count."""

    def _make_conv(self, provider="anthropic"):
        return LLMConversation(provider=provider, api_key="test-key")

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            LLMConversation(provider="bogus")

    def test_initialize_sets_system(self):
        conv = self._make_conv()
        conv.initialize("You are an expert.")
        self.assertEqual(conv._system, "You are an expert.")

    def test_messages_empty_before_send(self):
        conv = self._make_conv()
        conv.initialize("sys")
        self.assertEqual(conv.messages, [])
        self.assertEqual(conv.turn_count, 0)

    def test_send_appends_user_and_assistant(self):
        conv = self._make_conv("anthropic")
        conv.initialize("sys")
        mock_client = _MockAnthropicClient(["Hello ", "world"])
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = conv.send("Hi there")
        self.assertEqual(result, "Hello world")
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0], {"role": "user", "content": "Hi there"})
        self.assertEqual(
            conv.messages[1], {"role": "assistant", "content": "Hello world"}
        )
        self.assertEqual(conv.turn_count, 1)

    def test_send_multiple_turns_accumulates(self):
        conv = self._make_conv("anthropic")
        conv.initialize("sys")
        mock_client = _MockAnthropicClient(["resp"])
        with patch("anthropic.Anthropic", return_value=mock_client):
            conv.send("turn1")
            conv.send("turn2")
        self.assertEqual(conv.turn_count, 2)
        self.assertEqual(len(conv.messages), 4)

    def test_system_set_once_not_in_messages(self):
        conv = self._make_conv("anthropic")
        conv.initialize("fence content")
        mock_client = _MockAnthropicClient(["ok"])
        with patch("anthropic.Anthropic", return_value=mock_client):
            conv.send("q")
        # system must never appear in _messages
        for msg in conv.messages:
            self.assertNotEqual(msg.get("content"), "fence content")


# ── TestStreaming ─────────────────────────────────────────────────────────────


@_skip_no_anthropic
@_skip_no_openai
class TestStreaming(unittest.TestCase):
    """on_token callback and silent collection."""

    def test_on_token_called_per_chunk_anthropic(self):
        conv = LLMConversation(provider="anthropic", api_key="k")
        conv.initialize("sys")
        received = []
        mock_client = _MockAnthropicClient(["Hello ", "world"])
        with patch("anthropic.Anthropic", return_value=mock_client):
            conv.send("q", on_token=received.append)
        self.assertEqual(received, ["Hello ", "world"])

    def test_on_token_none_silent(self):
        conv = LLMConversation(provider="anthropic", api_key="k")
        conv.initialize("sys")
        mock_client = _MockAnthropicClient(["silent"])
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = conv.send("q", on_token=None)
        self.assertEqual(result, "silent")

    def test_on_token_called_per_chunk_openai(self):
        conv = LLMConversation(provider="openai", api_key="k")
        conv.initialize("sys")
        received = []
        mock_client = _MockOpenAIClient(["Hello ", "world"])
        with patch("openai.OpenAI", return_value=mock_client):
            conv.send("q", on_token=received.append)
        self.assertEqual(received, ["Hello ", "world"])

    def test_local_provider_uses_openai_path(self):
        conv = LLMConversation(provider="local")
        conv.initialize("sys")
        received = []
        mock_client = _MockOpenAIClient(["local-resp"])
        with patch("openai.OpenAI", return_value=mock_client):
            result = conv.send("q", on_token=received.append)
        self.assertEqual(result, "local-resp")
        self.assertEqual(received, ["local-resp"])


# ── TestOpenAIFallback ────────────────────────────────────────────────────────


@_skip_no_openai
class TestOpenAIFallback(unittest.TestCase):
    """max_completion_tokens → max_tokens on BadRequestError."""

    def test_fallback_on_bad_request(self):
        import openai

        conv = LLMConversation(provider="openai", api_key="k")
        conv.initialize("sys")
        bad_error = openai.BadRequestError(
            message="max_completion_tokens not supported",
            response=MagicMock(status_code=400),
            body={"error": {"message": "max_completion_tokens not supported"}},
        )
        mock_client = _MockOpenAIClient(["fallback-ok"], raise_first=bad_error)
        with patch("openai.OpenAI", return_value=mock_client):
            result = conv.send("q")
        self.assertEqual(result, "fallback-ok")
        self.assertEqual(mock_client.chat.completions._call_count, 2)


# ── TestCompaction ────────────────────────────────────────────────────────────


@_skip_no_anthropic
class TestCompaction(unittest.TestCase):
    """Compaction trigger, turn_count not incremented, summary block placement."""

    def _make_conv_with_mock(
        self, provider="anthropic", compact_every=2, keep_recent_turns=1
    ):
        conv = LLMConversation(
            provider=provider,
            api_key="k",
            compact_every=compact_every,
            keep_recent_turns=keep_recent_turns,
        )
        conv.initialize("sys")
        return conv

    def test_compaction_triggered_at_n_turns(self):
        conv = self._make_conv_with_mock(compact_every=2, keep_recent_turns=1)
        mock_client = _MockAnthropicClient(["resp"])
        compact_called = []

        original_compact = conv._compact

        def mock_compact():
            compact_called.append(True)
            original_compact()

        conv._compact = mock_compact

        with patch("anthropic.Anthropic", return_value=mock_client):
            conv.send("turn1")  # turn_count=1, not a multiple of 2 → no compact
            self.assertEqual(compact_called, [])
            conv.send("turn2")  # turn_count=2, 2 % 2 == 0 → compact
        self.assertEqual(len(compact_called), 1)

    def test_turn_count_not_incremented_by_compaction(self):
        conv = self._make_conv_with_mock(compact_every=2, keep_recent_turns=1)
        # Mock _compact to be a no-op (avoids needing a second mock client)
        conv._compact = lambda: None
        mock_client = _MockAnthropicClient(["resp"])
        with patch("anthropic.Anthropic", return_value=mock_client):
            conv.send("t1")
            conv.send("t2")
        self.assertEqual(conv.turn_count, 2)

    def test_compaction_replaces_old_messages_with_summary_block(self):
        conv = self._make_conv_with_mock(compact_every=4, keep_recent_turns=1)
        conv.initialize("sys")
        conv._messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {
                "role": "user",
                "content": "q3",
            },  # recent (keep_recent_turns=1 → keep 2 msgs)
            {"role": "assistant", "content": "a3"},
        ]
        conv._turn_count = 3

        mock_client = _MockAnthropicClient(["compact summary"])
        with patch("anthropic.Anthropic", return_value=mock_client):
            conv._compact()

        self.assertEqual(len(conv.messages), 4)  # 2 summary + 2 recent
        self.assertEqual(conv.messages[0]["role"], "user")
        self.assertEqual(conv.messages[0]["content"], "Summarize our session so far.")
        self.assertIn("[Session summary]", conv.messages[1]["content"])
        self.assertIn("compact summary", conv.messages[1]["content"])
        # Recent messages preserved
        self.assertEqual(conv.messages[2]["content"], "q3")
        self.assertEqual(conv.messages[3]["content"], "a3")

    def test_compaction_does_not_crash_on_failure(self):
        conv = self._make_conv_with_mock(compact_every=2, keep_recent_turns=1)
        conv._messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        conv._turn_count = 2
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            def _fail(**kw):
                raise RuntimeError("API down")

            conv._call_non_streaming = _fail
            conv._compact()
        # Messages unchanged
        self.assertEqual(len(conv.messages), 4)


# ── TestDiskArchive ───────────────────────────────────────────────────────────


@_skip_no_anthropic
class TestDiskArchive(unittest.TestCase):
    """JSONL archive written only when history_path is set."""

    def test_archive_written_on_compaction(self):
        with tempfile.TemporaryDirectory() as td:
            hp = pathlib.Path(td) / "history.jsonl"
            conv = LLMConversation(
                provider="anthropic",
                api_key="k",
                compact_every=4,
                keep_recent_turns=1,
                history_path=hp,
            )
            conv.initialize("sys")
            conv._messages = [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
                {"role": "assistant", "content": "a3"},
            ]
            conv._turn_count = 3
            mock_client = _MockAnthropicClient(["summary"])
            with patch("anthropic.Anthropic", return_value=mock_client):
                conv._compact()

            self.assertTrue(hp.exists())
            lines = hp.read_text().strip().splitlines()
            self.assertGreaterEqual(len(lines), 4)  # at least 4 old messages archived
            entry = json.loads(lines[0])
            self.assertIn("role", entry)
            self.assertIn("content", entry)
            self.assertIn("ts", entry)

    def test_no_archive_when_history_path_none(self):
        conv = LLMConversation(
            provider="anthropic",
            api_key="k",
            compact_every=4,
            keep_recent_turns=1,
            history_path=None,
        )
        conv.initialize("sys")
        conv._messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ]
        conv._turn_count = 3
        mock_client = _MockAnthropicClient(["summary"])
        with patch("anthropic.Anthropic", return_value=mock_client):
            conv._compact()  # Should not raise, should not create any file


# ── TestPersistence ───────────────────────────────────────────────────────────


class TestPersistence(unittest.TestCase):
    """to_dict / from_dict round-trip."""

    def test_round_trip_restores_all_state(self):
        with tempfile.TemporaryDirectory() as td:
            hp = pathlib.Path(td) / "hist.jsonl"
            conv = LLMConversation(
                provider="anthropic",
                api_key="orig-key",
                model="claude-opus-4-6",
                compact_every=5,
                keep_recent_turns=3,
                history_path=hp,
            )
            conv.initialize("fence text")
            conv._messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
            conv._turn_count = 1

            d = conv.to_dict()
            restored = LLMConversation.from_dict(d, api_key="new-key")

            self.assertEqual(restored._provider, "anthropic")
            self.assertEqual(restored._model, "claude-opus-4-6")
            self.assertEqual(restored._system, "fence text")
            self.assertEqual(restored._messages, conv._messages)
            self.assertEqual(restored.turn_count, 1)
            self.assertEqual(restored._compact_every, 5)
            self.assertEqual(restored._keep_recent_turns, 3)
            self.assertEqual(restored._api_key, "new-key")

    def test_from_dict_api_key_override(self):
        conv = LLMConversation(provider="anthropic", api_key="orig")
        conv.initialize("sys")
        d = conv.to_dict()
        restored = LLMConversation.from_dict(d, api_key="override")
        self.assertEqual(restored._api_key, "override")

    def test_from_dict_model_override(self):
        conv = LLMConversation(provider="anthropic", model="claude-opus-4-6")
        conv.initialize("sys")
        d = conv.to_dict()
        restored = LLMConversation.from_dict(d, model="claude-sonnet-4-6")
        self.assertEqual(restored._model, "claude-sonnet-4-6")

    def test_to_dict_does_not_include_api_key(self):
        conv = LLMConversation(provider="anthropic", api_key="sk-secret")
        conv.initialize("sys")
        d = conv.to_dict()
        self.assertNotIn("api_key", d)
        self.assertNotIn("sk-secret", str(d))


# ── TestInteractiveIntegration ────────────────────────────────────────────────


class TestInteractiveIntegration(unittest.TestCase):
    """Integration tests: LLMConversation wired into InteractiveSession."""

    def _make_session(self, mock_conv=None):
        """Build an InteractiveSession with a mocked _conv and a temp session store."""
        from rocinsight.ai_analysis.interactive import InteractiveSession, SessionStore
        import tempfile

        store_dir = tempfile.mkdtemp()
        session = InteractiveSession(
            source_dir="/tmp/fake_src",
            tier0_result=None,
            recommendations=[],
            database_path="",
            llm_provider=None,  # no real LLM; we inject mock directly
            llm_api_key=None,
            llm_model=None,
            session_store=SessionStore(store_dir),
        )
        if mock_conv is not None:
            session._conv = mock_conv
        return session

    def test_conv_send_called_for_annotate_profiling_plan(self):
        mock_conv = MagicMock()
        mock_conv.send.return_value = "some annotation"
        session = self._make_session(mock_conv)

        # Provide minimal tier0 result
        import types

        plan = types.SimpleNamespace(
            programming_model="HIP",
            kernel_count=2,
            suggested_counters=[],
            risk_areas=[],
            detected_patterns=[],
        )
        session._tier0 = plan

        session._llm_annotate_profiling_plan([("counter", "rocprofv3 --pmc A -- ./app")])
        mock_conv.send.assert_called_once()
        call_msg = mock_conv.send.call_args[0][0]
        self.assertIn("Annotate this profiling plan", call_msg)

    def test_conv_send_called_for_optimize_via_tier0(self):
        mock_conv = MagicMock()
        mock_conv.send.return_value = "use MFMA"
        session = self._make_session(mock_conv)

        # _optimize_via_tier0 reads self._recs; give it empty list
        session._recs = []
        # patch _offer_apply_suggestions and _offer_run_ai_commands to be no-ops
        session._offer_apply_suggestions = MagicMock()
        session._offer_run_ai_commands = MagicMock()
        session._extract_ai_commands = MagicMock(return_value=[])

        import types

        plan = types.SimpleNamespace(
            source_files=[],
            detected_patterns=[],
            suggested_counters=[],
        )
        session._tier0 = plan
        session._optimize_via_tier0(llm_provider="anthropic")
        mock_conv.send.assert_called_once()
        call_msg = mock_conv.send.call_args[0][0]
        self.assertIn("optimization recommendations", call_msg)

    def test_conv_send_called_for_request_optimization_suggestions(self):
        mock_conv = MagicMock()
        mock_conv.send.return_value = "FILE: foo.cpp\nuse LDS"
        session = self._make_session(mock_conv)

        session._request_optimization_suggestions([("foo.cpp", "// kernel code")])
        mock_conv.send.assert_called_once()
        # First call sends full source content
        call_msg = mock_conv.send.call_args[0][0]
        self.assertIn("foo.cpp", call_msg)
        self.assertIn("kernel code", call_msg)

    def test_repeated_call_does_not_resend_source_content(self):
        """Second call for the same files sends a short follow-up, not the full source."""
        mock_conv = MagicMock()
        mock_conv.send.return_value = "FILE: foo.cpp\nuse LDS"
        session = self._make_session(mock_conv)

        summaries = [("foo.cpp", "// kernel code")]
        session._request_optimization_suggestions(summaries)
        session._request_optimization_suggestions(summaries)

        self.assertEqual(mock_conv.send.call_count, 2)
        first_msg = mock_conv.send.call_args_list[0][0][0]
        second_msg = mock_conv.send.call_args_list[1][0][0]
        # First call contains source content, second does not
        self.assertIn("kernel code", first_msg)
        self.assertNotIn("kernel code", second_msg)
        self.assertIn("already shared", second_msg)

    def test_new_file_sends_only_new_content(self):
        """Adding a new file on second call sends only the new file's content."""
        mock_conv = MagicMock()
        mock_conv.send.return_value = "FILE: bar.cpp\nuse streams"
        session = self._make_session(mock_conv)

        session._request_optimization_suggestions([("foo.cpp", "// foo")])
        mock_conv.reset_mock()
        mock_conv.send.return_value = "FILE: bar.cpp\nuse streams"

        session._request_optimization_suggestions(
            [("foo.cpp", "// foo"), ("bar.cpp", "// bar")]
        )
        mock_conv.send.assert_called_once()
        msg = mock_conv.send.call_args[0][0]
        self.assertNotIn("// foo", msg)  # already in conversation
        self.assertIn("// bar", msg)  # new file — must be sent

    def test_post_rewrite_summary_appended_to_conv(self):
        """_apply_suggestions_via_llm notifies _conv after writing a file."""
        from rocinsight.ai_analysis.interactive import InteractiveSession, SessionStore
        import pathlib
        import tempfile

        store_dir = tempfile.mkdtemp()
        src_dir = tempfile.mkdtemp()
        src_file = pathlib.Path(src_dir) / "kernel.cpp"
        src_file.write_text("// original")

        mock_conv = MagicMock()
        mock_conv.send.return_value = "summary sent"

        session = InteractiveSession(
            source_dir=src_dir,
            tier0_result=None,
            recommendations=[],
            database_path="",
            llm_provider=None,
            llm_api_key=None,
            llm_model=None,
            session_store=SessionStore(store_dir),
        )
        session._conv = mock_conv
        session._llm_provider = "anthropic"

        # _pick_source_file prompts interactively — return the file directly
        session._pick_source_file = MagicMock(return_value=src_file)

        # _apply_suggestions_via_llm instantiates LLMAnalyzer inline and calls
        # _call_anthropic — patch it at the source so no real API call is made
        mock_analyzer = MagicMock()
        mock_analyzer._call_anthropic.return_value = "// rewritten by LLM"

        import rocinsight.ai_analysis.interactive as imod

        with patch(
            "rocinsight.ai_analysis.llm_analyzer.LLMAnalyzer", return_value=mock_analyzer
        ):
            with patch.object(imod, "_input", return_value="y"):
                session._apply_suggestions_via_llm(
                    "use LDS tiling for better cache reuse", "anthropic"
                )

        # _conv.send must have been called with the post-rewrite notification
        rewrite_calls = [
            call for call in mock_conv.send.call_args_list if "rewritten" in str(call)
        ]
        self.assertTrue(
            len(rewrite_calls) >= 1,
            "Expected _conv.send() to be called with post-rewrite summary",
        )

    def test_session_context_is_importable(self):
        """SessionContext must be importable from interactive module."""
        from rocinsight.ai_analysis.interactive import SessionContext

        ctx = SessionContext()
        self.assertEqual(ctx.iteration, 0)

    def test_workflow_session_has_no_conv(self):
        """WorkflowSession must not own a _conv attribute."""
        from rocinsight.ai_analysis.interactive import WorkflowSession

        ws = WorkflowSession(app_command="./app")
        self.assertFalse(
            hasattr(ws, "_conv"),
            "WorkflowSession must not own _conv",
        )


if __name__ == "__main__":
    unittest.main()
