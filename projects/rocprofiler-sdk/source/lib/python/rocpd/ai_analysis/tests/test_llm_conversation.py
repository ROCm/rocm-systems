# ai_analysis/tests/test_llm_conversation.py
"""Tests for LLMConversation persistent streaming session."""
from __future__ import annotations
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rocpd.ai_analysis.llm_conversation import LLMConversation


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
        self.assertEqual(conv.messages[1], {"role": "assistant", "content": "Hello world"})
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

class TestCompaction(unittest.TestCase):
    """Compaction trigger, turn_count not incremented, summary block placement."""

    def _make_conv_with_mock(self, provider="anthropic", compact_every=2, keep_recent_turns=1):
        conv = LLMConversation(
            provider=provider, api_key="k",
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
            {"role": "user", "content": "q3"},  # recent (keep_recent_turns=1 → keep 2 msgs)
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
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            def _fail(**kw):
                raise RuntimeError("API down")
            conv._call_non_streaming = _fail
            conv._compact()
        # Messages unchanged
        self.assertEqual(len(conv.messages), 4)


# ── TestDiskArchive ───────────────────────────────────────────────────────────

class TestDiskArchive(unittest.TestCase):
    """JSONL archive written only when history_path is set."""

    def test_archive_written_on_compaction(self):
        with tempfile.TemporaryDirectory() as td:
            hp = pathlib.Path(td) / "history.jsonl"
            conv = LLMConversation(
                provider="anthropic", api_key="k",
                compact_every=4, keep_recent_turns=1,
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
            provider="anthropic", api_key="k",
            compact_every=4, keep_recent_turns=1,
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
                provider="anthropic", api_key="orig-key",
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


if __name__ == "__main__":
    unittest.main()
