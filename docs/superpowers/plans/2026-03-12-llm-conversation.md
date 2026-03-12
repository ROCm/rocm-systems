# LLMConversation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-call `LLMAnalyzer` instantiation in `InteractiveSession` with a persistent multi-turn `LLMConversation` that streams tokens, compacts via LLM summarization, and archives history to disk.

**Architecture:** New `LLMConversation` class handles a single multi-turn conversation for the lifetime of an `InteractiveSession`; the fence document is the system prompt set once at init; streaming eliminates per-request timeouts; LLM-based compaction keeps the context window bounded. `WorkflowSession` is unchanged — its only LLM call (`_llm_rewrite_file`) stays a standalone `LLMAnalyzer` call.

**Tech Stack:** Python 3, `anthropic` SDK (streaming via `client.messages.stream()`), `openai` SDK (streaming via `create(stream=True)`), standard library `json`/`pathlib`/`datetime`.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `ai_analysis/llm_conversation.py` | **CREATE** | `LLMConversation` class — streaming, compaction, history, persistence |
| `ai_analysis/llm_analyzer.py` | **MODIFY** | Add module-level `load_reference_guide() -> str` |
| `ai_analysis/__init__.py` | **MODIFY** | Export `LLMConversation` |
| `ai_analysis/interactive.py` | **MODIFY** | `InteractiveSession` uses `_conv`; update `SessionData`; remove `SessionContext` machinery |
| `ai_analysis/tests/test_llm_conversation.py` | **CREATE** | Unit + integration tests for `LLMConversation` |
| `ai_analysis/tests/test_interactive_context.py` | **DELETE** | Superseded by `test_llm_conversation.py` |
| `analyze.py` | **MODIFY** | Register `--llm-compact-every N`; pass to `InteractiveSession` |

All paths relative to:
`projects/rocprofiler-sdk/source/lib/python/rocpd/`

Test run command (from `/tmp` to avoid circular import):
```bash
ROCPD_SYS=/opt/rocm-7.2.0/lib/python3.12/site-packages
ROCPD_SRC=/home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python
TEST_FILE=/home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/tests/test_llm_conversation.py
cd /tmp && PYTHONPATH="${ROCPD_SYS}:${ROCPD_SRC}" python3 -m pytest $TEST_FILE --noconftest -v
```

---

## Chunk 1: LLMConversation class

### Task 1: Create `llm_conversation.py` with streaming send (all three providers)

**Files:**
- Create: `ai_analysis/llm_conversation.py`
- Test: `ai_analysis/tests/test_llm_conversation.py`

- [ ] **Step 1: Write the failing tests for LLMConversation core**

Create `ai_analysis/tests/test_llm_conversation.py`:

```python
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
            # Simulate compaction failure by raising in _call_non_streaming
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
ROCPD_SYS=/opt/rocm-7.2.0/lib/python3.12/site-packages
ROCPD_SRC=/home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python
cd /tmp && PYTHONPATH="${ROCPD_SYS}:${ROCPD_SRC}" python3 -m pytest \
  /home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/tests/test_llm_conversation.py \
  --noconftest -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'LLMConversation' from 'rocpd.ai_analysis.llm_conversation'`

- [ ] **Step 3: Create `ai_analysis/llm_conversation.py`**

```python
# ai_analysis/llm_conversation.py
"""Persistent multi-turn LLM conversation with streaming, compaction, and disk archive."""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .llm_analyzer import DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENAI_MODEL
from .exceptions import LLMAuthenticationError, LLMRateLimitError

_DEFAULT_LOCAL_URL = "http://localhost:11434/v1"
_DEFAULT_LOCAL_MODEL = "codellama:13b"


class LLMConversation:
    """Persistent multi-turn LLM session with streaming and LLM-based compaction.

    Usage:
        conv = LLMConversation(provider="anthropic", api_key="sk-ant-...")
        conv.initialize("You are an expert AMD GPU engineer.\\n\\n" + fence)
        response = conv.send("What is the bottleneck?", on_token=print_fn)
    """

    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        compact_every: int = 10,
        keep_recent_turns: int = 6,
        history_path: Optional[Path] = None,
    ) -> None:
        valid = {"anthropic", "openai", "local"}
        if provider not in valid:
            raise ValueError(
                f"Unknown provider: {provider!r}. Must be one of: {', '.join(sorted(valid))}"
            )
        self._provider = provider
        self._api_key = api_key
        self._model = model
        self._compact_every = compact_every
        self._keep_recent_turns = keep_recent_turns
        self._history_path = Path(history_path) if history_path else None
        self._system: str = ""
        self._messages: List[Dict[str, str]] = []
        self._turn_count: int = 0

    def initialize(self, system_prompt: str) -> None:
        """Set the system prompt (fence + role). Must be called before send()."""
        self._system = system_prompt

    def send(
        self,
        user_message: str,
        *,
        max_tokens: int = 4096,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Append user turn, stream response, increment turn_count, check compaction."""
        self._messages.append({"role": "user", "content": user_message})
        result = self._stream_response(max_tokens=max_tokens, on_token=on_token)
        self._messages.append({"role": "assistant", "content": result})
        self._turn_count += 1
        if self._turn_count > 0 and self._turn_count % self._compact_every == 0:
            self._compact()
        return result

    # ── Streaming ─────────────────────────────────────────────────────────────

    def _stream_response(
        self,
        max_tokens: int,
        on_token: Optional[Callable[[str], None]],
    ) -> str:
        if self._provider == "anthropic":
            return self._stream_anthropic(max_tokens=max_tokens, on_token=on_token)
        return self._stream_openai(max_tokens=max_tokens, on_token=on_token)

    def _stream_anthropic(
        self,
        max_tokens: int,
        on_token: Optional[Callable[[str], None]],
    ) -> str:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise LLMAuthenticationError(
                "No Anthropic API key. Set ANTHROPIC_API_KEY environment variable."
            )
        model = self._model or os.environ.get("ROCPD_LLM_MODEL") or DEFAULT_ANTHROPIC_MODEL
        client = _anthropic.Anthropic(api_key=api_key)
        result = ""
        try:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=self._system,
                messages=self._messages,
            ) as stream:
                for text in stream.text_stream:
                    if on_token:
                        on_token(text)
                    result += text
        except _anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(f"Anthropic authentication failed: {e}")
        except _anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic rate limit exceeded: {e}")
        except (LLMAuthenticationError, LLMRateLimitError):
            raise
        except Exception as e:
            if result:
                warnings.warn(
                    f"[LLMConversation] Streaming error mid-response: {e}", stacklevel=3
                )
            else:
                raise
        return result

    def _stream_openai(
        self,
        max_tokens: int,
        on_token: Optional[Callable[[str], None]],
    ) -> str:
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        if self._provider == "local":
            base_url = os.environ.get("ROCPD_LLM_LOCAL_URL", _DEFAULT_LOCAL_URL)
            model = self._model or os.environ.get("ROCPD_LLM_LOCAL_MODEL", _DEFAULT_LOCAL_MODEL)
            client = _openai.OpenAI(api_key="ignored", base_url=base_url)
        else:
            api_key = self._api_key or os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise LLMAuthenticationError(
                    "No OpenAI API key. Set OPENAI_API_KEY environment variable."
                )
            model = self._model or os.environ.get("ROCPD_LLM_MODEL") or DEFAULT_OPENAI_MODEL
            client = _openai.OpenAI(api_key=api_key)

        messages_with_system = [{"role": "system", "content": self._system}] + self._messages
        result = ""
        try:
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages_with_system,
                    max_completion_tokens=max_tokens,
                    stream=True,
                )
            except _openai.BadRequestError as e:
                if "max_completion_tokens" in str(e):
                    stream = client.chat.completions.create(
                        model=model,
                        messages=messages_with_system,
                        max_tokens=max_tokens,
                        stream=True,
                    )
                else:
                    raise
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    if on_token:
                        on_token(delta)
                    result += delta
        except _openai.AuthenticationError as e:
            raise LLMAuthenticationError(f"OpenAI authentication failed: {e}")
        except _openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}")
        except (LLMAuthenticationError, LLMRateLimitError):
            raise
        except Exception as e:
            if result:
                warnings.warn(
                    f"[LLMConversation] Streaming error mid-response: {e}", stacklevel=3
                )
            else:
                raise
        return result

    # ── Compaction ────────────────────────────────────────────────────────────

    _COMPACTION_PROMPT = (
        "Summarize the key context from this session so far. Include:\n"
        "- What app is being profiled and its source files\n"
        "- Profiling runs done (trace types, counter sets collected)\n"
        "- Performance issues identified (bottlenecks, percentages)\n"
        "- Code optimizations applied and their observed effect\n"
        "- Current state of the application\n"
        "Be concise (max 300 words)."
    )

    def _compact(self) -> None:
        """LLM-summarize oldest messages; replace with summary block + recent turns."""
        keep = self._keep_recent_turns * 2
        if len(self._messages) <= keep:
            return
        old_messages = self._messages[:-keep] if keep > 0 else list(self._messages)
        recent_messages = self._messages[-keep:] if keep > 0 else []

        if self._history_path and old_messages:
            self._append_to_archive(old_messages)

        try:
            summary = self._call_non_streaming(
                messages=old_messages + [{"role": "user", "content": self._COMPACTION_PROMPT}],
                max_tokens=600,
            )
            summary_block = [
                {"role": "user", "content": "Summarize our session so far."},
                {"role": "assistant", "content": f"[Session summary] {summary}"},
            ]
            self._messages = summary_block + recent_messages
        except Exception as e:
            warnings.warn(
                f"[LLMConversation] Compaction failed, skipping: {e}", stacklevel=2
            )

    def _call_non_streaming(self, messages: List[Dict], max_tokens: int) -> str:
        """Non-streaming API call used for compaction. Does NOT increment _turn_count."""
        if self._provider == "anthropic":
            try:
                import anthropic as _anthropic
            except ImportError:
                raise ImportError("anthropic package not installed.")
            api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            model = self._model or os.environ.get("ROCPD_LLM_MODEL") or DEFAULT_ANTHROPIC_MODEL
            client = _anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=self._system,
                messages=messages,
            )
            return resp.content[0].text if resp.content else ""

        # openai or local
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("openai package not installed.")
        if self._provider == "local":
            client = _openai.OpenAI(
                api_key="ignored",
                base_url=os.environ.get("ROCPD_LLM_LOCAL_URL", _DEFAULT_LOCAL_URL),
            )
            model = self._model or os.environ.get("ROCPD_LLM_LOCAL_MODEL", _DEFAULT_LOCAL_MODEL)
        else:
            client = _openai.OpenAI(api_key=self._api_key or os.environ.get("OPENAI_API_KEY", ""))
            model = self._model or os.environ.get("ROCPD_LLM_MODEL") or DEFAULT_OPENAI_MODEL
        full_messages = [{"role": "system", "content": self._system}] + messages
        try:
            resp = client.chat.completions.create(
                model=model, messages=full_messages, max_completion_tokens=max_tokens,
            )
        except _openai.BadRequestError as e:
            if "max_completion_tokens" in str(e):
                resp = client.chat.completions.create(
                    model=model, messages=full_messages, max_tokens=max_tokens,
                )
            else:
                raise
        return resp.choices[0].message.content or ""

    # ── Disk archive ──────────────────────────────────────────────────────────

    def _append_to_archive(self, messages: List[Dict]) -> None:
        """Append messages to JSONL archive (append-only)."""
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).isoformat()
            with self._history_path.open("a", encoding="utf-8") as f:
                for msg in messages:
                    entry = {
                        "role": msg["role"],
                        "content": msg["content"],
                        "turn": self._turn_count,
                        "ts": ts,
                    }
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            warnings.warn(
                f"[LLMConversation] Failed to write history archive: {e}", stacklevel=3
            )

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for SessionData persistence. Does NOT include api_key."""
        return {
            "provider": self._provider,
            "model": self._model,
            "compact_every": self._compact_every,
            "keep_recent_turns": self._keep_recent_turns,
            "history_path": str(self._history_path) if self._history_path else None,
            "system": self._system,
            "messages": list(self._messages),
            "turn_count": self._turn_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], **kwargs: Any) -> "LLMConversation":
        """Restore from serialized state.

        kwargs:
            api_key: override stored api_key (api_key is never stored)
            model:   override stored model
        """
        history_path = d.get("history_path")
        conv = cls(
            provider=d["provider"],
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model") or d.get("model"),
            compact_every=d.get("compact_every", 10),
            keep_recent_turns=d.get("keep_recent_turns", 6),
            history_path=Path(history_path) if history_path else None,
        )
        conv._system = d.get("system", "")
        conv._messages = list(d.get("messages", []))
        conv._turn_count = d.get("turn_count", 0)
        return conv

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def messages(self) -> List[Dict]:
        return list(self._messages)
```

- [ ] **Step 4: Copy `llm_conversation.py` to system package so other tests can import it**

```bash
cp /home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/llm_conversation.py \
   /opt/rocm-7.2.0/lib/python3.12/site-packages/rocpd/ai_analysis/llm_conversation.py
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
ROCPD_SYS=/opt/rocm-7.2.0/lib/python3.12/site-packages
ROCPD_SRC=/home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python
cd /tmp && PYTHONPATH="${ROCPD_SYS}:${ROCPD_SRC}" python3 -m pytest \
  /home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/tests/test_llm_conversation.py \
  --noconftest -v
```

Expected: All tests pass. If `BadRequestError` mock constructor fails, use `MagicMock` for `response` and `body` args.

- [ ] **Step 6: Commit**

```bash
cd /home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev
git add projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/llm_conversation.py \
        projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/tests/test_llm_conversation.py
git commit -m "feat: add LLMConversation streaming persistent session class"
```

---

## Chunk 2: load_reference_guide() + __init__.py export

### Task 2: Add `load_reference_guide()` to `llm_analyzer.py` and export `LLMConversation` from `__init__.py`

**Files:**
- Modify: `ai_analysis/llm_analyzer.py`
- Modify: `ai_analysis/__init__.py`

- [ ] **Step 1: Add `load_reference_guide()` module-level function to `llm_analyzer.py`**

After the existing `get_reference_guide_path()` function (line ~103), add:

```python
def load_reference_guide() -> str:
    """Load the LLM fence document.

    Same path lookup order as get_reference_guide_path():
    ROCPD_LLM_REFERENCE_GUIDE env var → module share/ dir → /opt/rocm/share/...

    Raises:
        ReferenceGuideNotFoundError: If guide file not found.
    """
    return get_reference_guide_path().read_text()
```

- [ ] **Step 2: Export `LLMConversation` and `load_reference_guide` from `__init__.py`**

In `ai_analysis/__init__.py`, extend the existing `from .llm_analyzer import LLMAnalyzer, AnalysisContext` line (line 59):

```python
from .llm_analyzer import LLMAnalyzer, AnalysisContext, load_reference_guide
```

Add a new import for `LLMConversation`:
```python
from .llm_conversation import LLMConversation
```

Add `"LLMConversation"` and `"load_reference_guide"` to `__all__`.

- [ ] **Step 3: Add `compact_every > 0` guard to `LLMConversation.__init__`**

In `llm_conversation.py`, change the `compact_every` assignment:
```python
self._compact_every = max(1, compact_every)
```

And update the trigger check in `send()` accordingly (no change needed — `max(1, ...)` prevents zero division).

- [ ] **Step 4: Copy modified files to system package**

```bash
cp /home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/llm_analyzer.py \
   /opt/rocm-7.2.0/lib/python3.12/site-packages/rocpd/ai_analysis/llm_analyzer.py
cp /home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/__init__.py \
   /opt/rocm-7.2.0/lib/python3.12/site-packages/rocpd/ai_analysis/__init__.py
```

- [ ] **Step 5: Verify import works**

```bash
ROCPD_SYS=/opt/rocm-7.2.0/lib/python3.12/site-packages
cd /tmp && PYTHONPATH="${ROCPD_SYS}" python3 -c \
  "from rocpd.ai_analysis import LLMConversation, load_reference_guide; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
cd /home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev
git add projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/llm_analyzer.py \
        projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/__init__.py \
        projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/llm_conversation.py
git commit -m "feat: expose load_reference_guide() and LLMConversation in ai_analysis public API"
```

---

## Chunk 3: interactive.py migration

### Task 3: Update `SessionData` and add `_conv` to `InteractiveSession`

**Files:**
- Modify: `ai_analysis/interactive.py`

This is the largest task. It has multiple sub-steps. Complete them in order.

#### Sub-task 3a: Add `_print_token` module-level helper and update `SessionData`

- [ ] **Step 1: Add `_print_token` near the top of `interactive.py`** (after the module-level imports, before the dataclasses)

```python
def _print_token(t: str) -> None:
    """Stream a single LLM token to stdout without newline."""
    print(t, end="", flush=True)
```

- [ ] **Step 2: Update `SessionData` — replace `context` field with `conversation`**

In `SessionData` (lines ~55–79):

Change:
```python
context: Optional[Dict[str, Any]] = None          # NEW field
```
To:
```python
conversation: Optional[Dict[str, Any]] = None     # serialized LLMConversation
```

Update `from_dict` — change:
```python
context=d.get("context"),               # None if key absent (backward compat)
```
To:
```python
conversation=d.get("conversation"),     # None if absent (backward compatible)
```

Update the `cls(...)` call in `from_dict` to pass `conversation=` instead of `context=`.

- [ ] **Step 3: Add `compact_every` parameter to `InteractiveSession.__init__` and initialize `_conv`**

In `InteractiveSession.__init__` signature (line ~292), add `compact_every: int = 10` after `resume_session_id`:

```python
def __init__(
    self,
    source_dir: str,
    tier0_result: Optional[Any],
    recommendations: List[Dict[str, Any]],
    database_path: str,
    llm_provider: Optional[str],
    llm_api_key: Optional[str],
    llm_model: Optional[str],
    llm_local: Optional[str] = None,
    llm_local_model: Optional[str] = None,
    session_store: Optional[SessionStore] = None,
    resume_session_id: Optional[str] = None,
    compact_every: int = 10,
) -> None:
```

In the body, add `self._compact_every = compact_every` after `self._store = session_store or SessionStore()`.

Replace `self._ctx = SessionContext()` with `self._conv: Optional[LLMConversation] = None`.

Add imports at top of `interactive.py` (after existing imports):
```python
from .llm_conversation import LLMConversation
from .llm_analyzer import load_reference_guide
```

#### Sub-task 3b: Update `_init_session` to restore/create `_conv`

- [ ] **Step 4: Replace `_ctx` restoration in `_init_session` with `_conv` restoration**

Current `_init_session` (lines ~323–348) sets `self._ctx` from `loaded.context`. Replace the entire method with:

```python
def _init_session(self, resume_id: Optional[str]) -> SessionData:
    # Explicit resume
    if resume_id:
        loaded = self._store.load(resume_id)
        if loaded:
            self._conv = self._restore_or_create_conv(loaded)
            return loaded

    # Auto-detect previous session for this source dir
    existing = self._store.find_by_source_dir(self._source_dir)
    if existing:
        chosen = self._prompt_resume(existing)
        if chosen:
            self._conv = self._restore_or_create_conv(chosen)
            return chosen

    # New session
    now = datetime.now(timezone.utc).isoformat()
    new_session = SessionData(
        session_id=SessionStore.make_session_id(self._source_dir),
        source_dir=self._source_dir,
        created_at=now,
        last_updated=now,
    )
    self._conv = self._make_fresh_conv(new_session.session_id)
    return new_session

def _restore_or_create_conv(self, loaded: SessionData) -> Optional["LLMConversation"]:
    """Restore _conv from a loaded session, or create fresh if absent."""
    raw_conv = loaded.conversation
    if raw_conv:
        return LLMConversation.from_dict(
            raw_conv, api_key=self._llm_api_key, model=self._llm_model
        )
    return self._make_fresh_conv(loaded.session_id)

def _make_fresh_conv(self, session_id: str) -> Optional["LLMConversation"]:
    """Create a new LLMConversation for a session, or None if no LLM configured."""
    if not self._llm_provider:
        return None
    hp = pathlib.Path(f"~/.rocpd/sessions/{session_id}_history.jsonl").expanduser()
    conv = LLMConversation(
        provider=self._llm_provider,
        api_key=self._llm_api_key,
        model=self._llm_model,
        compact_every=self._compact_every,
        history_path=hp,
    )
    try:
        fence = load_reference_guide()
    except Exception as e:
        warnings.warn(f"[LLMConversation] Could not load reference guide: {e}", stacklevel=3)
        fence = ""
    conv.initialize(
        "You are an expert AMD GPU performance engineer "
        "helping optimize a HIP/ROCm application.\n\n" + fence
    )
    return conv
```

#### Sub-task 3c: Update save points

- [ ] **Step 5: Replace `SessionContext` serialization at all three save points**

There are three save sites:

**Site 1** — `[s]` branch (line ~482):
Change:
```python
self._session.context = asdict(self._ctx)   # flush context before save
self._store.save(self._session)
```
To:
```python
if self._conv:
    self._session.conversation = self._conv.to_dict()
self._store.save(self._session)
```

**Site 2** — `_save_and_quit` (line ~502):
Change:
```python
self._session.context = asdict(self._ctx)     # flush context before save
self._store.save(self._session)
```
To:
```python
if self._conv:
    self._session.conversation = self._conv.to_dict()
self._store.save(self._session)
```

**Site 3** — `_offer_run_ai_commands` auto-save (line ~935):
Change:
```python
self._session.context = asdict(self._ctx)
self._store.save(self._session)
```
To:
```python
if self._conv:
    self._session.conversation = self._conv.to_dict()
self._store.save(self._session)
```

#### Sub-task 3d: Replace LLM call sites with `_conv.send()`

- [ ] **Step 6: Replace `_llm_annotate_profiling_plan` with `_conv.send()`**

Current method (lines ~699–736) instantiates `LLMAnalyzer` and calls `analyzer.annotate_profiling_plan(metadata)`. Replace the body with:

```python
def _llm_annotate_profiling_plan(self, cmds: List[tuple]) -> List[tuple]:
    """Send tier0 metadata to LLM for annotation via persistent conversation."""
    if self._conv is None:
        return cmds
    try:
        plan = self._tier0
        if plan is None:
            return cmds
        patterns = getattr(plan, "detected_patterns", [])
        import json as _json
        metadata = {
            "programming_model":  getattr(plan, "programming_model", "HIP"),
            "kernel_count":       getattr(plan, "kernel_count", 0),
            "suggested_counters": getattr(plan, "suggested_counters", []),
            "risk_areas":         getattr(plan, "risk_areas", []),
            "detected_patterns":  [
                {"id":          (p.get("pattern_id") if isinstance(p, dict) else getattr(p, "pattern_id", "")),
                 "severity":    (p.get("severity")   if isinstance(p, dict) else getattr(p, "severity",   "")),
                 "description": (p.get("description") if isinstance(p, dict) else getattr(p, "description", ""))}
                for p in patterns
            ],
            "suggested_commands": [cmd for _, cmd in cmds],
        }
        user_msg = (
            f"Annotate this profiling plan (max 200 words, plain text only — no markdown): "
            f"{_json.dumps(metadata)}"
        )
        _print()
        _print("  ── LLM Profiling Advice ────────────────────────────", style="cyan")
        note = self._conv.send(user_msg, on_token=_print_token)
        _print()
    except Exception as exc:
        _print(f"  (LLM annotation skipped: {exc})", style="dim")
    return cmds
```

- [ ] **Step 7: Replace `_optimize_via_tier0` LLM call with `_conv.send()`**

Current method (lines ~1091–1182) instantiates `LLMAnalyzer` and calls `analyzer._call_openai/anthropic/local`. Replace the LLM portion:

Keep all the metadata-building code up to and including the `metadata` dict construction. Remove the `model = ...` / `analyzer = LLMAnalyzer(...)` block. Replace the `with _Spinner(...)` block and the `if note:` block with:

```python
        if self._conv is None:
            _print("  (No LLM configured — skipping AI optimization)", style="dim")
            return

        user_msg = (
            "Based on these detected GPU source patterns, provide concrete "
            "optimization recommendations (max 300 words, plain text only — no markdown headers):\n"
            + _json.dumps(metadata, indent=2)
        )
        _print()
        _print("  ── AI Optimization Suggestions ──────────────────────", style="cyan")
        try:
            note = self._conv.send(user_msg, on_token=_print_token)
            _print()
        except Exception as exc:
            _print(f"\n  (LLM optimization failed: {exc})", style="red")
            return
        if note:
            self._offer_apply_suggestions(note, self._llm_provider)
            structured = [
                c.get("full_command", "")
                for rec in self._recs
                for c in rec.get("commands", [])
                if c.get("full_command")
            ]
            ai_cmds = self._extract_ai_commands(note, structured)
            self._offer_run_ai_commands(ai_cmds)
        else:
            _print("  (LLM returned no suggestions)", style="yellow")
```

Also remove the outer `try/except Exception as exc:` wrapper (the per-call error handling is now inside send(), and specific `LLMAuthenticationError`/`LLMRateLimitError` propagate to existing callers).

- [ ] **Step 8: Replace `_request_optimization_suggestions` LLM call with `_conv.send()`**

Current method (lines ~1415–1459) instantiates `LLMAnalyzer` and calls `analyzer.suggest_optimizations(...)`. Replace with:

```python
def _request_optimization_suggestions(
    self, summaries: List[tuple], llm_provider: Optional[str] = None
) -> Dict[str, str]:
    """Send source file summaries to LLM; return {filename: suggestion_text}."""
    if self._conv is None:
        return {}
    try:
        file_list = ", ".join(name for name, _ in summaries)
        combined = "\n\n".join(f"=== {name} ===\n{content}" for name, content in summaries)
        user_msg = (
            f"Analyze these AMD GPU source files and provide concrete, actionable "
            f"optimization suggestions. Focus on: memory coalescing, wave occupancy, "
            f"unnecessary hipDeviceSynchronize, blocking hipMemcpy, MFMA usage, LDS "
            f"utilization, loop structure, kernel launch parameters. Be specific — "
            f"reference actual patterns visible in the code. Use plain text only — "
            f"no markdown headers. Start each file section with exactly: FILE: <filename>\n\n"
            f"{combined}"
        )
        _print()
        _print("  ── AI Optimization Suggestions ──────────────────────", style="cyan")
        raw = self._conv.send(user_msg, on_token=_print_token)
        _print()

        result: Dict[str, str] = {}
        if raw and raw.lstrip().startswith("FILE:"):
            raw = "\n" + raw.lstrip()
        for block in re.split(r"\nFILE:\s*", raw or ""):
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n", 1)
            if len(lines) == 2:
                result[lines[0].strip()] = lines[1].strip()
        if not result and raw and raw.strip():
            first_name = summaries[0][0] if summaries else "response"
            result[first_name] = raw.strip()
        return result
    except Exception as exc:
        _print(f"  [DEBUG] exception in LLM call: {type(exc).__name__}: {exc}", style="red")
        return {}
```

- [ ] **Step 9: Add post-rewrite summary turn in `_apply_suggestions_via_llm`**

After the successful file-write section in `_apply_suggestions_via_llm` (after the backup is created and new content written), add:

```python
        # Notify the persistent conversation about the rewrite
        if self._conv:
            try:
                self._conv.send(
                    f"File `{chosen.name}` was rewritten applying the above optimizations. "
                    f"Compilation: pending.",
                    on_token=None,
                )
            except Exception:
                pass  # post-rewrite summary is advisory; never crash here
```

Find the line that writes the rewritten file (writes `new_content` to `chosen`) and add this block immediately after.

#### Sub-task 3e: Remove `SessionContext` machinery

- [ ] **Step 10: Remove `SessionContext` dataclass and all `_update_ctx_*` / `_format_context_block` methods**

**Delete these class/method definitions:**
- The `@dataclass` `SessionContext` class (lines ~40–52)
- `_update_ctx_analysis()` method (~778)
- `_update_ctx_suggestion()` method (~798)
- `_update_ctx_command()` method (~804)
- `_format_context_block()` method (~810)

**Remove all remaining call sites** (search for `_update_ctx` and `_format_context_block` to find them all):
```bash
grep -n "_update_ctx\|_format_context_block\|self._ctx" \
  projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/interactive.py
```
Each hit must be removed or already replaced by the `_conv.send()` calls in Steps 6–9.

**`asdict` import:** `asdict` is still used in `SessionData.to_dict()` (line 65 — `return asdict(self)`). Do NOT remove that import. The only `asdict()` calls to remove are `asdict(self._ctx)` at the three save points (already replaced in Steps 5).

**`SessionContext` removal from import list:** If `interactive.py` re-exports `SessionContext` in `__all__` or uses it in type annotations, remove those references too.

**`self._ctx` attribute:** Remove `self._ctx = SessionContext()` line from `InteractiveSession.__init__` (already identified in Sub-task 3a Step 3, but verify it's gone).

- [ ] **Step 11: Remove `test_interactive_context.py`**

```bash
rm projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/tests/test_interactive_context.py
```

- [ ] **Step 12: Run interactive tests to verify migration**

Copy `interactive.py` to system path first:
```bash
cp /home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/interactive.py \
   /opt/rocm-7.2.0/lib/python3.12/site-packages/rocpd/ai_analysis/interactive.py
```

Run existing interactive tests:
```bash
ROCPD_SYS=/opt/rocm-7.2.0/lib/python3.12/site-packages
ROCPD_SRC=/home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python
cd /tmp && PYTHONPATH="${ROCPD_SYS}:${ROCPD_SRC}" python3 -m pytest \
  /home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/tests/test_interactive.py \
  --noconftest -v
```

Expected: All tests pass. `test_interactive_context.py` no longer exists so no import errors.

- [ ] **Step 13: Commit**

```bash
cd /home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev
git add projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/interactive.py
git rm projects/rocprofiler-sdk/source/lib/python/rocpd/ai_analysis/tests/test_interactive_context.py
git commit -m "feat: replace SessionContext with LLMConversation in InteractiveSession"
```

---

## Chunk 4: analyze.py CLI flag

### Task 4: Register `--llm-compact-every N` and thread it to `InteractiveSession`

**Files:**
- Modify: `analyze.py`

- [ ] **Step 1: Add `--llm-compact-every` argument to `add_args()` in `analyze.py`**

In `add_args()` (around line ~4956, after `--llm-thinking`), add to the `llm_options` group:

```python
llm_options.add_argument(
    "--llm-compact-every",
    metavar="N",
    type=int,
    default=10,
    dest="llm_compact_every",
    help=(
        "Compact the LLM conversation context every N assistant turns by summarizing "
        "older messages (default: 10). Lower values use less memory; higher values "
        "preserve more context. Only applies to --interactive sessions."
    ),
)
```

- [ ] **Step 2: Add `"llm_compact_every"` to `valid_args` in `process_args()`**

In `process_args()` (line ~4999):
```python
valid_args = ["source_dir", "prompt", "top_kernels", "format", "min_duration",
              "llm", "llm_api_key", "llm_model", "llm_thinking", "verbose", "interactive",
              "resume_session", "llm_local", "llm_local_model", "llm_compact_every"]
```

- [ ] **Step 3: Pass `compact_every` to `_run_interactive_session` and `InteractiveSession`**

In `_run_interactive_session()` signature (line ~4802), add `compact_every: int = 10` parameter:

```python
def _run_interactive_session(
    recommendations: List[Dict[str, Any]],
    tier0_result: Optional[Any] = None,
    database_path: str = "",
    source_dir: str = "",
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_local: Optional[str] = None,
    llm_local_model: Optional[str] = None,
    resume_session: Optional[str] = None,
    compact_every: int = 10,
) -> None:
```

Pass it to `InteractiveSession(...)`:
```python
    InteractiveSession(
        source_dir=source_dir,
        tier0_result=tier0_result,
        recommendations=recommendations,
        database_path=database_path,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_local=llm_local,
        llm_local_model=llm_local_model,
        session_store=SessionStore(),
        resume_session_id=resume_session,
        compact_every=compact_every,
    ).run()
```

In `execute()` (line ~5110), update the `_run_interactive_session` call to pass `compact_every`:
```python
        _run_interactive_session(
            recommendations=result_store.get("recommendations", []),
            tier0_result=result_store.get("tier0_result"),
            database_path=result_store.get("database_path", database_path),
            source_dir=kwargs.get("source_dir", ""),
            llm_provider=_interactive_llm_provider,
            llm_api_key=_interactive_llm_api_key,
            llm_model=_interactive_llm_model,
            llm_local=kwargs.get("llm_local"),
            llm_local_model=kwargs.get("llm_local_model"),
            resume_session=kwargs.get("resume_session"),
            compact_every=kwargs.get("llm_compact_every", 10),
        )
```

- [ ] **Step 4: Verify `--llm-compact-every` appears in help output**

```bash
ROCPD_SYS=/opt/rocm-7.2.0/lib/python3.12/site-packages
ROCPD_SRC=/home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python
cd /tmp && PYTHONPATH="${ROCPD_SYS}:${ROCPD_SRC}" python3 -m rocpd analyze --help | grep compact
```

Expected: `--llm-compact-every N  Compact the LLM conversation context every N assistant turns`

- [ ] **Step 5: Commit**

```bash
cd /home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev
git add projects/rocprofiler-sdk/source/lib/python/rocpd/analyze.py
git commit -m "feat: add --llm-compact-every CLI flag for LLMConversation compaction interval"
```
