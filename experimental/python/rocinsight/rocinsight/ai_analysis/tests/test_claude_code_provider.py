# ai_analysis/tests/test_claude_code_provider.py
"""Tests for the claude-code LLM provider.

Two-tier auth chain (as of 2026-03-25):
  Tier 1 — ANTHROPIC_API_KEY via ``anthropic`` SDK (direct API call, no CLI)
  Tier 2 — ``claude -p`` subprocess  (stored Claude Code CLI credentials)

Covers:
  - PROVIDER_REGISTRY entry
  - _get_api_key_from_env must not raise ValueError for 'claude-code'
  - Alias map (sonnet/opus/haiku → full model IDs)
  - LLMAnalyzer._call_claude_code: API-key tier used first; CLI is fallback
  - LLMAnalyzer._call_claude_cli_subprocess: JSON/plain-text parsing, error paths
  - LLMAnalyzer._call_claude_code_api_fallback: model alias resolution, auth error
  - LLMConversation._call_claude_code_turn: same two-tier chain
  - LLMConversation._call_claude_cli_subprocess: JSON/plain-text, error paths
  - interactive.py dispatch: claude-code must not fall through to _call_local
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rocinsight.ai_analysis.llm_analyzer import LLMAnalyzer, PROVIDER_REGISTRY
from rocinsight.ai_analysis.llm_conversation import LLMConversation

_has_anthropic = importlib.util.find_spec("anthropic") is not None

_skip_no_anthropic = unittest.skipUnless(_has_anthropic, "anthropic package not installed")


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_anthropic_response(text: str):
    block = SimpleNamespace(text=text, type="text")
    return SimpleNamespace(content=[block])


# ── PROVIDER_REGISTRY ─────────────────────────────────────────────────────────

class TestProviderRegistry(unittest.TestCase):
    def test_claude_code_in_registry(self):
        self.assertIn("claude-code", PROVIDER_REGISTRY)

    def test_claude_code_description_mentions_auth(self):
        desc = PROVIDER_REGISTRY["claude-code"]
        self.assertIn("Claude", desc)
        self.assertIn("ANTHROPIC_API_KEY", desc)


# ── LLMAnalyzer._get_api_key_from_env ────────────────────────────────────────

class TestGetApiKeyClaudeCode(unittest.TestCase):
    """_get_api_key_from_env must not raise ValueError for 'claude-code'."""

    def test_no_value_error_when_no_env_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            analyzer = LLMAnalyzer(provider="claude-code")
            key = analyzer._get_api_key_from_env(raise_if_missing=False)
            self.assertEqual(key, "")

    def test_returns_anthropic_key_when_set(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-123"}):
            analyzer = LLMAnalyzer(provider="claude-code")
            key = analyzer._get_api_key_from_env(raise_if_missing=False)
            self.assertEqual(key, "sk-test-123")


# ── LLMAnalyzer alias map ────────────────────────────────────────────────────

class TestAliasMap(unittest.TestCase):
    def test_analyzer_alias_map_covers_all_aliases(self):
        m = LLMAnalyzer._CLAUDE_CODE_ALIAS_MAP
        self.assertIn("sonnet", m)
        self.assertIn("opus", m)
        self.assertIn("haiku", m)

    def test_conversation_alias_map_matches_analyzer(self):
        self.assertEqual(
            LLMAnalyzer._CLAUDE_CODE_ALIAS_MAP,
            LLMConversation._CLAUDE_CODE_ALIAS_MAP,
        )

    def test_aliases_map_to_versioned_model_ids(self):
        m = LLMAnalyzer._CLAUDE_CODE_ALIAS_MAP
        for alias, model_id in m.items():
            self.assertIn("claude-", model_id, f"{alias} maps to non-Claude model ID: {model_id}")
            self.assertNotEqual(alias, model_id, "Alias must differ from its resolved ID")


# ── LLMAnalyzer._call_claude_code: two-tier priority ─────────────────────────

@_skip_no_anthropic
class TestClaudeCodeAnalyzerTwoTier(unittest.TestCase):
    """_call_claude_code: API key is tier 1, CLI subprocess is tier 2."""

    def test_api_key_tier_used_first(self):
        """When ANTHROPIC_API_KEY is set, _call_claude_code_api_fallback is called
        and CLI subprocess is never invoked."""
        analyzer = LLMAnalyzer(provider="claude-code")

        with patch.object(analyzer, "_call_claude_code_api_fallback",
                          return_value="api result") as mock_api, \
             patch.object(analyzer, "_call_claude_cli_subprocess") as mock_cli:
            result = analyzer._call_claude_code("system", "user")

        mock_api.assert_called_once()
        mock_cli.assert_not_called()
        self.assertEqual(result, "api result")

    def test_cli_fallback_when_no_api_key(self):
        """When _call_claude_code_api_fallback raises LLMAuthenticationError
        (no API key), CLI subprocess tier is tried."""
        from rocinsight.ai_analysis.exceptions import LLMAuthenticationError

        analyzer = LLMAnalyzer(provider="claude-code")

        with patch.object(analyzer, "_call_claude_code_api_fallback",
                          side_effect=LLMAuthenticationError("no key")), \
             patch.object(analyzer, "_call_claude_cli_subprocess",
                          return_value="cli result") as mock_cli:
            result = analyzer._call_claude_code("system", "user")

        mock_cli.assert_called_once()
        self.assertEqual(result, "cli result")

    def test_auth_error_raised_when_both_tiers_fail(self):
        """When both API key and CLI fail, LLMAuthenticationError is raised."""
        from rocinsight.ai_analysis.exceptions import (
            AnalysisError, LLMAuthenticationError,
        )

        analyzer = LLMAnalyzer(provider="claude-code")

        with patch.object(analyzer, "_call_claude_code_api_fallback",
                          side_effect=LLMAuthenticationError("no key")), \
             patch.object(analyzer, "_call_claude_cli_subprocess",
                          side_effect=AnalysisError("CLI not found")):
            with self.assertRaises(LLMAuthenticationError):
                analyzer._call_claude_code("system", "user")

    def test_rate_limit_propagates_without_cli_fallback(self):
        """LLMRateLimitError from the API key tier must propagate immediately —
        the CLI is not a workaround for rate limiting."""
        from rocinsight.ai_analysis.exceptions import LLMRateLimitError

        analyzer = LLMAnalyzer(provider="claude-code")

        with patch.object(analyzer, "_call_claude_code_api_fallback",
                          side_effect=LLMRateLimitError("rate limited")), \
             patch.object(analyzer, "_call_claude_cli_subprocess") as mock_cli:
            with self.assertRaises(LLMRateLimitError):
                analyzer._call_claude_code("system", "user")

        mock_cli.assert_not_called()

    def test_analysis_error_propagates_without_cli_fallback(self):
        """AnalysisError from the API key tier must propagate immediately."""
        from rocinsight.ai_analysis.exceptions import AnalysisError

        analyzer = LLMAnalyzer(provider="claude-code")

        with patch.object(analyzer, "_call_claude_code_api_fallback",
                          side_effect=AnalysisError("bad data")), \
             patch.object(analyzer, "_call_claude_cli_subprocess") as mock_cli:
            with self.assertRaises(AnalysisError):
                analyzer._call_claude_code("system", "user")

        mock_cli.assert_not_called()


# ── LLMAnalyzer._call_claude_cli_subprocess ───────────────────────────────────

class TestClaudeCliSubprocess(unittest.TestCase):
    """Unit tests for _call_claude_cli_subprocess in LLMAnalyzer."""

    def _make_analyzer(self):
        return LLMAnalyzer(provider="claude-code")

    def test_returns_result_from_json_output(self):
        analyzer = self._make_analyzer()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"result": "analysis text", "is_error": false}'
        mock_proc.stderr = ""

        with patch("subprocess.run", return_value=mock_proc):
            result = analyzer._call_claude_cli_subprocess("sys", "usr", "sonnet")

        self.assertEqual(result, "analysis text")

    def test_returns_plain_text_when_not_json(self):
        analyzer = self._make_analyzer()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "  plain text response  "
        mock_proc.stderr = ""

        with patch("subprocess.run", return_value=mock_proc):
            result = analyzer._call_claude_cli_subprocess("sys", "usr", "sonnet")

        self.assertEqual(result, "plain text response")

    def test_raises_on_file_not_found(self):
        from rocinsight.ai_analysis.exceptions import AnalysisError

        analyzer = self._make_analyzer()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaises(AnalysisError):
                analyzer._call_claude_cli_subprocess("sys", "usr", "sonnet")

    def test_raises_on_nonzero_exit(self):
        from rocinsight.ai_analysis.exceptions import AnalysisError

        analyzer = self._make_analyzer()
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "authentication error"

        with patch("subprocess.run", return_value=mock_proc):
            with self.assertRaises(AnalysisError):
                analyzer._call_claude_cli_subprocess("sys", "usr", "sonnet")

    def test_raises_on_is_error_true(self):
        from rocinsight.ai_analysis.exceptions import AnalysisError

        analyzer = self._make_analyzer()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"is_error": true, "result": "something went wrong"}'
        mock_proc.stderr = ""

        with patch("subprocess.run", return_value=mock_proc):
            with self.assertRaises(AnalysisError):
                analyzer._call_claude_cli_subprocess("sys", "usr", "sonnet")


# ── LLMAnalyzer._call_claude_code_api_fallback ───────────────────────────────

@_skip_no_anthropic
class TestClaudeCodeApiFallback(unittest.TestCase):
    """_call_claude_code_api_fallback uses ANTHROPIC_API_KEY and maps model aliases."""

    def test_uses_anthropic_api_key(self):
        analyzer = LLMAnalyzer(provider="claude-code")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response("hello")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             patch("anthropic.Anthropic", return_value=mock_client):
            result = analyzer._call_claude_code_api_fallback("sys", "usr", "sonnet")

        self.assertEqual(result, "hello")
        create_call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(create_call_kwargs["model"], "claude-sonnet-4-6")

    def test_maps_opus_alias(self):
        analyzer = LLMAnalyzer(provider="claude-code")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response("ok")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             patch("anthropic.Anthropic", return_value=mock_client):
            analyzer._call_claude_code_api_fallback("sys", "usr", "opus")

        create_call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(create_call_kwargs["model"], "claude-opus-4-6")

    def test_passes_through_full_model_id(self):
        analyzer = LLMAnalyzer(provider="claude-code")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response("ok")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             patch("anthropic.Anthropic", return_value=mock_client):
            analyzer._call_claude_code_api_fallback("sys", "usr", "claude-opus-4-6")

        create_call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(create_call_kwargs["model"], "claude-opus-4-6")

    def test_raises_auth_error_when_no_key(self):
        from rocinsight.ai_analysis.exceptions import LLMAuthenticationError

        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            analyzer = LLMAnalyzer(provider="claude-code")
            with self.assertRaises(LLMAuthenticationError):
                analyzer._call_claude_code_api_fallback("sys", "usr", "sonnet")
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved


# ── LLMConversation._call_claude_code_turn: two-tier priority ─────────────────

@_skip_no_anthropic
class TestConversationClaudeCodeTwoTier(unittest.TestCase):
    """_call_claude_code_turn: API key is tier 1, CLI subprocess is tier 2."""

    def test_api_key_tier_used_when_key_present(self):
        """When ANTHROPIC_API_KEY is set, CLI subprocess is never invoked."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response("api conv result")

        conv = LLMConversation(provider="claude-code")
        conv.initialize("system prompt")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), \
             patch("anthropic.Anthropic", return_value=mock_client), \
             patch.object(conv, "_call_claude_cli_subprocess") as mock_cli:
            result = conv._call_claude_code_turn("user message", "sonnet")

        mock_cli.assert_not_called()
        self.assertEqual(result, "api conv result")

    def test_cli_fallback_when_no_api_key(self):
        """When ANTHROPIC_API_KEY is not set, CLI subprocess tier is used."""
        conv = LLMConversation(provider="claude-code")
        conv.initialize("system prompt")

        with patch.dict(os.environ, {}, clear=False), \
             patch.object(conv, "_call_claude_cli_subprocess",
                          return_value="cli conv result") as mock_cli:
            os.environ.pop("ANTHROPIC_API_KEY", None)
            result = conv._call_claude_code_turn("user message", "sonnet")

        mock_cli.assert_called_once()
        self.assertEqual(result, "cli conv result")

    def test_auth_error_when_both_tiers_fail(self):
        """When no API key and CLI fails → LLMAuthenticationError."""
        from rocinsight.ai_analysis.exceptions import LLMAuthenticationError

        conv = LLMConversation(provider="claude-code")
        conv.initialize("system")

        with patch.dict(os.environ, {}, clear=False), \
             patch.object(conv, "_call_claude_cli_subprocess",
                          side_effect=RuntimeError("no CLI")):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with self.assertRaises(LLMAuthenticationError):
                conv._call_claude_code_turn("msg", "sonnet")


# ── LLMConversation._call_claude_cli_subprocess ───────────────────────────────

class TestConversationCliSubprocess(unittest.TestCase):
    """Unit tests for _call_claude_cli_subprocess in LLMConversation."""

    def _make_conv(self):
        conv = LLMConversation(provider="claude-code")
        conv.initialize("system prompt")
        return conv

    def test_returns_result_from_json_output(self):
        conv = self._make_conv()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"result": "conv text", "is_error": false}'
        mock_proc.stderr = ""

        with patch("subprocess.run", return_value=mock_proc):
            result = conv._call_claude_cli_subprocess("usr", "sonnet")

        self.assertEqual(result, "conv text")

    def test_returns_plain_text_when_not_json(self):
        conv = self._make_conv()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "  plain text  "
        mock_proc.stderr = ""

        with patch("subprocess.run", return_value=mock_proc):
            result = conv._call_claude_cli_subprocess("usr", "sonnet")

        self.assertEqual(result, "plain text")

    def test_raises_on_file_not_found(self):
        conv = self._make_conv()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaises(RuntimeError):
                conv._call_claude_cli_subprocess("usr", "sonnet")


# ── Interactive.py dispatch: claude-code must not fall through to _call_local ─

class TestInteractiveDispatchClaudeCode(unittest.TestCase):
    """Regression: interactive.py _llm_rewrite_file must route claude-code to
    _call_claude_code, not _call_local (Ollama)."""

    def _make_analyzer_with_mock(self):
        from rocinsight.ai_analysis.llm_analyzer import LLMAnalyzer

        call_tracker = {}

        class _TrackedAnalyzer(LLMAnalyzer):
            def _call_claude_code(self, *a, **kw):
                call_tracker["method"] = "claude_code"
                return "rewritten content"

            def _call_local(self, *a, **kw):
                call_tracker["method"] = "local"
                return "local result"

            def _call_anthropic(self, *a, **kw):
                call_tracker["method"] = "anthropic"
                return "anthropic result"

        return _TrackedAnalyzer(provider="claude-code"), call_tracker

    def test_llm_rewrite_file_uses_claude_code_not_local(self):
        """_llm_rewrite_file with provider='claude-code' must call _call_claude_code."""
        import pathlib
        import tempfile

        from rocinsight.ai_analysis.interactive import WorkflowSession

        analyzer, tracker = self._make_analyzer_with_mock()

        with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
            f.write("int main(){}")
            tmp = pathlib.Path(f.name)

        try:
            session = WorkflowSession.__new__(WorkflowSession)
            session._llm_provider = "claude-code"
            session._llm_api_key = None
            session._llm_model = "sonnet"
            session._source_paths = [tmp.parent]

            with patch(
                "rocinsight.ai_analysis.llm_analyzer.LLMAnalyzer",
                return_value=analyzer,
            ):
                result = session._llm_rewrite_file(tmp, "optimize this")

            self.assertEqual(tracker.get("method"), "claude_code",
                             "claude-code provider must route to _call_claude_code, "
                             f"but routed to: {tracker.get('method')}")
            self.assertIsNotNone(result)
        finally:
            tmp.unlink(missing_ok=True)

    def test_llm_rewrite_file_does_not_use_local_for_claude_code(self):
        """Regression: 'else → _call_local' must not be reachable for claude-code."""
        import pathlib
        import tempfile

        from rocinsight.ai_analysis.interactive import WorkflowSession

        analyzer, tracker = self._make_analyzer_with_mock()

        with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
            f.write("void kernel(){}")
            tmp = pathlib.Path(f.name)

        try:
            session = WorkflowSession.__new__(WorkflowSession)
            session._llm_provider = "claude-code"
            session._llm_api_key = None
            session._llm_model = "sonnet"
            session._source_paths = [tmp.parent]

            with patch(
                "rocinsight.ai_analysis.llm_analyzer.LLMAnalyzer",
                return_value=analyzer,
            ):
                session._llm_rewrite_file(tmp, "suggestions")

            self.assertNotEqual(tracker.get("method"), "local",
                                "claude-code must not fall through to Ollama/_call_local")
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
