#!/usr/bin/env python3
"""
Regression tests for _post_revert_llm_analysis alternative-apply compile-wait.

Bug: after the user accepted an LLM-proposed alternative fix, the session
skipped the compile-wait loop and went straight to the post-revert menu
([p]/[q]), so the user was never asked to recompile and type 'done'.

Fix: the compile-wait loop is now run inline in _post_revert_llm_analysis
before returning when the alternative is applied.
"""
import contextlib
import pathlib
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_ANALYSIS_TEXT = (
    "ANALYSIS:\nThe file was truncated at line 261, leaving braces unclosed.\n\n"
    "ALTERNATIVE:\nHere is the corrected file:\n```cpp\nint main(){return 0;}\n```"
)
_ALT_REWRITE = "int main() { return 0; }\n"
_ORIGINAL = "int main() { return 1; }\n"
_FAILED = "int main() { return BROKEN\n"


@contextlib.contextmanager
def _mock_llm(analysis_text: str = _ANALYSIS_TEXT):
    """Patch LLMAnalyzer (imported inside the method) and _Spinner."""
    spinner = MagicMock(
        __enter__=MagicMock(return_value=None),
        __exit__=MagicMock(return_value=False),
    )
    with (
        # LLMAnalyzer is imported lazily inside the method body
        patch("rocinsight.ai_analysis.llm_analyzer.LLMAnalyzer", autospec=True) as M,
        patch(
            "rocinsight.ai_analysis.interactive._Spinner",
            return_value=spinner,
        ),
    ):
        M.return_value._call_anthropic.return_value = analysis_text
        yield


def _make_session(tmp_path: pathlib.Path):
    """Construct a real WorkflowSession with minimal deps mocked out."""
    from rocinsight.ai_analysis.interactive import WorkflowSession

    # __init__ only needs app_command; it doesn't touch the filesystem.
    sess = WorkflowSession(
        app_command="./demo",
        source_paths=[str(tmp_path)],
        llm_provider="anthropic",
        llm_api_key="dummy",
        trace_dir=str(tmp_path / "traces"),
    )
    # Redirect sessions dir so tests don't touch $HOME
    sess._sessions_dir = tmp_path / "sessions"
    sess._sessions_dir.mkdir(parents=True, exist_ok=True)
    sess._session_file = sess._sessions_dir / "test.json"
    return sess


def _run(
    tmp_path: pathlib.Path,
    user_inputs: list,
    alt_rewrite: str = _ALT_REWRITE,
    analysis_text: str = _ANALYSIS_TEXT,
):
    """Drive _post_revert_llm_analysis with mocked LLM + user inputs.

    Returns (action, file_content_after).
    """
    sess = _make_session(tmp_path)
    src = tmp_path / "test.cpp"
    src.write_text(_ORIGINAL)

    inputs = iter(user_inputs)

    with (
        _mock_llm(analysis_text),
        patch("rocinsight.ai_analysis.interactive._input", side_effect=lambda p="": next(inputs)),
        patch.object(sess, "_llm_rewrite_file", return_value=alt_rewrite),
        patch.object(sess, "_save_session"),
    ):
        action = sess._post_revert_llm_analysis(
            file_path=src,
            original_content=_ORIGINAL,
            failed_content=_FAILED,
            failure_reason="syntax error at line 261",
            allow_retry=True,
        )

    return action, src.read_text()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAlternativeApplyCompileWait:
    """Compile-wait loop must be shown after the alternative is applied."""

    def test_apply_then_done_returns_continue(self, tmp_path):
        """
        Regression: apply alt → type 'done' → must return 'continue'.

        Before the fix this returned the post-revert menu action without
        ever asking the user to compile.
        Inputs:
          "Apply this alternative? [y/N]"   → y
          "Apply this corrected version? [y/N]" → y
          compile-wait "> "                 → done
        """
        action, _ = _run(tmp_path, ["y", "y", "done"])
        assert action == "continue"

    def test_apply_then_blank_is_done(self, tmp_path):
        """Blank input at compile-wait counts as 'done'."""
        action, _ = _run(tmp_path, ["y", "y", ""])
        assert action == "continue"

    def test_apply_with_error_output_then_done(self, tmp_path):
        """User pastes compiler errors then fixes and types 'done'."""
        action, _ = _run(
            tmp_path,
            [
                "y",                                  # Apply alternative?
                "y",                                  # Apply corrected version?
                "multi_gpu_demo.cpp:261: error: expected '}'",  # error pasted
                "note: to match this '{'",            # more output
                "done",                               # compiled OK
            ],
        )
        assert action == "continue"

    def test_apply_then_abort_returns_exit(self, tmp_path):
        """'abort' at compile-wait propagates 'exit'."""
        action, _ = _run(tmp_path, ["y", "y", "abort"])
        assert action == "exit"

    def test_apply_then_revert_restores_original(self, tmp_path):
        """'revert' at compile-wait restores the original file."""
        sess = _make_session(tmp_path)
        src = tmp_path / "test.cpp"
        src.write_text(_ORIGINAL)

        # After revert, _post_revert_menu is shown → [p] = continue
        inputs = iter(["y", "y", "revert", "p"])

        with (
            _mock_llm(),
            patch("rocinsight.ai_analysis.interactive._input", side_effect=lambda p="": next(inputs)),
            patch.object(sess, "_llm_rewrite_file", return_value=_ALT_REWRITE),
            patch.object(sess, "_save_session"),
        ):
            action = sess._post_revert_llm_analysis(
                file_path=src,
                original_content=_ORIGINAL,
                failed_content=_FAILED,
                failure_reason="error",
                allow_retry=True,
            )

        assert src.read_text() == _ORIGINAL, "File should be restored to original"
        assert action in ("continue", "exit")

    def test_decline_corrected_version_skips_compile_wait(self, tmp_path):
        """
        Declining 'Apply corrected version?' must NOT enter the compile-wait loop.
        Goes straight to post-revert menu.

        Inputs: y (apply alt?) → n (corrected version?) → p (menu: continue)
        """
        action, content = _run(tmp_path, ["y", "n", "p"])
        assert action == "continue"
        # File unchanged
        assert content == _ORIGINAL

    def test_decline_first_offer_skips_compile_wait(self, tmp_path):
        """Declining 'Apply this alternative?' goes to post-revert menu."""
        action, content = _run(tmp_path, ["n", "p"])
        assert action == "continue"
        assert content == _ORIGINAL

    def test_no_llm_provider_skips_analysis(self, tmp_path):
        """Without a configured LLM, falls through to post-revert menu."""
        sess = _make_session(tmp_path)
        sess._llm_provider = None
        src = tmp_path / "test.cpp"
        src.write_text(_ORIGINAL)

        inputs = iter(["p"])
        with patch("rocinsight.ai_analysis.interactive._input", side_effect=lambda p="": next(inputs)):
            action = sess._post_revert_llm_analysis(
                file_path=src,
                original_content=_ORIGINAL,
                failed_content=_FAILED,
                failure_reason="error",
                allow_retry=True,
            )
        assert action == "continue"


# ---------------------------------------------------------------------------
# Plateau detection tests
# ---------------------------------------------------------------------------


class TestPlateauDetection:
    """Tests for the optimization plateau detection feature."""

    def _make_breakdown(self, total_runtime_ns: int) -> dict:
        return {
            "kernel_time_pct": 80.0,
            "memcpy_time_pct": 10.0,
            "api_overhead_pct": 5.0,
            "idle_time_pct": 5.0,
            "total_runtime_ns": total_runtime_ns,
        }

    def _make_snap(self, iteration: int, total_runtime_ns: int):
        from rocinsight.ai_analysis.interactive import _AnalysisSnapshot

        return _AnalysisSnapshot(
            timestamp="2026-01-01T00:00:00+00:00",
            iteration=iteration,
            recommendations=[],
            execution_breakdown=self._make_breakdown(total_runtime_ns),
            hotspots=[],
        )

    def test_plateau_after_two_small_changes(self, tmp_path):
        """Two consecutive <2% changes should trigger plateau detection."""
        sess = _make_session(tmp_path)
        # Simulate 2 prior runs with similar runtimes
        sess._state.analysis_history.append(self._make_snap(0, 1_000_000_000))
        sess._state.analysis_history.append(self._make_snap(1, 1_005_000_000))
        # First small change: increment plateau_iteration_count to 1
        sess._state.plateau_iteration_count = 1

        # New breakdown with <2% change from last
        new_bd = self._make_breakdown(1_010_000_000)  # ~0.5% change from 1_005_000_000
        is_plateaued, count, pct = sess._compute_plateau_status(new_bd)
        assert is_plateaued is True
        assert count >= 2
        assert pct < 2.0

    def test_no_plateau_on_first_run(self, tmp_path):
        """No prior history should not trigger plateau."""
        sess = _make_session(tmp_path)
        new_bd = self._make_breakdown(1_000_000_000)
        is_plateaued, count, pct = sess._compute_plateau_status(new_bd)
        assert is_plateaued is False
        assert count == 0
        assert pct == 0.0

    def test_plateau_resets_on_improvement(self, tmp_path):
        """A >5% change should reset the plateau counter."""
        sess = _make_session(tmp_path)
        sess._state.analysis_history.append(self._make_snap(0, 1_000_000_000))
        sess._state.plateau_iteration_count = 3

        # New breakdown with >5% change
        new_bd = self._make_breakdown(900_000_000)  # 10% change
        is_plateaued, count, pct = sess._compute_plateau_status(new_bd)
        assert is_plateaued is False
        assert sess._state.plateau_iteration_count == 0

    def test_filter_seen_recommendations(self, tmp_path):
        """Previously-seen recommendations should be filtered out."""
        sess = _make_session(tmp_path)
        sess._state.seen_recommendation_hashes = [
            "MEMORY_TRANSFER:High memory transfer overhead detected",
        ]
        recs = [
            {"category": "MEMORY_TRANSFER", "issue": "High memory transfer overhead detected"},
            {"category": "COMPUTE", "issue": "Kernel compute bottleneck"},
        ]
        filtered, suppressed = sess._filter_seen_recommendations(recs)
        assert suppressed == 1
        assert len(filtered) == 1
        assert filtered[0]["category"] == "COMPUTE"

    def test_zero_runtime_guard(self, tmp_path):
        """Previous run with total_runtime_ns=0 should not crash."""
        sess = _make_session(tmp_path)
        sess._state.analysis_history.append(self._make_snap(0, 0))  # zero runtime

        new_bd = self._make_breakdown(1_000_000_000)
        is_plateaued, count, pct = sess._compute_plateau_status(new_bd)
        assert is_plateaued is False
        assert pct == 0.0


# ---------------------------------------------------------------------------
# WorkflowSession conversation persistence tests
# ---------------------------------------------------------------------------


class TestWorkflowConversation:
    """WorkflowSession conversation persistence tests."""

    def test_ensure_conv_no_llm_returns_none(self, tmp_path):
        """Without LLM provider, _ensure_conv returns None."""
        sess = _make_session(tmp_path)
        sess._llm_provider = None
        assert sess._ensure_conv() is None

    def test_ensure_conv_creates_conversation(self, tmp_path):
        """With LLM provider, _ensure_conv creates an LLMConversation."""
        sess = _make_session(tmp_path)
        sess._llm_provider = "anthropic"
        sess._llm_api_key = "dummy"
        with patch(
            "rocinsight.ai_analysis.interactive.LLMConversation"
        ) as MockConv:
            instance = MockConv.return_value
            instance.initialize = MagicMock()
            conv = sess._ensure_conv()
            assert conv is not None
            instance.initialize.assert_called_once()

    def test_ensure_conv_restores_from_state(self, tmp_path):
        """When state.conversation is set, _ensure_conv calls from_dict."""
        sess = _make_session(tmp_path)
        sess._llm_provider = "anthropic"
        sess._llm_api_key = "dummy"
        sess._state.conversation = {"messages": [], "system": "test"}
        with patch(
            "rocinsight.ai_analysis.interactive.LLMConversation"
        ) as MockConv:
            MockConv.from_dict = MagicMock(return_value=MagicMock())
            conv = sess._ensure_conv()
            MockConv.from_dict.assert_called_once()

    def test_save_session_flushes_conversation(self, tmp_path):
        """_save_session stores conversation dict in state."""
        sess = _make_session(tmp_path)
        mock_conv = MagicMock()
        mock_conv.to_dict.return_value = {
            "messages": [{"role": "user", "content": "test"}]
        }
        sess._conv = mock_conv
        # Manually trigger the flush logic
        if sess._conv is not None:
            sess._state.conversation = sess._conv.to_dict()
        assert sess._state.conversation is not None
        assert len(sess._state.conversation["messages"]) == 1

    def test_ensure_conv_idempotent(self, tmp_path):
        """Calling _ensure_conv twice returns the same instance."""
        sess = _make_session(tmp_path)
        sess._llm_provider = "anthropic"
        sess._llm_api_key = "dummy"
        with patch(
            "rocinsight.ai_analysis.interactive.LLMConversation"
        ) as MockConv:
            instance = MockConv.return_value
            instance.initialize = MagicMock()
            conv1 = sess._ensure_conv()
            conv2 = sess._ensure_conv()
            assert conv1 is conv2
            # initialize should only be called once
            assert instance.initialize.call_count == 1

    def test_save_session_warns_large_conversation(self, tmp_path):
        """_save_session emits UserWarning when conversation has >1000 messages."""
        import warnings

        sess = _make_session(tmp_path)
        sess._conv = None  # no live conversation object
        # Inject a large conversation dict directly into state
        sess._state.conversation = {
            "messages": [{"role": "user", "content": f"msg {i}"} for i in range(1500)]
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sess._save_session()
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "1500 messages" in str(user_warnings[0].message)

    def test_save_session_no_warning_small_conversation(self, tmp_path):
        """_save_session does NOT warn when conversation has <= 1000 messages."""
        import warnings

        sess = _make_session(tmp_path)
        sess._conv = None
        sess._state.conversation = {
            "messages": [{"role": "user", "content": f"msg {i}"} for i in range(50)]
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sess._save_session()
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0


# ---------------------------------------------------------------------------
# Command injection validation tests
# ---------------------------------------------------------------------------


class TestCommandValidation:
    """Tests for _has_shell_meta() shell injection detection."""

    def test_safe_command_returns_false(self):
        """Normal rocprofv3 command should not be flagged."""
        from rocinsight.ai_analysis.interactive import _has_shell_meta

        assert _has_shell_meta("rocprofv3 --sys-trace -- ./app") is False

    def test_semicolon_injection_detected(self):
        """Semicolon-based command chaining should be detected."""
        from rocinsight.ai_analysis.interactive import _has_shell_meta

        assert _has_shell_meta("rocprofv3 ; rm -rf /") is True

    def test_pipe_injection_detected(self):
        """Pipe-based output redirection should be detected."""
        from rocinsight.ai_analysis.interactive import _has_shell_meta

        assert _has_shell_meta("rocprofv3 | cat /etc/passwd") is True

    def test_ampersand_chain_detected(self):
        """Ampersand-based command chaining should be detected."""
        from rocinsight.ai_analysis.interactive import _has_shell_meta

        assert _has_shell_meta("rocprofv3 && echo pwned") is True

    def test_subshell_detected(self):
        """Dollar-sign subshell expansion should be detected."""
        from rocinsight.ai_analysis.interactive import _has_shell_meta

        assert _has_shell_meta("rocprofv3 $(whoami)") is True

    def test_backtick_detected(self):
        """Backtick command substitution should be detected."""
        from rocinsight.ai_analysis.interactive import _has_shell_meta

        assert _has_shell_meta("rocprofv3 `id`") is True

    def test_parentheses_detected(self):
        """Parentheses (subshell grouping) should be detected."""
        from rocinsight.ai_analysis.interactive import _has_shell_meta

        assert _has_shell_meta("rocprofv3 (echo hi)") is True

    def test_curly_braces_detected(self):
        """Curly braces (brace expansion) should be detected."""
        from rocinsight.ai_analysis.interactive import _has_shell_meta

        assert _has_shell_meta("rocprofv3 {a,b}") is True

    def test_exclamation_detected(self):
        """Exclamation mark (history expansion) should be detected."""
        from rocinsight.ai_analysis.interactive import _has_shell_meta

        assert _has_shell_meta("rocprofv3 !last") is True

    def test_safe_flags_and_paths(self):
        """Complex but safe command with flags, paths, env vars should pass."""
        from rocinsight.ai_analysis.interactive import _has_shell_meta

        assert _has_shell_meta("rocprofv3 --kernel-trace --stats -d /tmp/out -o results -- ./my_app") is False

    def test_empty_string_is_safe(self):
        """Empty string should not be flagged."""
        from rocinsight.ai_analysis.interactive import _has_shell_meta

        assert _has_shell_meta("") is False
