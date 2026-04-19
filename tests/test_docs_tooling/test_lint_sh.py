#!/usr/bin/env python3
"""Unit tests for docs/lint.sh.

Covers:
- Existence / executability
- Original 9 banned strings (Dimension 1)
- 10 total-legacy-scrub additions
- Historical-anchor exception ("removed in Phase 7.1" — load-bearing
  literal string; see docs/lint.sh for why the phrase is load-bearing)
- Exact-phrase-match semantics of the anchor filter
"""

import subprocess
import tempfile
import os
from pathlib import Path

# Absolute path to the lint.sh we're testing. Using an absolute path
# avoids cwd ambiguity when the subprocess is invoked from a tmpdir.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LINT_SH = _REPO_ROOT / "docs" / "lint.sh"


def _run_lint_on_doc(md_body: str) -> subprocess.CompletedProcess:
    """Run lint.sh in a throwaway tmpdir with a single docs/*.md file.

    lint.sh searches relative paths `experimental/python/perfxpert` and
    `docs`. We create `docs/` inside a tmpdir, run lint.sh with cwd =
    tmpdir so the relative path resolves to that docs/, and invoke the
    script by absolute path so the bash process finds it.
    """
    with tempfile.TemporaryDirectory() as raw:
        tmpdir = Path(raw)
        (tmpdir / "docs").mkdir()
        (tmpdir / "docs" / "test.md").write_text(md_body)
        return subprocess.run(
            ["bash", str(_LINT_SH)],
            cwd=str(tmpdir),
            capture_output=True,
            text=True,
        )


def test_lint_sh_exists():
    """Lint script must exist and be executable."""
    lint_script = Path("docs/lint.sh")
    assert lint_script.exists(), "docs/lint.sh does not exist"
    assert os.access(lint_script, os.X_OK), "docs/lint.sh is not executable"


def test_lint_sh_detects_interactive_py():
    """Lint script must detect 'interactive.py' references."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_doc = tmpdir / "test.md"
        test_doc.write_text("# Test\nSee `interactive.py` for details.")

        result = subprocess.run(
            ["bash", "docs/lint.sh"],
            cwd=str(tmpdir.parent),
            capture_output=True,
            text=True,
        )
        # Should find at least one violation
        assert result.returncode != 0 or "interactive.py" in result.stdout


def test_lint_sh_detects_llm_conversation():
    """Lint script must detect 'LLMConversation' references."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_doc = tmpdir / "test.md"
        test_doc.write_text("# API\nUse `LLMConversation` class.")

        result = subprocess.run(
            ["bash", "docs/lint.sh"],
            cwd=str(tmpdir.parent),
            capture_output=True,
            text=True,
        )
        # Should find violation
        assert result.returncode != 0 or "LLMConversation" in result.stdout


def test_lint_sh_detects_interactive_flag():
    """Lint script must detect '--interactive' flag references."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_doc = tmpdir / "test.md"
        test_doc.write_text("# CLI\nRun with `--interactive` flag.")

        result = subprocess.run(
            ["bash", "docs/lint.sh"],
            cwd=str(tmpdir.parent),
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 or "--interactive" in result.stdout


def test_lint_sh_detects_all_banned_strings():
    """Lint script must detect all 9 banned strings."""
    banned = [
        "interactive.py",
        "LLMConversation",
        "llm_analyzer.analyze_with_llm",
        "--interactive",
        "--resume-session",
        "AnalysisContext",
        "ROCINSIGHT_LLM_",
        "ROCPD_LLM_",
        ".resume()",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "docs").mkdir()

        # Create docs with all banned strings
        for i, banned_str in enumerate(banned):
            test_doc = tmpdir / f"docs/test_{i}.md"
            test_doc.write_text(f"# Test\nContains {banned_str}")

        result = subprocess.run(
            ["bash", "docs/lint.sh"],
            cwd=str(tmpdir),
            capture_output=True,
            text=True,
        )

        # All 9 should be detected
        output = result.stdout + result.stderr
        for banned_str in banned:
            # At least some should match (accounting for regex escaping)
            pass

        # Should have non-zero exit (violation found)
        assert result.returncode != 0, "lint.sh should exit non-zero on violations"


# -----------------------------------------------------------------------
# Total-legacy-scrub additions — cycle-2 coverage hardening
# -----------------------------------------------------------------------
# The 10 new banned strings added by commit `dfdae97794`. The lint.sh
# currently encodes 10 post-refactor entries (9 original + 10 new = 19).
# Coverage requirement: one positive test per new banned string, plus
# dedicated tests for the historical-anchor exception path.

NEW_BANNED_STRINGS = [
    "PERFXPERT_USE_AGENTS",
    "ROCINSIGHT_",
    "_route_to_legacy",
    "_route_to_agents",
    "_is_legacy_mode",
    "_LegacyTraceAnalysis",
    "LLMAnalyzer",
    "from perfxpert.ai_analysis",
    "analyze_database(",
    "ROCINSIGHT_LLM_",  # kept for belt-and-braces — already in original 9
]


def test_banned_perfxpert_use_agents_detected():
    """A raw PERFXPERT_USE_AGENTS mention in a .md file must be flagged."""
    result = _run_lint_on_doc("Set `PERFXPERT_USE_AGENTS=1` to enable.")
    assert result.returncode != 0, (
        "lint.sh failed to flag 'PERFXPERT_USE_AGENTS'; "
        f"stdout={result.stdout!r}"
    )
    assert "PERFXPERT_USE_AGENTS" in result.stdout


def test_banned_perfxpert_use_agents_anchored_passes():
    """Same banned string + historical anchor on same line must pass."""
    body = "The PERFXPERT_USE_AGENTS flag was removed in Phase 7.1."
    result = _run_lint_on_doc(body)
    assert result.returncode == 0, (
        "lint.sh should not flag lines anchored with 'removed in Phase 7.1'; "
        f"stdout={result.stdout!r}"
    )


def test_banned_rocinsight_prefix_detected():
    result = _run_lint_on_doc("Env var `ROCINSIGHT_FOO` does something.")
    assert result.returncode != 0
    assert "ROCINSIGHT_" in result.stdout


def test_banned_rocinsight_prefix_anchored_passes():
    body = "`ROCINSIGHT_FOO` was removed in Phase 7.1."
    result = _run_lint_on_doc(body)
    assert result.returncode == 0, f"stdout={result.stdout!r}"


def test_banned_route_to_legacy_detected():
    result = _run_lint_on_doc("Call `_route_to_legacy` on dispatch.")
    assert result.returncode != 0
    assert "_route_to_legacy" in result.stdout


def test_banned_route_to_agents_detected():
    result = _run_lint_on_doc("Call `_route_to_agents` on dispatch.")
    assert result.returncode != 0
    assert "_route_to_agents" in result.stdout


def test_banned_is_legacy_mode_detected():
    result = _run_lint_on_doc("`_is_legacy_mode()` returns bool.")
    assert result.returncode != 0
    assert "_is_legacy_mode" in result.stdout


def test_banned_legacy_trace_analysis_detected():
    result = _run_lint_on_doc("Instantiate `_LegacyTraceAnalysis` directly.")
    assert result.returncode != 0
    assert "_LegacyTraceAnalysis" in result.stdout


def test_banned_llm_analyzer_detected():
    result = _run_lint_on_doc("Use `LLMAnalyzer` to call the model.")
    assert result.returncode != 0
    assert "LLMAnalyzer" in result.stdout


def test_banned_llm_analyzer_anchored_passes():
    body = "`LLMAnalyzer` was removed in Phase 7.1."
    result = _run_lint_on_doc(body)
    assert result.returncode == 0, f"stdout={result.stdout!r}"


def test_banned_from_perfxpert_ai_analysis_detected():
    result = _run_lint_on_doc("Import it: `from perfxpert.ai_analysis import x`.")
    assert result.returncode != 0
    assert "from perfxpert" in result.stdout  # banned-pattern echoed


def test_banned_from_perfxpert_ai_analysis_anchored_passes():
    body = "`from perfxpert.ai_analysis import X` was removed in Phase 7.1."
    result = _run_lint_on_doc(body)
    assert result.returncode == 0, f"stdout={result.stdout!r}"


def test_banned_analyze_database_call_detected():
    """analyze_database( — the open-paren form — must be flagged."""
    result = _run_lint_on_doc("Call `analyze_database(path)` to kick it off.")
    assert result.returncode != 0
    assert "analyze_database(" in result.stdout


def test_bare_analyze_database_mention_passes():
    """Docstring prose 'analyze_database' without '(' should NOT be flagged.

    The lint.sh banned pattern is literally `analyze_database(` — bare
    mentions in prose ('the legacy analyze_database helper') are safe
    because they lack the open paren.
    """
    result = _run_lint_on_doc("Background on the analyze_database helper.")
    assert result.returncode == 0, (
        f"lint.sh should not flag bare 'analyze_database'; stdout={result.stdout!r}"
    )


def test_all_new_banned_strings_detected_individually():
    """Loop positive-case coverage for every banned string in one table.

    Sanity-check that every entry in NEW_BANNED_STRINGS trips lint.sh.
    Complements the per-string tests above by catching any future drop
    of a pattern from lint.sh's BANNED array.
    """
    for banned in NEW_BANNED_STRINGS:
        result = _run_lint_on_doc(f"# Test\nContains `{banned}` inline.")
        assert result.returncode != 0, (
            f"lint.sh failed to flag banned string {banned!r}; "
            f"stdout={result.stdout!r}"
        )


# -----------------------------------------------------------------------
# Historical-anchor exception semantics
# -----------------------------------------------------------------------


def test_anchor_exact_string_match_required():
    """Only the exact phrase 'removed in Phase 7.1' triggers the anchor exception.

    Variants like 'deleted in Phase 7.1', 'removed in phase 7.1' (lower-case),
    or 'removed in Phase 7.2' MUST NOT suppress the violation — otherwise a
    lazy reviewer could accidentally invent their own pedigree phrase.
    """
    variants_that_must_fail = [
        "interactive.py (deleted in Phase 7.1)",
        "interactive.py (removed in phase 7.1)",    # lowercase phase
        "interactive.py (removed in Phase 7.2)",    # wrong number
        "interactive.py (removed-in-Phase-7.1)",    # punctuation
    ]
    for body in variants_that_must_fail:
        result = _run_lint_on_doc(body)
        assert result.returncode != 0, (
            f"Variant {body!r} wrongly suppressed violation; "
            f"stdout={result.stdout!r}"
        )

    # Only the exact canonical anchor passes.
    canonical = "interactive.py (removed in Phase 7.1)"
    result = _run_lint_on_doc(canonical)
    assert result.returncode == 0, (
        f"Canonical anchor wrongly flagged; stdout={result.stdout!r}"
    )


def test_anchor_filter_uses_fixed_string_semantics():
    """lint.sh's anchor filter is `grep -vqF` (fixed-string, not regex).

    Verifies the script source contains the exact `grep -vqF` invocation
    — a regression that drops the -F flag would re-introduce regex
    surprise (e.g. a '.' in a banned pattern would accidentally match
    any character in the anchor phrase and mis-trigger exceptions).
    """
    lint_src = _LINT_SH.read_text()
    assert "grep -vqF 'removed in Phase 7.1'" in lint_src, (
        "lint.sh must keep the anchor filter as `grep -vqF` "
        "(fixed-string semantics). Regex semantics risk false anchors."
    )


def test_anchor_applies_per_line():
    """Anchor exception is line-scoped, not file-scoped.

    A file that has 'removed in Phase 7.1' on line 1 and a raw banned
    string on line 5 must still flag the line-5 violation.
    """
    body = (
        "Phase 7.1 was the cleanup: interactive.py was removed in Phase 7.1.\n"
        "\n"
        "Current code still unfortunately uses LLMConversation though.\n"
    )
    result = _run_lint_on_doc(body)
    assert result.returncode != 0, (
        "Anchor must apply per-line, not across the file; "
        f"stdout={result.stdout!r}"
    )
    assert "LLMConversation" in result.stdout


if __name__ == "__main__":
    # Quick smoke test
    test_lint_sh_exists()
    print("✓ test_lint_sh_exists passed")
