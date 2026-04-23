"""Test that integration docs required by the multi-backend plan exist.

Task 0 of the multi-backend plan (B4): ensure `docs/integration/mcp-server.md`
is present so later tasks can cross-reference it without dangling links.
This test guards that port: if the file goes missing or someone
accidentally deletes the integration/ subtree, CI fails loudly.
"""

from __future__ import annotations

from pathlib import Path


# Anchor at the perfxpert app root (tests/ -> perfxpert app root).
_APP_ROOT = Path(__file__).resolve().parents[2]


def test_mcp_server_doc_exists() -> None:
    """`docs/integration/mcp-server.md` must exist (docs-integration port)."""
    target = _APP_ROOT / "docs" / "integration" / "mcp-server.md"
    assert target.is_file(), f"Missing {target}. Task 0 of multi-backend plan port failed."


def test_mcp_server_doc_is_non_empty() -> None:
    """Sanity check — the port should have copied a meaningful doc."""
    target = _APP_ROOT / "docs" / "integration" / "mcp-server.md"
    assert target.stat().st_size > 1_000, (
        f"{target} is smaller than expected (<1KB); port likely truncated."
    )


def test_integration_readme_exists() -> None:
    """README pointer should exist alongside mcp-server.md."""
    target = _APP_ROOT / "docs" / "integration" / "README.md"
    assert target.is_file(), f"Missing {target}. Task 0 port failed (README companion)."
