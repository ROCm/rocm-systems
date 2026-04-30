"""Packaging checks for generated opencode artifacts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_generated_opencode_binary_license_is_not_package_data():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"_bundled/opencode"' not in text
    assert '"_bundled/**/*"' not in text
    assert '"_bundled/opencode_LICENSE"' not in text
    assert '"_bundled/opencode_config/*"' in text
    assert not (REPO_ROOT / "perfxpert" / "_bundled" / "opencode_LICENSE").exists()
