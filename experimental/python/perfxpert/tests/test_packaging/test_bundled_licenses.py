"""Packaging checks for bundled third-party license files."""

from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_bundled_opencode_license_is_package_data_and_present():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]["perfxpert"]
    assert "_bundled/opencode_LICENSE" in package_data
    license_path = REPO_ROOT / "perfxpert" / "_bundled" / "opencode_LICENSE"
    assert "MIT License" in license_path.read_text(encoding="utf-8")
