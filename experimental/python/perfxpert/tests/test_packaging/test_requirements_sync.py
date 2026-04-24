"""Checks that fallback requirements files mirror pyproject metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-r "):
            continue
        names.add(line.split(";", 1)[0].split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].split(">", 1)[0])
    return names


def _pyproject_names(extra: str | None = None) -> set[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    values = list(data["project"].get("dependencies", []))
    if extra:
        values.extend(data["project"]["optional-dependencies"][extra])
    return {
        value.split(";", 1)[0].split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].split(">", 1)[0]
        for value in values
    }


def test_requirements_txt_matches_full_runtime_dependencies():
    assert _requirement_names(REPO_ROOT / "requirements.txt") == _pyproject_names("all")


def test_requirements_dev_txt_matches_dev_dependencies():
    expected = _pyproject_names("all") | _pyproject_names("dev")
    actual = _requirement_names(REPO_ROOT / "requirements.txt") | _requirement_names(REPO_ROOT / "requirements-dev.txt")
    assert actual == expected
