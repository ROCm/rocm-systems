"""Suite-wide isolation for user-local PerfXpert state."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_knowledge_store(monkeypatch, tmp_path):
    """Never let retention tests or producer hooks touch a developer's home."""

    monkeypatch.setenv(
        "PERFXPERT_KNOWLEDGE_ROOT",
        str(tmp_path / "perfxpert-knowledge"),
    )
    monkeypatch.setenv(
        "PERFXPERT_PROJECT_ROOT",
        str(tmp_path / "project"),
    )
