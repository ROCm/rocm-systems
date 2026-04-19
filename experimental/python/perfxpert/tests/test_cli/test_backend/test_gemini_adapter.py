"""Tests for `perfxpert.cli._backend.gemini.GeminiAdapter` (Task 5).

Covers:

* Tool-name template is single-underscore (B1).
* Never touches GEMINI.md (I3).
* context.fileName list-merge preserves existing entries.
* New entry appended when absent.
* Existing mcpServers preserved.
* Installs are idempotent.
* verify_mcp_live returns healthy when settings.json contains our entry.
* uninstall list-removes entry.
* spawn uses execvpe.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from perfxpert.cli._backend.gemini import GeminiAdapter
from perfxpert.cli._backend.protocol import (
    BackendAdapter,
    ConfigClobber,
    InstallReport,
    Plan,
    UninstallReport,
)


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("PERFXPERT_ASSUME_CONSENT", "1")
    monkeypatch.setenv("PERFXPERT_SKIP_LIVE_CHECK", "1")
    return tmp_path


@pytest.fixture
def project_cwd(isolated_home: Path) -> Path:
    cwd = isolated_home / "proj"
    cwd.mkdir()
    return cwd


def test_adapter_conforms_to_protocol() -> None:
    assert isinstance(GeminiAdapter(), BackendAdapter)


def test_tool_name_template_is_single_underscore() -> None:
    assert GeminiAdapter.tool_name_template == "mcp_perfxpert_{tool}"


def test_spawn_strategy_is_execvpe() -> None:
    assert GeminiAdapter.spawn_strategy == "execvpe"


# ---------------------------------------------------------------------------
# plan.
# ---------------------------------------------------------------------------


def test_plan_lists_settings_and_agents(project_cwd: Path, isolated_home: Path) -> None:
    plan = GeminiAdapter().plan(project_cwd)
    assert isinstance(plan, Plan)
    target_names = {p.name for p in plan.targets}
    assert "settings.json" in target_names
    assert "AGENTS.md" in target_names
    # Plan text mentions settings.json path + preserve-existing.
    joined = "\n".join(plan.actions)
    assert "settings.json" in joined
    assert "preserve" in joined.lower() or "context.fileName" in joined


def test_plan_never_mentions_gemini_md(
    project_cwd: Path, isolated_home: Path
) -> None:
    """I3: the whole point — GEMINI.md never appears in the plan."""
    plan = GeminiAdapter().plan(project_cwd)
    for action in plan.actions:
        assert "GEMINI.md" not in action
    for target in plan.targets:
        assert "GEMINI.md" not in str(target)


# ---------------------------------------------------------------------------
# install — settings.json merging.
# ---------------------------------------------------------------------------


def test_install_writes_mcp_servers_perfxpert(
    project_cwd: Path, isolated_home: Path
) -> None:
    adapter = GeminiAdapter()
    adapter.install(project_cwd)
    settings = isolated_home / ".gemini" / "settings.json"
    data = json.loads(settings.read_text())
    assert data["mcpServers"]["perfxpert"]["command"] == "perfxpert-mcp"


def test_install_list_appends_context_filename(
    project_cwd: Path, isolated_home: Path
) -> None:
    """practical §3.3: list-append preserves existing entries + adds ours."""
    settings = isolated_home / ".gemini" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(
        json.dumps(
            {
                "context": {
                    "fileName": ["~/.gemini/my-context.md", "/abs/file.md"]
                }
            }
        )
    )
    adapter = GeminiAdapter()
    adapter.install(project_cwd)
    data = json.loads(settings.read_text())
    files = data["context"]["fileName"]
    # Existing entries preserved in order.
    assert files[0] == "~/.gemini/my-context.md"
    assert files[1] == "/abs/file.md"
    # Our AGENTS.md appended.
    assert any(".perfxpert/AGENTS.md" in str(f) for f in files)


def test_install_does_not_duplicate_context_filename_on_rerun(
    project_cwd: Path, isolated_home: Path
) -> None:
    adapter = GeminiAdapter()
    adapter.install(project_cwd)
    adapter.install(project_cwd)  # idempotent.
    settings = isolated_home / ".gemini" / "settings.json"
    data = json.loads(settings.read_text())
    files = data["context"]["fileName"]
    perfxpert_entries = [f for f in files if ".perfxpert/AGENTS.md" in str(f)]
    assert len(perfxpert_entries) == 1


def test_install_preserves_existing_mcp_servers(
    project_cwd: Path, isolated_home: Path
) -> None:
    settings = isolated_home / ".gemini" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(
        json.dumps({"mcpServers": {"other": {"command": "other-bin", "args": []}}})
    )
    GeminiAdapter().install(project_cwd)
    data = json.loads(settings.read_text())
    assert data["mcpServers"]["other"]["command"] == "other-bin"
    assert data["mcpServers"]["perfxpert"]["command"] == "perfxpert-mcp"


def test_install_refuses_clobber(
    project_cwd: Path, isolated_home: Path
) -> None:
    settings = isolated_home / ".gemini" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "perfxpert": {"command": "different-bin", "args": []}
                }
            }
        )
    )
    with pytest.raises(ConfigClobber):
        GeminiAdapter().install(project_cwd)


def test_install_idempotent(
    project_cwd: Path, isolated_home: Path
) -> None:
    adapter = GeminiAdapter()
    r1 = adapter.install(project_cwd)
    r2 = adapter.install(project_cwd)
    assert isinstance(r1, InstallReport)
    assert isinstance(r2, InstallReport)


# ---------------------------------------------------------------------------
# I3: never touches GEMINI.md.
# ---------------------------------------------------------------------------


def test_install_never_touches_gemini_md(
    project_cwd: Path, isolated_home: Path
) -> None:
    """I3: GEMINI.md must not be created, even if user has one.

    We seed a GEMINI.md and assert byte-identical state after install.
    """
    gemini_md = project_cwd / "GEMINI.md"
    gemini_md.write_text("user's own content — do NOT edit\n")
    snapshot = gemini_md.read_bytes()
    GeminiAdapter().install(project_cwd)
    assert gemini_md.read_bytes() == snapshot


# ---------------------------------------------------------------------------
# verify_mcp_live.
# ---------------------------------------------------------------------------


def test_verify_mcp_live_healthy_after_install(
    project_cwd: Path, isolated_home: Path
) -> None:
    GeminiAdapter().install(project_cwd)
    # Remove the SKIP env so verify actually runs.
    os.environ.pop("PERFXPERT_SKIP_LIVE_CHECK", None)
    report = GeminiAdapter().verify_mcp_live(project_cwd)
    assert report.mcp_healthy is True
    assert report.gate_hook_installed is True  # gate hook installed by install()


def test_verify_mcp_live_unhealthy_when_entry_missing(
    project_cwd: Path, isolated_home: Path
) -> None:
    settings = isolated_home / ".gemini" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"mcpServers": {"other": {"command": "o"}}}))
    report = GeminiAdapter().verify_mcp_live(project_cwd)
    assert report.mcp_healthy is False
    assert "missing" in (report.error or "").lower()


def test_verify_mcp_live_returns_error_when_settings_absent(
    project_cwd: Path, isolated_home: Path
) -> None:
    report = GeminiAdapter().verify_mcp_live(project_cwd)
    assert report.mcp_healthy is False
    assert "not present" in (report.error or "")


# ---------------------------------------------------------------------------
# uninstall.
# ---------------------------------------------------------------------------


def test_uninstall_removes_mcp_entry_and_context_filename(
    project_cwd: Path, isolated_home: Path
) -> None:
    # Seed existing entries we want preserved.
    settings = isolated_home / ".gemini" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(
        json.dumps(
            {
                "mcpServers": {"other": {"command": "other-bin", "args": []}},
                "context": {"fileName": ["/user/other.md"]},
            }
        )
    )
    adapter = GeminiAdapter()
    adapter.install(project_cwd)
    adapter.uninstall(project_cwd)

    data = json.loads(settings.read_text())
    # Other entries preserved.
    assert data["mcpServers"]["other"]["command"] == "other-bin"
    # perfxpert entries removed.
    assert "perfxpert" not in data["mcpServers"]
    assert "/user/other.md" in data["context"]["fileName"]
    assert not any(".perfxpert/AGENTS.md" in str(f) for f in data["context"]["fileName"])


def test_uninstall_returns_report(
    project_cwd: Path, isolated_home: Path
) -> None:
    adapter = GeminiAdapter()
    adapter.install(project_cwd)
    report = adapter.uninstall(project_cwd)
    assert isinstance(report, UninstallReport)


# ---------------------------------------------------------------------------
# spawn.
# ---------------------------------------------------------------------------


def test_spawn_uses_execvpe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict = {}

    def _fake_execvpe(name, argv, env):
        called["name"] = name
        called["argv"] = list(argv)
        raise RuntimeError("stopped")

    monkeypatch.setattr("os.execvpe", _fake_execvpe)
    monkeypatch.setattr("os.chdir", lambda _p: None)
    with pytest.raises(RuntimeError, match="stopped"):
        GeminiAdapter().spawn(["hello"], {"K": "V"}, tmp_path)
    assert called["name"] == "gemini"
    assert called["argv"] == ["gemini", "hello"]
