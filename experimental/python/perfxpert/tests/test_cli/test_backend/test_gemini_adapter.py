"""Tests for `perfxpert.cli._backend.gemini.GeminiAdapter` (Task 5)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from perfxpert.cli._backend.gemini import GeminiAdapter
from perfxpert.cli._backend.protocol import (
    BackendAdapter,
    ConfigClobber,
    GateHookUnsupported,
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


def _project_settings(project_cwd: Path) -> Path:
    return project_cwd / ".gemini" / "settings.json"


def test_adapter_conforms_to_protocol() -> None:
    assert isinstance(GeminiAdapter(), BackendAdapter)


def test_tool_name_template_is_single_underscore() -> None:
    assert GeminiAdapter.tool_name_template == "mcp_perfxpert_{tool}"


def test_spawn_strategy_is_execvpe() -> None:
    assert GeminiAdapter.spawn_strategy == "execvpe"


def test_plan_lists_project_settings_and_agents(project_cwd: Path) -> None:
    plan = GeminiAdapter().plan(project_cwd)
    assert isinstance(plan, Plan)
    assert _project_settings(project_cwd) in plan.targets
    assert project_cwd / ".perfxpert" / "AGENTS.md" in plan.targets
    joined = "\n".join(plan.actions)
    assert ".gemini/settings.json" in joined
    assert "context.fileName" in joined
    assert "BeforeTool/AfterTool" in joined


def test_plan_never_mentions_gemini_md(project_cwd: Path) -> None:
    plan = GeminiAdapter().plan(project_cwd)
    for action in plan.actions:
        assert "GEMINI.md" not in action
    for target in plan.targets:
        assert "GEMINI.md" not in str(target)


def test_install_writes_project_mcp_servers_perfxpert(project_cwd: Path) -> None:
    GeminiAdapter().install(project_cwd)
    data = json.loads(_project_settings(project_cwd).read_text())
    assert data["mcpServers"]["perfxpert"]["command"] == "perfxpert-mcp"


def test_install_does_not_write_user_global_settings_when_not_needed(
    project_cwd: Path, isolated_home: Path
) -> None:
    GeminiAdapter().install(project_cwd)
    assert not (isolated_home / ".gemini" / "settings.json").exists()


def test_install_list_appends_context_filename(project_cwd: Path) -> None:
    settings = _project_settings(project_cwd)
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "context": {
                    "fileName": ["~/.gemini/my-context.md", "/abs/file.md"]
                }
            }
        )
    )
    GeminiAdapter().install(project_cwd)
    data = json.loads(settings.read_text())
    files = data["context"]["fileName"]
    assert files[0] == "~/.gemini/my-context.md"
    assert files[1] == "/abs/file.md"
    assert any(".perfxpert/AGENTS.md" in str(entry) for entry in files)


def test_install_does_not_duplicate_context_filename_on_rerun(
    project_cwd: Path,
) -> None:
    adapter = GeminiAdapter()
    adapter.install(project_cwd)
    adapter.install(project_cwd)
    data = json.loads(_project_settings(project_cwd).read_text())
    files = data["context"]["fileName"]
    perfxpert_entries = [entry for entry in files if ".perfxpert/AGENTS.md" in str(entry)]
    assert len(perfxpert_entries) == 1


def test_install_preserves_existing_mcp_servers(project_cwd: Path) -> None:
    settings = _project_settings(project_cwd)
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps({"mcpServers": {"other": {"command": "other-bin", "args": []}}})
    )
    GeminiAdapter().install(project_cwd)
    data = json.loads(settings.read_text())
    assert data["mcpServers"]["other"]["command"] == "other-bin"
    assert data["mcpServers"]["perfxpert"]["command"] == "perfxpert-mcp"


def test_install_preserves_existing_perfxpert_subkeys(project_cwd: Path) -> None:
    settings = _project_settings(project_cwd)
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "perfxpert": {
                        "command": "perfxpert-mcp",
                        "args": [],
                        "env": {"KEEP": "1"},
                        "timeout": 1234,
                    }
                }
            }
        )
    )
    GeminiAdapter().install(project_cwd)
    data = json.loads(settings.read_text())
    entry = data["mcpServers"]["perfxpert"]
    assert entry["env"] == {"KEEP": "1"}
    assert entry["timeout"] == 1234
    assert entry["command"] == "perfxpert-mcp"


def test_install_refuses_clobber(project_cwd: Path) -> None:
    settings = _project_settings(project_cwd)
    settings.parent.mkdir(parents=True)
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


def test_install_idempotent(project_cwd: Path) -> None:
    adapter = GeminiAdapter()
    r1 = adapter.install(project_cwd)
    r2 = adapter.install(project_cwd)
    assert isinstance(r1, InstallReport)
    assert isinstance(r2, InstallReport)


def test_install_fails_closed_when_gate_hook_disabled(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERFXPERT_GATE_HOOK", "0")

    with pytest.raises(GateHookUnsupported):
        GeminiAdapter().install(project_cwd)

    assert not _project_settings(project_cwd).exists()
    assert not (project_cwd / ".perfxpert" / "AGENTS.md").exists()


def test_install_fails_closed_when_hook_settings_shape_is_invalid(
    project_cwd: Path,
) -> None:
    settings = _project_settings(project_cwd)
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"hooks": []}))

    with pytest.raises(GateHookUnsupported):
        GeminiAdapter().install(project_cwd)

    data = json.loads(settings.read_text())
    assert "mcpServers" not in data
    assert "context" not in data
    assert not (project_cwd / ".perfxpert" / "AGENTS.md").exists()


def test_install_does_not_touch_legacy_user_settings(project_cwd: Path, isolated_home: Path) -> None:
    legacy = isolated_home / ".gemini" / "settings.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "perfxpert": {"command": "perfxpert-mcp", "args": []}
                },
                "context": {
                    "fileName": [str(project_cwd / ".perfxpert" / "AGENTS.md")]
                },
                "allowedTools": ["mcp_perfxpert_*"],
                "tools": {"allowed": ["mcp_perfxpert_*"]},
            }
        ),
        encoding="utf-8",
    )
    before = legacy.read_text()

    GeminiAdapter().install(project_cwd)
    assert legacy.read_text() == before


def test_install_does_not_touch_legacy_user_settings_with_auth_fields(
    project_cwd: Path, isolated_home: Path
) -> None:
    legacy = isolated_home / ".gemini" / "settings.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "perfxpert": {
                        "command": "perfxpert-mcp",
                        "args": [],
                        "timeout": 30000,
                    }
                },
                "context": {
                    "fileName": [str(project_cwd / ".perfxpert" / "AGENTS.md")]
                },
                "tools": {"allowed": ["mcp_perfxpert_*"]},
                "security": {"auth": {"selectedType": "oauth-personal"}},
                "theme": "ansi",
            }
        )
    )
    before = legacy.read_text()

    GeminiAdapter().install(project_cwd)
    assert legacy.read_text() == before


def test_install_does_not_touch_legacy_nested_tools_allowed_for_this_project(
    project_cwd: Path, isolated_home: Path
) -> None:
    legacy = isolated_home / ".gemini" / "settings.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "perfxpert": {"command": "perfxpert-mcp", "args": []}
                },
                "context": {
                    "fileName": [str(project_cwd / ".perfxpert" / "AGENTS.md")]
                },
                "tools": {"allowed": ["mcp_perfxpert_*"]},
            }
        )
    )
    before = legacy.read_text()

    GeminiAdapter().install(project_cwd)
    assert legacy.read_text() == before


def test_install_does_not_touch_other_projects_legacy_user_state(
    project_cwd: Path, isolated_home: Path
) -> None:
    legacy = isolated_home / ".gemini" / "settings.json"
    other_agents = isolated_home / "other" / ".perfxpert" / "AGENTS.md"
    other_agents.parent.mkdir(parents=True)
    other_agents.write_text("other project cache\n")
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "perfxpert": {"command": "perfxpert-mcp", "args": []}
                },
                "context": {"fileName": [str(other_agents)]},
                "allowedTools": ["mcp_perfxpert_*"],
                "tools": {"allowed": ["mcp_perfxpert_*"]},
            },
            indent=2,
        )
        + "\n"
    )
    before = legacy.read_text()

    GeminiAdapter().install(project_cwd)

    assert legacy.read_text() == before


def test_install_does_not_touch_manual_global_perfxpert_entry_without_legacy_markers(
    project_cwd: Path, isolated_home: Path
) -> None:
    legacy = isolated_home / ".gemini" / "settings.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "perfxpert": {
                        "command": "perfxpert-mcp",
                        "args": [],
                        "timeout": 30000,
                    }
                },
                "security": {"auth": {"selectedType": "oauth-personal"}},
            },
            indent=2,
        )
        + "\n"
    )
    before = legacy.read_text()

    GeminiAdapter().install(project_cwd)

    assert legacy.read_text() == before


def test_install_never_touches_gemini_md(project_cwd: Path) -> None:
    gemini_md = project_cwd / "GEMINI.md"
    gemini_md.write_text("user's own content\n")
    snapshot = gemini_md.read_bytes()
    GeminiAdapter().install(project_cwd)
    assert gemini_md.read_bytes() == snapshot


def test_verify_mcp_live_healthy_after_install(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    GeminiAdapter().install(project_cwd)
    os.environ.pop("PERFXPERT_SKIP_LIVE_CHECK", None)

    def _fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["gemini", "mcp", "list"],
            returncode=0,
            stdout="Configured MCP servers:\n✓ perfxpert: perfxpert-mcp (stdio) - Connected\n",
            stderr="",
        )

    monkeypatch.setattr("subprocess.run", _fake_run)
    report = GeminiAdapter().verify_mcp_live(project_cwd)
    assert report.mcp_healthy is True
    assert report.mcp_listed is True
    assert report.gate_hook_installed is True


def test_verify_mcp_live_treats_missing_list_entry_as_advisory(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    GeminiAdapter().install(project_cwd)

    def _fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["gemini", "mcp", "list"],
            returncode=0,
            stdout="Configured MCP servers:\n✓ other: other-mcp (stdio) - Connected\n",
            stderr="",
        )

    monkeypatch.setattr("subprocess.run", _fake_run)
    report = GeminiAdapter().verify_mcp_live(project_cwd)
    assert report.mcp_healthy is True
    assert report.mcp_listed is False
    assert "advisory" in (report.error or "").lower()


def test_install_succeeds_when_mcp_list_probe_is_advisory(
    project_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["gemini", "mcp", "list"],
            returncode=0,
            stdout="Configured MCP servers:\n",
            stderr="",
        )

    monkeypatch.setattr("subprocess.run", _fake_run)

    report = GeminiAdapter().install(project_cwd)

    assert isinstance(report, InstallReport)


def test_verify_mcp_live_rejects_malformed_perfxpert_entry(project_cwd: Path) -> None:
    settings = _project_settings(project_cwd)
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"mcpServers": {"perfxpert": "junk"}}))
    report = GeminiAdapter().verify_mcp_live(project_cwd)
    assert report.mcp_healthy is False
    assert "malformed" in (report.error or "").lower()


def test_verify_mcp_live_rejects_wrong_command_entry(project_cwd: Path) -> None:
    settings = _project_settings(project_cwd)
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps({"mcpServers": {"perfxpert": {"command": "wrong-bin"}}})
    )
    report = GeminiAdapter().verify_mcp_live(project_cwd)
    assert report.mcp_healthy is False
    assert "malformed" in (report.error or "").lower()


def test_verify_mcp_live_returns_error_when_settings_absent(project_cwd: Path) -> None:
    report = GeminiAdapter().verify_mcp_live(project_cwd)
    assert report.mcp_healthy is False
    assert "not present" in (report.error or "")


def test_uninstall_removes_mcp_entry_and_context_filename(project_cwd: Path) -> None:
    settings = _project_settings(project_cwd)
    settings.parent.mkdir(parents=True)
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
    assert data["mcpServers"]["other"]["command"] == "other-bin"
    assert "perfxpert" not in data["mcpServers"]
    assert "/user/other.md" in data["context"]["fileName"]
    assert not any(".perfxpert/AGENTS.md" in str(entry) for entry in data["context"]["fileName"])


def test_uninstall_does_not_touch_legacy_user_settings(project_cwd: Path, isolated_home: Path) -> None:
    legacy = isolated_home / ".gemini" / "settings.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "perfxpert": {"command": "perfxpert-mcp", "args": []}
                },
                "context": {
                    "fileName": [str(project_cwd / ".perfxpert" / "AGENTS.md")]
                },
                "allowedTools": ["mcp_perfxpert_*"],
                "tools": {"allowed": ["mcp_perfxpert_*"]},
            }
        )
    )
    before = legacy.read_text()
    adapter = GeminiAdapter()
    adapter.install(project_cwd)
    adapter.uninstall(project_cwd)
    assert legacy.read_text() == before


def test_uninstall_returns_report(project_cwd: Path) -> None:
    adapter = GeminiAdapter()
    adapter.install(project_cwd)
    report = adapter.uninstall(project_cwd)
    assert isinstance(report, UninstallReport)


def test_spawn_uses_execvpe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, object] = {}

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
