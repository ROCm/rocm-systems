"""CLI tests for ``perfxpert workflow``."""

from __future__ import annotations

import json

import pytest

from perfxpert import __main__ as main_mod
from perfxpert.integrations.external_workflow import ExternalWorkflowRuntimeError


def test_workflow_import_refuses_without_interactive(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("PERFXPERT_TUI_INTERACTIVE", "1")

    with pytest.raises(SystemExit) as exc:
        main_mod.main(["workflow", "import", str(tmp_path)])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "TUI-interactive only" in captured.err


def test_workflow_import_refuses_outside_tui_session(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    monkeypatch.delenv("PERFXPERT_IN_AGENT_SESSION", raising=False)
    monkeypatch.delenv("PERFXPERT_TUI_INTERACTIVE", raising=False)

    with pytest.raises(SystemExit) as exc:
        main_mod.main(["workflow", "import", str(tmp_path), "--interactive"])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "inside perfxpert-code" in captured.err


def test_workflow_import_rejects_agent_session_without_tui_marker(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("PERFXPERT_IN_AGENT_SESSION", "codex")
    monkeypatch.delenv("PERFXPERT_TUI_INTERACTIVE", raising=False)

    with pytest.raises(SystemExit) as exc:
        main_mod.main(["workflow", "import", str(tmp_path), "--interactive"])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "inside perfxpert-code" in captured.err


def test_workflow_import_json_for_local_adapter(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("PERFXPERT_TUI_INTERACTIVE", "1")
    source = tmp_path / "external"
    source.mkdir()
    (source / "README.md").write_text(
        "The profiler uses rocprof-compute counters and maps stalls to source line.\n",
        encoding="utf-8",
    )
    (source / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"adapter": {"command": "adapter-mcp", "args": ["serve"]}}}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        main_mod.main(
            [
                "workflow",
                "import",
                str(source),
                "--interactive",
                "--cache-root",
                str(tmp_path / "cache"),
                "--json",
            ]
        )

    assert exc.value.code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["interactive_only"] is True
    assert plan["execution_allowed"] is False
    assert plan["mcp_servers"][0]["name"] == "adapter"
    assert any(cap["kind"] == "profiling" for cap in plan["capabilities"])
    assert plan["manifest_path"]


def test_workflow_import_summary_for_local_adapter(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("PERFXPERT_TUI_INTERACTIVE", "1")
    source = tmp_path / "external"
    source.mkdir()
    (source / "README.md").write_text("The adapter can replay kernel launches.\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main_mod.main(
            [
                "workflow",
                "import",
                str(source),
                "--interactive",
                "--cache-root",
                str(tmp_path / "cache"),
            ]
        )

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Imported external workflow adapter:" in out
    assert "Capabilities:" in out
    assert "Activation: advisory TUI context only" in out


def test_workflow_import_runtime_error_returns_one(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("PERFXPERT_TUI_INTERACTIVE", "1")

    def fail_import(*args, **kwargs):
        raise ExternalWorkflowRuntimeError("clone failed")

    monkeypatch.setattr("perfxpert.cli.workflow_cmd.inspect_external_workflow", fail_import)

    with pytest.raises(SystemExit) as exc:
        main_mod.main(["workflow", "import", str(tmp_path), "--interactive"])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "clone failed" in captured.err
