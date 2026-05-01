"""CLI tests for ``perfxpert workflow``."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
from pathlib import Path

import pytest

from perfxpert import __main__ as main_mod
from perfxpert.cli._tui_session import (
    bind_tui_session_env,
    cleanup_tui_session_env,
)
from perfxpert.integrations.external_workflow import ExternalWorkflowRuntimeError


@pytest.fixture
def tui_session(monkeypatch) -> None:
    monkeypatch.setattr("perfxpert.cli.workflow_cmd._in_perfxpert_tui_session", lambda: True)
    monkeypatch.setattr(
        "perfxpert.integrations.external_workflow._has_active_tui_session",
        lambda: True,
    )


def _unix_socket_available(tmp_path: Path) -> bool:
    socket_path = tmp_path / "probe.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(socket_path))
    except OSError:
        return False
    finally:
        listener.close()
        try:
            socket_path.unlink()
        except OSError:
            pass
    return True


def test_workflow_import_help_hides_internal_controls(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main_mod.main(["workflow", "import", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--interactive" not in out
    assert "--allow-network" not in out
    assert "--cache-root" not in out
    assert "--json" not in out
    assert "--no-persist" not in out


def test_workflow_import_refuses_without_interactive(tmp_path, capsys, tui_session) -> None:
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
    assert "inside the TUI" in captured.err


def test_workflow_import_rejects_agent_session_without_tui_marker(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("PERFXPERT_IN_AGENT_SESSION", "codex")
    monkeypatch.delenv("PERFXPERT_TUI_INTERACTIVE", raising=False)

    with pytest.raises(SystemExit) as exc:
        main_mod.main(["workflow", "import", str(tmp_path), "--interactive"])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "inside the TUI" in captured.err


def test_workflow_import_rejects_forged_tui_marker_without_token(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("PERFXPERT_TUI_INTERACTIVE", "1")
    monkeypatch.delenv("PERFXPERT_TUI_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("PERFXPERT_TUI_SESSION_SOCKET", raising=False)
    monkeypatch.delenv("PERFXPERT_TUI_SESSION_TOKEN_FILE", raising=False)

    with pytest.raises(SystemExit) as exc:
        main_mod.main(["workflow", "import", str(tmp_path), "--interactive"])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "inside the TUI" in captured.err


def test_workflow_import_rejects_forged_token_without_launcher_ancestor(tmp_path, capsys, monkeypatch) -> None:
    if not _unix_socket_available(tmp_path):
        pytest.skip("Unix socket bind is not available in this test sandbox")
    env: dict[str, str] = {}
    bind_tui_session_env(env)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    try:
        with pytest.raises(SystemExit) as exc:
            main_mod.main(["workflow", "import", str(tmp_path), "--interactive"])

        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "inside the TUI" in captured.err
    finally:
        cleanup_tui_session_env(env)


def test_workflow_import_rejects_socket_that_only_replies_ok(tmp_path, capsys, monkeypatch) -> None:
    if not _unix_socket_available(tmp_path):
        pytest.skip("Unix socket bind is not available in this test sandbox")
    socket_path = tmp_path / "fake.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    socket_path.chmod(0o600)
    listener.listen(1)
    stop = threading.Event()

    def fake_server() -> None:
        listener.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                try:
                    conn.recv(512)
                    conn.sendall(b"ok")
                except OSError:
                    pass

    thread = threading.Thread(target=fake_server, daemon=True)
    thread.start()
    monkeypatch.setenv("PERFXPERT_TUI_INTERACTIVE", "1")
    monkeypatch.setenv("PERFXPERT_TUI_SESSION_TOKEN", "forged")
    monkeypatch.setenv("PERFXPERT_TUI_SESSION_SOCKET", str(socket_path))
    try:
        with pytest.raises(SystemExit) as exc:
            main_mod.main(["workflow", "import", str(tmp_path), "--interactive"])
    finally:
        stop.set()
        listener.close()
        thread.join(timeout=1.0)

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "inside the TUI" in captured.err


def test_workflow_import_rejects_path_shadowed_parent_socket(tmp_path) -> None:
    if not _unix_socket_available(tmp_path):
        pytest.skip("Unix socket bind is not available in this test sandbox")
    source = tmp_path / "external"
    source.mkdir()
    (source / "README.md").write_text("The adapter can use rocprof-compute counters.\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_parent = fake_bin / "perfxpert-code"
    socket_path = tmp_path / "fake-parent.sock"
    package_root = Path(__file__).resolve().parents[2]
    fake_parent.write_text(
        f"""#!/usr/bin/env python3
import os
import socket
import subprocess
import sys
import threading

# Include the expected entrypoint strings to prove PATH shadowing alone is not
# enough to satisfy the launcher identity check.
_entrypoint_reference = "perfxpert.cli.opencode_launcher:main"

listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
listener.bind({str(socket_path)!r})
os.chmod({str(socket_path)!r}, 0o600)
listener.listen(1)
stop = threading.Event()

def server():
    listener.settimeout(0.2)
    while not stop.is_set():
        try:
            conn, _ = listener.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        with conn:
            try:
                conn.recv(512)
                conn.sendall(b"ok")
            except OSError:
                pass

thread = threading.Thread(target=server, daemon=True)
thread.start()
env = os.environ.copy()
env["PYTHONPATH"] = {str(package_root)!r}
env["PERFXPERT_TUI_INTERACTIVE"] = "1"
env["PERFXPERT_TUI_SESSION_TOKEN"] = "forged"
env["PERFXPERT_TUI_SESSION_SOCKET"] = {str(socket_path)!r}
try:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "perfxpert",
            "workflow",
            "import",
            sys.argv[1],
            "--interactive",
            "--no-persist",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
finally:
    stop.set()
    listener.close()
    thread.join(timeout=1.0)
print(proc.stdout, end="")
print(proc.stderr, end="", file=sys.stderr)
raise SystemExit(proc.returncode)
""",
        encoding="utf-8",
    )
    fake_parent.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(fake_bin), env.get("PATH", "")])

    proc = subprocess.run(
        [str(fake_parent), str(source)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "inside the TUI" in proc.stderr


def test_workflow_import_accepts_real_launcher_descendant_tui_session(tmp_path) -> None:
    if not _unix_socket_available(tmp_path):
        pytest.skip("Unix socket bind is not available in this test sandbox")
    source = tmp_path / "external"
    source.mkdir()
    (source / "README.md").write_text("The adapter can use rocprof-compute counters.\n", encoding="utf-8")
    launcher = tmp_path / "perfxpert-code"
    launcher.write_text(
        """#!/usr/bin/env python3
from perfxpert.cli.opencode_launcher import main

raise SystemExit(main())
""",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    fake_opencode = tmp_path / "fake-opencode"
    fake_config = tmp_path / "fake-config"
    fake_config.mkdir()
    for name in ("opencode.json", "AGENTS.md", "mcp.json", "amd-theme.json"):
        (fake_config / name).write_text("{}\n", encoding="utf-8")
    fake_opencode.write_text(
        """#!/usr/bin/env python3
import os
import subprocess
import sys

proc = subprocess.run(
    [
        sys.executable,
        "-m",
        "perfxpert",
        "workflow",
        "import",
        os.environ["PERFXPERT_TEST_SOURCE"],
        "--interactive",
        "--no-persist",
        "--json",
    ],
    env=os.environ.copy(),
    text=True,
    capture_output=True,
    check=False,
)
print(proc.stdout, end="")
print(proc.stderr, end="", file=sys.stderr)
raise SystemExit(proc.returncode)
""",
        encoding="utf-8",
    )
    fake_opencode.chmod(0o755)

    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "",
                "from perfxpert.cli import _tui_session",
                "from perfxpert.cli import opencode_launcher",
                "",
                'opencode_launcher.resolve_opencode_binary = lambda: Path(os.environ["PERFXPERT_TEST_FAKE_OPENCODE"])',
                'opencode_launcher.resolve_config_dir = lambda: Path(os.environ["PERFXPERT_TEST_CONFIG_DIR"])',
                '_tui_session._trusted_launcher_paths = lambda: (Path(os.environ["PERFXPERT_TEST_TRUSTED_LAUNCHER"]),)',
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    package_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        os.pathsep.join([str(tmp_path), str(package_root)])
        if not existing_pythonpath
        else os.pathsep.join([str(tmp_path), str(package_root), existing_pythonpath])
    )
    env["PERFXPERT_CODE_NO_BANNER"] = "1"
    env["PERFXPERT_TEST_FAKE_OPENCODE"] = str(fake_opencode)
    env["PERFXPERT_TEST_CONFIG_DIR"] = str(fake_config)
    env["PERFXPERT_TEST_SOURCE"] = str(source)
    env["PERFXPERT_TEST_TRUSTED_LAUNCHER"] = str(launcher)

    proc = subprocess.run(
        [str(launcher)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["interactive_only"] is True
    assert plan["execution_allowed"] is False


def test_workflow_import_json_for_local_adapter(tmp_path, capsys, tui_session) -> None:
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


def test_workflow_import_summary_for_local_adapter(tmp_path, capsys, tui_session) -> None:
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


def test_workflow_import_url_without_network_consent_is_usage_error(capsys, tui_session) -> None:
    with pytest.raises(SystemExit) as exc:
        main_mod.main(
            [
                "workflow",
                "import",
                "https://github.com/example/perf-workflow",
                "--interactive",
            ]
        )

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "explicit network consent" in captured.err
    assert "--allow-network" not in captured.err


def test_workflow_import_url_with_network_consent_forwards_to_inspector(
    tmp_path, capsys, monkeypatch, tui_session
) -> None:
    captured = {}

    def fake_import(source, **kwargs):
        captured["source"] = source
        captured.update(kwargs)
        return {
            "adapter_id": "adapter-123",
            "capabilities": [],
            "mcp_servers": [],
            "knowledge_links": [],
            "manifest_path": None,
        }

    monkeypatch.setattr("perfxpert.cli.workflow_cmd.inspect_external_workflow", fake_import)

    with pytest.raises(SystemExit) as exc:
        main_mod.main(
            [
                "workflow",
                "import",
                "https://github.com/example/perf-workflow",
                "--interactive",
                "--allow-network",
                "--cache-root",
                str(tmp_path / "cache"),
            ]
        )

    assert exc.value.code == 0
    assert captured["source"] == "https://github.com/example/perf-workflow"
    assert captured["allow_network"] is True
    assert captured["interactive"] is True


def test_workflow_import_runtime_error_returns_one(tmp_path, capsys, monkeypatch, tui_session) -> None:

    def fail_import(*args, **kwargs):
        raise ExternalWorkflowRuntimeError("clone failed")

    monkeypatch.setattr("perfxpert.cli.workflow_cmd.inspect_external_workflow", fail_import)

    with pytest.raises(SystemExit) as exc:
        main_mod.main(["workflow", "import", str(tmp_path), "--interactive"])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "clone failed" in captured.err
