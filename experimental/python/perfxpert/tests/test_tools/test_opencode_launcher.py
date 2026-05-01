"""Unit tests for perfxpert.cli.opencode_launcher."""

from importlib.metadata import version
from pathlib import Path
from unittest import mock

import pytest

from perfxpert.cli._tui_session import TUI_SESSION_SOCKET_ENV, TUI_SESSION_TOKEN_ENV
from perfxpert.cli import opencode_launcher


@pytest.fixture(autouse=True)
def _disable_repo_local_patched_binary(monkeypatch):
    monkeypatch.setattr(
        opencode_launcher,
        "_repo_local_patched_opencode_paths",
        lambda: [],
    )


@pytest.fixture(autouse=True)
def _stub_tui_session_socket_binding(monkeypatch, tmp_path):
    def fake_bind(env):
        env["PERFXPERT_TUI_INTERACTIVE"] = "1"
        env[TUI_SESSION_TOKEN_ENV] = "test-token"
        env[TUI_SESSION_SOCKET_ENV] = str(tmp_path / "auth.sock")
        return True

    monkeypatch.setattr(opencode_launcher, "_bind_tui_session_env", fake_bind)
    monkeypatch.setattr(opencode_launcher, "_cleanup_tui_session_env", lambda env: None)


def test_version_flag_short_circuit(capsys):
    rc = opencode_launcher.main(["--version"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "AMD" in captured.out
    assert version("perfxpert") in captured.out


def test_v_short_flag(capsys):
    rc = opencode_launcher.main(["-V"])
    assert rc == 0


def test_perfxpert_code_help_hides_workflow(capsys, monkeypatch):
    def fail_resolve():
        raise FileNotFoundError("missing bundled opencode")

    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", fail_resolve)

    rc = opencode_launcher.main(["--help"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "perfxpert-owned subcommands" in out
    assert "workflow" not in out


def test_route_workflow_import_to_perfxpert_dispatch():
    kind, argv = opencode_launcher.route_subcommand(["workflow", "import", "./tool", "--interactive"])

    assert kind == "perfxpert"
    assert argv == ["workflow", "import", "./tool", "--interactive"]


def test_resolve_config_dir_returns_bundled_path():
    p = opencode_launcher.resolve_config_dir()
    assert p.exists()
    assert (p / "opencode.json").exists()
    assert (p / "amd-theme.json").exists()
    assert (p / "AGENTS.md").exists()
    assert (p / "mcp.json").exists()


def test_resolve_user_binary_uses_override(tmp_path: Path, monkeypatch):
    fake_bin = tmp_path / "fake-opencode"
    fake_bin.write_text("#!/bin/sh\necho fake\n")
    fake_bin.chmod(0o755)
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", str(fake_bin))
    assert opencode_launcher.resolve_user_opencode_binary() == fake_bin


def test_resolve_binary_prefers_repo_local_patched_build(tmp_path: Path, monkeypatch):
    local_bin = tmp_path / "repo-opencode"
    local_bin.write_text("#!/bin/sh\necho repo\n")
    local_bin.chmod(0o755)
    bundled_bin = tmp_path / "bundled-opencode"
    bundled_bin.write_text("#!/bin/sh\necho bundled\n")
    bundled_bin.chmod(0o755)

    monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
    monkeypatch.setattr(
        opencode_launcher,
        "_repo_local_patched_opencode_paths",
        lambda: [local_bin],
    )

    import contextlib

    @contextlib.contextmanager
    def _fake_as_file(_):
        yield bundled_bin

    monkeypatch.setattr(opencode_launcher.resources, "as_file", _fake_as_file)

    assert opencode_launcher.resolve_opencode_binary() == local_bin


def test_resolve_user_binary_raises_when_override_missing(tmp_path: Path, monkeypatch):
    """The explicit upstream-opencode escape hatch validates its override."""
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", str(tmp_path / "nonexistent"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        opencode_launcher.resolve_user_opencode_binary()


def test_resolve_user_binary_raises_when_override_not_executable(tmp_path: Path, monkeypatch):
    """The explicit upstream-opencode escape hatch validates execute bit."""
    fake_bin = tmp_path / "fake-opencode"
    fake_bin.write_text("#!/bin/sh\necho fake\n")
    fake_bin.chmod(0o644)
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", str(fake_bin))
    with pytest.raises(FileNotFoundError, match="not executable"):
        opencode_launcher.resolve_user_opencode_binary()


def test_prepare_runtime_config_dir_skips_subdirectories(tmp_path: Path):
    """_prepare_runtime_config_dir must not crash on subdirectories in src_config_dir."""
    src = tmp_path / "src_config"
    src.mkdir()
    (src / "opencode.json").write_text('{"config": true}')
    # Add a subdirectory — should be silently skipped
    subdir = src / "subdir"
    subdir.mkdir()
    (subdir / "nested.json").write_text("{}")

    from perfxpert.cli.opencode_launcher import _prepare_runtime_config_dir

    # Should not raise; only files are copied
    out = _prepare_runtime_config_dir(src)
    assert (out / "opencode.json").exists()
    # subdirectory itself must NOT be copied as a file
    assert not (out / "subdir").is_file()


def test_banner_is_printed_to_stderr(monkeypatch):
    # Stub subprocess to avoid actually launching opencode
    monkeypatch.setattr(
        opencode_launcher.subprocess,
        "run",
        mock.MagicMock(return_value=mock.MagicMock(returncode=0)),
    )
    monkeypatch.setattr(
        opencode_launcher,
        "resolve_opencode_binary",
        lambda: Path("/bin/true"),
    )
    monkeypatch.delenv("PERFXPERT_CODE_NO_BANNER", raising=False)
    # Track that print_banner is called
    banner_called = []

    original_print_banner = opencode_launcher.print_banner

    def track_banner(stream=None):
        import sys

        if stream is None:
            stream = sys.stderr
        banner_called.append(True)
        original_print_banner(stream)

    monkeypatch.setattr(opencode_launcher, "print_banner", track_banner)
    opencode_launcher.main([])
    assert len(banner_called) > 0


def test_banner_suppressed_by_env(monkeypatch, capsys):
    monkeypatch.setattr(
        opencode_launcher.subprocess,
        "run",
        mock.MagicMock(return_value=mock.MagicMock(returncode=0)),
    )
    monkeypatch.setattr(
        opencode_launcher,
        "resolve_opencode_binary",
        lambda: Path("/bin/true"),
    )
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")
    opencode_launcher.main([])
    captured = capsys.readouterr()
    assert "AMD ROCm PerfXpert" not in captured.err


def test_recursion_guard_env_set(monkeypatch):
    captured_env = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(
        opencode_launcher,
        "resolve_opencode_binary",
        lambda: Path("/bin/true"),
    )
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")
    opencode_launcher.main([])
    assert captured_env.get("PERFXPERT_IN_OPENCODE_SESSION") == "1"


def test_default_tui_sets_interactive_marker(monkeypatch):
    captured_env = {}
    cwd = Path.cwd()

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    opencode_launcher.main([])

    assert captured_env.get("PERFXPERT_TUI_INTERACTIVE") == "1"
    assert captured_env.get(TUI_SESSION_TOKEN_ENV)
    assert not Path(captured_env[TUI_SESSION_SOCKET_ENV]).exists()
    assert captured_env.get("PERFXPERT_WORKLOAD_CWD") == str(cwd)


def test_tui_subcommand_sets_interactive_marker(monkeypatch):
    captured_env = {}
    captured_cmd = []

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    opencode_launcher.main(["tui"])

    assert captured_env.get("PERFXPERT_TUI_INTERACTIVE") == "1"
    assert captured_env.get(TUI_SESSION_TOKEN_ENV)
    assert not Path(captured_env[TUI_SESSION_SOCKET_ENV]).exists()
    assert captured_cmd[1:] == []


def test_interactive_launch_fails_when_tui_authority_cannot_bind(monkeypatch, capsys):
    run_called = False

    def fake_run(cmd, **kwargs):
        nonlocal run_called
        run_called = True
        return mock.MagicMock(returncode=0)

    def fail_bind(env):
        raise RuntimeError("Linux peer-credential sockets unavailable")

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setattr(opencode_launcher, "_bind_tui_session_env", fail_bind)
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    rc = opencode_launcher.main(["tui"])

    assert rc == 1
    assert run_called is False
    assert "cannot start TUI workflow import authority" in capsys.readouterr().err


def test_option_value_tui_does_not_select_tui_subcommand(monkeypatch):
    captured_env = {}
    captured_cmd = []

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    opencode_launcher.main(["--model", "tui", "run", "optimize ./app"])

    assert "PERFXPERT_TUI_INTERACTIVE" not in captured_env
    assert TUI_SESSION_TOKEN_ENV not in captured_env
    assert captured_cmd[1:] == ["--model", "tui", "run", "--agent", "perfxpert", "optimize ./app"]


def test_value_flags_do_not_route_or_inject_from_their_values(monkeypatch):
    captured_env = {}
    captured_cmd = []

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    opencode_launcher.main(
        [
            "--prompt",
            "doctor",
            "-m",
            "tui",
            "--command",
            "claude",
            "--cors",
            "http://localhost:3000",
            "--mdns-domain",
            "tui.local",
            "--password",
            "run",
            "-p",
            "tui",
            "run",
            "optimize ./app",
        ]
    )

    assert "PERFXPERT_TUI_INTERACTIVE" not in captured_env
    assert captured_cmd[1:] == [
        "--prompt",
        "doctor",
        "-m",
        "tui",
        "--command",
        "claude",
        "--cors",
        "http://localhost:3000",
        "--mdns-domain",
        "tui.local",
        "--password",
        "run",
        "-p",
        "tui",
        "run",
        "--agent",
        "perfxpert",
        "optimize ./app",
    ]


def test_format_flag_before_run_still_routes_and_injects_agent(monkeypatch):
    captured_env = {}
    captured_cmd = []

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    rc = opencode_launcher.main(["--format", "json", "run", "optimize ./app"])

    assert rc == 0
    assert "PERFXPERT_TUI_INTERACTIVE" not in captured_env
    assert captured_cmd[1:] == ["--format", "json", "run", "--agent", "perfxpert", "optimize ./app"]


def test_bare_help_passthrough_does_not_require_tui_authority(monkeypatch):
    captured_env = {}
    captured_cmd = []
    bind_called = False

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    def fail_bind(env):
        nonlocal bind_called
        bind_called = True
        raise RuntimeError("Linux peer-credential sockets unavailable")

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setattr(opencode_launcher, "_bind_tui_session_env", fail_bind)
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    rc = opencode_launcher.main(["--help"])

    assert rc == 0
    assert bind_called is False
    assert captured_cmd[1:] == ["--help"]
    assert "PERFXPERT_TUI_INTERACTIVE" not in captured_env


def test_tui_help_passthrough_does_not_require_tui_authority(monkeypatch):
    captured_env = {}
    captured_cmd = []
    bind_called = False

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    def fail_bind(env):
        nonlocal bind_called
        bind_called = True
        raise RuntimeError("Linux peer-credential sockets unavailable")

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setattr(opencode_launcher, "_bind_tui_session_env", fail_bind)
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    rc = opencode_launcher.main(["tui", "--help"])

    assert rc == 0
    assert bind_called is False
    assert captured_cmd[1:] == ["--help"]
    assert "PERFXPERT_TUI_INTERACTIVE" not in captured_env


def test_help_shaped_option_value_does_not_trigger_wrapper_help(monkeypatch, capsys):
    captured_cmd = []
    captured_env = {}

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    rc = opencode_launcher.main(["--model", "--help", "run", "optimize ./app"])

    assert rc == 0
    assert "AMD ROCm PerfXpert" not in capsys.readouterr().out
    assert "PERFXPERT_TUI_INTERACTIVE" not in captured_env
    assert captured_cmd[1:] == ["--model", "--help", "run", "--agent", "perfxpert", "optimize ./app"]


def test_agent_shaped_option_value_does_not_suppress_run_agent_injection(monkeypatch):
    captured_cmd = []

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    rc = opencode_launcher.main(["--model", "--agent", "run", "optimize ./app"])

    assert rc == 0
    assert captured_cmd[1:] == ["--model", "--agent", "run", "--agent", "perfxpert", "optimize ./app"]


def test_run_help_is_forwarded_without_agent_injection(monkeypatch):
    captured_cmd = []

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    rc = opencode_launcher.main(["run", "--help"])

    assert rc == 0
    assert captured_cmd[1:] == ["run", "--help"]


@pytest.mark.parametrize("subcommand", ["completion", "upgrade"])
def test_opencode_maintenance_subcommands_do_not_bind_tui_authority(subcommand, monkeypatch):
    captured_cmd = []
    bind_called = False

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        return mock.MagicMock(returncode=0)

    def fail_bind(env):
        nonlocal bind_called
        bind_called = True
        raise RuntimeError("Linux peer-credential sockets unavailable")

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setattr(opencode_launcher, "_bind_tui_session_env", fail_bind)
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    rc = opencode_launcher.main([subcommand])

    assert rc == 0
    assert bind_called is False
    assert captured_cmd[1:] == [subcommand]


def test_upgrade_method_flag_before_subcommand_does_not_bind_tui_authority(monkeypatch):
    captured_cmd = []
    bind_called = False

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        return mock.MagicMock(returncode=0)

    def fail_bind(env):
        nonlocal bind_called
        bind_called = True
        raise RuntimeError("Linux peer-credential sockets unavailable")

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setattr(opencode_launcher, "_bind_tui_session_env", fail_bind)
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    rc = opencode_launcher.main(["--method", "npm", "upgrade"])

    assert rc == 0
    assert bind_called is False
    assert captured_cmd[1:] == ["--method", "npm", "upgrade"]


def test_tui_subcommand_workload_arg_sets_workload_cwd(tmp_path: Path, monkeypatch):
    captured_env = {}
    captured_cmd = []

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    opencode_launcher.main(["tui", str(tmp_path)])

    assert captured_env.get("PERFXPERT_TUI_INTERACTIVE") == "1"
    assert captured_env.get("PERFXPERT_WORKLOAD_CWD") == str(tmp_path)
    assert captured_cmd[1:] == [str(tmp_path)]


def test_dir_option_sets_workload_cwd(tmp_path: Path, monkeypatch):
    captured_env = {}
    captured_cmd = []

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    opencode_launcher.main(["--dir", str(tmp_path), "tui"])

    assert captured_env.get("PERFXPERT_TUI_INTERACTIVE") == "1"
    assert captured_env.get("PERFXPERT_WORKLOAD_CWD") == str(tmp_path)
    assert captured_cmd[1:] == [str(tmp_path)]


def test_dir_shaped_option_value_does_not_set_workload_cwd(tmp_path: Path, monkeypatch):
    captured_env = {}
    cwd = Path.cwd()

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    opencode_launcher.main(["--model", "--dir", "run", "optimize ./app"])

    assert captured_env.get("PERFXPERT_WORKLOAD_CWD") == str(cwd)


def test_opencode_default_cwd_override_sets_workload_cwd(tmp_path: Path, monkeypatch):
    captured_env = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    opencode_launcher.main([str(tmp_path)])

    assert captured_env.get("PERFXPERT_WORKLOAD_CWD") == str(tmp_path)


def test_noninteractive_run_does_not_set_interactive_marker(monkeypatch):
    captured_env = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", lambda: Path("/bin/true"))
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")
    monkeypatch.setenv("PERFXPERT_TUI_INTERACTIVE", "1")
    monkeypatch.setenv("PERFXPERT_TUI_SESSION_TOKEN", "stale")
    monkeypatch.setenv("PERFXPERT_TUI_SESSION_SOCKET", "/tmp/stale")
    monkeypatch.setenv("PERFXPERT_TUI_SESSION_TOKEN_FILE", "/tmp/old-stale")

    opencode_launcher.main(["run", "optimize ./app"])

    assert captured_env.get("PERFXPERT_IN_OPENCODE_SESSION") == "1"
    assert "PERFXPERT_TUI_INTERACTIVE" not in captured_env
    assert TUI_SESSION_TOKEN_ENV not in captured_env
    assert TUI_SESSION_SOCKET_ENV not in captured_env
    assert "PERFXPERT_TUI_SESSION_TOKEN_FILE" not in captured_env


def test_user_opencode_path_scrubs_stale_tui_env(tmp_path: Path, monkeypatch):
    captured_env = {}
    fake_bin = tmp_path / "opencode"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return mock.MagicMock(returncode=0)

    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", str(fake_bin))
    monkeypatch.setenv("PERFXPERT_TUI_INTERACTIVE", "1")
    monkeypatch.setenv("PERFXPERT_TUI_SESSION_TOKEN", "stale")
    monkeypatch.setenv("PERFXPERT_TUI_SESSION_SOCKET", "/tmp/stale")
    monkeypatch.setenv("PERFXPERT_TUI_SESSION_TOKEN_FILE", "/tmp/old-stale")
    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)

    rc = opencode_launcher.main(["opencode", "run", "hello"])

    assert rc == 0
    assert "PERFXPERT_TUI_INTERACTIVE" not in captured_env
    assert TUI_SESSION_TOKEN_ENV not in captured_env
    assert TUI_SESSION_SOCKET_ENV not in captured_env
    assert "PERFXPERT_TUI_SESSION_TOKEN_FILE" not in captured_env


def test_workflow_subcommand_dispatches_without_resolving_opencode(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return mock.MagicMock(returncode=0)

    def fail_resolve():
        raise AssertionError("workflow dispatch should not resolve opencode")

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(opencode_launcher, "resolve_opencode_binary", fail_resolve)

    rc = opencode_launcher.main(["workflow", "import", "./adapter", "--interactive"])

    assert rc == 0
    assert calls
    assert calls[0][1:4] == ["-m", "perfxpert", "workflow"]


def test_amd_red_in_banner(monkeypatch):
    """Banner includes AMD red ANSI color code."""
    # Verify the function itself contains the AMD red color code
    import inspect

    source = inspect.getsource(opencode_launcher.print_banner)
    # The source will have the escaped form \\033 when inspected
    assert "38;5;196m" in source  # AMD red color code in the function


# -- Fix 4: doctor autodiscovery of well-known opencode paths ---------------


def test_resolve_user_binary_autodiscovers_home_opencode_bin(tmp_path: Path, monkeypatch):
    """`~/.opencode/bin/opencode` is the upstream installer's default;
    the explicit `perfxpert-code opencode` path must find it without PATH munging."""
    fake_home = tmp_path
    fake_bin_dir = fake_home / ".opencode" / "bin"
    fake_bin_dir.mkdir(parents=True)
    fake_bin = fake_bin_dir / "opencode"
    fake_bin.write_text("#!/bin/sh\necho fake\n")
    fake_bin.chmod(0o755)

    # Isolate: no override, no bundled binary, no PATH hit.
    monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(opencode_launcher.shutil, "which", lambda _: None)

    # Force the bundled-resource branch to miss (no bundled binary in the test wheel).
    import contextlib

    @contextlib.contextmanager
    def _fake_as_file(_):
        yield tmp_path / "no_such_bundled_path"

    monkeypatch.setattr(opencode_launcher.resources, "as_file", _fake_as_file)

    resolved = opencode_launcher.resolve_user_opencode_binary()
    assert resolved == fake_bin


@pytest.mark.parametrize(
    "subpath",
    [
        ".opencode/bin/opencode",
        ".local/bin/opencode",
    ],
)
def test_resolve_user_binary_autodiscovers_multiple_wellknown_paths(tmp_path: Path, monkeypatch, subpath):
    """Each well-known upstream location must be auto-discovered for the escape hatch."""
    fake_home = tmp_path
    fake_bin = fake_home / subpath
    fake_bin.parent.mkdir(parents=True, exist_ok=True)
    fake_bin.write_text("#!/bin/sh\necho fake\n")
    fake_bin.chmod(0o755)

    monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(opencode_launcher.shutil, "which", lambda _: None)

    import contextlib

    @contextlib.contextmanager
    def _fake_as_file(_):
        yield tmp_path / "no_such_bundled_path"

    monkeypatch.setattr(opencode_launcher.resources, "as_file", _fake_as_file)

    assert opencode_launcher.resolve_user_opencode_binary() == fake_bin


def test_resolve_default_binary_missing_suggests_wrapper(monkeypatch, tmp_path: Path):
    """The default path must point users back to the bundled submodule build."""
    monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
    # Ensure no well-known path resolves
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(opencode_launcher.shutil, "which", lambda _: None)

    import contextlib

    @contextlib.contextmanager
    def _fake_as_file(_):
        yield tmp_path / "no_such_bundled_path"

    monkeypatch.setattr(opencode_launcher.resources, "as_file", _fake_as_file)

    with pytest.raises(FileNotFoundError) as exc:
        opencode_launcher.resolve_opencode_binary()
    msg = str(exc.value)
    assert "bundled patched opencode binary not found" in msg
    assert "pip-install-from-git.sh" in msg
    assert "perfxpert-code opencode" in msg


def test_resolve_user_binary_missing_suggests_upstream_install(monkeypatch, tmp_path: Path):
    """The explicit upstream-opencode escape hatch gives upstream install help."""
    monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(opencode_launcher.shutil, "which", lambda _: None)

    with pytest.raises(FileNotFoundError) as exc:
        opencode_launcher.resolve_user_opencode_binary()
    assert "opencode.ai/install.sh" in str(exc.value)


def test_wellknown_paths_list_includes_home_opencode():
    """Sanity: the well-known paths helper lists `~/.opencode/bin/opencode`."""
    paths = opencode_launcher._wellknown_opencode_paths()
    # The upstream installer's default must be listed.
    home_opencode = Path.home() / ".opencode" / "bin" / "opencode"
    assert home_opencode in paths
