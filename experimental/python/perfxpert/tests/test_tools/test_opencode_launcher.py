"""Unit tests for perfxpert.cli.opencode_launcher."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from perfxpert.cli import opencode_launcher


def test_version_flag_short_circuit(capsys):
    rc = opencode_launcher.main(["--version"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "AMD" in captured.out


def test_v_short_flag(capsys):
    rc = opencode_launcher.main(["-V"])
    assert rc == 0


def test_resolve_config_dir_returns_bundled_path():
    p = opencode_launcher.resolve_config_dir()
    assert p.exists()
    assert (p / "opencode.json").exists()
    assert (p / "amd-theme.json").exists()
    assert (p / "AGENTS.md").exists()
    assert (p / "mcp.json").exists()


def test_resolve_binary_uses_override(tmp_path: Path, monkeypatch):
    fake_bin = tmp_path / "fake-opencode"
    fake_bin.write_text("#!/bin/sh\necho fake\n")
    fake_bin.chmod(0o755)
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", str(fake_bin))
    assert opencode_launcher.resolve_opencode_binary() == fake_bin


def test_resolve_binary_falls_back_when_override_missing(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", str(tmp_path / "nonexistent"))
    # Should print a warning and try bundled / PATH. If neither present, raises.
    try:
        opencode_launcher.resolve_opencode_binary()
    except FileNotFoundError:
        pass  # acceptable when no bundled/PATH opencode
    captured = capsys.readouterr()
    assert "WARNING" in captured.err or "not found" in captured.err.lower()


def test_resolve_binary_falls_back_when_override_not_executable(tmp_path: Path, monkeypatch, capsys):
    fake_bin = tmp_path / "fake-opencode"
    fake_bin.write_text("echo nope\n")
    fake_bin.chmod(0o644)
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", str(fake_bin))
    monkeypatch.setattr(opencode_launcher.resources, "files", lambda *_: (_ for _ in ()).throw(ModuleNotFoundError))
    monkeypatch.setattr(opencode_launcher.shutil, "which", lambda *_args, **_kwargs: None)

    with pytest.raises(FileNotFoundError):
        opencode_launcher.resolve_opencode_binary()

    captured = capsys.readouterr()
    assert "not executable" in captured.err.lower()


def test_banner_is_printed_to_stderr(tmp_path: Path, monkeypatch):
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
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setattr(
        opencode_launcher,
        "_runtime_cache_root",
        lambda: cache_root,
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


def test_banner_suppressed_by_env(tmp_path: Path, monkeypatch, capsys):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
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
    monkeypatch.setattr(
        opencode_launcher,
        "_runtime_cache_root",
        lambda: cache_root,
    )
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")
    opencode_launcher.main([])
    captured = capsys.readouterr()
    assert "AMD ROCm PerfXpert" not in captured.err


def test_recursion_guard_env_set(tmp_path: Path, monkeypatch):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
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
    monkeypatch.setattr(
        opencode_launcher,
        "_runtime_cache_root",
        lambda: cache_root,
    )
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")
    opencode_launcher.main([])
    assert captured_env.get("PERFXPERT_IN_OPENCODE_SESSION") == "1"


def test_recursion_guard_rejects_nested_launch(monkeypatch, capsys):
    monkeypatch.setenv("PERFXPERT_IN_OPENCODE_SESSION", "1")
    monkeypatch.setattr(
        opencode_launcher.subprocess,
        "run",
        mock.MagicMock(side_effect=AssertionError("subprocess should not run")),
    )

    rc = opencode_launcher.main([])

    assert rc == 1
    captured = capsys.readouterr()
    assert "recursion detected" in captured.err


def test_prepare_runtime_config_dir_is_read_only(tmp_path: Path):
    staged = opencode_launcher._prepare_runtime_config_dir(
        opencode_launcher.resolve_config_dir(),
        tmp_path / "runtime",
    )

    assert staged == tmp_path / "runtime"
    assert (staged.stat().st_mode & 0o200) == 0
    for name in ("opencode.json", "amd-theme.json", "AGENTS.md", "mcp.json"):
        assert ((staged / name).stat().st_mode & 0o200) == 0


def test_main_launches_from_staged_runtime_dir(tmp_path: Path, monkeypatch):
    staged = tmp_path / "runtime"
    staged.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    captured = {}
    original_tempdir = tempfile.TemporaryDirectory

    monkeypatch.setattr(
        opencode_launcher,
        "resolve_opencode_binary",
        lambda: Path("/bin/true"),
    )
    monkeypatch.setattr(
        opencode_launcher,
        "resolve_config_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        opencode_launcher,
        "_prepare_runtime_config_dir",
        lambda src, runtime_dir=None: staged,
    )
    monkeypatch.setattr(
        opencode_launcher,
        "_runtime_cache_root",
        lambda: cache_root,
    )
    monkeypatch.setattr(
        opencode_launcher.tempfile,
        "TemporaryDirectory",
        lambda *args, **kwargs: original_tempdir(dir=str(tmp_path)),
    )

    def fake_run(cmd, **kwargs):
        captured["cwd"] = kwargs.get("cwd")
        return mock.MagicMock(returncode=0)

    monkeypatch.setattr(opencode_launcher.subprocess, "run", fake_run)
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    rc = opencode_launcher.main([])

    assert rc == 0
    assert captured["cwd"] == str(staged)


def test_runtime_cache_root_falls_back_without_home(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setattr(opencode_launcher.Path, "home", lambda: (_ for _ in ()).throw(RuntimeError("no home")))
    monkeypatch.setattr(opencode_launcher.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(opencode_launcher.os, "getuid", lambda: 1234)

    cache_root = opencode_launcher._runtime_cache_root()

    assert cache_root == tmp_path / "perfxpert-1234" / "perfxpert"
    assert cache_root.exists()


def test_main_reports_launch_oserror(tmp_path: Path, monkeypatch, capsys):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setattr(
        opencode_launcher,
        "resolve_opencode_binary",
        lambda: Path("/bin/true"),
    )
    monkeypatch.setattr(
        opencode_launcher,
        "_runtime_cache_root",
        lambda: cache_root,
    )
    monkeypatch.setattr(
        opencode_launcher.subprocess,
        "run",
        mock.MagicMock(side_effect=PermissionError("Permission denied")),
    )
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    rc = opencode_launcher.main([])

    assert rc == 1
    captured = capsys.readouterr()
    assert "Permission denied" in captured.err


def test_amd_red_in_banner(monkeypatch):
    """Banner includes AMD red ANSI color code."""
    # Verify the function itself contains the AMD red color code
    import inspect
    source = inspect.getsource(opencode_launcher.print_banner)
    # The source will have the escaped form \\033 when inspected
    assert "38;5;196m" in source  # AMD red color code in the function
