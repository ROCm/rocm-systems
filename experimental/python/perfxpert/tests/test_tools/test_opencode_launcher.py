"""Unit tests for perfxpert.cli.opencode_launcher."""

from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from unittest import mock

import pytest

from perfxpert.cli import opencode_launcher

_ORIGINAL_MANAGED_PATCHED_OPENCODE_PATHS = (
    opencode_launcher._managed_patched_opencode_paths
)
_PERFXPERT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _disable_repo_local_patched_binary(monkeypatch):
    monkeypatch.setattr(
        opencode_launcher,
        "_repo_local_patched_opencode_paths",
        lambda: [],
    )
    monkeypatch.setattr(
        opencode_launcher,
        "_managed_patched_opencode_paths",
        _ORIGINAL_MANAGED_PATCHED_OPENCODE_PATHS,
    )
    monkeypatch.setattr(
        opencode_launcher,
        "_managed_patched_opencode_paths",
        lambda: [],
    )


def test_version_flag_short_circuit(capsys):
    rc = opencode_launcher.main(["--version"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "AMD" in captured.out
    assert version("perfxpert") in captured.out


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

    monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
    monkeypatch.delenv("PERFXPERT_PATCHED_OPENCODE_PATH", raising=False)
    monkeypatch.setattr(
        opencode_launcher,
        "_repo_local_patched_opencode_paths",
        lambda: [local_bin],
    )

    assert opencode_launcher.resolve_opencode_binary() == local_bin


def test_resolve_binary_uses_managed_cache_artifact(tmp_path: Path, monkeypatch):
    cached_bin = tmp_path / "cache" / "perfxpert" / "opencode" / "opencode"
    cached_bin.parent.mkdir(parents=True)
    cached_bin.write_text("#!/bin/sh\necho cache\n")
    cached_bin.chmod(0o755)

    monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
    monkeypatch.delenv("PERFXPERT_PATCHED_OPENCODE_PATH", raising=False)
    monkeypatch.setattr(
        opencode_launcher,
        "_managed_patched_opencode_paths",
        lambda: [cached_bin],
    )

    assert opencode_launcher.resolve_opencode_binary() == cached_bin


def test_patch_manifest_hash_constant_matches_current_manifest():
    manifest = _PERFXPERT_ROOT / ".patches" / "SHA256SUMS"
    assert sha256(manifest.read_bytes()).hexdigest() == (
        opencode_launcher._PATCH_MANIFEST_SHA256
    )


def test_resolve_validated_binary_accepts_tagged_managed_cache_artifact(
    tmp_path: Path, monkeypatch
):
    cached_bin = tmp_path / "cache" / "perfxpert" / "opencode" / "opencode"
    cached_bin.parent.mkdir(parents=True)
    cached_bin.write_text("#!/bin/sh\necho opencode 1.0.0\n")
    cached_bin.chmod(0o755)
    cached_bin.with_name(
        cached_bin.name + ".perfxpert-patch-manifest-sha256"
    ).write_text(opencode_launcher._PATCH_MANIFEST_SHA256, encoding="utf-8")

    monkeypatch.delenv("PERFXPERT_PATCHED_OPENCODE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(
        opencode_launcher,
        "_repo_local_patched_opencode_paths",
        lambda: [],
    )
    monkeypatch.setattr(
        opencode_launcher,
        "_managed_patched_opencode_paths",
        _ORIGINAL_MANAGED_PATCHED_OPENCODE_PATHS,
    )

    path, version = opencode_launcher.resolve_validated_opencode_binary()

    assert path == cached_bin
    assert version == "opencode 1.0.0"


def test_resolve_validated_binary_rejects_untagged_managed_cache_artifact(
    tmp_path: Path, monkeypatch
):
    cached_bin = tmp_path / "cache" / "perfxpert" / "opencode" / "opencode"
    cached_bin.parent.mkdir(parents=True)
    cached_bin.write_text("#!/bin/sh\necho opencode 1.0.0\n")
    cached_bin.chmod(0o755)

    monkeypatch.delenv("PERFXPERT_PATCHED_OPENCODE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(
        opencode_launcher,
        "_repo_local_patched_opencode_paths",
        lambda: [],
    )
    monkeypatch.setattr(
        opencode_launcher,
        "_managed_patched_opencode_paths",
        _ORIGINAL_MANAGED_PATCHED_OPENCODE_PATHS,
    )

    with pytest.raises(
        opencode_launcher.OpencodeBinaryValidationError,
        match="missing patch metadata",
    ):
        opencode_launcher.resolve_validated_opencode_binary()


def test_resolve_validated_binary_rejects_broken_managed_cache_artifact(
    tmp_path: Path, monkeypatch
):
    cached_bin = tmp_path / "cache" / "perfxpert" / "opencode" / "opencode"
    cached_bin.parent.mkdir(parents=True)
    cached_bin.write_text("#!/bin/sh\nexit 42\n")
    cached_bin.chmod(0o755)
    cached_bin.with_name(
        cached_bin.name + ".perfxpert-patch-manifest-sha256"
    ).write_text(opencode_launcher._PATCH_MANIFEST_SHA256, encoding="utf-8")

    monkeypatch.delenv("PERFXPERT_PATCHED_OPENCODE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(
        opencode_launcher,
        "_repo_local_patched_opencode_paths",
        lambda: [],
    )
    monkeypatch.setattr(
        opencode_launcher,
        "_managed_patched_opencode_paths",
        _ORIGINAL_MANAGED_PATCHED_OPENCODE_PATHS,
    )

    with pytest.raises(
        opencode_launcher.OpencodeBinaryValidationError,
        match="smoke test failed",
    ):
        opencode_launcher.resolve_validated_opencode_binary()


def test_managed_cache_paths_honor_xdg_without_home_fallback(
    tmp_path: Path, monkeypatch
):
    xdg = tmp_path / "xdg-cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    monkeypatch.setattr(
        opencode_launcher,
        "_managed_patched_opencode_paths",
        _ORIGINAL_MANAGED_PATCHED_OPENCODE_PATHS,
    )

    assert opencode_launcher._managed_patched_opencode_paths() == [
        xdg / "perfxpert" / "opencode" / "opencode"
    ]


def test_resolve_binary_uses_explicit_patched_artifact(tmp_path: Path, monkeypatch):
    patched_bin = tmp_path / "patched-opencode"
    patched_bin.write_text("#!/bin/sh\necho patched\n")
    patched_bin.chmod(0o755)

    monkeypatch.setenv("PERFXPERT_PATCHED_OPENCODE_PATH", str(patched_bin))
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", str(tmp_path / "upstream-opencode"))

    assert opencode_launcher.resolve_opencode_binary() == patched_bin


def test_resolve_binary_validates_explicit_patched_artifact(
    tmp_path: Path, monkeypatch
):
    patched_bin = tmp_path / "patched-opencode"
    patched_bin.write_text("#!/bin/sh\necho patched\n")
    patched_bin.chmod(0o644)

    monkeypatch.setenv("PERFXPERT_PATCHED_OPENCODE_PATH", str(patched_bin))

    with pytest.raises(FileNotFoundError, match="not executable"):
        opencode_launcher.resolve_opencode_binary()


def test_resolve_user_binary_raises_when_override_missing(tmp_path: Path, monkeypatch):
    """The explicit upstream-opencode escape hatch validates its override."""
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", str(tmp_path / "nonexistent"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        opencode_launcher.resolve_user_opencode_binary()


def test_resolve_user_binary_raises_when_override_not_executable(
    tmp_path: Path, monkeypatch
):
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

    # Isolate: no override and no PATH hit.
    monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(opencode_launcher.shutil, "which", lambda _: None)

    resolved = opencode_launcher.resolve_user_opencode_binary()
    assert resolved == fake_bin


@pytest.mark.parametrize(
    "subpath",
    [
        ".opencode/bin/opencode",
        ".local/bin/opencode",
    ],
)
def test_resolve_user_binary_autodiscovers_multiple_wellknown_paths(
    tmp_path: Path, monkeypatch, subpath
):
    """Each well-known upstream location must be auto-discovered for the escape hatch."""
    fake_home = tmp_path
    fake_bin = fake_home / subpath
    fake_bin.parent.mkdir(parents=True, exist_ok=True)
    fake_bin.write_text("#!/bin/sh\necho fake\n")
    fake_bin.chmod(0o755)

    monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(opencode_launcher.shutil, "which", lambda _: None)

    assert opencode_launcher.resolve_user_opencode_binary() == fake_bin


def test_resolve_default_binary_missing_suggests_wrapper(monkeypatch, tmp_path: Path):
    """The default path must point users back to the patched submodule build."""
    monkeypatch.delenv("PERFXPERT_OPENCODE_PATH", raising=False)
    monkeypatch.delenv("PERFXPERT_PATCHED_OPENCODE_PATH", raising=False)
    # Ensure no well-known path resolves
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(opencode_launcher.shutil, "which", lambda _: None)

    with pytest.raises(FileNotFoundError) as exc:
        opencode_launcher.resolve_opencode_binary()
    msg = str(exc.value)
    assert "patched opencode binary not found" in msg
    assert "perfxpert-code install-patches" in msg
    assert "build-patched-opencode.sh" in msg
    assert "PERFXPERT_PATCHED_OPENCODE_PATH" in msg
    assert "perfxpert-code opencode" in msg


def test_resolve_user_binary_missing_suggests_upstream_install(
    monkeypatch, tmp_path: Path
):
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
