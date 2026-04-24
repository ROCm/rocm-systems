from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace

import setuptools


_COUNTER = itertools.count()
_SETUP_PY = Path(__file__).resolve().parents[2] / "setup.py"
_BUILD_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build-bundled-opencode.sh"


def _load_setup_module(monkeypatch):
    monkeypatch.setattr(setuptools, "setup", lambda *args, **kwargs: None)
    name = f"perfxpert_setup_test_{next(_COUNTER)}"
    spec = importlib.util.spec_from_file_location(name, _SETUP_PY)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ensure_bun_on_path_refuses_network_download(
    monkeypatch, capsys
) -> None:
    module = _load_setup_module(monkeypatch)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    assert module._ensure_bun_on_path() is None
    assert "refusing to download build tooling" in capsys.readouterr().err


def test_opencode_dir_is_populated_requires_git_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_setup_module(monkeypatch)
    opencode_dir = tmp_path / "opencode"
    opencode_dir.mkdir(parents=True)
    (opencode_dir / "package.json").write_text("{}\n")
    monkeypatch.setattr(module, "_OPENCODE_DIR", opencode_dir)

    assert module._opencode_dir_is_populated() is False

    (opencode_dir / ".git").write_text("gitdir: ../.git/modules/opencode\n")
    assert module._opencode_dir_is_populated() is True


def test_ensure_opencode_checkout_uses_scoped_submodule_update(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_setup_module(monkeypatch)
    repo_root = tmp_path / "repo"
    package_root = repo_root / "experimental" / "python" / "perfxpert"
    opencode_dir = package_root / "opencode"
    opencode_dir.mkdir(parents=True)
    (repo_root / ".git").mkdir()
    (repo_root / ".gitmodules").write_text(
        '[submodule "experimental/python/perfxpert/opencode"]\n'
        "\tpath = experimental/python/perfxpert/opencode\n"
        "\turl = https://github.com/sst/opencode.git\n"
    )
    monkeypatch.setattr(module, "_HERE", package_root)
    monkeypatch.setattr(module, "_OPENCODE_DIR", opencode_dir)

    rel_path = "experimental/python/perfxpert/opencode"
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"submodule.perfxpert.path {rel_path}\n",
            stderr="",
        ),
    )

    calls: list[tuple[list[str], Path | None]] = []

    def _fake_run_git(args: list[str], cwd: Path | None = None, timeout: int = 300) -> bool:
        calls.append((args, cwd))
        (opencode_dir / "package.json").write_text("{}\n")
        (opencode_dir / ".git").write_text("gitdir: ../.git/modules/opencode\n")
        return True

    monkeypatch.setattr(module, "_run_git", _fake_run_git)

    assert module._ensure_opencode_checkout() is True
    assert calls == [
        (
            ["submodule", "update", "--init", "--depth", "1", "--", rel_path],
            repo_root,
        )
    ]


def test_ensure_opencode_checkout_refuses_direct_network_clone(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    module = _load_setup_module(monkeypatch)
    package_root = tmp_path / "pkg"
    opencode_dir = package_root / "opencode"
    opencode_dir.mkdir(parents=True)
    monkeypatch.setattr(module, "_HERE", package_root)
    monkeypatch.setattr(module, "_OPENCODE_DIR", opencode_dir)

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _unexpected_run_git(*args, **kwargs):
        calls.append((args, kwargs))
        return False

    monkeypatch.setattr(module, "_run_git", _unexpected_run_git)

    assert module._ensure_opencode_checkout() is False
    assert calls == []
    assert "will not clone from the network" in capsys.readouterr().err


def test_build_script_has_portable_size_and_sha_helpers() -> None:
    text = _BUILD_SCRIPT.read_text()
    assert "stat -c%s" in text
    assert "stat -f%z" in text
    assert "sha256sum" in text
    assert "shasum -a 256" in text


def test_build_script_retries_when_frozen_lockfile_install_fails() -> None:
    text = _BUILD_SCRIPT.read_text()
    assert "bun install --frozen-lockfile --ignore-scripts" in text
    assert "frozen lockfile install failed; retrying" in text
    assert "bun install --ignore-scripts" in text
