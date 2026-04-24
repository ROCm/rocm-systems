from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace

import setuptools


_COUNTER = itertools.count()
_SETUP_PY = Path(__file__).resolve().parents[2] / "setup.py"
_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
_BUILD_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build-bundled-opencode.sh"
_BUILD_SCRIPT_PS1 = Path(__file__).resolve().parents[2] / "scripts" / "build-bundled-opencode.ps1"


def _load_setup_module(monkeypatch):
    monkeypatch.setattr(setuptools, "setup", lambda *args, **kwargs: None)
    name = f"perfxpert_setup_test_{next(_COUNTER)}"
    spec = importlib.util.spec_from_file_location(name, _SETUP_PY)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pyproject_packages_top_level_bundled_opencode_binary() -> None:
    text = _PYPROJECT.read_text()

    assert '"_bundled/opencode"' in text
    assert '"_bundled/opencode.exe"' in text
    assert '"_bundled/**/*"' in text


def test_ensure_bun_on_path_bootstraps_user_local_bun(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.os, "pathsep", ":")
    home = tmp_path / "home"
    monkeypatch.setattr(module.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin")
    calls: list[dict[str, str]] = []

    def _fake_which(name: str, path: str | None = None) -> str | None:
        if name == "bun":
            if path and str(home / ".bun" / "bin") in path:
                return str(home / ".bun" / "bin" / "bun")
            return None
        if name == "unzip":
            return "/usr/bin/unzip"
        if name in {"bash", "curl", "git"}:
            return f"/usr/bin/{name}"
        return None

    monkeypatch.setattr(module.shutil, "which", _fake_which)

    def _fake_install(env: dict[str, str]) -> int:
        calls.append(env)
        return 0

    monkeypatch.setattr(module, "_run_bun_install_script", _fake_install)

    path = module._ensure_bun_on_path()

    assert str(home / ".bun" / "bin") in path
    assert calls and calls[0]["BUN_INSTALL"] == str(home / ".bun")
    assert "bootstrapping bun" in capsys.readouterr().err


def test_ensure_bun_on_path_fails_when_unzip_missing(monkeypatch) -> None:
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module.sys, "platform", "linux")

    def _fake_which(name: str, path: str | None = None) -> str | None:
        return None

    monkeypatch.setattr(module.shutil, "which", _fake_which)

    try:
        module._ensure_bun_on_path()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected SystemExit when unzip is missing")


def test_ensure_bun_on_path_accepts_existing_bun_without_bootstrap_tools(
    monkeypatch,
) -> None:
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setenv("PATH", "/opt/bun/bin")

    def _fake_which(name: str, path: str | None = None) -> str | None:
        if name == "bun":
            return "/opt/bun/bin/bun"
        return None

    monkeypatch.setattr(module.shutil, "which", _fake_which)
    monkeypatch.setattr(
        module,
        "_run_bun_install_script",
        lambda _env: (_ for _ in ()).throw(AssertionError("unexpected bootstrap")),
    )

    assert module._ensure_bun_on_path() == "/opt/bun/bin"


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


def test_run_opencode_build_fails_when_checkout_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module.sys, "platform", "linux")
    build_script = tmp_path / "build-bundled-opencode.sh"
    build_script.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(module, "_BUILD_SCRIPT", build_script)
    monkeypatch.setattr(module, "_BUNDLE_PATH", tmp_path / "missing-opencode")
    monkeypatch.setattr(module, "_ensure_opencode_checkout", lambda: False)
    monkeypatch.setattr(module.shutil, "which", lambda name, path=None: f"/usr/bin/{name}")

    try:
        module._run_opencode_build()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected SystemExit when opencode checkout is unavailable")


def test_run_opencode_build_fails_when_build_script_missing(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module, "_BUILD_SCRIPT", tmp_path / "missing-build.sh")
    monkeypatch.setattr(module, "_BUNDLE_PATH", tmp_path / "missing-opencode")

    try:
        module._run_opencode_build()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected SystemExit when build script is missing")


def test_run_opencode_build_fails_when_build_script_fails(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module.sys, "platform", "linux")
    build_script = tmp_path / "build-bundled-opencode.sh"
    build_script.write_text("#!/bin/sh\nexit 3\n")
    monkeypatch.setattr(module, "_BUILD_SCRIPT", build_script)
    monkeypatch.setattr(module, "_BUNDLE_PATH", tmp_path / "missing-opencode")
    monkeypatch.setattr(module, "_ensure_opencode_checkout", lambda: True)
    monkeypatch.setattr(module, "_ensure_bun_on_path", lambda: "/usr/bin")
    monkeypatch.setattr(module.shutil, "which", lambda name, path=None: f"/usr/bin/{name}")

    class _Result:
        returncode = 3

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: _Result())

    try:
        module._run_opencode_build()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected SystemExit when bundled opencode build fails")


def test_setup_selects_powershell_build_script_on_windows(monkeypatch) -> None:
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module.sys, "platform", "win32")
    assert module._build_script_path().name == "build-bundled-opencode.ps1"
    assert module._bundle_path().name == "opencode.exe"


def test_setup_uses_powershell_command_on_windows(monkeypatch, tmp_path: Path) -> None:
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module.sys, "platform", "win32")

    def _fake_which(name: str, path: str | None = None) -> str | None:
        if name in {"pwsh", "pwsh.exe", "powershell", "powershell.exe"}:
            return "C:/Program Files/PowerShell/7/pwsh.exe"
        return None

    monkeypatch.setattr(module.shutil, "which", _fake_which)
    script = tmp_path / "build-bundled-opencode.ps1"
    cmd = module._build_script_command(script)

    assert cmd[0].endswith("pwsh.exe")
    assert cmd[1:6] == ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"]
    assert cmd[6] == str(script)


def test_windows_bun_bootstrap_uses_powershell(monkeypatch) -> None:
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setenv("PERFXPERT_BUN_INSTALL_PS1_URL", "https://example.invalid/install.ps1")

    def _fake_which(name: str, path: str | None = None) -> str | None:
        if name in {"pwsh", "pwsh.exe", "powershell", "powershell.exe"}:
            return "pwsh.exe"
        return None

    class _Result:
        returncode = 0

    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _Result()

    monkeypatch.setattr(module.shutil, "which", _fake_which)
    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    assert module._run_bun_install_script({}) == 0
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:6] == ["pwsh.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"]
    assert "install.ps1" in cmd[6]


def test_copy_bundled_binary_to_build_lib_preserves_windows_exe(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_setup_module(monkeypatch)
    bundled = tmp_path / "src" / "opencode.exe"
    bundled.parent.mkdir()
    bundled.write_text("binary\n")
    build_lib = tmp_path / "build"

    monkeypatch.setattr(module, "_bundle_path", lambda: bundled)

    module._copy_bundled_binary_to_build_lib(str(build_lib))

    copied = build_lib / "perfxpert" / "_bundled" / "opencode.exe"
    assert copied.read_text() == "binary\n"


def test_copy_bundled_binary_to_build_lib_is_noop_when_skipped(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module, "_bundle_path", lambda: tmp_path / "missing-opencode")

    module._copy_bundled_binary_to_build_lib(str(tmp_path / "build"))

    assert not (tmp_path / "build").exists()


def test_build_script_has_portable_size_and_sha_helpers() -> None:
    text = _BUILD_SCRIPT.read_text()
    assert "stat -c%s" in text
    assert "stat -f%z" in text
    assert "sha256sum" in text
    assert "shasum -a 256" in text


def test_windows_build_script_has_native_bundle_contract() -> None:
    text = _BUILD_SCRIPT_PS1.read_text()
    assert "Get-FileHash -Algorithm SHA256" in text
    assert "Get-NormalizedTextSha256Lower" in text
    assert "New-NormalizedPatchFile" in text
    assert "Invoke-OpencodeBuild" in text
    assert "opencode-windows-x64" in text
    assert "opencode.exe" in text
    assert "bun run build --single" in text
    assert "bun run build --single | Out-Host" in text
    assert "bun install --force --ignore-scripts" in text
    assert '(Join-Path $Dist "opencode-windows-x64\\bin\\opencode.exe")' in text


def test_windows_build_script_hashes_crlf_patches_against_lf_manifest() -> None:
    text = _BUILD_SCRIPT_PS1.read_text()
    assert '$Text.Replace("`r`n", "`n").Replace("`r", "`n")' in text
    assert "$Actual = Get-NormalizedTextSha256Lower $PatchPath" in text


def test_windows_build_script_applies_lf_normalized_patch_copies() -> None:
    text = _BUILD_SCRIPT_PS1.read_text()
    assert "$NormalizedPatch = New-NormalizedPatchFile $Patch.FullName" in text
    assert "git -C $OpencodeDir apply --check $NormalizedPatch" in text
    assert "git -C $OpencodeDir apply $NormalizedPatch" in text
    assert "Remove-Item -Path $NormalizedPatch" in text


def test_windows_build_script_retries_after_bun_bin_remap_failure() -> None:
    text = _BUILD_SCRIPT_PS1.read_text()
    assert "build failed; running 'bun install --force --ignore-scripts'" in text
    assert "$BuildExit = Invoke-OpencodeBuild $OpencodePackage" in text
    assert "'bun run build --single' exited $BuildExit after force reinstall retry" in text


def test_opencode_patch_manifest_includes_direct_vite_windows_fix() -> None:
    manifest = (_PYPROJECT.parent / ".patches" / "SHA256SUMS").read_text()
    patch = (_PYPROJECT.parent / ".patches" / "0071-windows-direct-vite-build.patch").read_text()
    assert "0071-windows-direct-vite-build.patch" in manifest
    assert "node_modules\", \"vite\", \"bin\", \"vite.js\"" in patch
    assert "bun --cwd ${appDir} ${viteBin} build" in patch


def test_opencode_patch_manifest_skips_lifecycle_for_native_installs() -> None:
    manifest = (_PYPROJECT.parent / ".patches" / "SHA256SUMS").read_text()
    patch = (
        _PYPROJECT.parent / ".patches" / "0072-ignore-lifecycle-during-native-installs.patch"
    ).read_text()
    assert "0072-ignore-lifecycle-during-native-installs.patch" in manifest
    assert '--ignore-scripts --os="*" --cpu="*" @opentui/core' in patch
    assert '--ignore-scripts --os="*" --cpu="*" @parcel/watcher' in patch


def test_build_script_retries_when_frozen_lockfile_install_fails() -> None:
    text = _BUILD_SCRIPT.read_text()
    assert "bun install --frozen-lockfile --ignore-scripts" in text
    assert "frozen lockfile install failed; retrying" in text
    assert "bun install --ignore-scripts" in text
    assert "bun run build --single --skip-install" in text
