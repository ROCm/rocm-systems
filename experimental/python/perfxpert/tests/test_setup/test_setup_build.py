from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import setuptools


_COUNTER = itertools.count()
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_SETUP_PY = _PACKAGE_ROOT / "setup.py"
_PYPROJECT = _PACKAGE_ROOT / "pyproject.toml"
_PATCHED_BUILD_SCRIPT = _PACKAGE_ROOT / "scripts" / "build-patched-opencode.sh"


def _load_setup_call(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_setup(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(setuptools, "setup", _fake_setup)
    name = f"perfxpert_setup_test_{next(_COUNTER)}"
    spec = importlib.util.spec_from_file_location(name, _SETUP_PY)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, captured


def test_pyproject_does_not_package_generated_opencode_binary() -> None:
    text = _PYPROJECT.read_text(encoding="utf-8")

    assert '"_bundled/opencode"' not in text
    assert '"_bundled/**/*"' not in text
    assert '"_bundled/opencode_LICENSE"' not in text
    assert '"_bundled/opencode_config/*"' in text


def test_setup_py_registers_no_opencode_build_hooks(monkeypatch) -> None:
    module, captured = _load_setup_call(monkeypatch)

    assert captured == {"args": (), "kwargs": {}}
    assert not hasattr(module, "_run_opencode_build")
    assert not hasattr(module, "_ensure_bun_on_path")
    text = _SETUP_PY.read_text(encoding="utf-8")
    assert "build-patched-opencode.sh" not in text
    assert "build-bundled-opencode.sh" not in text
    assert "PERFXPERT_SKIP_BUNDLED_BUILD" not in text


def test_build_script_uses_patched_opencode_name_without_compat_wrapper() -> None:
    patched = _PATCHED_BUILD_SCRIPT.read_text(encoding="utf-8")

    assert _PATCHED_BUILD_SCRIPT.is_file()
    assert "build-patched-opencode" in patched
    assert "PERFXPERT_PATCHED_OPENCODE_PATH" in patched
    assert "perfxpert/_bundled" not in patched
    assert "PERFXPERT_BUNDLED_DIR" not in patched


def test_build_script_has_portable_size_and_sha_helpers() -> None:
    text = _PATCHED_BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "stat -c%s" in text
    assert "stat -f%z" in text
    assert "sha256sum" in text
    assert "shasum -a 256" in text


def test_build_script_retries_when_frozen_lockfile_install_fails() -> None:
    text = _PATCHED_BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "bun install --frozen-lockfile --ignore-scripts" in text
    assert "frozen lockfile install failed; retrying" in text
    assert "bun install --ignore-scripts" in text
    assert "bun run build --single --skip-install" in text
