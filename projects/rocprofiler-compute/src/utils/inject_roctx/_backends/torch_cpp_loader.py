# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Resolve and load the ``torch_trace_collector`` pybind11 extension.

The loader first looks for a prebuilt ``.so`` under the install prefix, then
builds the extension from the source tree, using the user cache as the build
directory when the source tree is read-only. Artifacts are named by the Python
and torch versions in use and a fingerprint of the C++ inputs, so an artifact
already present in the build directory is imported without running cmake
again. Set ``ROCPROFCOMPUTE_REBUILD_TORCH_TRACE=1`` to discard the build
directory and build again.
"""

import importlib.util
import os
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.inject_roctx._backends import torch_trace_fingerprint

_THIS_DIR = Path(__file__).resolve().parent
# parents[2] resolves to <repo>/src in dev and <install>/libexec/<project>
# in installed layouts; both host the torch_trace_collector sources at lib/.
_NATIVE_TOOL_ROOT = _THIS_DIR.parents[2]
_NATIVE_SOURCE_DIR = _NATIVE_TOOL_ROOT / "lib"
_SO_SOURCE_DIR = _NATIVE_SOURCE_DIR / "torch_trace_collector"
_SO_BUILDFILE = _SO_SOURCE_DIR / "CMakeLists.txt"

_INSTALL_TREE_PROJECT_NAME = "rocprofiler-compute"

TIER_PREBUILT = "prebuilt"
TIER_RUNTIME_BUILD = "runtime-build"

C_TIER_NAMES = frozenset((TIER_PREBUILT, TIER_RUNTIME_BUILD))

_source_fingerprint = torch_trace_fingerprint.source_fingerprint

_REBUILD_ENV_VAR = "ROCPROFCOMPUTE_REBUILD_TORCH_TRACE"

_Diagnostics = list[tuple[str, str]]


@dataclass
class LoadResult:
    """Outcome of a ``load()`` attempt: the module (or ``None`` for the Python
    fallback), the tier that produced it, and the diagnostic trail.
    """

    module: Optional[types.ModuleType]
    tier: Optional[str]
    diagnostics: _Diagnostics


def _safe_log(
    level: str,
    msg: str,
    diagnostics: Optional[_Diagnostics] = None,
) -> None:
    """Log via ``utils.logger`` if importable, otherwise stderr; also append the
    line to ``diagnostics`` when provided.
    """
    if diagnostics is not None:
        diagnostics.append((level, msg))
    try:
        from utils.logger import console_error, console_log, console_warning

        emit = {"log": console_log, "warning": console_warning, "error": console_error}[
            level
        ]
        emit("ml api trace loader", msg)
    except Exception:
        sys.stderr.write(f"[ml api trace loader] {level.upper()}: {msg}\n")


def format_load_diagnostic_trail(
    diagnostics: _Diagnostics,
    *,
    max_lines: int = 24,
) -> str:
    """Render diagnostics as indented lines, capped at ``max_lines``."""
    if not diagnostics:
        return ""
    rendered = [f"  [{lvl}] {msg}" for lvl, msg in diagnostics[-max_lines:]]
    return "\n".join(rendered)


def compute_tag() -> Optional[str]:
    """Return ``py{major}.{minor}_torch{version}_src{fingerprint}``, or ``None``."""
    try:
        import torch
    except Exception:
        return None

    torch_version = torch.__version__.split("+", 1)[0]
    fingerprint = _source_fingerprint()
    return (
        f"py{sys.version_info.major}.{sys.version_info.minor}"
        f"_torch{torch_version}_src{fingerprint}"
    )


def _import_module_from_path(name: str, path: Path) -> types.ModuleType:
    """Import a shared object from a filesystem path."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to build importlib spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_tree_prebuilt_candidates(tag: str) -> list[Path]:
    """Packager-baked .so candidates under <install-prefix>/lib*/<project>/."""
    # parents[4] reaches the install prefix from this file's location in
    # both layouts: <repo>/src/utils/inject_roctx/_backends in dev, and
    # <prefix>/libexec/<project>/utils/inject_roctx/_backends when installed.
    install_root = _THIS_DIR.parents[4]
    so_name = f"torch_trace_collector-{tag}.so"
    pattern = f"lib*/{_INSTALL_TREE_PROJECT_NAME}/{so_name}"
    return sorted(install_root.glob(pattern))


def _try_prebuilt(
    tag: str,
    diagnostics: Optional[_Diagnostics] = None,
) -> Optional[types.ModuleType]:
    for so_path in _install_tree_prebuilt_candidates(tag):
        if not so_path.exists():
            continue
        try:
            mod = _import_module_from_path("torch_trace_collector", so_path)
            _safe_log("log", f"loaded pre-built .so: {so_path}", diagnostics)
            return mod
        except Exception as e:
            _safe_log(
                "warning",
                f"pre-built .so at {so_path} failed to load: {e}",
                diagnostics,
            )
    return None


def _user_cache_dir(diagnostics: Optional[_Diagnostics] = None) -> Optional[Path]:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    d = Path(base) / "rocprofiler-compute" / "torch_trace_collector"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _safe_log(
            "log",
            f"user cache directory unavailable ({d}): {type(e).__name__}: {e}",
            diagnostics,
        )
        return None
    return d


_PREBUILT_HINT = (
    "ship a prebuilt torch_trace_collector-<tag>.so under "
    "<install-prefix>/lib*/" + _INSTALL_TREE_PROJECT_NAME + "/"
)


def _explain_cmake_failure(err: Exception) -> tuple[str, str]:
    """Classify a cmake failure into ``(reason, hint)``."""
    text = str(err).lower()
    if "could not find torch" in text or "torch_dir" in text:
        return (
            "libtorch package not visible to cmake",
            "ensure the running interpreter's torch wheel is fully "
            "installed; alternatively, " + _PREBUILT_HINT,
        )
    if "rocprofiler-sdk-roctx" in text or "roctx.h" in text:
        return (
            "rocprofiler-sdk-roctx headers/library not found",
            "set ROCM_PATH to your ROCm install root (default: "
            "/opt/rocm); alternatively, " + _PREBUILT_HINT,
        )
    if any(
        tok in text
        for tok in (
            "no cmake_cxx_compiler",
            "is not able to compile",
            "no such file",
            "command not found",
        )
    ):
        return (
            "host C++ compiler not found or non-functional",
            "ensure a working g++ or clang is on PATH; alternatively, "
            + _PREBUILT_HINT,
        )
    return (
        "cmake build failed",
        "see the cmake stderr above; if the failure is environmental, "
        + _PREBUILT_HINT,
    )


def _log_cmake_failure(
    err: Exception,
    diagnostics: Optional[_Diagnostics] = None,
) -> None:
    """Log a classified cmake failure."""
    reason, hint = _explain_cmake_failure(err)
    _safe_log("log", f"{reason}: {err}", diagnostics)
    _safe_log("log", f"to enable the C++ tier, {hint}", diagnostics)


def _cmake_executable() -> Optional[str]:
    """Return the cmake executable from ``$CMAKE`` or ``PATH``."""
    return shutil.which(os.environ.get("CMAKE", "cmake"))


def _is_writable(path: Path) -> bool:
    """Whether ``path`` is writable, or its parent when ``path`` does not exist."""
    target = path if path.exists() else path.parent
    return os.access(target, os.W_OK)


def _runtime_build_path(
    source_build_path: Path,
    diagnostics: Optional[_Diagnostics] = None,
) -> Optional[Path]:
    """Return ``source_build_path`` when it is writable, otherwise a directory of
    the same name under the user cache.
    """
    if _is_writable(source_build_path):
        return source_build_path
    cache_dir = _user_cache_dir(diagnostics)
    if cache_dir is None:
        return None
    _safe_log(
        "log",
        f"{source_build_path} is not writable; building under {cache_dir}",
        diagnostics,
    )
    return cache_dir / source_build_path.name


def _unlink_runtime_build_artifact(so_path: Path, build_path: Path) -> None:
    """Unlink ``so_path`` when it is a file under ``build_path``."""
    try:
        resolved = so_path.resolve()
        resolved.relative_to(build_path.resolve())
        if resolved.is_file():
            resolved.unlink()
    except (OSError, ValueError):
        return


def _try_runtime_build(
    tag: str,
    *,
    force_rebuild: bool = False,
    diagnostics: Optional[_Diagnostics] = None,
) -> Optional[types.ModuleType]:
    """Build the extension from the source tree and import the result."""
    missing_inputs = [
        p.name
        for p in torch_trace_fingerprint.required_input_paths()
        if not p.exists()
    ]
    if missing_inputs:
        _safe_log(
            "log",
            f"build inputs missing under {_NATIVE_SOURCE_DIR} "
            f"({', '.join(missing_inputs)}); skipping the runtime build",
            diagnostics,
        )
        return None

    cmake_exe = _cmake_executable()
    if cmake_exe is None:
        _safe_log("log", "cmake not on PATH; skipping the runtime build", diagnostics)
        return None

    try:
        from utils.native_tool_finder import NativeToolFinder
    except Exception as e:
        _safe_log(
            "log",
            f"native tool finder unavailable; skipping the runtime build: {e}",
            diagnostics,
        )
        return None

    source_build_path = (
        _NATIVE_TOOL_ROOT
        / NativeToolFinder.sources_dir_name
        / NativeToolFinder.sources_build_subdir_name
    )
    build_path = _runtime_build_path(source_build_path, diagnostics)
    if build_path is None:
        return None

    if force_rebuild:
        shutil.rmtree(build_path, ignore_errors=True)

    artifact_name = f"torch_trace_collector-{tag}.so"
    build_target = f"torch_trace_collector-{tag}"
    configure_options = (
        f"-DTORCH_TRACE_PYTHON={sys.executable}",
        f"-DTORCH_TRACE_SOURCE_FINGERPRINT={tag.rpartition('_src')[2]}",
        "-DBUILD_TORCH_TRACE_COLLECTOR=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    )

    for search_installed, reuse_built_artifact in (
        (not force_rebuild, not force_rebuild),
        (False, False),
    ):
        finder = NativeToolFinder(
            _NATIVE_TOOL_ROOT,
            artifact_name=artifact_name,
            build_target=build_target,
            build_path=build_path,
            configure_options=configure_options,
            cmake_executable=cmake_exe,
            search_installed=search_installed,
            reuse_built_artifact=reuse_built_artifact,
        )
        try:
            so_path = finder.get_artifact_path()
        except Exception as e:
            _log_cmake_failure(e, diagnostics)
            return None
        try:
            mod = _import_module_from_path("torch_trace_collector", so_path)
        except Exception as e:
            _safe_log(
                "warning",
                f"built .so at {so_path} failed to load: {e}",
                diagnostics,
            )
            _unlink_runtime_build_artifact(so_path, build_path)
            if not reuse_built_artifact:
                return None
            continue
        _safe_log("log", f"built and loaded .so: {so_path}", diagnostics)
        return mod
    return None


def load(force_python_fallback: bool = False) -> LoadResult:
    """Resolve the ``torch_trace_collector`` module and return a ``LoadResult``."""
    diagnostics: _Diagnostics = []

    if force_python_fallback:
        _safe_log(
            "log", "force_python_fallback=True; declining to load .so", diagnostics
        )
        return LoadResult(None, None, diagnostics)

    tag = compute_tag()
    if tag is None:
        _safe_log(
            "warning", "torch not importable; using Python-only injector", diagnostics
        )
        return LoadResult(None, None, diagnostics)

    if os.environ.get(_REBUILD_ENV_VAR) == "1":
        _safe_log(
            "warning",
            f"{_REBUILD_ENV_VAR}=1: discarding the build directory and "
            f"building again for tag {tag}",
            diagnostics,
        )
        mod = _try_runtime_build(
            tag,
            force_rebuild=True,
            diagnostics=diagnostics,
        )
        tier = TIER_RUNTIME_BUILD if mod is not None else None
        return LoadResult(mod, tier, diagnostics)

    mod = _try_prebuilt(tag, diagnostics=diagnostics)
    if mod is not None:
        return LoadResult(mod, TIER_PREBUILT, diagnostics)
    mod = _try_runtime_build(tag, diagnostics=diagnostics)
    if mod is not None:
        return LoadResult(mod, TIER_RUNTIME_BUILD, diagnostics)

    _safe_log(
        "log",
        "no torch_trace_collector .so available; using Python-only injector",
        diagnostics,
    )
    return LoadResult(None, None, diagnostics)
