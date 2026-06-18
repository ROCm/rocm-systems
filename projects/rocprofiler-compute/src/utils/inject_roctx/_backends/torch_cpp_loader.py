# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Resolve and load the ``roctx_recordfn`` pybind11 extension.

The loader first looks for a prebuilt ``.so`` under the install
prefix, then falls back to a cached cmake build under
``~/.cache/rocprofiler-compute/``. Cache entries are keyed by the
Python and torch versions in use and a fingerprint of the C++
inputs. Set ``ROCPROFCOMPUTE_REBUILD_ROCTX=1`` to force a fresh
build.
"""

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_THIS_DIR = Path(__file__).resolve().parent
# parents[2] resolves to <repo>/src in dev and <install>/libexec/<project>
# in installed layouts; both host the roctx_recordfn sources at lib/.
_SO_SOURCE_DIR = _THIS_DIR.parents[2] / "lib" / "roctx_recordfn"
_SO_SOURCE = _SO_SOURCE_DIR / "roctx_recordfn.cpp"
_SO_BUILDFILE = _SO_SOURCE_DIR / "CMakeLists.txt"

_INSTALL_TREE_PROJECT_NAME = "rocprofiler-compute"

TIER_PREBUILT = "prebuilt"
TIER_JIT = "jit"

C_TIER_NAMES = frozenset((TIER_PREBUILT, TIER_JIT))

_FINGERPRINT_INPUTS = (_SO_SOURCE, _SO_BUILDFILE)

_REBUILD_ENV_VAR = "ROCPROFCOMPUTE_REBUILD_ROCTX"

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


def _source_fingerprint() -> str:
    """First 12 hex chars of a SHA-256 over the source inputs, or ``"missing"``."""
    h = hashlib.sha256()
    seen = 0
    for path in _FINGERPRINT_INPUTS:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        h.update(f"{path.name}:{len(data)}\n".encode("ascii"))
        h.update(data)
        seen += 1
    if seen == 0:
        return "missing"
    return h.hexdigest()[:12]


def compute_tag() -> Optional[str]:
    """Return the cache tag for the active Python and torch versions, or ``None``."""
    try:
        import torch
    except Exception:
        return None

    py_major = sys.version_info.major
    py_minor = sys.version_info.minor
    torch_version = torch.__version__
    fingerprint = _source_fingerprint()
    return f"py{py_major}.{py_minor}_torch{torch_version}_src{fingerprint}"


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
    so_name = f"roctx_recordfn-{tag}.so"
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
            mod = _import_module_from_path("roctx_recordfn", so_path)
            _safe_log("log", f"loaded pre-built .so: {so_path}", diagnostics)
            return mod
        except Exception as e:
            _safe_log(
                "warning",
                f"pre-built .so at {so_path} failed to load: {e}",
                diagnostics,
            )
    return None


def _jit_cache_dir(diagnostics: Optional[_Diagnostics] = None) -> Optional[Path]:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    d = Path(base) / "rocprofiler-compute" / "roctx_recordfn"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _safe_log(
            "log",
            f"jit cache dir unavailable ({d}): {type(e).__name__}: {e}",
            diagnostics,
        )
        return None
    return d


_PREBUILT_HINT = (
    "ship a prebuilt roctx_recordfn-<tag>.so under "
    "<install-prefix>/lib*/" + _INSTALL_TREE_PROJECT_NAME + "/"
)


def _explain_cmake_failure(
    phase: str, err: Exception, stderr_tail: str
) -> tuple[str, str]:
    """Classify a cmake-tier failure into ``(reason, hint)``."""
    text = (str(err) + "\n" + (stderr_tail or "")).lower()
    if "could not find torch" in text or "torch_dir" in text:
        return (
            f"cmake {phase}: libtorch package not visible to cmake",
            "ensure the running interpreter's torch wheel is fully "
            "installed; alternatively, " + _PREBUILT_HINT,
        )
    if "rocprofiler-sdk-roctx" in text or "roctx.h" in text:
        return (
            f"cmake {phase}: rocprofiler-sdk-roctx headers/library not found",
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
            f"cmake {phase}: host C++ compiler not found or non-functional",
            "ensure a working g++ or clang is on PATH; alternatively, "
            + _PREBUILT_HINT,
        )
    return (
        f"cmake {phase} failed",
        "see the cmake stderr above; if the failure is environmental, "
        + _PREBUILT_HINT,
    )


def _log_cmake_failure(
    phase: str,
    err: Exception,
    stderr_tail: str,
    diagnostics: Optional[_Diagnostics] = None,
) -> None:
    """Log a classified cmake failure, including a stderr tail."""
    reason, hint = _explain_cmake_failure(phase, err, stderr_tail or "")
    _safe_log("log", f"cmake build failed: {reason}: {err}", diagnostics)
    if stderr_tail:
        tail = "\n".join(stderr_tail.strip().splitlines()[-12:])
        if tail:
            _safe_log("log", f"cmake stderr (tail):\n{tail}", diagnostics)
    _safe_log("log", f"to enable the C++ tier, {hint}", diagnostics)


def _jit_failure_marker(
    tag: str,
    diagnostics: Optional[_Diagnostics] = None,
) -> Optional[Path]:
    """Return the failure-marker path for ``tag``, or ``None`` if no cache dir."""
    cache_dir = _jit_cache_dir(diagnostics)
    if cache_dir is None:
        return None
    return cache_dir / f"roctx_recordfn-{tag}.build-failed"


def _record_jit_failure(
    tag: str,
    err: Exception,
    reason: str = TIER_JIT,
    stderr: str = "",
    diagnostics: Optional[_Diagnostics] = None,
) -> None:
    """Write a failure marker for the JIT tier."""
    try:
        marker = _jit_failure_marker(tag, diagnostics)
        if marker is None:
            return
        payload = f"{reason}: {type(err).__name__}: {err}\n"
        if stderr:
            tail = "\n".join(stderr.strip().splitlines()[-20:])
            if tail:
                payload += f"--- stderr tail ---\n{tail}\n"
        marker.write_text(payload)
    except Exception as exc:
        _safe_log(
            "log",
            f"jit failure-marker write skipped ({tag}): {type(exc).__name__}: {exc}",
            diagnostics,
        )


def _previous_jit_failure(
    tag: str,
    diagnostics: Optional[_Diagnostics] = None,
) -> Optional[str]:
    """Return the recorded failure summary for ``tag``, if any."""
    try:
        marker = _jit_failure_marker(tag, diagnostics)
        if marker is not None and marker.exists():
            return marker.read_text().strip() or None
    except Exception as exc:
        _safe_log(
            "log",
            f"jit failure-marker read skipped ({tag}): {type(exc).__name__}: {exc}",
            diagnostics,
        )
    return None


def _clear_jit_failure(tag: str) -> None:
    """Remove the failure marker for ``tag``."""
    marker = _jit_failure_marker(tag)
    if marker is None:
        return
    try:
        marker.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _install_cached_so(
    src_so: Path,
    cached_so: Path,
    diagnostics: Optional[_Diagnostics] = None,
) -> None:
    """Copy ``src_so`` onto ``cached_so`` for the next-run cache hit."""
    if not src_so.exists():
        return
    try:
        shutil.copy2(src_so, cached_so)
    except Exception as exc:
        _safe_log(
            "log",
            f"jit cache install skipped ({cached_so}): {type(exc).__name__}: {exc}",
            diagnostics,
        )


def _cmake_executable() -> Optional[str]:
    """Return the cmake executable from ``$CMAKE`` or ``PATH``."""
    return shutil.which(os.environ.get("CMAKE", "cmake"))


def _try_jit(
    tag: str,
    *,
    force_rebuild: bool = False,
    diagnostics: Optional[_Diagnostics] = None,
) -> Optional[types.ModuleType]:
    """Return the cached or freshly cmake-built ``roctx_recordfn`` module."""
    cache_dir = _jit_cache_dir(diagnostics)
    if cache_dir is None:
        return None
    cached_so = cache_dir / f"roctx_recordfn-{tag}.so"

    if not force_rebuild and cached_so.exists():
        try:
            mod = _import_module_from_path("roctx_recordfn", cached_so)
            _safe_log("log", f"loaded JIT-cached .so: {cached_so}", diagnostics)
            return mod
        except Exception as e:
            _safe_log(
                "warning",
                f"JIT-cached .so at {cached_so} failed to load: {e}",
                diagnostics,
            )
            try:
                cached_so.unlink()
            except Exception:
                pass

    if not _SO_SOURCE.exists() or not _SO_BUILDFILE.exists():
        _safe_log(
            "log",
            f"sources missing under {_SO_SOURCE_DIR}; skipping cmake tier",
            diagnostics,
        )
        return None

    cmake_exe = _cmake_executable()
    if cmake_exe is None:
        _safe_log("log", "cmake not on PATH; skipping cmake tier", diagnostics)
        return None

    prior = _previous_jit_failure(tag, diagnostics)
    if prior is not None:
        _safe_log(
            "log",
            f"skipping cmake build (prior failure cached for tag {tag}): {prior}",
            diagnostics,
        )
        return None

    build_dir = cache_dir / f"cmake-build-{tag}"
    try:
        build_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _log_cmake_failure("setup", e, "", diagnostics)
        _record_jit_failure(tag, e, diagnostics=diagnostics)
        return None

    configure_argv = [
        cmake_exe,
        "-S",
        str(_SO_SOURCE_DIR),
        "-B",
        str(build_dir),
        f"-DTORCH_TRACE_PYTHON={sys.executable}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    try:
        configure = subprocess.run(
            configure_argv,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as e:
        _log_cmake_failure("invoke", e, "", diagnostics)
        _record_jit_failure(tag, e, diagnostics=diagnostics)
        return None

    if configure.returncode != 0:
        err = RuntimeError(f"cmake configure exited with rc={configure.returncode}")
        _log_cmake_failure("configure", err, configure.stderr, diagnostics)
        _record_jit_failure(tag, err, stderr=configure.stderr, diagnostics=diagnostics)
        return None

    try:
        build = subprocess.run(
            [cmake_exe, "--build", str(build_dir), "-j"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as e:
        _log_cmake_failure("invoke", e, "", diagnostics)
        _record_jit_failure(tag, e, diagnostics=diagnostics)
        return None

    if build.returncode != 0:
        err = RuntimeError(f"cmake --build exited with rc={build.returncode}")
        _log_cmake_failure("build", err, build.stderr, diagnostics)
        _record_jit_failure(tag, err, stderr=build.stderr, diagnostics=diagnostics)
        return None

    produced = build_dir / f"roctx_recordfn-{tag}.so"
    if not produced.is_file():
        err = RuntimeError(
            f"cmake build succeeded but expected .so missing at {produced}"
        )
        _log_cmake_failure("missing-output", err, "", diagnostics)
        _record_jit_failure(tag, err, diagnostics=diagnostics)
        return None

    _install_cached_so(produced, cached_so, diagnostics)

    try:
        mod = _import_module_from_path("roctx_recordfn", cached_so)
    except Exception as e:
        _log_cmake_failure("load", e, "", diagnostics)
        _record_jit_failure(tag, e, diagnostics=diagnostics)
        return None

    _clear_jit_failure(tag)
    shutil.rmtree(build_dir, ignore_errors=True)

    _safe_log("log", f"cmake-built roctx_recordfn.so for {tag}", diagnostics)
    return mod


def load(force_python_fallback: bool = False) -> LoadResult:
    """Resolve the ``roctx_recordfn`` module and return a ``LoadResult``."""
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
            f"{_REBUILD_ENV_VAR}=1: bypassing prebuilt and JIT cache, "
            f"forcing fresh build for tag {tag}",
            diagnostics,
        )
        _clear_jit_failure(tag)
        mod = _try_jit(tag, force_rebuild=True, diagnostics=diagnostics)
        tier = TIER_JIT if mod is not None else None
        return LoadResult(mod, tier, diagnostics)

    for tier_name, step in (
        (TIER_PREBUILT, _try_prebuilt),
        (TIER_JIT, _try_jit),
    ):
        mod = step(tag, diagnostics=diagnostics)
        if mod is not None:
            return LoadResult(mod, tier_name, diagnostics)

    _safe_log(
        "log",
        "no roctx_recordfn .so available; using Python-only injector",
        diagnostics,
    )
    return LoadResult(None, None, diagnostics)
