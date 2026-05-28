# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Resolve and load the roctx_recordfn pybind11 extension.

Tier order: prebuilt, JIT cache, cmake build, torch.utils.cpp_extension.
Returns None when every tier misses (caller drops to the Python-only
injector). Cache keyed by SHA-256 fingerprint of the C++ inputs; failed
builds drop a tag-scoped negative-cache marker. Set
ROCPROFCOMPUTE_REBUILD_ROCTX=1 to bypass caches.
"""

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Optional

_THIS_DIR = Path(__file__).resolve().parent
_SO_SOURCE_DIR = _THIS_DIR.parent / "lib" / "roctx_recordfn"
_SO_SOURCE = _SO_SOURCE_DIR / "roctx_recordfn.cpp"
_SO_BUILDFILE = _SO_SOURCE_DIR / "CMakeLists.txt"

_INSTALL_TREE_PROJECT_NAME = "rocprofiler-compute"

_CMAKE_TIER_NAME = "cmake-build"
_CPPEXT_TIER_NAME = "cpp-extension"

TIER_EXPLICIT = "explicit"
TIER_PREBUILT = "prebuilt"
TIER_JIT_CACHED = "jit_cached"
TIER_CMAKE_BUILD = "cmake_build"
TIER_CPP_EXTENSION = "cpp_extension"

C_TIER_NAMES = frozenset((
    TIER_EXPLICIT,
    TIER_PREBUILT,
    TIER_JIT_CACHED,
    TIER_CMAKE_BUILD,
    TIER_CPP_EXTENSION,
))


_LAST_LOAD_DIAGNOSTICS: list[tuple[str, str]] = []
_LAST_LOADED_TIER: Optional[str] = None

# Any new build input MUST be appended or edits to it will load a stale .so.
_FINGERPRINT_INPUTS = (_SO_SOURCE, _SO_BUILDFILE)

_REBUILD_ENV_VAR = "ROCPROFCOMPUTE_REBUILD_ROCTX"
_EXPLICIT_SO_ENV_VAR = "ROCPROFCOMPUTE_ROCTX_RECORDFN_SO"


def _safe_log(level: str, msg: str) -> None:
    """Log via utils.logger if importable, else stderr; tee to the trail."""
    _LAST_LOAD_DIAGNOSTICS.append((level, msg))
    try:
        from utils.logger import console_error, console_log, console_warning

        emit = {"log": console_log, "warning": console_warning, "error": console_error}[
            level
        ]
        emit("torch trace loader", msg)
    except Exception:
        sys.stderr.write(f"[torch trace loader] {level.upper()}: {msg}\n")


def loaded_tier() -> Optional[str]:
    """Tier that won on the most recent load(), or None."""
    return _LAST_LOADED_TIER


def consume_diagnostics() -> tuple[Optional[str], list[tuple[str, str]]]:
    """Drain and return (tier_loaded, [(level, msg), ...])."""
    diagnostics = list(_LAST_LOAD_DIAGNOSTICS)
    _LAST_LOAD_DIAGNOSTICS.clear()
    return _LAST_LOADED_TIER, diagnostics


def format_load_diagnostic_trail(
    diagnostics: list[tuple[str, str]],
    *,
    max_lines: int = 24,
) -> str:
    """Render trail as indented lines, capped at max_lines."""
    if not diagnostics:
        return ""
    rendered = [f"  [{lvl}] {msg}" for lvl, msg in diagnostics[-max_lines:]]
    return "\n".join(rendered)


def _source_fingerprint() -> str:
    """First 12 hex of SHA-256 over _FINGERPRINT_INPUTS, or "missing"."""
    h = hashlib.sha256()
    seen = 0
    for path in _FINGERPRINT_INPUTS:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        # Length-delimit so cross-boundary edits can't collide hashes.
        h.update(f"{path.name}:{len(data)}\n".encode("ascii"))
        h.update(data)
        seen += 1
    if seen == 0:
        return "missing"
    return h.hexdigest()[:12]


def compute_tag() -> Optional[str]:
    """py{X}.{Y}_torch{Z}_abi{0|1}_src{12-hex}, or None if no torch."""
    try:
        import torch
        import torch._C
    except Exception:
        return None

    py_major = sys.version_info.major
    py_minor = sys.version_info.minor
    torch_version = torch.__version__
    try:
        cxx11_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    except Exception:
        cxx11_abi = 1
    fingerprint = _source_fingerprint()
    return (
        f"py{py_major}.{py_minor}_torch{torch_version}_abi{cxx11_abi}_src{fingerprint}"
    )


def _import_module_from_path(name: str, path: Path) -> types.ModuleType:
    """Import a .so from a filesystem path."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to build importlib spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_tree_prebuilt_candidates(tag: str) -> list[Path]:
    """Packager-baked .so candidates under <install-prefix>/lib*/<project>/."""
    install_root = _THIS_DIR.parent.parent.parent
    so_name = f"roctx_recordfn-{tag}.so"
    # lib* glob pattern handles CMAKE_INSTALL_LIBDIR variations (lib, lib64,
    # distro multiarch layouts) in the same way as NativeToolFinder.
    pattern = f"lib*/{_INSTALL_TREE_PROJECT_NAME}/{so_name}"
    return sorted(install_root.glob(pattern))


def _try_explicit_so(tag: str) -> Optional[types.ModuleType]:
    """Honour ROCPROFCOMPUTE_ROCTX_RECORDFN_SO; no build-tier fallback on miss."""
    env_path_str = os.environ.get(_EXPLICIT_SO_ENV_VAR)
    if not env_path_str:
        return None
    env_path = Path(env_path_str)
    if not env_path.is_file():
        _safe_log(
            "warning",
            f"{_EXPLICIT_SO_ENV_VAR}={env_path_str} but the file does "
            "not exist; skipping (no build-tier fallback)",
        )
        return None
    try:
        mod = _import_module_from_path("roctx_recordfn", env_path)
    except Exception as e:
        _safe_log(
            "warning",
            f"{_EXPLICIT_SO_ENV_VAR}={env_path_str} failed to import: "
            f"{e}; skipping (no build-tier fallback)",
        )
        return None
    _safe_log("log", f"loaded explicit .so from {_EXPLICIT_SO_ENV_VAR}={env_path}")
    return mod


def _try_prebuilt(tag: str) -> Optional[types.ModuleType]:
    for so_path in _install_tree_prebuilt_candidates(tag):
        if not so_path.exists():
            continue
        try:
            mod = _import_module_from_path("roctx_recordfn", so_path)
            _safe_log("log", f"loaded pre-built .so: {so_path}")
            return mod
        except Exception as e:
            _safe_log(
                "warning",
                f"pre-built .so at {so_path} failed to load: {e}",
            )
    return None


def _jit_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    d = Path(base) / "rocprofiler-compute" / "roctx_recordfn"
    d.mkdir(parents=True, exist_ok=True)
    return d


_PREBUILT_HINT = (
    "ship a prebuilt roctx_recordfn-{tag}.so under "
    "<install-prefix>/lib*/" + _INSTALL_TREE_PROJECT_NAME + "/ (resolved by the "
    "loader's prebuilt tier), or pin one explicitly via "
    "ROCPROFCOMPUTE_ROCTX_RECORDFN_SO=<path>"
)


def _explain_cppext_failure(err: Exception) -> tuple[str, str]:
    """Classify cpp_extension failure into (reason, hint). Never recommends ninja."""
    text = str(err).lower()
    if "ninja" in text:
        return (
            "torch.utils.cpp_extension.load on this PyTorch build "
            "requires the ninja build backend, which rocprof-compute "
            "does not list as a dependency",
            "install cmake to enable the cmake build tier (which does "
            "not need ninja); alternatively, " + _PREBUILT_HINT,
        )
    if "torch/extension.h" in text or "torch/torch.h" in text:
        return (
            "libtorch headers not found via torch.utils.cpp_extension",
            "ensure torch is fully installed (the official wheels "
            "ship the headers under site-packages/torch/include); "
            "alternatively, " + _PREBUILT_HINT,
        )
    if any(
        tok in text
        for tok in ("g++", "gcc", "clang", "no such file", "command not found")
    ):
        return (
            "host C++ compiler not found or non-functional",
            "ensure a working g++ or clang is on PATH; alternatively, "
            + _PREBUILT_HINT,
        )
    return (
        "torch.utils.cpp_extension.load raised an unrecognised error",
        "see the torch exception above; if the failure is environmental, "
        + _PREBUILT_HINT,
    )


def _explain_cmake_failure(
    phase: str, err: Exception, stderr_tail: str
) -> tuple[str, str]:
    """Classify cmake-tier failure into (reason, hint).

    phase: invoke/configure/build/missing-output/load.
    """
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


def _log_cppext_failure(err: Exception) -> None:
    """Log a classified cpp_extension failure at LOG level."""
    reason, hint = _explain_cppext_failure(err)
    _safe_log("log", f"cpp_extension JIT skipped: {reason}: {err}")
    _safe_log("log", f"to enable the C++ tier, {hint}")


def _log_cmake_failure(phase: str, err: Exception, stderr_tail: str) -> None:
    """Same as _log_cppext_failure plus a cmake stderr tail."""
    reason, hint = _explain_cmake_failure(phase, err, stderr_tail or "")
    _safe_log("log", f"cmake build skipped: {reason}: {err}")
    if stderr_tail:
        tail = "\n".join(stderr_tail.strip().splitlines()[-12:])
        if tail:
            _safe_log("log", f"cmake stderr (tail):\n{tail}")
    _safe_log("log", f"to enable the C++ tier, {hint}")


def _jit_compile_viable(cpp_ext: types.ModuleType) -> bool:
    """True if cpp_extension.load is likely to succeed without pulling ninja."""
    import inspect

    try:
        if "use_ninja" in inspect.signature(cpp_ext.load).parameters:
            return True
    except (TypeError, ValueError):
        pass
    return shutil.which("ninja") is not None


def _jit_failure_marker(tag: str) -> Path:
    """Tag-scoped negative-cache marker, co-located with the cached .so."""
    return _jit_cache_dir() / f"roctx_recordfn-{tag}.build-failed"


def _record_jit_failure(
    tag: str,
    err: Exception,
    reason: str = _CPPEXT_TIER_NAME,
    stderr: str = "",
) -> None:
    """Drop a tag-scoped marker shared by both build tiers."""
    try:
        payload = f"{reason}: {type(err).__name__}: {err}\n"
        if stderr:
            tail = "\n".join(stderr.strip().splitlines()[-20:])
            if tail:
                payload += f"--- stderr tail ---\n{tail}\n"
        _jit_failure_marker(tag).write_text(payload)
    except Exception as exc:
        _safe_log(
            "log",
            f"jit failure-marker write skipped ({tag}): {type(exc).__name__}: {exc}",
        )


def _previous_jit_failure(tag: str) -> Optional[str]:
    """Cached failure summary for tag, or None."""
    try:
        marker = _jit_failure_marker(tag)
        if marker.exists():
            return marker.read_text().strip() or None
    except Exception as exc:
        _safe_log(
            "log",
            f"jit failure-marker read skipped ({tag}): {type(exc).__name__}: {exc}",
        )
    return None


def _clear_jit_failure(tag: str) -> None:
    """Remove the failure marker (best-effort)."""
    try:
        _jit_failure_marker(tag).unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _install_cached_so(src_so: Path, cached_so: Path) -> None:
    """Best-effort copy of src_so onto cached_so for next-run JIT cache hit."""
    if not src_so.exists():
        return
    try:
        shutil.copy2(src_so, cached_so)
    except Exception as exc:
        _safe_log(
            "log",
            f"jit cache install skipped ({cached_so}): {type(exc).__name__}: {exc}",
        )


def _try_jit_cached(tag: str) -> Optional[types.ModuleType]:
    cache_dir = _jit_cache_dir()
    so_path = cache_dir / f"roctx_recordfn-{tag}.so"
    if not so_path.exists():
        return None
    try:
        mod = _import_module_from_path("roctx_recordfn", so_path)
        _safe_log("log", f"loaded JIT-cached .so: {so_path}")
        return mod
    except Exception as e:
        _safe_log("warning", f"JIT-cached .so at {so_path} failed to load: {e}")
        try:
            so_path.unlink()
        except Exception:
            pass
        return None


def _cmake_executable() -> Optional[str]:
    """cmake binary from $CMAKE or PATH, or None."""
    return shutil.which(os.environ.get("CMAKE", "cmake"))


def _try_cmake_build(tag: str) -> Optional[types.ModuleType]:
    """Build the .so via our CMakeLists.txt. No ninja required."""
    if not _SO_SOURCE.exists() or not _SO_BUILDFILE.exists():
        _safe_log(
            "log",
            f"sources missing under {_SO_SOURCE_DIR}; skipping cmake tier",
        )
        return None

    cmake_exe = _cmake_executable()
    if cmake_exe is None:
        _safe_log("log", "cmake not on PATH; skipping cmake tier")
        return None

    prior = _previous_jit_failure(tag)
    if prior is not None:
        _safe_log(
            "log",
            f"skipping cmake build (prior failure cached for tag {tag}): {prior}",
        )
        return None

    build_dir = _jit_cache_dir() / f"cmake-build-{tag}"
    try:
        build_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _log_cmake_failure("setup", e, "")
        _record_jit_failure(tag, e, reason=_CMAKE_TIER_NAME)
        return None

    # Pin python so the cmake-side tag matches compute_tag().
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
        _log_cmake_failure("invoke", e, "")
        _record_jit_failure(tag, e, reason=_CMAKE_TIER_NAME)
        return None

    if configure.returncode != 0:
        err = RuntimeError(f"cmake configure exited with rc={configure.returncode}")
        _log_cmake_failure("configure", err, configure.stderr)
        _record_jit_failure(
            tag,
            err,
            reason=_CMAKE_TIER_NAME,
            stderr=configure.stderr,
        )
        return None

    try:
        build = subprocess.run(
            [cmake_exe, "--build", str(build_dir), "-j"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as e:
        _log_cmake_failure("invoke", e, "")
        _record_jit_failure(tag, e, reason=_CMAKE_TIER_NAME)
        return None

    if build.returncode != 0:
        err = RuntimeError(f"cmake --build exited with rc={build.returncode}")
        _log_cmake_failure("build", err, build.stderr)
        _record_jit_failure(
            tag,
            err,
            reason=_CMAKE_TIER_NAME,
            stderr=build.stderr,
        )
        return None

    produced = build_dir / f"roctx_recordfn-{tag}.so"
    if not produced.is_file():
        err = RuntimeError(
            f"cmake build succeeded but expected .so missing at {produced}"
        )
        _log_cmake_failure("missing-output", err, "")
        _record_jit_failure(tag, err, reason=_CMAKE_TIER_NAME)
        return None

    cached_so = _jit_cache_dir() / f"roctx_recordfn-{tag}.so"
    _install_cached_so(produced, cached_so)

    try:
        mod = _import_module_from_path("roctx_recordfn", cached_so)
    except Exception as e:
        _log_cmake_failure("load", e, "")
        _record_jit_failure(tag, e, reason=_CMAKE_TIER_NAME)
        return None

    _clear_jit_failure(tag)
    shutil.rmtree(build_dir, ignore_errors=True)

    _safe_log("log", f"cmake-built roctx_recordfn.so for {tag}")
    return mod


def _try_jit_build(tag: str) -> Optional[types.ModuleType]:
    """Fallback build via torch.utils.cpp_extension.load.

    Skipped when it would need ninja.
    """
    if not _SO_SOURCE.exists():
        _safe_log("log", f"source not found at {_SO_SOURCE}; cannot JIT-compile")
        return None

    prior = _previous_jit_failure(tag)
    if prior is not None:
        _safe_log(
            "log",
            f"skipping JIT build (prior failure cached for tag {tag}): {prior}",
        )
        return None

    try:
        import torch.utils.cpp_extension as cpp_ext
    except Exception as e:
        _log_cppext_failure(e)
        _record_jit_failure(tag, e, reason=_CPPEXT_TIER_NAME)
        return None

    # Profile Mode Dependency Policy: never pull ninja.
    if not _jit_compile_viable(cpp_ext):
        err = RuntimeError(
            "the running PyTorch's torch.utils.cpp_extension.load "
            "requires the ninja build backend, which rocprof-compute "
            "does not depend on; the cpp_extension tier is skipped on "
            "this host (the cmake tier is the supported alternative)"
        )
        _log_cppext_failure(err)
        _record_jit_failure(tag, err, reason=_CPPEXT_TIER_NAME)
        return None

    rocm_path = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    extra_include_paths = [str(rocm_path / "include")]
    extra_ldflags = [
        f"-L{rocm_path / 'lib'}",
        "-lrocprofiler-sdk-roctx",
    ]
    # No feature probe here, so the .so uses TEST_INFO_2 (dormant in
    # production; only collides with PyTorch's own gtest binaries).
    extra_cflags = ["-O2", "-fvisibility=hidden", "-Wno-deprecated-declarations"]

    build_dir = _jit_cache_dir() / f"build-{tag}"
    build_dir.mkdir(parents=True, exist_ok=True)

    # use_ninja=False is removed on newer PyTorch; retry without it on TypeError.
    load_kwargs = dict(
        name="roctx_recordfn",
        sources=[str(_SO_SOURCE)],
        extra_include_paths=extra_include_paths,
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        build_directory=str(build_dir),
        with_cuda=False,
        verbose=False,
        use_ninja=False,
    )
    try:
        mod = cpp_ext.load(**load_kwargs)
    except TypeError as type_err:
        if "use_ninja" not in str(type_err):
            _log_cppext_failure(type_err)
            _record_jit_failure(tag, type_err, reason=_CPPEXT_TIER_NAME)
            return None
        load_kwargs.pop("use_ninja", None)
        try:
            mod = cpp_ext.load(**load_kwargs)
        except Exception as e:
            _log_cppext_failure(e)
            _record_jit_failure(tag, e, reason=_CPPEXT_TIER_NAME)
            return None
    except Exception as e:
        _log_cppext_failure(e)
        _record_jit_failure(tag, e, reason=_CPPEXT_TIER_NAME)
        return None

    _install_cached_so(
        build_dir / "roctx_recordfn.so",
        _jit_cache_dir() / f"roctx_recordfn-{tag}.so",
    )
    _clear_jit_failure(tag)

    _safe_log("log", f"JIT-compiled roctx_recordfn.so for {tag}")
    return mod


def load(force_python_fallback: bool = False) -> Optional[types.ModuleType]:
    """Return the roctx_recordfn module, or None for the Python fallback.

    ROCPROFCOMPUTE_ROCTX_RECORDFN_SO=<path> pins to that .so; no build-tier
    fallback on miss. ROCPROFCOMPUTE_REBUILD_ROCTX=1 skips cached tiers.
    """
    global _LAST_LOADED_TIER
    _LAST_LOAD_DIAGNOSTICS.clear()
    _LAST_LOADED_TIER = None

    if force_python_fallback:
        _safe_log("log", "force_python_fallback=True; declining to load .so")
        return None

    tag = compute_tag()
    if tag is None:
        _safe_log("warning", "torch not importable; using Python-only injector")
        return None

    # Explicit pin: no build-tier fallback on miss.
    if os.environ.get(_EXPLICIT_SO_ENV_VAR):
        explicit_mod = _try_explicit_so(tag)
        if explicit_mod is not None:
            _LAST_LOADED_TIER = TIER_EXPLICIT
        return explicit_mod

    if os.environ.get(_REBUILD_ENV_VAR) == "1":
        _safe_log(
            "warning",
            f"{_REBUILD_ENV_VAR}=1: bypassing prebuilt and JIT-cached "
            f".so, forcing fresh build for tag {tag}",
        )
        _clear_jit_failure(tag)
        for tier_name, step in (
            (TIER_CMAKE_BUILD, _try_cmake_build),
            (TIER_CPP_EXTENSION, _try_jit_build),
        ):
            mod = step(tag)
            if mod is not None:
                _LAST_LOADED_TIER = tier_name
                return mod
        return None

    for tier_name, step in (
        (TIER_PREBUILT, _try_prebuilt),
        (TIER_JIT_CACHED, _try_jit_cached),
        (TIER_CMAKE_BUILD, _try_cmake_build),
        (TIER_CPP_EXTENSION, _try_jit_build),
    ):
        mod = step(tag)
        if mod is not None:
            _LAST_LOADED_TIER = tier_name
            return mod

    _safe_log(
        "log",
        "no roctx_recordfn .so available; using Python-only injector",
    )
    return None
