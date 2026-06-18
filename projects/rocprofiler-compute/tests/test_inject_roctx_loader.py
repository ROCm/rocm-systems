# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ``utils.inject_roctx_loader``.

The tests stub the loader to verify tier ordering, cache fingerprint
stability, and the ``ROCPROFCOMPUTE_REBUILD_ROCTX`` override across the
prebuilt and JIT tiers. End-to-end builds are exercised in
``test_torch_trace_coverage``.
"""

import importlib
import sys

import common  # noqa: F401
import pytest

from utils import inject_roctx_loader

_FAKE_TAG = "py3.12_torch2.9_src000000000000"


# ---------------------------------------------------------------------------
# compute_tag / fingerprint
# ---------------------------------------------------------------------------


def test_compute_tag_returns_well_formed_string():
    """The cache tag follows ``py<X>.<Y>_torch<ver>_src<12-hex>``."""
    tag = inject_roctx_loader.compute_tag()
    if tag is None:
        pytest.skip("torch not importable")
    parts = tag.split("_")
    assert any(p.startswith("py") for p in parts)
    assert any(p.startswith("torch") for p in parts)
    src_components = [p for p in parts if p.startswith("src")]
    assert len(src_components) == 1, (
        f"expected exactly one '_src...' component in tag {tag!r}"
    )
    src_value = src_components[0][len("src") :]
    assert src_value == "missing" or (
        len(src_value) == 12 and all(c in "0123456789abcdef" for c in src_value)
    ), f"unexpected src component {src_value!r} in tag {tag!r}"


def test_compute_tag_is_stable_across_calls():
    """``compute_tag()`` returns the same value for repeated calls in one process."""
    assert inject_roctx_loader.compute_tag() == inject_roctx_loader.compute_tag()


def test_source_fingerprint_changes_when_inputs_change(tmp_path, monkeypatch):
    """A single-byte edit to any input changes the fingerprint."""
    cpp = tmp_path / "roctx_recordfn.cpp"
    cmake = tmp_path / "CMakeLists.txt"
    cpp.write_text("// fingerprint test source\n")
    cmake.write_text("# fingerprint test buildfile\n")
    monkeypatch.setattr(
        inject_roctx_loader,
        "_FINGERPRINT_INPUTS",
        (cpp, cmake),
    )

    baseline = inject_roctx_loader._source_fingerprint()
    assert len(baseline) == 12

    for input_path in (cpp, cmake):
        original = input_path.read_bytes()
        input_path.write_bytes(original + b"\n# mutation\n")
        mutated = inject_roctx_loader._source_fingerprint()
        assert mutated != baseline, (
            f"editing {input_path.name} did not change the fingerprint"
        )
        input_path.write_bytes(original)
        assert inject_roctx_loader._source_fingerprint() == baseline


def test_source_fingerprint_excludes_tool_version_file():
    """The tool ``VERSION`` file is not a fingerprint input."""
    for path in inject_roctx_loader._FINGERPRINT_INPUTS:
        assert path.name != "VERSION", (
            f"VERSION should not be in _FINGERPRINT_INPUTS; saw {path}"
        )


def test_source_fingerprint_is_missing_sentinel_when_no_inputs_readable(
    tmp_path,
    monkeypatch,
):
    """When no inputs are readable, the fingerprint is the ``"missing"`` sentinel."""
    monkeypatch.setattr(
        inject_roctx_loader,
        "_FINGERPRINT_INPUTS",
        (tmp_path / "does_not_exist.cpp",),
    )
    assert inject_roctx_loader._source_fingerprint() == "missing"


def test_source_fingerprint_is_length_delimited(tmp_path, monkeypatch):
    """Reshuffling bytes across input boundaries changes the fingerprint."""
    a = tmp_path / "a.cpp"
    b = tmp_path / "b.txt"
    monkeypatch.setattr(inject_roctx_loader, "_FINGERPRINT_INPUTS", (a, b))

    a.write_bytes(b"AB")
    b.write_bytes(b"C")
    fp1 = inject_roctx_loader._source_fingerprint()
    a.write_bytes(b"A")
    b.write_bytes(b"BC")
    fp2 = inject_roctx_loader._source_fingerprint()
    assert fp1 != fp2, "fingerprint collided across an input boundary"


def test_cmake_and_runtime_compute_identical_fingerprint():
    """The loader and CMake compute the same source fingerprint."""
    import subprocess

    cmake_dir = inject_roctx_loader._SO_SOURCE_DIR
    assert cmake_dir.is_dir(), f"missing cmake source dir at {cmake_dir}"

    snippet = (
        "import sys, pathlib; "
        f"sys.path.insert(0, str(pathlib.Path('{cmake_dir}/../..').resolve())); "
        "from utils.inject_roctx_loader import _source_fingerprint; "
        "print(_source_fingerprint())"
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        check=True,
        capture_output=True,
        text=True,
    )
    cmake_side = result.stdout.strip()
    runtime_side = inject_roctx_loader._source_fingerprint()
    assert cmake_side == runtime_side, (
        f"install-time fingerprint {cmake_side!r} != runtime {runtime_side!r}"
    )


# ---------------------------------------------------------------------------
# Source / buildfile hygiene
# ---------------------------------------------------------------------------


def test_roctx_recordfn_source_avoids_torch_umbrella_headers():
    """The source must not include ``<torch/{extension,all,torch}.h>``."""
    cpp_path = inject_roctx_loader._SO_SOURCE
    assert cpp_path.is_file(), f"missing C++ source at {cpp_path}"
    active_lines = [
        line
        for line in cpp_path.read_text().splitlines()
        if not line.lstrip().startswith("//")
    ]
    active_src = "\n".join(active_lines)

    forbidden = (
        "<torch/extension.h>",
        "<torch/all.h>",
        "<torch/torch.h>",
    )
    for header in forbidden:
        directive = f"#include {header}"
        assert directive not in active_src, f"must not include {header}"


def test_roctx_recordfn_source_uses_narrow_includes():
    """The source includes only the required narrow PyTorch headers."""
    cpp_path = inject_roctx_loader._SO_SOURCE
    src = cpp_path.read_text()

    required = (
        "#include <ATen/record_function.h>",
        "#include <c10/util/ThreadLocalDebugInfo.h>",
        "#include <pybind11/pybind11.h>",
        "#include <pybind11/stl.h>",
    )
    for directive in required:
        assert directive in src, f"must include {directive}"


def test_cmake_buildfile_does_not_override_output_name():
    """``CMakeLists.txt`` must not set ``OUTPUT_NAME``."""
    cmake_path = inject_roctx_loader._SO_BUILDFILE
    assert cmake_path.is_file(), f"missing CMakeLists.txt at {cmake_path}"
    active_lines = [
        line
        for line in cmake_path.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    active_src = "\n".join(active_lines)

    assert "OUTPUT_NAME" not in active_src, "must not set OUTPUT_NAME"


def test_cmake_buildfile_strips_lib_prefix():
    """``CMakeLists.txt`` sets ``PREFIX ""`` so the artifact is not ``lib``-prefixed."""
    import re

    cmake_path = inject_roctx_loader._SO_BUILDFILE
    active_lines = [
        line
        for line in cmake_path.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    active_src = "\n".join(active_lines)

    assert re.search(r'PREFIX\s+""', active_src), 'must set PREFIX ""'


def test_loader_and_cmake_agree_on_artifact_filename_shape():
    """The loader and CMake produce identical artifact filenames."""
    import inspect

    src = inspect.getsource(inject_roctx_loader._try_jit)
    assert 'f"roctx_recordfn-{tag}.so"' in src, (
        "artifact filename has changed; update this test accordingly"
    )


def test_loader_source_never_recommends_installing_ninja():
    """The loader source must not recommend installing ninja."""
    import inspect as _stdlib_inspect

    src = _stdlib_inspect.getsource(inject_roctx_loader).lower()
    forbidden = (
        "pip install ninja",
        "apt install ninja",
        "apt-get install ninja",
        "install ninja",
        "add ninja to requirements",
        "ninja>=",
        "ninja==",
    )
    for token in forbidden:
        assert token not in src, f"loader source must not contain {token!r}"


# ---------------------------------------------------------------------------
# Tier-name surface
# ---------------------------------------------------------------------------


def test_c_tier_names_matches_the_tier_ladder():
    """``C_TIER_NAMES`` enumerates exactly the prebuilt and JIT tiers."""
    assert inject_roctx_loader.C_TIER_NAMES == frozenset((
        inject_roctx_loader.TIER_PREBUILT,
        inject_roctx_loader.TIER_JIT,
    ))


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def test_force_python_fallback_returns_none():
    assert inject_roctx_loader.load(force_python_fallback=True) is None


def test_install_cached_so_overwrites_stale_artifact(tmp_path):
    """``_install_cached_so`` overwrites an existing cached ``.so``."""
    src = tmp_path / "build" / "roctx_recordfn.so"
    src.parent.mkdir()
    src.write_bytes(b"new content")
    cached = tmp_path / "cache" / "roctx_recordfn-tag.so"
    cached.parent.mkdir()
    cached.write_bytes(b"STALE content -- must be overwritten")

    inject_roctx_loader._install_cached_so(src, cached)
    assert cached.read_bytes() == b"new content", (
        "cache copy did not overwrite the existing cached .so"
    )


def test_install_cached_so_is_a_noop_when_src_missing(tmp_path):
    """``_install_cached_so`` is a no-op when the source ``.so`` is missing."""
    src = tmp_path / "build" / "does_not_exist.so"
    cached = tmp_path / "cache" / "roctx_recordfn-tag.so"
    cached.parent.mkdir()
    cached.write_bytes(b"good content")

    inject_roctx_loader._install_cached_so(src, cached)
    assert cached.read_bytes() == b"good content", (
        "missing src must leave cached .so untouched"
    )


def test_jit_cache_dir_is_creatable(monkeypatch, tmp_path):
    """``_jit_cache_dir`` creates the directory on first call and is idempotent."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    d = inject_roctx_loader._jit_cache_dir()
    assert d.exists() and d.is_dir()
    assert inject_roctx_loader._jit_cache_dir() == d


def test_cmake_executable_honors_env_var_then_falls_back(monkeypatch):
    """``$CMAKE`` takes precedence over ``PATH``."""
    seen = []

    def fake_which(name):
        seen.append(name)
        return f"/fake/bin/{name}"

    monkeypatch.setattr(inject_roctx_loader.shutil, "which", fake_which)
    monkeypatch.setenv("CMAKE", "my-custom-cmake")
    assert inject_roctx_loader._cmake_executable() == "/fake/bin/my-custom-cmake"
    assert seen[-1] == "my-custom-cmake"

    monkeypatch.delenv("CMAKE", raising=False)
    assert inject_roctx_loader._cmake_executable() == "/fake/bin/cmake"
    assert seen[-1] == "cmake"


def test_no_prebuilt_returns_none_for_unknown_tag(monkeypatch):
    """``_try_prebuilt`` returns ``None`` when no candidate matches the tag."""
    monkeypatch.setattr(
        inject_roctx_loader,
        "_install_tree_prebuilt_candidates",
        lambda _tag: [],
    )
    assert inject_roctx_loader._try_prebuilt("py3.10_torch2.9") is None


# ---------------------------------------------------------------------------
# JIT failure marker
# ---------------------------------------------------------------------------


def test_jit_failure_marker_round_trip(monkeypatch, tmp_path):
    """``_previous_jit_failure`` reads the reason from ``_record_jit_failure``."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    tag = _FAKE_TAG

    assert inject_roctx_loader._previous_jit_failure(tag) is None
    inject_roctx_loader._record_jit_failure(tag, RuntimeError("ninja missing"))
    recorded = inject_roctx_loader._previous_jit_failure(tag)
    assert recorded is not None
    assert "ninja missing" in recorded
    assert "RuntimeError" in recorded


def test_jit_failure_marker_cleared_on_demand(monkeypatch, tmp_path):
    """_clear_jit_failure() must remove the marker and be idempotent."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    tag = _FAKE_TAG

    inject_roctx_loader._record_jit_failure(tag, RuntimeError("x"))
    assert inject_roctx_loader._previous_jit_failure(tag) is not None
    inject_roctx_loader._clear_jit_failure(tag)
    assert inject_roctx_loader._previous_jit_failure(tag) is None
    inject_roctx_loader._clear_jit_failure(tag)


# ---------------------------------------------------------------------------
# _try_jit: cache-hit path
# ---------------------------------------------------------------------------


def test_try_jit_returns_cache_hit_without_invoking_cmake(monkeypatch, tmp_path):
    """An on-disk cached ``.so`` is loaded directly, no cmake invocation."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cache_dir = inject_roctx_loader._jit_cache_dir()
    cached_so = cache_dir / f"roctx_recordfn-{_FAKE_TAG}.so"
    cached_so.write_bytes(b"stub-so")

    sentinel = object()
    monkeypatch.setattr(
        inject_roctx_loader,
        "_import_module_from_path",
        lambda _n, _p: sentinel,
    )

    def fail_subprocess(*_a, **_k):
        pytest.fail("subprocess.run must not fire on cache hit")

    monkeypatch.setattr(inject_roctx_loader.subprocess, "run", fail_subprocess)

    assert inject_roctx_loader._try_jit(_FAKE_TAG) is sentinel


def test_try_jit_force_rebuild_bypasses_cache(monkeypatch, tmp_path):
    """``force_rebuild=True`` ignores the cache and reaches the build path."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cache_dir = inject_roctx_loader._jit_cache_dir()
    cached_so = cache_dir / f"roctx_recordfn-{_FAKE_TAG}.so"
    cached_so.write_bytes(b"stale")

    def must_not_import(_n, _p):
        pytest.fail("cache hit must not be consulted under force_rebuild")

    monkeypatch.setattr(
        inject_roctx_loader, "_import_module_from_path", must_not_import
    )
    monkeypatch.setattr(inject_roctx_loader, "_SO_SOURCE", tmp_path / "missing.cpp")

    assert inject_roctx_loader._try_jit(_FAKE_TAG, force_rebuild=True) is None


# ---------------------------------------------------------------------------
# _try_jit: cmake build path
# ---------------------------------------------------------------------------


class _StubCompleted:
    """Minimal ``subprocess.CompletedProcess`` stand-in."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _set_so_inputs_present(monkeypatch, tmp_path):
    """Point the loader at synthetic source files."""
    src_dir = tmp_path / "roctx_recordfn"
    src_dir.mkdir(parents=True, exist_ok=True)
    cpp = src_dir / "roctx_recordfn.cpp"
    cml = src_dir / "CMakeLists.txt"
    cpp.write_text("// stub\n")
    cml.write_text("# stub\n")
    monkeypatch.setattr(inject_roctx_loader, "_SO_SOURCE_DIR", src_dir)
    monkeypatch.setattr(inject_roctx_loader, "_SO_SOURCE", cpp)
    monkeypatch.setattr(inject_roctx_loader, "_SO_BUILDFILE", cml)
    return src_dir


def test_try_jit_skips_when_sources_missing(monkeypatch, tmp_path):
    """``_try_jit`` returns ``None`` when sources are missing (no cache hit)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(inject_roctx_loader, "_SO_SOURCE", tmp_path / "nonexistent.cpp")
    monkeypatch.setattr(
        inject_roctx_loader, "_SO_BUILDFILE", tmp_path / "nonexistent.txt"
    )

    def fail_subprocess(*_a, **_k):
        pytest.fail("subprocess.run must not be called when sources are missing")

    monkeypatch.setattr(inject_roctx_loader.subprocess, "run", fail_subprocess)
    assert inject_roctx_loader._try_jit(_FAKE_TAG) is None


def test_try_jit_skips_when_cmake_not_on_path(monkeypatch, tmp_path):
    """Absence of cmake on ``PATH`` does not write a failure marker."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(inject_roctx_loader, "_cmake_executable", lambda: None)

    def fail_subprocess(*_a, **_k):
        pytest.fail("subprocess.run must not be called when cmake is absent")

    monkeypatch.setattr(inject_roctx_loader.subprocess, "run", fail_subprocess)
    assert inject_roctx_loader._try_jit(_FAKE_TAG) is None
    assert inject_roctx_loader._previous_jit_failure(_FAKE_TAG) is None


def test_try_jit_short_circuits_on_prior_failure(monkeypatch, tmp_path):
    """A prior failure marker vetoes ``_try_jit``."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(inject_roctx_loader, "_cmake_executable", lambda: "/fake/cmake")
    inject_roctx_loader._record_jit_failure(
        _FAKE_TAG,
        RuntimeError("earlier failure"),
    )

    def fail_subprocess(*_a, **_k):
        pytest.fail("subprocess.run must not fire when marker is present")

    monkeypatch.setattr(inject_roctx_loader.subprocess, "run", fail_subprocess)
    assert inject_roctx_loader._try_jit(_FAKE_TAG) is None


def test_try_jit_passes_runtime_python_to_cmake(monkeypatch, tmp_path):
    """``_try_jit`` passes ``-DTORCH_TRACE_PYTHON=sys.executable`` to cmake."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(inject_roctx_loader, "_cmake_executable", lambda: "/fake/cmake")

    invocations = []

    def fake_run(argv, **_kw):
        invocations.append(list(argv))
        return _StubCompleted(returncode=0)

    monkeypatch.setattr(inject_roctx_loader.subprocess, "run", fake_run)
    cache_dir = inject_roctx_loader._jit_cache_dir()
    build_dir = cache_dir / f"cmake-build-{_FAKE_TAG}"
    produced = build_dir / f"roctx_recordfn-{_FAKE_TAG}.so"
    produced.parent.mkdir(parents=True, exist_ok=True)
    produced.write_bytes(b"stub-so")
    monkeypatch.setattr(
        inject_roctx_loader,
        "_import_module_from_path",
        lambda _n, _p: object(),
    )

    inject_roctx_loader._try_jit(_FAKE_TAG)

    assert len(invocations) == 2, f"expected two cmake invocations, saw {invocations!r}"
    configure_argv = invocations[0]
    runtime_python_flag = f"-DTORCH_TRACE_PYTHON={sys.executable}"
    assert runtime_python_flag in configure_argv, (
        f"-DTORCH_TRACE_PYTHON must equal sys.executable; saw {configure_argv!r}"
    )
    build_argv = invocations[1]
    assert build_argv[1] == "--build", (
        f"second invocation must be `cmake --build`, saw {build_argv!r}"
    )


def test_try_jit_records_failure_marker_on_configure_failure(monkeypatch, tmp_path):
    """A non-zero configure return code writes a JIT failure marker."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(inject_roctx_loader, "_cmake_executable", lambda: "/fake/cmake")

    def fake_run(argv, **_kw):
        return _StubCompleted(
            returncode=1,
            stderr="CMake Error: Could not find Torch (missing: TORCH_DIR)\n",
        )

    monkeypatch.setattr(inject_roctx_loader.subprocess, "run", fake_run)
    assert inject_roctx_loader._try_jit(_FAKE_TAG) is None
    marker = inject_roctx_loader._previous_jit_failure(_FAKE_TAG)
    assert marker is not None
    assert inject_roctx_loader.TIER_JIT in marker
    assert "Could not find Torch" in marker, (
        f"marker must preserve stderr tail; saw {marker!r}"
    )


def test_try_jit_records_failure_marker_on_build_failure(monkeypatch, tmp_path):
    """A non-zero build return code writes a JIT failure marker."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(inject_roctx_loader, "_cmake_executable", lambda: "/fake/cmake")

    call_count = [0]

    def fake_run(argv, **_kw):
        call_count[0] += 1
        if call_count[0] == 1:
            return _StubCompleted(returncode=0)
        return _StubCompleted(
            returncode=2,
            stderr="error: undefined reference to `roctxRangePushA'\n",
        )

    monkeypatch.setattr(inject_roctx_loader.subprocess, "run", fake_run)
    assert inject_roctx_loader._try_jit(_FAKE_TAG) is None
    marker = inject_roctx_loader._previous_jit_failure(_FAKE_TAG)
    assert marker is not None
    assert inject_roctx_loader.TIER_JIT in marker
    assert "undefined reference" in marker


def test_try_jit_records_failure_marker_when_so_missing(monkeypatch, tmp_path):
    """A successful build with no ``.so`` output writes a failure marker."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(inject_roctx_loader, "_cmake_executable", lambda: "/fake/cmake")
    monkeypatch.setattr(
        inject_roctx_loader.subprocess,
        "run",
        lambda *a, **k: _StubCompleted(returncode=0),
    )
    assert inject_roctx_loader._try_jit(_FAKE_TAG) is None
    marker = inject_roctx_loader._previous_jit_failure(_FAKE_TAG)
    assert marker is not None
    assert inject_roctx_loader.TIER_JIT in marker
    assert "missing" in marker


def test_try_jit_cleans_build_dir_on_success(monkeypatch, tmp_path):
    """The build directory is removed after a successful build."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(inject_roctx_loader, "_cmake_executable", lambda: "/fake/cmake")
    monkeypatch.setattr(
        inject_roctx_loader.subprocess,
        "run",
        lambda *a, **k: _StubCompleted(returncode=0),
    )

    cache_dir = inject_roctx_loader._jit_cache_dir()
    build_dir = cache_dir / f"cmake-build-{_FAKE_TAG}"
    produced = build_dir / f"roctx_recordfn-{_FAKE_TAG}.so"
    produced.parent.mkdir(parents=True, exist_ok=True)
    produced.write_bytes(b"stub-so")
    leftover = build_dir / "CMakeFiles"
    leftover.mkdir(exist_ok=True)
    (leftover / "stale.o").write_bytes(b"x")

    monkeypatch.setattr(
        inject_roctx_loader,
        "_import_module_from_path",
        lambda _n, _p: object(),
    )

    result = inject_roctx_loader._try_jit(_FAKE_TAG)
    assert result is not None
    assert not build_dir.exists(), (
        "build dir must be removed on success to bound cache disk usage"
    )
    cached_so = cache_dir / f"roctx_recordfn-{_FAKE_TAG}.so"
    assert cached_so.exists() and cached_so.read_bytes() == b"stub-so", (
        "produced .so must have been copied to the cache before cleanup"
    )


def test_try_jit_keeps_build_dir_on_failure(monkeypatch, tmp_path):
    """Preserve the build dir on failure for post-mortem inspection."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(inject_roctx_loader, "_cmake_executable", lambda: "/fake/cmake")
    monkeypatch.setattr(
        inject_roctx_loader.subprocess,
        "run",
        lambda *a, **k: _StubCompleted(returncode=1, stderr="boom\n"),
    )

    assert inject_roctx_loader._try_jit(_FAKE_TAG) is None
    build_dir = inject_roctx_loader._jit_cache_dir() / f"cmake-build-{_FAKE_TAG}"
    assert build_dir.exists(), "build dir must survive a failure"


# ---------------------------------------------------------------------------
# explain_cmake_failure classification
# ---------------------------------------------------------------------------


def test_explain_cmake_failure_classifies_torch_not_found():
    reason, hint = inject_roctx_loader._explain_cmake_failure(
        "configure",
        RuntimeError("rc=1"),
        "CMake Error: Could not find Torch (missing: TORCH_DIR)\n",
    )
    assert "libtorch" in reason.lower()
    assert "torch" in hint.lower()


def test_explain_cmake_failure_classifies_roctx_not_found():
    reason, hint = inject_roctx_loader._explain_cmake_failure(
        "configure",
        RuntimeError("rc=1"),
        "find_library failed: rocprofiler-sdk-roctx not found\n",
    )
    assert "roctx" in reason.lower() or "rocprofiler-sdk-roctx" in reason.lower()
    assert "rocm_path" in hint.lower() or "/opt/rocm" in hint.lower()


def test_explain_cmake_failure_classifies_missing_cxx_compiler():
    reason, hint = inject_roctx_loader._explain_cmake_failure(
        "configure",
        RuntimeError("rc=1"),
        "No CMAKE_CXX_COMPILER could be found.\n",
    )
    assert "compiler" in reason.lower()
    assert "g++" in hint.lower() or "clang" in hint.lower()


def test_explain_cmake_failure_never_recommends_installing_ninja():
    """cmake-tier hints must not recommend installing ninja."""
    samples = [
        (RuntimeError("rc=1"), "CMake Error: ninja not found"),
        (RuntimeError("rc=1"), "Could not find Torch"),
        (RuntimeError("rc=1"), ""),
    ]
    for err, stderr in samples:
        reason, hint = inject_roctx_loader._explain_cmake_failure(
            "configure",
            err,
            stderr,
        )
        forbidden = ("install ninja", "pip install ninja", "apt install ninja")
        joined = (reason + " " + hint).lower()
        for token in forbidden:
            assert token not in joined, (
                f"cmake-tier hint must not recommend installing ninja "
                f"(found {token!r} in: {hint!r})"
            )


# ---------------------------------------------------------------------------
# load(): tier ordering and REBUILD override
# ---------------------------------------------------------------------------


def test_rebuild_env_var_clears_failure_marker(monkeypatch, tmp_path):
    """``ROCPROFCOMPUTE_REBUILD_ROCTX=1`` clears any cached failure marker."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv(inject_roctx_loader._REBUILD_ENV_VAR, "1")

    inject_roctx_loader._record_jit_failure(_FAKE_TAG, RuntimeError("stale failure"))
    assert inject_roctx_loader._previous_jit_failure(_FAKE_TAG) is not None

    sentinel = object()
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_prebuilt",
        lambda _t: pytest.fail("prebuilt must be skipped under REBUILD"),
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_jit",
        lambda _t, force_rebuild=False: sentinel,
    )

    result = inject_roctx_loader.load()
    assert result is sentinel
    assert inject_roctx_loader._previous_jit_failure(_FAKE_TAG) is None, (
        "REBUILD must clear the negative cache before the forced rebuild"
    )


def test_rebuild_env_var_skips_prebuilt_and_forces_jit_rebuild(monkeypatch):
    """REBUILD=1 routes directly to the JIT build, skipping prebuilt and cache."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    calls = []
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_prebuilt",
        lambda tag: calls.append(("prebuilt", tag)) or None,
    )
    sentinel = object()

    def _jit(tag, force_rebuild=False):
        calls.append(("jit", tag, force_rebuild))
        return sentinel

    monkeypatch.setattr(inject_roctx_loader, "_try_jit", _jit)
    monkeypatch.setenv(inject_roctx_loader._REBUILD_ENV_VAR, "1")

    result = inject_roctx_loader.load()
    assert result is sentinel
    assert [c[0] for c in calls] == ["jit"], (
        f"expected only _try_jit to fire under REBUILD; saw {calls!r}"
    )
    assert calls[0][1] == _FAKE_TAG, "tag must propagate to the build step"
    assert calls[0][2] is True, "REBUILD must pass force_rebuild=True"


def test_rebuild_env_var_returns_none_when_build_fails(monkeypatch):
    """Under REBUILD, JIT failing yields ``None`` with no fallback."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_prebuilt",
        lambda _t: pytest.fail("_try_prebuilt called despite rebuild env var"),
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_jit",
        lambda _t, force_rebuild=False: None,
    )
    monkeypatch.setenv(inject_roctx_loader._REBUILD_ENV_VAR, "1")
    assert inject_roctx_loader.load() is None


def test_default_load_path_still_tries_prebuilt_first(monkeypatch):
    """The prebuilt tier short-circuits the default load path."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_prebuilt",
        lambda tag: calls.append("prebuilt") or sentinel,
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_jit",
        lambda tag, force_rebuild=False: calls.append("jit") or None,
    )
    assert inject_roctx_loader.load() is sentinel
    assert calls == ["prebuilt"], (
        f"expected prebuilt to short-circuit before jit; saw {calls!r}"
    )


def test_default_load_path_walks_prebuilt_then_jit(monkeypatch):
    """Tiers are tried in order: prebuilt, then jit."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    calls = []
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_prebuilt",
        lambda tag: calls.append("prebuilt") or None,
    )
    sentinel = object()
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_jit",
        lambda tag, force_rebuild=False: calls.append("jit") or sentinel,
    )
    assert inject_roctx_loader.load() is sentinel
    assert calls == ["prebuilt", "jit"], (
        f"expected order prebuilt -> jit, saw {calls!r}"
    )


def test_load_does_not_raise_when_torch_missing(monkeypatch):
    """``load()`` returns ``None`` when ``torch`` is not importable."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: None)
    assert inject_roctx_loader.load() is None


def test_load_returns_module_or_none_no_raise():
    """``load()`` returns a module exposing the documented API, or ``None``."""
    mod = inject_roctx_loader.load()
    if mod is not None:
        for sym in (
            "install",
            "uninstall",
            "is_installed",
            "push_user_scope",
            "pop_user_scope",
            "dump_stats",
        ):
            assert hasattr(mod, sym), f"loaded module is missing {sym}"


# ---------------------------------------------------------------------------
# loaded_tier / consume_diagnostics / format_load_diagnostic_trail
# ---------------------------------------------------------------------------


def test_loaded_tier_records_successful_step(monkeypatch):
    """``loaded_tier()`` reports the name of the tier that returned a module."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    sentinel = object()
    monkeypatch.setattr(inject_roctx_loader, "_try_prebuilt", lambda _t: None)
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_jit",
        lambda _t, force_rebuild=False: sentinel,
    )
    assert inject_roctx_loader.load() is sentinel
    assert inject_roctx_loader.loaded_tier() == inject_roctx_loader.TIER_JIT
    assert inject_roctx_loader.TIER_JIT in inject_roctx_loader.C_TIER_NAMES


def test_loaded_tier_is_none_when_all_tiers_miss(monkeypatch):
    """``loaded_tier()`` returns ``None`` when every tier returns ``None``."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    monkeypatch.setattr(inject_roctx_loader, "_try_prebuilt", lambda _t: None)
    monkeypatch.setattr(
        inject_roctx_loader, "_try_jit", lambda _t, force_rebuild=False: None
    )
    assert inject_roctx_loader.load() is None
    assert inject_roctx_loader.loaded_tier() is None


def test_load_resets_diagnostics_on_entry(monkeypatch):
    """``load()`` clears prior diagnostics on entry."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    inject_roctx_loader._LAST_LOAD_DIAGNOSTICS.clear()
    inject_roctx_loader._LAST_LOAD_DIAGNOSTICS.append(("log", "STALE LINE"))

    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_prebuilt",
        lambda _t: object(),
    )
    monkeypatch.setattr(
        inject_roctx_loader, "_try_jit", lambda _t, force_rebuild=False: None
    )

    inject_roctx_loader.load()
    _tier, diagnostics = inject_roctx_loader.consume_diagnostics()
    for level, msg in diagnostics:
        assert "STALE LINE" not in msg, (
            f"pre-existing diagnostic leaked across load() boundary: ({level}, {msg!r})"
        )


def test_safe_log_tees_into_accumulator(monkeypatch):
    """Each ``_safe_log`` call appends to ``_LAST_LOAD_DIAGNOSTICS``."""
    inject_roctx_loader._LAST_LOAD_DIAGNOSTICS.clear()
    inject_roctx_loader._safe_log("log", "tier A skipped")
    inject_roctx_loader._safe_log("warning", "tier B failed")
    inject_roctx_loader._safe_log("log", "final fallback engaged")

    captured = list(inject_roctx_loader._LAST_LOAD_DIAGNOSTICS)
    assert [lvl for lvl, _ in captured] == ["log", "warning", "log"]
    assert [msg for _, msg in captured] == [
        "tier A skipped",
        "tier B failed",
        "final fallback engaged",
    ]


def test_consume_diagnostics_drains_and_returns_tier(monkeypatch):
    """``consume_diagnostics`` drains the trail and keeps the tier scalar."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    monkeypatch.setattr(inject_roctx_loader, "_try_prebuilt", lambda _t: None)
    monkeypatch.setattr(
        inject_roctx_loader, "_try_jit", lambda _t, force_rebuild=False: object()
    )

    inject_roctx_loader.load()
    tier, trail = inject_roctx_loader.consume_diagnostics()
    assert tier == inject_roctx_loader.TIER_JIT
    assert isinstance(trail, list)

    tier2, trail2 = inject_roctx_loader.consume_diagnostics()
    assert tier2 == tier, "tier scalar must persist across drains"
    assert trail2 == [], "second consume must see no leftover lines"


def test_consume_diagnostics_returns_python_tier_failure_trail(monkeypatch):
    """When every tier misses the trail includes the terminal fallback line."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    monkeypatch.setattr(inject_roctx_loader, "_try_prebuilt", lambda _t: None)
    monkeypatch.setattr(
        inject_roctx_loader, "_try_jit", lambda _t, force_rebuild=False: None
    )

    inject_roctx_loader.load()
    tier, trail = inject_roctx_loader.consume_diagnostics()
    assert tier is None
    joined = " ".join(msg for _, msg in trail).lower()
    assert "python-only injector" in joined or "no roctx_recordfn" in joined, (
        f"terminal-fallback line missing from trail: "
        f"{[(lvl, msg) for lvl, msg in trail]!r}"
    )


def test_format_load_diagnostic_trail_handles_empty():
    """An empty trail renders to an empty string."""
    assert inject_roctx_loader.format_load_diagnostic_trail([]) == ""


def test_format_load_diagnostic_trail_caps_lines():
    """``format_load_diagnostic_trail`` caps output and keeps the latest lines."""
    trail = [("log", f"line {i}") for i in range(100)]
    rendered = inject_roctx_loader.format_load_diagnostic_trail(
        trail,
        max_lines=12,
    )
    lines = rendered.splitlines()
    assert len(lines) == 12, (
        f"expected max_lines=12 to cap output, saw {len(lines)} lines"
    )
    assert "line 99" in rendered, "must keep the trailing (latest) lines"
    assert "line 0" not in rendered, "must drop the leading (oldest) lines"


def test_format_load_diagnostic_trail_includes_level_per_line():
    """Each rendered line carries its level (INFO vs WARNING)."""
    trail = [
        ("log", "skipped tier A"),
        ("warning", "tier B failed"),
    ]
    rendered = inject_roctx_loader.format_load_diagnostic_trail(trail)
    assert "[log]" in rendered
    assert "[warning]" in rendered
    assert "skipped tier A" in rendered
    assert "tier B failed" in rendered


# ---------------------------------------------------------------------------
# Python fallback integration
# ---------------------------------------------------------------------------


def test_python_fallback_path_still_works_without_so(monkeypatch):
    """With the loader returning ``None``, the Python fallback still works."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not importable")
    monkeypatch.setattr(inject_roctx_loader, "load", lambda **kw: None)
    if "utils.inject_roctx" in sys.modules:
        del sys.modules["utils.inject_roctx"]
    inject_roctx = importlib.import_module("utils.inject_roctx")
    try:
        assert inject_roctx.using_c_tier() is False
        assert inject_roctx.dump_recordfn_stats() is None
        inject_roctx._push_scope("py.tier.test", "#1@test:1")
        inject_roctx._pop_scope()
    finally:
        sys.modules.pop("utils.inject_roctx", None)


def test_python_fallback_uses_python_dispatch_sentinel(monkeypatch):
    """When no user frame is available the fallback emits ``python.dispatch:0``."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not importable; utils.inject_roctx module-load exits")
    monkeypatch.setattr(inject_roctx_loader, "load", lambda **kw: None)
    if "utils.inject_roctx" in sys.modules:
        del sys.modules["utils.inject_roctx"]
    inject_roctx = importlib.import_module("utils.inject_roctx")
    try:
        monkeypatch.setattr("inspect.currentframe", lambda: None)
        assert inject_roctx.resolve_user_caller_location() == "python.dispatch:0"
        import inspect as _stdlib_inspect

        src = _stdlib_inspect.getsource(inject_roctx)
        assert "dispatcher:0" not in src, "legacy 'dispatcher:0' sentinel still present"
    finally:
        sys.modules.pop("utils.inject_roctx", None)


def test_import_does_not_apply_global_patches(monkeypatch):
    """Importing ``utils.inject_roctx`` does not patch PyTorch."""
    monkeypatch.setattr(inject_roctx_loader, "load", lambda **kw: None)
    if "utils.inject_roctx" in sys.modules:
        del sys.modules["utils.inject_roctx"]

    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not importable")

    import torch as _torch

    pre = {
        "compile": getattr(_torch, "compile", None),
    }

    inject_roctx = importlib.import_module("utils.inject_roctx")
    try:
        post = {
            "compile": getattr(_torch, "compile", None),
        }
        assert hasattr(inject_roctx, "install_global_wraps")
        assert post["compile"] is pre["compile"], "torch.compile was replaced on import"
    finally:
        sys.modules.pop("utils.inject_roctx", None)
