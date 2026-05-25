# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.inject_roctx_loader and the Python-tier fallback.

No GPU required. The loader's resolution order, tag computation, and the
Python tier are exercised by stubbing the loader. The JIT compile path
itself is covered end-to-end by test_torch_trace_worker_thread.py.

CI: registered with ctest; not in test_categories.yaml so it runs only
    on the full ctest invocation (and on the coverage job). No GPU,
    ctest timeout 300 s.
Coverage: included in the coverage XML.
Why: pins loader contracts (tier order, cache fingerprint, REBUILD
    env var, no-ninja policy) without depending on a real cmake build
    or a GPU, so most loader regressions fail fast in CI.
"""

import importlib
import sys

import common  # noqa: F401  -- adds src/ to sys.path
import pytest

from utils import inject_roctx_loader


def test_compute_tag_returns_well_formed_string():
    """tag is "py<X>.<Y>_torch<ver>_abi<0|1>_src<12-hex>" or None."""
    tag = inject_roctx_loader.compute_tag()
    if tag is None:
        pytest.skip("torch not importable")
    parts = tag.split("_")
    assert any(p.startswith("py") for p in parts)
    assert any(p.startswith("torch") for p in parts)
    assert any(p.startswith("abi") for p in parts)
    # _src{12 hex} guards against stale .so loads after a source edit.
    src_components = [p for p in parts if p.startswith("src")]
    assert len(src_components) == 1, (
        f"expected exactly one '_src...' component in tag {tag!r}"
    )
    src_value = src_components[0][len("src") :]
    # 12-char lowercase hex, or "missing" if no inputs were readable.
    assert src_value == "missing" or (
        len(src_value) == 12 and all(c in "0123456789abcdef" for c in src_value)
    ), f"unexpected src component {src_value!r} in tag {tag!r}"


def test_compute_tag_is_stable_across_calls():
    assert inject_roctx_loader.compute_tag() == inject_roctx_loader.compute_tag()


def test_source_fingerprint_changes_when_inputs_change(tmp_path, monkeypatch):
    """A one-byte edit to any fingerprint input must change the tag,
    so the on-disk cache misses after a source edit.
    """
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
    """VERSION must not be in _FINGERPRINT_INPUTS (would invalidate
    the cache on every release; escape hatch is the REBUILD env var).
    """
    for path in inject_roctx_loader._FINGERPRINT_INPUTS:
        assert path.name != "VERSION", (
            f"VERSION should not be in _FINGERPRINT_INPUTS; saw {path}"
        )


def test_source_fingerprint_is_missing_sentinel_when_no_inputs_readable(
    tmp_path,
    monkeypatch,
):
    """If every fingerprint input is unreadable the tag still has a
    stable shape, so prebuilt resolution by filename keeps working.
    """
    monkeypatch.setattr(
        inject_roctx_loader,
        "_FINGERPRINT_INPUTS",
        (tmp_path / "does_not_exist.cpp",),
    )
    assert inject_roctx_loader._source_fingerprint() == "missing"


def test_source_fingerprint_is_length_delimited(tmp_path, monkeypatch):
    """Moving bytes across an input boundary (``"AB"+"C"`` ->
    ``"A"+"BC"``) must change the hash; the per-file length prefix
    is what prevents that collision.
    """
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


def test_force_python_fallback_returns_none():
    assert inject_roctx_loader.load(force_python_fallback=True) is None


def test_install_cached_so_overwrites_stale_artifact(tmp_path):
    """The cache copy must overwrite an existing cached .so; otherwise
    a forced rebuild would load fresh in-process but leave the stale
    binary for the next run.
    """
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


def test_cmake_and_runtime_compute_identical_fingerprint():
    """Pin the loader and CMake to the same _source_fingerprint().
    Drift here makes a prebuilt .so silently invisible and forces the
    Python fallback.
    """
    import pathlib
    import subprocess

    cmake_dir = (
        pathlib
        .Path(
            inject_roctx_loader.__file__,
        )
        .resolve()
        .parent
        / "roctx_recordfn"
    )
    assert cmake_dir.is_dir(), (
        f"expected the cmake source dir at {cmake_dir}; if the layout "
        f"moved, the CMakeLists.txt sys.path computation must move too"
    )

    # Mirror exactly the inline -c argument from CMakeLists.txt.
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
        f"install-time fingerprint {cmake_side!r} != runtime "
        f"fingerprint {runtime_side!r}; prebuilt .so will never be "
        f"resolved. Check the import path in "
        f"src/utils/roctx_recordfn/CMakeLists.txt."
    )


def test_roctx_recordfn_source_avoids_torch_umbrella_headers():
    """Forbid <torch/extension.h>, <torch/all.h>, and <torch/torch.h>.
    Some ROCm nightly wheels strip the umbrella; only narrow ATen +
    c10 + pybind11 includes are portable.
    """
    from pathlib import Path

    cpp_path = (
        Path(inject_roctx_loader.__file__).resolve().parent
        / "roctx_recordfn"
        / "roctx_recordfn.cpp"
    )
    assert cpp_path.is_file(), f"expected the C++ source at {cpp_path}"
    # Strip line comments so the rationale prose doesn't false-positive.
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
        assert directive not in active_src, (
            f"roctx_recordfn.cpp must not include {header}; use the "
            f"narrow <ATen/...>, <c10/...>, <pybind11/...> includes."
        )


def test_cmake_buildfile_does_not_override_output_name():
    """CMakeLists must not pin OUTPUT_NAME (would strip the tag from
    the artifact filename, making it invisible to the tag-keyed
    runtime probes).
    """
    from pathlib import Path

    cmake_path = (
        Path(inject_roctx_loader.__file__).resolve().parent
        / "roctx_recordfn"
        / "CMakeLists.txt"
    )
    assert cmake_path.is_file(), f"expected CMakeLists.txt at {cmake_path}"
    # Strip line comments so rationale text doesn't false-positive.
    active_lines = [
        line
        for line in cmake_path.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    active_src = "\n".join(active_lines)

    assert "OUTPUT_NAME" not in active_src, (
        "CMakeLists.txt must not set OUTPUT_NAME on roctx_recordfn "
        "(would strip the tag and hide the artifact from the loader)."
    )


def test_cmake_buildfile_strips_lib_prefix():
    """Pin PREFIX="" so the artifact is `roctx_recordfn-${tag}.so`,
    matching the loader's tag-keyed probes (not the CMake default
    `libroctx_recordfn-${tag}.so`).
    """
    import re
    from pathlib import Path

    cmake_path = (
        Path(inject_roctx_loader.__file__).resolve().parent
        / "roctx_recordfn"
        / "CMakeLists.txt"
    )
    active_lines = [
        line
        for line in cmake_path.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    active_src = "\n".join(active_lines)

    # Accept PREFIX "" or PREFIX="" (set_target_properties syntax).
    assert re.search(r'PREFIX\s+""', active_src), (
        'CMakeLists.txt must set PREFIX "" on roctx_recordfn so the '
        "artifact matches the loader's tag-keyed probes."
    )


def test_loader_and_cmake_agree_on_artifact_filename_shape():
    """Pin the filename shape used by both the loader and CMakeLists,
    catching drift where one side rotates and the other doesn't.
    """
    import inspect

    # The loader-side expected-output line lives in _try_cmake_build:
    #     produced = build_dir / "prebuilt" / f"roctx_recordfn-{tag}.so"
    # We don't want to invoke _try_cmake_build (it would spawn cmake);
    # we just inspect its source for the literal expectation. This is
    # fragile to refactors but exactly the kind of bug we want pinned
    # -- if someone changes the filename shape on one side and not
    # the other, the test fails and the message points at the right
    # commit.
    src = inspect.getsource(inject_roctx_loader._try_cmake_build)
    assert 'f"roctx_recordfn-{tag}.so"' in src, (
        "inject_roctx_loader._try_cmake_build no longer references "
        "the literal `roctx_recordfn-{tag}.so` filename shape. If "
        "this name shape moved (e.g. into a module-level constant), "
        "update this test to compare against the new source of "
        "truth. The corresponding cmake-side contract is enforced "
        "by test_cmake_buildfile_does_not_override_output_name and "
        "test_cmake_buildfile_strips_lib_prefix."
    )


def test_roctx_recordfn_source_uses_narrow_includes():
    """Counterpart to the umbrella-header ban: positively pin the
    narrow includes that replace it. Without this, a future refactor
    could pass the negative test by deleting <torch/extension.h>
    while also accidentally dropping <pybind11/pybind11.h>, leaving
    the source uncompilable for the opposite reason."""
    from pathlib import Path

    cpp_path = (
        Path(inject_roctx_loader.__file__).resolve().parent
        / "roctx_recordfn"
        / "roctx_recordfn.cpp"
    )
    src = cpp_path.read_text()

    required = (
        # ATen RecordFunction API.
        "#include <ATen/record_function.h>",
        # c10 ThreadLocalDebugInfo for the autograd-worker chain.
        "#include <c10/util/ThreadLocalDebugInfo.h>",
        # pybind11 module surface (replaces <torch/extension.h>).
        "#include <pybind11/pybind11.h>",
        # STL casters needed for std::vector<std::string> return from
        # stop_capture(); without this pybind11 has no registered
        # caster and the module-init fails at import time.
        "#include <pybind11/stl.h>",
    )
    for directive in required:
        assert directive in src, f"roctx_recordfn.cpp must include {directive}"


def test_loader_source_never_recommends_installing_ninja():
    """Static audit: the loader's source must not recommend installing
    ninja (CONTRIBUTING.md: Profile Mode Dependency Policy). Naming
    ninja in prose is fine; action verbs targeting installation are not.
    """
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
        assert token not in src, (
            f"loader source contains {token!r}; this violates the "
            f"Profile Mode Dependency Policy (CONTRIBUTING.md). "
            f"Recovery messaging must direct users at the prebuilt "
            f".so path, never at expanding the dependency surface."
        )


def test_explain_cppext_failure_never_recommends_installing_ninja():
    """Recovery hints on ninja-flavoured failures must steer the user
    at the cmake tier or the prebuilt path, never at installing ninja.
    """
    samples = [
        RuntimeError("Ninja is required to load C++ extensions"),
        RuntimeError("ninja: command not found"),
        RuntimeError("Could not find ninja on PATH"),
    ]
    for err in samples:
        reason, hint = inject_roctx_loader._explain_cppext_failure(err)
        assert "ninja" in reason.lower(), (
            f"ninja-flavoured failure must be classified as such: {reason!r}"
        )
        forbidden = ("install ninja", "pip install ninja", "apt install ninja")
        joined = (reason + " " + hint).lower()
        for token in forbidden:
            assert token not in joined, (
                f"recovery hint must not recommend installing ninja "
                f"(found {token!r} in: {hint!r})"
            )
        assert "cmake" in hint.lower() or "prebuilt" in hint.lower(), (
            f"ninja-class hint must mention cmake or prebuilt: {hint!r}"
        )


def test_explain_cppext_failure_classifies_compiler_and_header_cases():
    """Non-ninja errors should still classify into actionable buckets
    so the diagnostic line names the actual missing prerequisite."""
    cases = [
        (RuntimeError("g++: command not found"), "compiler"),
        (RuntimeError("clang: not found"), "compiler"),
        (RuntimeError("fatal error: torch/extension.h: No such file"), "libtorch"),
    ]
    for err, expected_keyword in cases:
        reason, hint = inject_roctx_loader._explain_cppext_failure(err)
        assert expected_keyword in reason.lower(), (
            f"{err!r}: reason {reason!r} missing {expected_keyword!r}"
        )
        assert "prebuilt" in hint.lower(), (
            f"{err!r}: hint should mention prebuilt fallback: {hint!r}"
        )


def test_jit_compile_viable_true_when_use_ninja_in_signature(monkeypatch):
    """When cpp_extension.load accepts use_ninja=False the JIT tier
    is viable without ninja (distutils backend).
    """

    class FakeCppExt:
        @staticmethod
        def load(name, sources, use_ninja=False, **kwargs):
            pass

    assert inject_roctx_loader._jit_compile_viable(FakeCppExt) is True


def test_jit_compile_viable_falls_back_to_path_probe(monkeypatch):
    """When use_ninja is absent from the signature, viability falls
    back to ninja being already on PATH; the loader never pulls it in.
    """

    class FakeCppExt:
        @staticmethod
        def load(name, sources, **kwargs):
            pass

    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda _exe: None)
    assert inject_roctx_loader._jit_compile_viable(FakeCppExt) is False

    # ninja-on-PATH: viable (we consume ninja if it's already there).
    monkeypatch.setattr(_shutil, "which", lambda exe: "/usr/bin/ninja")
    assert inject_roctx_loader._jit_compile_viable(FakeCppExt) is True


def test_jit_failure_marker_round_trip(monkeypatch, tmp_path):
    """record -> previous reads back the recorded reason; lets sibling
    processes in a PMC sweep skip repeated doomed build attempts.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    tag = "py3.12_torch2.9_abi1_src000000000000"

    assert inject_roctx_loader._previous_jit_failure(tag) is None
    inject_roctx_loader._record_jit_failure(tag, RuntimeError("ninja missing"))
    recorded = inject_roctx_loader._previous_jit_failure(tag)
    assert recorded is not None
    assert "ninja missing" in recorded
    assert "RuntimeError" in recorded


def test_jit_failure_marker_cleared_on_demand(monkeypatch, tmp_path):
    """_clear_jit_failure() must remove the marker and be idempotent."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    tag = "py3.12_torch2.9_abi1_src000000000000"

    inject_roctx_loader._record_jit_failure(tag, RuntimeError("x"))
    assert inject_roctx_loader._previous_jit_failure(tag) is not None
    inject_roctx_loader._clear_jit_failure(tag)
    assert inject_roctx_loader._previous_jit_failure(tag) is None
    # No-op on already-cleared marker.
    inject_roctx_loader._clear_jit_failure(tag)


def test_rebuild_env_var_clears_failure_marker(monkeypatch, tmp_path):
    """ROCPROFCOMPUTE_REBUILD_ROCTX=1 must clear a stale failure
    marker so the explicit retry isn't vetoed by a prior failure.
    """
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
        "_try_jit_cached",
        lambda _t: pytest.fail("cached must be skipped under REBUILD"),
    )
    # cmake tier returns None so control reaches cpp_extension.
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_cmake_build",
        lambda _t: None,
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_jit_build",
        lambda _t: sentinel,
    )

    result = inject_roctx_loader.load()
    assert result is sentinel
    assert inject_roctx_loader._previous_jit_failure(_FAKE_TAG) is None, (
        "REBUILD env var must clear the negative cache so the forced "
        "rebuild path actually runs"
    )


def test_install_cached_so_is_a_noop_when_src_missing(tmp_path):
    """If the JIT build dir doesn't contain the expected .so (compile
    bailed before linking, build_directory mis-pointed, ...) the cache
    must remain untouched. Otherwise a partial state would replace a
    previously-good cached .so with nothing."""
    src = tmp_path / "build" / "does_not_exist.so"
    cached = tmp_path / "cache" / "roctx_recordfn-tag.so"
    cached.parent.mkdir()
    cached.write_bytes(b"good content")

    inject_roctx_loader._install_cached_so(src, cached)
    assert cached.read_bytes() == b"good content", (
        "missing src must leave cached .so untouched"
    )


# Synthetic tag for routing tests; stubs compute_tag so these tests
# don't need a torch import.
_FAKE_TAG = "py3.12_torch2.9_abi1_src000000000000"


def test_rebuild_env_var_skips_prebuilt_and_cache(monkeypatch):
    """REBUILD=1 routes directly to the build tiers (cmake then
    cpp_extension), skipping prebuilt and JIT cache.
    """
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    calls = []
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_prebuilt",
        lambda tag: calls.append(("prebuilt", tag)) or None,
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_jit_cached",
        lambda tag: calls.append(("cached", tag)) or None,
    )
    sentinel = object()
    # cmake tier succeeds; cpp_extension must NOT be reached.
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_cmake_build",
        lambda tag: calls.append(("cmake_build", tag)) or sentinel,
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_jit_build",
        lambda tag: (
            calls.append(("jit_build", tag))
            or pytest.fail(
                "cpp_extension tier must not be reached when cmake succeeds",
            )
        ),
    )
    monkeypatch.setenv(inject_roctx_loader._REBUILD_ENV_VAR, "1")

    result = inject_roctx_loader.load()
    assert result is sentinel
    assert [c[0] for c in calls] == ["cmake_build"], (
        f"expected only _try_cmake_build to fire under the rebuild "
        f"env var (cmake succeeded), saw {calls!r}"
    )
    assert calls[0][1] == _FAKE_TAG, "tag must propagate to the build step"


def test_rebuild_env_var_falls_through_to_cppext_when_cmake_fails(monkeypatch):
    """Under REBUILD, cmake -> None must fall through to cpp_extension
    (else ninja-but-no-cmake hosts have no forced-rebuild path).
    """
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.setattr(inject_roctx_loader, "_try_prebuilt", lambda _t: None)
    monkeypatch.setattr(inject_roctx_loader, "_try_jit_cached", lambda _t: None)
    monkeypatch.setattr(inject_roctx_loader, "_try_cmake_build", lambda _t: None)
    sentinel = object()
    monkeypatch.setattr(inject_roctx_loader, "_try_jit_build", lambda _t: sentinel)
    monkeypatch.setenv(inject_roctx_loader._REBUILD_ENV_VAR, "1")
    assert inject_roctx_loader.load() is sentinel


def test_rebuild_env_var_returns_none_when_build_fails(monkeypatch):
    """Under REBUILD with both build tiers failing, return None and
    do NOT fall back to prebuilt / cached.
    """
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_prebuilt",
        lambda _t: pytest.fail(
            "_try_prebuilt called despite rebuild env var",
        ),
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_jit_cached",
        lambda _t: pytest.fail(
            "_try_jit_cached called despite rebuild env var",
        ),
    )
    monkeypatch.setattr(inject_roctx_loader, "_try_cmake_build", lambda _t: None)
    monkeypatch.setattr(inject_roctx_loader, "_try_jit_build", lambda _t: None)
    monkeypatch.setenv(inject_roctx_loader._REBUILD_ENV_VAR, "1")
    assert inject_roctx_loader.load() is None


def test_default_load_path_still_tries_prebuilt_first(monkeypatch):
    """Default path: prebuilt short-circuits before any other tier."""
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
        "_try_jit_cached",
        lambda tag: calls.append("cached") or None,
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_cmake_build",
        lambda tag: calls.append("cmake_build") or None,
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_jit_build",
        lambda tag: calls.append("jit_build") or None,
    )
    assert inject_roctx_loader.load() is sentinel
    assert calls == ["prebuilt"], (
        f"expected prebuilt to short-circuit before cached/cmake/jit; saw {calls!r}"
    )


def test_default_load_path_walks_all_four_tiers_in_order(monkeypatch):
    """Order: prebuilt -> cached -> cmake_build -> cpp_extension."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    calls = []
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_prebuilt",
        lambda tag: calls.append("prebuilt") or None,
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_jit_cached",
        lambda tag: calls.append("cached") or None,
    )
    sentinel = object()
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_cmake_build",
        lambda tag: calls.append("cmake_build") or sentinel,
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_jit_build",
        lambda tag: (
            calls.append("jit_build")
            or pytest.fail(
                "cpp_extension tier must not be reached when cmake succeeds",
            )
        ),
    )
    assert inject_roctx_loader.load() is sentinel
    assert calls == ["prebuilt", "cached", "cmake_build"], (
        f"expected order prebuilt -> cached -> cmake_build, saw {calls!r}"
    )


def test_default_load_path_falls_through_to_cppext_when_cmake_misses(monkeypatch):
    """Default path: cmake -> None must fall through to cpp_extension."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    monkeypatch.setattr(inject_roctx_loader, "_try_prebuilt", lambda _t: None)
    monkeypatch.setattr(inject_roctx_loader, "_try_jit_cached", lambda _t: None)
    monkeypatch.setattr(inject_roctx_loader, "_try_cmake_build", lambda _t: None)
    sentinel = object()
    monkeypatch.setattr(inject_roctx_loader, "_try_jit_build", lambda _t: sentinel)
    assert inject_roctx_loader.load() is sentinel


def test_no_prebuilt_returns_none_for_unknown_tag(tmp_path, monkeypatch):
    """Empty / mismatched prebuilt dir -> None, no exception."""
    fake_dir = tmp_path / "prebuilt"
    fake_dir.mkdir()
    # Decoy with a non-matching tag.
    (fake_dir / "roctx_recordfn-py9.99_torchUNRELATED_abi1.so").write_bytes(b"")
    monkeypatch.setattr(inject_roctx_loader, "_PREBUILT_DIR", fake_dir)
    assert inject_roctx_loader._try_prebuilt("py3.10_torch2.9_abi1") is None


def test_jit_cache_dir_is_creatable(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    d = inject_roctx_loader._jit_cache_dir()
    assert d.exists() and d.is_dir()
    inject_roctx_loader._jit_cache_dir()  # idempotent


def test_load_does_not_raise_when_torch_missing(monkeypatch):
    """torch missing -> None, no exception."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: None)
    assert inject_roctx_loader.load() is None


def test_load_returns_module_or_none_no_raise():
    """load() never raises. Sanity-check the module API when available."""
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


def test_python_fallback_path_is_silent_about_so(monkeypatch):
    """Stub the loader and re-import inject_roctx: using_c_tier()=False,
    dump_recordfn_stats()=None, push/pop still works.
    """
    # inject_roctx.py exits on missing torch; skip cleanly here.
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not importable; utils.inject_roctx module-load exits")
    monkeypatch.setattr(inject_roctx_loader, "load", lambda **kw: None)
    # Reset module state so subsequent tests start clean.
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
    """No user frame -> 'python.dispatch:0' sentinel (distinct from
    C++ tier leaf labels so analyzers can identify the tier).
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not importable; utils.inject_roctx module-load exits")
    monkeypatch.setattr(inject_roctx_loader, "load", lambda **kw: None)
    if "utils.inject_roctx" in sys.modules:
        del sys.modules["utils.inject_roctx"]
    inject_roctx = importlib.import_module("utils.inject_roctx")
    try:
        # No frame -> while-loop never enters -> sentinel returned.
        monkeypatch.setattr("inspect.currentframe", lambda: None)
        assert inject_roctx.resolve_user_caller_location() == "python.dispatch:0"
        # Legacy literal must not survive.
        import inspect as _stdlib_inspect

        src = _stdlib_inspect.getsource(inject_roctx)
        assert "dispatcher:0" not in src, "legacy 'dispatcher:0' sentinel still present"
    finally:
        sys.modules.pop("utils.inject_roctx", None)


def test_import_does_not_apply_global_patches(monkeypatch):
    """Plain ``import utils.inject_roctx`` (no __main__) must not
    mutate global PyTorch state; patches live behind install_global_wraps().
    """
    monkeypatch.setattr(inject_roctx_loader, "load", lambda **kw: None)
    if "utils.inject_roctx" in sys.modules:
        del sys.modules["utils.inject_roctx"]

    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not importable")

    # Snapshot a few process-global callables we are known to patch.
    import torch as _torch

    pre = {
        "compile": getattr(_torch, "compile", None),
    }

    inject_roctx = importlib.import_module("utils.inject_roctx")
    try:
        post = {
            "compile": getattr(_torch, "compile", None),
        }
        # install_global_wraps() must exist and be a no-op until called.
        assert hasattr(inject_roctx, "install_global_wraps")
        # No global patch applied on plain import.
        assert post["compile"] is pre["compile"], (
            "import-time side effect: torch.compile was replaced by "
            "patch_compile_callable() without entering __main__"
        )
    finally:
        sys.modules.pop("utils.inject_roctx", None)


# cmake build tier unit tests. Stub _SO_SOURCE / _SO_BUILDFILE and
# intercept subprocess.run / shutil.which to stay torch- and cmake-
# independent.


class _StubCompleted:
    """Minimal subprocess.CompletedProcess stand-in for monkeypatching."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _set_so_inputs_present(monkeypatch, tmp_path):
    """Point the loader at synthetic source files so the cmake tier's
    "sources missing" early return doesn't fire.
    """
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


def test_cmake_executable_honors_env_var_then_falls_back(monkeypatch):
    """$CMAKE wins over PATH; without $CMAKE, fall back to PATH."""
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


def test_try_cmake_build_skips_when_sources_missing(monkeypatch, tmp_path):
    """Missing C++ source or CMakeLists.txt -> clean short-circuit."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        inject_roctx_loader,
        "_SO_SOURCE",
        tmp_path / "nonexistent.cpp",
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_SO_BUILDFILE",
        tmp_path / "nonexistent.txt",
    )

    def fail_subprocess(*_a, **_k):
        pytest.fail("subprocess.run must not be called when sources are missing")

    monkeypatch.setattr(inject_roctx_loader.subprocess, "run", fail_subprocess)
    assert inject_roctx_loader._try_cmake_build(_FAKE_TAG) is None


def test_try_cmake_build_skips_when_cmake_not_on_path(monkeypatch, tmp_path):
    """No cmake on PATH falls through cleanly and must NOT write a
    failure marker (absence-of-cmake is not a build failure).
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(inject_roctx_loader, "_cmake_executable", lambda: None)

    def fail_subprocess(*_a, **_k):
        pytest.fail("subprocess.run must not be called when cmake is absent")

    monkeypatch.setattr(inject_roctx_loader.subprocess, "run", fail_subprocess)
    assert inject_roctx_loader._try_cmake_build(_FAKE_TAG) is None
    assert inject_roctx_loader._previous_jit_failure(_FAKE_TAG) is None, (
        "cmake-not-on-PATH must not veto cpp_extension via the marker"
    )


def test_try_cmake_build_short_circuits_on_prior_failure(monkeypatch, tmp_path):
    """A negative-cache marker vetoes _try_cmake_build before cmake fires."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(
        inject_roctx_loader,
        "_cmake_executable",
        lambda: "/fake/cmake",
    )
    inject_roctx_loader._record_jit_failure(
        _FAKE_TAG,
        RuntimeError("earlier failure"),
        reason=inject_roctx_loader._CPPEXT_TIER_NAME,
    )

    def fail_subprocess(*_a, **_k):
        pytest.fail("subprocess.run must not fire when marker is present")

    monkeypatch.setattr(inject_roctx_loader.subprocess, "run", fail_subprocess)
    assert inject_roctx_loader._try_cmake_build(_FAKE_TAG) is None


def test_try_cmake_build_passes_runtime_python_to_cmake(monkeypatch, tmp_path):
    """Pin -DTORCH_TRACE_PYTHON=sys.executable so the cmake-side tag
    matches the loader's compute_tag(); otherwise the .so is invisible.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(
        inject_roctx_loader,
        "_cmake_executable",
        lambda: "/fake/cmake",
    )

    invocations = []

    def fake_run(argv, **_kw):
        invocations.append(list(argv))
        return _StubCompleted(returncode=0)

    monkeypatch.setattr(inject_roctx_loader.subprocess, "run", fake_run)
    # Pre-create the expected output so the existence check passes.
    cache_dir = inject_roctx_loader._jit_cache_dir()
    build_dir = cache_dir / f"cmake-build-{_FAKE_TAG}"
    produced = build_dir / "prebuilt" / f"roctx_recordfn-{_FAKE_TAG}.so"
    produced.parent.mkdir(parents=True, exist_ok=True)
    produced.write_bytes(b"stub-so")
    monkeypatch.setattr(
        inject_roctx_loader,
        "_import_module_from_path",
        lambda _n, _p: object(),
    )

    inject_roctx_loader._try_cmake_build(_FAKE_TAG)

    assert len(invocations) == 2, f"expected two cmake invocations, saw {invocations!r}"
    configure_argv = invocations[0]
    assert "-DTORCH_TRACE_PREBUILT=ON" in configure_argv
    runtime_python_flag = f"-DTORCH_TRACE_PYTHON={sys.executable}"
    assert runtime_python_flag in configure_argv, (
        f"-DTORCH_TRACE_PYTHON must equal sys.executable; saw {configure_argv!r}"
    )
    build_argv = invocations[1]
    assert build_argv[1] == "--build", (
        f"second invocation must be `cmake --build`, saw {build_argv!r}"
    )


def test_try_cmake_build_records_failure_marker_on_configure_failure(
    monkeypatch,
    tmp_path,
):
    """cmake configure rc!=0 must record a marker tagged with the
    cmake-build discriminator so cpp_extension sees and respects it.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(
        inject_roctx_loader,
        "_cmake_executable",
        lambda: "/fake/cmake",
    )

    def fake_run(argv, **_kw):
        return _StubCompleted(
            returncode=1,
            stderr="CMake Error: Could not find Torch (missing: TORCH_DIR)\n",
        )

    monkeypatch.setattr(inject_roctx_loader.subprocess, "run", fake_run)
    assert inject_roctx_loader._try_cmake_build(_FAKE_TAG) is None
    marker = inject_roctx_loader._previous_jit_failure(_FAKE_TAG)
    assert marker is not None
    assert inject_roctx_loader._CMAKE_TIER_NAME in marker
    assert "Could not find Torch" in marker, (
        f"marker must preserve stderr tail; saw {marker!r}"
    )


def test_try_cmake_build_records_failure_marker_on_build_failure(monkeypatch, tmp_path):
    """cmake --build rc!=0 must record the same marker as configure failure."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(
        inject_roctx_loader,
        "_cmake_executable",
        lambda: "/fake/cmake",
    )

    call_count = [0]

    def fake_run(argv, **_kw):
        call_count[0] += 1
        if call_count[0] == 1:  # configure
            return _StubCompleted(returncode=0)
        # build
        return _StubCompleted(
            returncode=2,
            stderr="error: undefined reference to `roctxRangePushA'\n",
        )

    monkeypatch.setattr(inject_roctx_loader.subprocess, "run", fake_run)
    assert inject_roctx_loader._try_cmake_build(_FAKE_TAG) is None
    marker = inject_roctx_loader._previous_jit_failure(_FAKE_TAG)
    assert marker is not None
    assert inject_roctx_loader._CMAKE_TIER_NAME in marker
    assert "undefined reference" in marker


def test_try_cmake_build_records_failure_marker_when_so_missing(monkeypatch, tmp_path):
    """cmake reported success but the expected .so is missing -- record
    a failure so subsequent processes don't repeat a doomed build.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(
        inject_roctx_loader,
        "_cmake_executable",
        lambda: "/fake/cmake",
    )
    monkeypatch.setattr(
        inject_roctx_loader.subprocess,
        "run",
        lambda *a, **k: _StubCompleted(returncode=0),
    )
    assert inject_roctx_loader._try_cmake_build(_FAKE_TAG) is None
    marker = inject_roctx_loader._previous_jit_failure(_FAKE_TAG)
    assert marker is not None
    assert inject_roctx_loader._CMAKE_TIER_NAME in marker
    assert ".so missing" in marker or "missing" in marker


def test_try_cmake_build_cleans_build_dir_on_success(monkeypatch, tmp_path):
    """On success the build dir is removed (the .so is now cached;
    per-tag cells would otherwise accumulate hundreds of MB).
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(
        inject_roctx_loader,
        "_cmake_executable",
        lambda: "/fake/cmake",
    )
    monkeypatch.setattr(
        inject_roctx_loader.subprocess,
        "run",
        lambda *a, **k: _StubCompleted(returncode=0),
    )

    cache_dir = inject_roctx_loader._jit_cache_dir()
    build_dir = cache_dir / f"cmake-build-{_FAKE_TAG}"
    produced = build_dir / "prebuilt" / f"roctx_recordfn-{_FAKE_TAG}.so"
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

    result = inject_roctx_loader._try_cmake_build(_FAKE_TAG)
    assert result is not None
    assert not build_dir.exists(), (
        "build dir must be removed on success to bound cache disk usage"
    )
    cached_so = cache_dir / f"roctx_recordfn-{_FAKE_TAG}.so"
    assert cached_so.exists() and cached_so.read_bytes() == b"stub-so", (
        "produced .so must have been copied to the cache before cleanup"
    )


def test_try_cmake_build_keeps_build_dir_on_failure(monkeypatch, tmp_path):
    """Preserve the build dir on failure for post-mortem inspection."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(
        inject_roctx_loader,
        "_cmake_executable",
        lambda: "/fake/cmake",
    )
    monkeypatch.setattr(
        inject_roctx_loader.subprocess,
        "run",
        lambda *a, **k: _StubCompleted(returncode=1, stderr="boom\n"),
    )

    assert inject_roctx_loader._try_cmake_build(_FAKE_TAG) is None
    build_dir = inject_roctx_loader._jit_cache_dir() / f"cmake-build-{_FAKE_TAG}"
    assert build_dir.exists(), "build dir must survive a failure"


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
    """cmake-tier hints must never recommend installing ninja."""
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


# Diagnostic accumulator + loaded_tier API: contract relied on by
# inject_roctx.py to fold the per-tier trail into one user WARNING.


def test_loaded_tier_records_successful_step(monkeypatch):
    """When a tier returns non-None, loaded_tier() reports its name.
    The TIER_* string constants are public tokens; pinned here.
    """
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    sentinel = object()
    monkeypatch.setattr(inject_roctx_loader, "_try_prebuilt", lambda _t: None)
    monkeypatch.setattr(inject_roctx_loader, "_try_jit_cached", lambda _t: None)
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_cmake_build",
        lambda _t: sentinel,
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_jit_build",
        lambda _t: pytest.fail(
            "cppext tier must not run when cmake succeeds",
        ),
    )
    assert inject_roctx_loader.load() is sentinel
    assert inject_roctx_loader.loaded_tier() == inject_roctx_loader.TIER_CMAKE_BUILD
    assert inject_roctx_loader.TIER_CMAKE_BUILD in inject_roctx_loader.C_TIER_NAMES


def test_loaded_tier_is_none_when_all_tiers_miss(monkeypatch):
    """Every tier -> None makes loaded_tier() return None."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    for name in (
        "_try_prebuilt",
        "_try_jit_cached",
        "_try_cmake_build",
        "_try_jit_build",
    ):
        monkeypatch.setattr(inject_roctx_loader, name, lambda _t: None)
    assert inject_roctx_loader.load() is None
    assert inject_roctx_loader.loaded_tier() is None


def test_load_resets_diagnostics_on_entry(monkeypatch):
    """Stale diagnostics from a prior load() must not bleed in."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    inject_roctx_loader._LAST_LOAD_DIAGNOSTICS.clear()
    inject_roctx_loader._LAST_LOAD_DIAGNOSTICS.append(("log", "STALE LINE"))

    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_prebuilt",
        lambda _t: object(),
    )
    monkeypatch.setattr(inject_roctx_loader, "_try_jit_cached", lambda _t: None)
    monkeypatch.setattr(inject_roctx_loader, "_try_cmake_build", lambda _t: None)
    monkeypatch.setattr(inject_roctx_loader, "_try_jit_build", lambda _t: None)

    inject_roctx_loader.load()
    _tier, diagnostics = inject_roctx_loader.consume_diagnostics()
    for level, msg in diagnostics:
        assert "STALE LINE" not in msg, (
            f"pre-existing diagnostic leaked across load() boundary: ({level}, {msg!r})"
        )


def test_safe_log_tees_into_accumulator(monkeypatch):
    """Every _safe_log call appends to _LAST_LOAD_DIAGNOSTICS."""
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
    """Returns (tier, trail), drains the trail, tier scalar persists."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    monkeypatch.setattr(inject_roctx_loader, "_try_prebuilt", lambda _t: None)
    monkeypatch.setattr(inject_roctx_loader, "_try_jit_cached", lambda _t: None)
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_cmake_build",
        lambda _t: object(),
    )
    monkeypatch.setattr(inject_roctx_loader, "_try_jit_build", lambda _t: None)

    inject_roctx_loader.load()
    tier, trail = inject_roctx_loader.consume_diagnostics()
    assert tier == inject_roctx_loader.TIER_CMAKE_BUILD
    # Stub _try_* paths don't log; the contract is one-way drain.
    assert isinstance(trail, list)

    tier2, trail2 = inject_roctx_loader.consume_diagnostics()
    assert tier2 == tier, "tier scalar must persist across drains"
    assert trail2 == [], "second consume must see no leftover lines"


def test_consume_diagnostics_returns_python_tier_failure_trail(monkeypatch):
    """All-miss: the trail includes the terminal fallback line so
    inject_roctx.py's WARNING can embed it.
    """
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    monkeypatch.setattr(inject_roctx_loader, "_try_prebuilt", lambda _t: None)
    monkeypatch.setattr(inject_roctx_loader, "_try_jit_cached", lambda _t: None)
    monkeypatch.setattr(inject_roctx_loader, "_try_cmake_build", lambda _t: None)
    monkeypatch.setattr(inject_roctx_loader, "_try_jit_build", lambda _t: None)

    inject_roctx_loader.load()
    tier, trail = inject_roctx_loader.consume_diagnostics()
    assert tier is None
    joined = " ".join(msg for _, msg in trail).lower()
    assert "python-only injector" in joined or "no roctx_recordfn" in joined, (
        f"terminal-fallback line missing from trail: "
        f"{[(lvl, msg) for lvl, msg in trail]!r}"
    )


def test_format_load_diagnostic_trail_handles_empty():
    """Empty trail -> empty string."""
    assert inject_roctx_loader.format_load_diagnostic_trail([]) == ""


def test_format_load_diagnostic_trail_caps_lines():
    """The cap bounds output on a pathological diagnostic burst and
    keeps the most recent lines (nearest the failure).
    """
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


def test_cmake_failure_marker_vetoes_subsequent_cppext_attempt(monkeypatch, tmp_path):
    """The shared marker: a cmake failure must veto the cpp_extension
    tier on subsequent calls so per-PMC-pass noise stays bounded.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(
        inject_roctx_loader,
        "_cmake_executable",
        lambda: "/fake/cmake",
    )
    monkeypatch.setattr(
        inject_roctx_loader.subprocess,
        "run",
        lambda *a, **k: _StubCompleted(returncode=1, stderr="boom\n"),
    )

    assert inject_roctx_loader._try_cmake_build(_FAKE_TAG) is None
    assert inject_roctx_loader._previous_jit_failure(_FAKE_TAG) is not None

    # cppext tier must short-circuit on the cmake-recorded marker
    # without importing cpp_extension.
    cppext_import_attempts = []
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def watching_import(name, *a, **kw):
        if "cpp_extension" in name:
            cppext_import_attempts.append(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(
        "builtins.__import__",
        watching_import,
    )
    assert inject_roctx_loader._try_jit_build(_FAKE_TAG) is None
    assert cppext_import_attempts == [], (
        f"cpp_extension import attempted despite shared marker: "
        f"{cppext_import_attempts!r}"
    )
