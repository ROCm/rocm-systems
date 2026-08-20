# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.inject_roctx._backends.torch_cpp_loader. No GPU.
CMake build is end-to-end in test_torch_trace_coverage; this file stubs the
loader to pin contracts: tier order, artifact fingerprint, REBUILD env var,
no-ninja policy.
"""

import importlib
import re
import sys

import common  # noqa: F401
import pytest

from utils.inject_roctx._backends import torch_cpp_loader as inject_roctx_loader
from utils.inject_roctx._backends import torch_trace_fingerprint

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
    import torch

    torch_components = [p for p in parts if p.startswith("torch")]
    assert len(torch_components) == 1, (
        f"expected exactly one '_torch...' component in tag {tag!r}"
    )
    assert torch_components[0] == "torch" + torch.__version__.split("+", 1)[0]


def test_compute_tag_is_stable_across_calls():
    """``compute_tag()`` returns the same value for repeated calls in one process."""
    assert inject_roctx_loader.compute_tag() == inject_roctx_loader.compute_tag()


def test_source_fingerprint_changes_when_inputs_change(tmp_path, monkeypatch):
    """A single-byte edit to any input changes the fingerprint."""
    cpp = tmp_path / "torch_trace_collector.cpp"
    cmake = tmp_path / "CMakeLists.txt"
    cpp.write_text("// fingerprint test source\n")
    cmake.write_text("# fingerprint test buildfile\n")
    monkeypatch.setattr(
        torch_trace_fingerprint,
        "fingerprint_input_paths",
        lambda: (cpp, cmake),
    )

    baseline = torch_trace_fingerprint.source_fingerprint()
    assert len(baseline) == 12

    for input_path in (cpp, cmake):
        original = input_path.read_bytes()
        input_path.write_bytes(original + b"\n# mutation\n")
        mutated = torch_trace_fingerprint.source_fingerprint()
        assert mutated != baseline, (
            f"editing {input_path.name} did not change the fingerprint"
        )
        input_path.write_bytes(original)
        assert torch_trace_fingerprint.source_fingerprint() == baseline


def test_source_fingerprint_excludes_tool_version_file():
    """The tool ``VERSION`` file is not a fingerprint input."""
    for path in torch_trace_fingerprint.fingerprint_input_paths():
        assert path.name != "VERSION", (
            f"VERSION should not be in fingerprint_input_paths; saw {path}"
        )


def test_source_fingerprint_is_missing_sentinel_when_no_inputs_readable(
    tmp_path,
    monkeypatch,
):
    """When no inputs are readable, the fingerprint is the ``"missing"`` sentinel."""
    monkeypatch.setattr(
        torch_trace_fingerprint,
        "fingerprint_input_paths",
        lambda: (tmp_path / "does_not_exist.cpp",),
    )
    assert torch_trace_fingerprint.source_fingerprint() == "missing"


def test_source_fingerprint_is_length_delimited(tmp_path, monkeypatch):
    """Reshuffling bytes across input boundaries changes the fingerprint."""
    a = tmp_path / "a.cpp"
    b = tmp_path / "b.txt"
    monkeypatch.setattr(
        torch_trace_fingerprint, "fingerprint_input_paths", lambda: (a, b)
    )

    a.write_bytes(b"AB")
    b.write_bytes(b"C")
    fp1 = torch_trace_fingerprint.source_fingerprint()
    a.write_bytes(b"A")
    b.write_bytes(b"BC")
    fp2 = torch_trace_fingerprint.source_fingerprint()
    assert fp1 != fp2, "fingerprint collided across an input boundary"


# ---------------------------------------------------------------------------
# Source / buildfile hygiene
# ---------------------------------------------------------------------------


def torch_trace_collector_module_sources():
    """The module's own C++ source and header files, from the fingerprint inputs.

    The inputs also carry shared headers from outside the module directory, so
    the parent is part of the filter.
    """
    return sorted(
        p
        for p in torch_trace_fingerprint.fingerprint_input_paths()
        if p.suffix in (".cpp", ".h") and p.parent == inject_roctx_loader._SO_SOURCE_DIR
    )


def cmake_declared_sources():
    """The C++ files CMakeLists.txt names, from its header list and add_library
    calls.

    Parsed from the build file rather than globbed, so this is independent of
    how the loader collects its fingerprint inputs. Globbing here would restate
    the implementation and pass regardless of what it missed.
    """
    text = inject_roctx_loader._SO_BUILDFILE.read_text()
    names = set()

    # Longest extension first, and a trailing non-word guard, so ".hpp" is not
    # truncated to ".h" by the alternation.
    header_block = re.search(r"set\(_ttc_headers(.*?)\)", text, re.DOTALL)
    if header_block:
        names.update(re.findall(r"[\w.-]+\.(?:hpp|hxx|h)(?!\w)", header_block.group(1)))

    for args in re.findall(r"add_library\((.*?)\)", text, re.DOTALL):
        names.update(re.findall(r"[\w.-]+\.(?:cpp|cxx|cc)(?!\w)", args))

    return names


def test_fingerprint_inputs_cover_every_build_input():
    """Every file the build consumes is a fingerprint input."""
    src_dir = inject_roctx_loader._SO_SOURCE_DIR
    if not src_dir.is_dir():
        pytest.skip(f"module sources not present at {src_dir}")

    declared = cmake_declared_sources()
    assert declared, (
        f"no sources parsed from {inject_roctx_loader._SO_BUILDFILE}; "
        "the build file changed shape and this parser needs updating"
    )

    input_names = {p.name for p in torch_trace_fingerprint.fingerprint_input_paths()}
    missing = declared - input_names
    assert not missing, f"build inputs absent from the fingerprint: {sorted(missing)}"

    # None of these is a compiled source, so none is reachable from add_library.
    # The module includes synchronized.hpp and gsl_assert.h from the shared utils.
    for required in (
        "CMakeLists.txt",
        "synchronized.hpp",
        "gsl_assert.h",
    ):
        assert required in input_names, f"{required} is not a fingerprint input"


def test_required_input_paths_cover_every_build_input():
    """Every file the build consumes is a required input."""
    src_dir = inject_roctx_loader._SO_SOURCE_DIR
    if not src_dir.is_dir():
        pytest.skip(f"module sources not present at {src_dir}")

    declared = cmake_declared_sources()
    assert declared, (
        f"no sources parsed from {inject_roctx_loader._SO_BUILDFILE}; "
        "the build file changed shape and this parser needs updating"
    )

    required_names = {p.name for p in torch_trace_fingerprint.required_input_paths()}
    missing = declared - required_names
    assert not missing, f"build inputs absent from the required list: {sorted(missing)}"

    for required in (
        "CMakeLists.txt",
        "synchronized.hpp",
        "gsl_assert.h",
    ):
        assert required in required_names, f"{required} is not a required input"


def test_required_input_paths_include_missing_files(tmp_path, monkeypatch):
    """``required_input_paths()`` includes collector files that are not on disk."""
    monkeypatch.setattr(torch_trace_fingerprint, "_SO_SOURCE_DIR", tmp_path)
    monkeypatch.setattr(
        torch_trace_fingerprint, "_SO_BUILDFILE", tmp_path / "CMakeLists.txt"
    )
    names = {path.name for path in torch_trace_fingerprint.required_input_paths()}
    for name in torch_trace_fingerprint._COLLECTOR_SOURCE_NAMES:
        assert name in names
    for name in torch_trace_fingerprint._COLLECTOR_HEADER_NAMES:
        assert name in names
    assert "CMakeLists.txt" in names
    assert "synchronized.hpp" in names
    assert "gsl_assert.h" in names
    assert not (tmp_path / "torch_trace_collector.cpp").exists()
    assert not (tmp_path / "leaf_context.h").exists()


def test_required_input_paths_exist_in_the_source_tree():
    """Every required input exists in the source tree."""
    src_dir = inject_roctx_loader._SO_SOURCE_DIR
    if not src_dir.is_dir():
        pytest.skip(f"module sources not present at {src_dir}")
    missing = [
        path.name
        for path in torch_trace_fingerprint.required_input_paths()
        if not path.exists()
    ]
    assert not missing, f"required inputs are absent: {missing}"


def test_torch_trace_collector_source_avoids_torch_umbrella_headers():
    """No module source may include ``<torch/{extension,all,torch}.h>``."""
    sources = torch_trace_collector_module_sources()
    assert sources, f"no C++ sources under {inject_roctx_loader._SO_SOURCE_DIR}"

    forbidden = (
        "<torch/extension.h>",
        "<torch/all.h>",
        "<torch/torch.h>",
    )
    for path in sources:
        active_lines = [
            line
            for line in path.read_text().splitlines()
            if not line.lstrip().startswith("//")
        ]
        active_src = "\n".join(active_lines)
        for header in forbidden:
            directive = f"#include {header}"
            assert directive not in active_src, f"{path.name} must not include {header}"


def test_torch_trace_collector_source_uses_narrow_includes():
    """The module includes the required narrow PyTorch headers."""
    combined = "\n".join(
        path.read_text() for path in torch_trace_collector_module_sources()
    )

    required = (
        "#include <ATen/record_function.h>",
        "#include <c10/util/ThreadLocalDebugInfo.h>",
        "#include <pybind11/pybind11.h>",
        "#include <pybind11/stl.h>",
    )
    for directive in required:
        assert directive in combined, f"module must include {directive}"


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


def test_cmake_buildfile_hardcodes_cxx11_abi():
    """Official ROCm libtorch 2.10+ is CXX11 ABI."""
    cmake_path = inject_roctx_loader._SO_BUILDFILE
    active_lines = [
        line
        for line in cmake_path.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    active_src = "\n".join(active_lines)

    assert "PUBLIC _GLIBCXX_USE_CXX11_ABI=1" in active_src, (
        "the collector must compile with CXX11 ABI to match ROCm libtorch 2.10+"
    )
    assert "TORCH_TRACE_CXX11_ABI" not in active_src, (
        "ABI must not be a configure cache variable"
    )


def test_loader_and_cmake_agree_on_artifact_filename_shape():
    """The loader and CMake produce identical artifact filenames."""
    import inspect

    src = inspect.getsource(inject_roctx_loader._try_runtime_build)
    assert 'f"torch_trace_collector-{tag}.so"' in src, (
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
    """``C_TIER_NAMES`` enumerates exactly the prebuilt and runtime-build tiers."""
    assert inject_roctx_loader.C_TIER_NAMES == frozenset((
        inject_roctx_loader.TIER_PREBUILT,
        inject_roctx_loader.TIER_RUNTIME_BUILD,
    ))


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def test_force_python_fallback_returns_none():
    assert inject_roctx_loader.load(force_python_fallback=True).module is None


def test_user_cache_dir_is_creatable(monkeypatch, tmp_path):
    """``_user_cache_dir`` creates the directory on first call and is idempotent."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    d = inject_roctx_loader._user_cache_dir()
    assert d.exists() and d.is_dir()
    assert inject_roctx_loader._user_cache_dir() == d


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
# _try_runtime_build: build directory and finder handoff
# ---------------------------------------------------------------------------


def _set_so_inputs_present(monkeypatch, tmp_path):
    """Point the loader at synthetic source files."""
    src_dir = tmp_path / "torch_trace_collector"
    src_dir.mkdir(parents=True, exist_ok=True)
    cpp = src_dir / "torch_trace_collector.cpp"
    module_cpp = src_dir / "torch_trace_collector_module.cpp"
    cml = src_dir / "CMakeLists.txt"
    cpp.write_text("// stub\n")
    module_cpp.write_text("// stub\n")
    cml.write_text("# stub\n")
    monkeypatch.setattr(inject_roctx_loader, "_SO_SOURCE_DIR", src_dir)
    monkeypatch.setattr(inject_roctx_loader, "_SO_BUILDFILE", cml)
    monkeypatch.setattr(
        torch_trace_fingerprint,
        "fingerprint_input_paths",
        lambda: (cpp, module_cpp, cml),
    )
    monkeypatch.setattr(
        torch_trace_fingerprint,
        "required_input_paths",
        lambda: (cpp, module_cpp, cml),
    )
    return src_dir


def _patch_finder(monkeypatch, *, artifact_path=None, error=None):
    """Replace ``NativeToolFinder`` with a subclass that records how the loader
    constructed it and resolves without running cmake.

    Returns the list of instances the loader created.
    """
    from utils.native_tool_finder import NativeToolFinder

    instances = []

    class _RecordingFinder(NativeToolFinder):
        def __init__(self, root_path, **kwargs):
            super().__init__(root_path, **kwargs)
            instances.append(self)

        def get_artifact_path(self):
            if error is not None:
                raise error
            return artifact_path

    monkeypatch.setattr("utils.native_tool_finder.NativeToolFinder", _RecordingFinder)
    return instances


def test_importing_the_loader_pulls_in_neither_the_finder_nor_torch():
    """The finder is imported only when a runtime build runs, so the loader
    remains importable where the ``utils`` package is absent.
    """
    import os
    import subprocess

    code = (
        "import sys\n"
        "import utils.inject_roctx._backends.torch_cpp_loader\n"
        "assert 'utils.native_tool_finder' not in sys.modules, 'finder imported'\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "PYTHONPATH": str(inject_roctx_loader._NATIVE_TOOL_ROOT),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_is_writable_uses_the_parent_for_a_missing_directory(tmp_path):
    """Writability of a missing directory is decided by its parent."""
    assert inject_roctx_loader._is_writable(tmp_path / "_build") is True
    assert inject_roctx_loader._is_writable(tmp_path / "absent" / "_build") is False


def test_runtime_build_path_uses_the_source_tree_when_writable(tmp_path):
    """A writable build directory in the source tree is used unchanged."""
    source_build_path = tmp_path / "lib" / "_build"
    source_build_path.parent.mkdir(parents=True)
    assert (
        inject_roctx_loader._runtime_build_path(source_build_path) == source_build_path
    )


def test_runtime_build_path_falls_back_to_the_user_cache(monkeypatch, tmp_path):
    """A read-only source tree moves the build directory under the user cache."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(inject_roctx_loader, "_is_writable", lambda _p: False)

    build_path = inject_roctx_loader._runtime_build_path(tmp_path / "lib" / "_build")

    assert build_path is not None
    assert build_path.name == "_build"
    assert str(tmp_path / "cache") in str(build_path), (
        f"build directory must live under the user cache; saw {build_path}"
    )


def test_runtime_build_skips_when_a_required_source_is_absent(
    monkeypatch, tmp_path
):
    """A missing required source skips the runtime build."""
    src_dir = tmp_path / "torch_trace_collector"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "CMakeLists.txt").write_text("# stub\n")
    monkeypatch.setattr(torch_trace_fingerprint, "_SO_SOURCE_DIR", src_dir)
    monkeypatch.setattr(
        torch_trace_fingerprint, "_SO_BUILDFILE", src_dir / "CMakeLists.txt"
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_cmake_executable",
        lambda: pytest.fail("cmake was invoked"),
    )
    diagnostics: list[tuple[str, str]] = []
    assert (
        inject_roctx_loader._try_runtime_build(_FAKE_TAG, diagnostics=diagnostics)
        is None
    )
    joined = " ".join(msg for _, msg in diagnostics)
    assert "torch_trace_collector.cpp" in joined, (
        f"absent collector source not reported; saw {diagnostics!r}"
    )
    assert "leaf_context.h" in joined, (
        f"absent collector header not reported; saw {diagnostics!r}"
    )


def test_runtime_build_skips_when_cmake_not_on_path(monkeypatch, tmp_path):
    """Absence of cmake on ``PATH`` skips the tier without constructing a finder."""
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(inject_roctx_loader, "_cmake_executable", lambda: None)
    instances = _patch_finder(monkeypatch)

    assert inject_roctx_loader._try_runtime_build(_FAKE_TAG) is None
    assert instances == [], "cmake absence must be detected before the finder runs"


def test_runtime_build_names_the_tagged_artifact_and_pins_the_interpreter(
    monkeypatch, tmp_path
):
    """The finder is asked for the tagged artifact and target, and cmake is told
    which interpreter and source fingerprint to build against.
    """
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(inject_roctx_loader, "_cmake_executable", lambda: "/fake/cmake")
    monkeypatch.setattr(
        inject_roctx_loader,
        "_runtime_build_path",
        lambda _p, diagnostics=None: tmp_path / "_build",
    )
    so_path = tmp_path / f"torch_trace_collector-{_FAKE_TAG}.so"
    so_path.write_bytes(b"stub-so")
    instances = _patch_finder(monkeypatch, artifact_path=so_path)
    sentinel = object()
    monkeypatch.setattr(
        inject_roctx_loader,
        "_import_module_from_path",
        lambda _n, _p: sentinel,
    )

    assert inject_roctx_loader._try_runtime_build(_FAKE_TAG) is sentinel
    assert len(instances) == 1, f"expected one finder, saw {instances!r}"
    finder = instances[0]
    assert finder.artifact_name == f"torch_trace_collector-{_FAKE_TAG}.so"
    assert finder.build_target == f"torch_trace_collector-{_FAKE_TAG}", (
        "the build must be scoped to the extension target"
    )
    assert f"-DTORCH_TRACE_PYTHON={sys.executable}" in finder.configure_options, (
        f"-DTORCH_TRACE_PYTHON must equal sys.executable; "
        f"saw {finder.configure_options!r}"
    )
    assert (
        f"-DTORCH_TRACE_SOURCE_FINGERPRINT={_FAKE_TAG.rpartition('_src')[2]}"
        in finder.configure_options
    ), (
        f"-DTORCH_TRACE_SOURCE_FINGERPRINT must be the tag's src suffix; "
        f"saw {finder.configure_options!r}"
    )
    assert "-DBUILD_TORCH_TRACE_COLLECTOR=ON" in finder.configure_options, (
        "the extension must be required so an absent torch is an error"
    )
    assert not any(
        opt.startswith("-DTORCH_TRACE_CXX11_ABI=") for opt in finder.configure_options
    ), (
        "CXX11 ABI is hardcoded in CMakeLists.txt; the loader must not pass it; "
        f"saw {finder.configure_options!r}"
    )
    assert finder.cmake_executable == "/fake/cmake"
    assert finder.search_installed is True
    assert finder.reuse_built_artifact is True, (
        "the tag identifies the sources, so an artifact already built under it "
        "must be imported without configuring again"
    )


def test_runtime_build_force_rebuild_discards_the_build_directory(
    monkeypatch, tmp_path
):
    """``force_rebuild`` discards the build directory and skips the installed search."""
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(inject_roctx_loader, "_cmake_executable", lambda: "/fake/cmake")
    build_path = tmp_path / "_build"
    stale = build_path / "CMakeCache.txt"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("stale cache\n")
    monkeypatch.setattr(
        inject_roctx_loader,
        "_runtime_build_path",
        lambda _p, diagnostics=None: build_path,
    )
    so_path = tmp_path / f"torch_trace_collector-{_FAKE_TAG}.so"
    so_path.write_bytes(b"stub-so")
    instances = _patch_finder(monkeypatch, artifact_path=so_path)
    monkeypatch.setattr(
        inject_roctx_loader,
        "_import_module_from_path",
        lambda _n, _p: object(),
    )

    result = inject_roctx_loader._try_runtime_build(_FAKE_TAG, force_rebuild=True)

    assert result is not None
    assert not stale.exists(), "force_rebuild must discard the stale build directory"
    assert instances[0].search_installed is False, (
        "force_rebuild must not resolve to an installed artifact"
    )
    assert instances[0].reuse_built_artifact is False, (
        "force_rebuild must not reuse an artifact from the build directory"
    )


def test_runtime_build_returns_none_and_classifies_a_cmake_failure(
    monkeypatch, tmp_path
):
    """A failing build is reported in the diagnostic trail and yields ``None``."""
    _set_so_inputs_present(monkeypatch, tmp_path)
    monkeypatch.setattr(inject_roctx_loader, "_cmake_executable", lambda: "/fake/cmake")
    monkeypatch.setattr(
        inject_roctx_loader,
        "_runtime_build_path",
        lambda _p, diagnostics=None: tmp_path / "_build",
    )
    _patch_finder(
        monkeypatch,
        error=RuntimeError(
            "Failed to execute command: cmake\n"
            "CMake Error: Could not find Torch (missing: TORCH_DIR)"
        ),
    )

    diagnostics: list[tuple[str, str]] = []
    result = inject_roctx_loader._try_runtime_build(_FAKE_TAG, diagnostics=diagnostics)

    assert result is None
    joined = " ".join(msg for _, msg in diagnostics).lower()
    assert "libtorch" in joined, (
        f"the cmake failure was not reported to the user: {diagnostics!r}"
    )


# ---------------------------------------------------------------------------
# explain_cmake_failure classification
# ---------------------------------------------------------------------------


def test_explain_cmake_failure_classifies_torch_not_found():
    reason, hint = inject_roctx_loader._explain_cmake_failure(
        RuntimeError("CMake Error: Could not find Torch (missing: TORCH_DIR)")
    )
    assert "libtorch" in reason.lower()
    assert "torch" in hint.lower()


def test_explain_cmake_failure_classifies_roctx_not_found():
    reason, hint = inject_roctx_loader._explain_cmake_failure(
        RuntimeError("find_library failed: rocprofiler-sdk-roctx not found")
    )
    assert "roctx" in reason.lower() or "rocprofiler-sdk-roctx" in reason.lower()
    assert "rocm_path" in hint.lower() or "/opt/rocm" in hint.lower()


def test_explain_cmake_failure_classifies_missing_cxx_compiler():
    reason, hint = inject_roctx_loader._explain_cmake_failure(
        RuntimeError("No CMAKE_CXX_COMPILER could be found.")
    )
    assert "compiler" in reason.lower()
    assert "g++" in hint.lower() or "clang" in hint.lower()


def test_explain_cmake_failure_never_recommends_installing_ninja():
    """cmake-tier hints must not recommend installing ninja."""
    samples = [
        RuntimeError("CMake Error: ninja not found"),
        RuntimeError("Could not find Torch"),
        RuntimeError("rc=1"),
    ]
    for err in samples:
        reason, hint = inject_roctx_loader._explain_cmake_failure(err)
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


def test_rebuild_env_var_skips_prebuilt_and_forces_a_rebuild(monkeypatch):
    """REBUILD=1 routes directly to the runtime build, skipping the prebuilt tier."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    calls = []
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_prebuilt",
        lambda tag, diagnostics=None: calls.append(("prebuilt", tag)) or None,
    )
    sentinel = object()

    def _runtime_build(tag, force_rebuild=False, diagnostics=None, **_kwargs):
        calls.append(("runtime-build", tag, force_rebuild))
        return sentinel

    monkeypatch.setattr(inject_roctx_loader, "_try_runtime_build", _runtime_build)
    monkeypatch.setenv(inject_roctx_loader._REBUILD_ENV_VAR, "1")

    result = inject_roctx_loader.load()
    assert result.module is sentinel
    assert [c[0] for c in calls] == ["runtime-build"], (
        f"expected only _try_runtime_build to fire under REBUILD; saw {calls!r}"
    )
    assert calls[0][1] == _FAKE_TAG, "tag must propagate to the build step"
    assert calls[0][2] is True, "REBUILD must pass force_rebuild=True"


def test_rebuild_env_var_returns_none_when_build_fails(monkeypatch):
    """Under REBUILD, a failing build yields ``None`` with no fallback."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_prebuilt",
        lambda _t, diagnostics=None: pytest.fail("_try_prebuilt called"),
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_runtime_build",
        lambda _t, force_rebuild=False, diagnostics=None, **_kwargs: None,
    )
    monkeypatch.setenv(inject_roctx_loader._REBUILD_ENV_VAR, "1")
    assert inject_roctx_loader.load().module is None


def test_default_load_path_still_tries_prebuilt_first(monkeypatch):
    """The prebuilt tier short-circuits the default load path."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_prebuilt",
        lambda tag, diagnostics=None: calls.append("prebuilt") or sentinel,
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_runtime_build",
        lambda tag, force_rebuild=False, diagnostics=None, **_kwargs: (
            calls.append("runtime-build") or None
        ),
    )
    assert inject_roctx_loader.load().module is sentinel
    assert calls == ["prebuilt"], (
        f"expected prebuilt to short-circuit before the build; saw {calls!r}"
    )


def test_default_load_path_walks_prebuilt_then_runtime_build(monkeypatch):
    """Tiers are tried in order: prebuilt, then runtime build."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    calls = []
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_prebuilt",
        lambda tag, diagnostics=None: calls.append("prebuilt") or None,
    )
    sentinel = object()
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_runtime_build",
        lambda tag, force_rebuild=False, diagnostics=None, **_kwargs: (
            calls.append("runtime-build") or sentinel
        ),
    )
    assert inject_roctx_loader.load().module is sentinel
    assert calls == ["prebuilt", "runtime-build"], (
        f"expected order prebuilt -> runtime-build, saw {calls!r}"
    )


def test_load_does_not_raise_when_torch_missing(monkeypatch):
    """``load()`` returns ``None`` when ``torch`` is not importable."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: None)
    assert inject_roctx_loader.load().module is None


def test_load_returns_module_or_none_no_raise():
    """``load()`` returns a module exposing the documented API, or ``None``."""
    mod = inject_roctx_loader.load().module
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
# LoadResult: tier / diagnostics / format_load_diagnostic_trail
# ---------------------------------------------------------------------------


def test_load_result_records_successful_tier(monkeypatch):
    """``LoadResult.tier`` reports the name of the tier that returned a module."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    sentinel = object()
    monkeypatch.setattr(
        inject_roctx_loader, "_try_prebuilt", lambda _t, diagnostics=None: None
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_runtime_build",
        lambda _t, force_rebuild=False, diagnostics=None, **_kwargs: sentinel,
    )
    result = inject_roctx_loader.load()
    assert result.module is sentinel
    assert result.tier == inject_roctx_loader.TIER_RUNTIME_BUILD
    assert inject_roctx_loader.TIER_RUNTIME_BUILD in inject_roctx_loader.C_TIER_NAMES


def test_load_result_tier_is_none_when_all_tiers_miss(monkeypatch):
    """``LoadResult.tier`` is ``None`` when every tier returns ``None``."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    monkeypatch.setattr(
        inject_roctx_loader, "_try_prebuilt", lambda _t, diagnostics=None: None
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_runtime_build",
        lambda _t, force_rebuild=False, diagnostics=None, **_kwargs: None,
    )
    result = inject_roctx_loader.load()
    assert result.module is None
    assert result.tier is None


def test_load_returns_independent_diagnostics(monkeypatch):
    """Each ``load()`` call returns its own diagnostics list."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    monkeypatch.setattr(
        inject_roctx_loader, "_try_prebuilt", lambda _t, diagnostics=None: object()
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_runtime_build",
        lambda _t, force_rebuild=False, diagnostics=None, **_kwargs: None,
    )

    first = inject_roctx_loader.load().diagnostics
    second = inject_roctx_loader.load().diagnostics
    assert first is not second, "each load() must return its own diagnostics list"


def test_safe_log_appends_to_provided_diagnostics():
    """Each ``_safe_log`` call appends to the provided diagnostics list."""
    diagnostics: list[tuple[str, str]] = []
    inject_roctx_loader._safe_log("log", "tier A skipped", diagnostics)
    inject_roctx_loader._safe_log("warning", "tier B failed", diagnostics)
    inject_roctx_loader._safe_log("log", "final fallback engaged", diagnostics)

    assert [lvl for lvl, _ in diagnostics] == ["log", "warning", "log"]
    assert [msg for _, msg in diagnostics] == [
        "tier A skipped",
        "tier B failed",
        "final fallback engaged",
    ]


def test_load_result_carries_tier_and_diagnostics(monkeypatch):
    """``load()`` returns the tier scalar and a diagnostics list."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    monkeypatch.setattr(
        inject_roctx_loader, "_try_prebuilt", lambda _t, diagnostics=None: None
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_runtime_build",
        lambda _t, force_rebuild=False, diagnostics=None, **_kwargs: object(),
    )

    result = inject_roctx_loader.load()
    assert result.tier == inject_roctx_loader.TIER_RUNTIME_BUILD
    assert isinstance(result.diagnostics, list)


def test_load_result_returns_python_tier_failure_trail(monkeypatch):
    """When every tier misses the trail includes the terminal fallback line."""
    monkeypatch.setattr(inject_roctx_loader, "compute_tag", lambda: _FAKE_TAG)
    monkeypatch.delenv(inject_roctx_loader._REBUILD_ENV_VAR, raising=False)
    monkeypatch.setattr(
        inject_roctx_loader, "_try_prebuilt", lambda _t, diagnostics=None: None
    )
    monkeypatch.setattr(
        inject_roctx_loader,
        "_try_runtime_build",
        lambda _t, force_rebuild=False, diagnostics=None, **_kwargs: None,
    )

    result = inject_roctx_loader.load()
    assert result.tier is None
    joined = " ".join(msg for _, msg in result.diagnostics).lower()
    assert "python-only injector" in joined or "no torch_trace_collector" in joined, (
        f"terminal-fallback line missing from trail: "
        f"{[(lvl, msg) for lvl, msg in result.diagnostics]!r}"
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
    importlib.import_module("utils.inject_roctx")
    try:
        from utils.inject_roctx import core
        from utils.inject_roctx._backends import torch as _torch_backend

        assert _torch_backend.using_c_tier() is False
        assert _torch_backend.dump_recordfn_stats() is None

        pushed: list[str] = []
        original_io = core.get_python_tier_io()
        core.set_python_tier_io(push=pushed.append, pop=lambda: None)
        try:
            core._push_scope("py.tier.test", "#1@test:1")
            core._pop_scope()
        finally:
            core.set_python_tier_io(*original_io)
        assert pushed, "Python-tier rangePush was not invoked in fallback mode"
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
    importlib.import_module("utils.inject_roctx")
    try:
        from utils.inject_roctx import core

        monkeypatch.setattr("inspect.currentframe", lambda: None)
        assert core.resolve_user_caller_location() == "python.dispatch:0"
        import inspect as _stdlib_inspect

        src = _stdlib_inspect.getsource(core)
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

    importlib.import_module("utils.inject_roctx")
    try:
        post = {
            "compile": getattr(_torch, "compile", None),
        }
        from utils.inject_roctx import core

        for sym in (
            "install_global_wraps",
            "_push_scope",
            "_pop_scope",
            "resolve_user_caller_location",
        ):
            assert hasattr(core, sym), f"core symbol missing: {sym}"
        from utils.inject_roctx._backends import torch as _torch_backend

        for sym in (
            "install_function_apply_wrappers",
            "using_c_tier",
            "dump_recordfn_stats",
        ):
            assert hasattr(_torch_backend, sym), f"torch backend symbol missing: {sym}"
        assert post["compile"] is pre["compile"], "torch.compile was replaced on import"
    finally:
        sys.modules.pop("utils.inject_roctx", None)
