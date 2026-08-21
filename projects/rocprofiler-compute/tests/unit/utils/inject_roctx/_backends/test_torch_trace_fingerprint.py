# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.inject_roctx._backends.torch_trace_fingerprint. No GPU.

The artifact tag has one implementation shared by the loader and by the
collector's CMakeLists.txt, so these pin its shape, its inputs, and the
sentinel it produces when the sources cannot be read.
"""

import os
import subprocess
import sys

import common  # noqa: F401
import pytest

from utils.inject_roctx._backends import torch_trace_fingerprint

# ---------------------------------------------------------------------------
# source_fingerprint
# ---------------------------------------------------------------------------


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
    assert (
        torch_trace_fingerprint.source_fingerprint()
        == torch_trace_fingerprint.MISSING_FINGERPRINT
    )


def test_source_fingerprint_is_missing_sentinel_when_one_input_is_unreadable(
    tmp_path,
    monkeypatch,
):
    """A partially readable tree must not produce a real-looking fingerprint.

    Hashing the readable subset would name an artifact that does not correspond
    to the sources it was built from.
    """
    readable = tmp_path / "torch_trace_collector.cpp"
    readable.write_text("// present\n")
    monkeypatch.setattr(
        torch_trace_fingerprint,
        "fingerprint_input_paths",
        lambda: (readable, tmp_path / "absent.h"),
    )
    assert (
        torch_trace_fingerprint.source_fingerprint()
        == torch_trace_fingerprint.MISSING_FINGERPRINT
    )


def test_source_fingerprint_separates_input_boundaries(tmp_path, monkeypatch):
    """Reshuffling bytes across input boundaries changes the fingerprint."""
    first = tmp_path / "a.cpp"
    second = tmp_path / "b.txt"
    monkeypatch.setattr(
        torch_trace_fingerprint,
        "fingerprint_input_paths",
        lambda: (first, second),
    )

    first.write_bytes(b"AB")
    second.write_bytes(b"C")
    fingerprint_one = torch_trace_fingerprint.source_fingerprint()
    first.write_bytes(b"A")
    second.write_bytes(b"BC")
    fingerprint_two = torch_trace_fingerprint.source_fingerprint()
    assert fingerprint_one != fingerprint_two, (
        "fingerprint collided across an input boundary"
    )


# ---------------------------------------------------------------------------
# required_input_paths
# ---------------------------------------------------------------------------


def test_required_input_paths_all_exist_in_the_source_tree():
    """Every named required input is present, so the loader's pre-build check
    does not veto a build over a stale name list.
    """
    required = torch_trace_fingerprint.required_input_paths()
    missing = [p for p in required if not p.exists()]
    assert not missing, f"required inputs missing from the source tree: {missing}"


def test_required_input_paths_are_a_subset_of_the_fingerprint_inputs():
    """Anything required to build is also hashed into the tag."""
    fingerprinted = set(torch_trace_fingerprint.fingerprint_input_paths())
    required = torch_trace_fingerprint.required_input_paths()
    unhashed = [p for p in required if p not in fingerprinted]
    assert not unhashed, f"required inputs absent from the fingerprint: {unhashed}"


def test_required_input_paths_cover_every_collector_header():
    """The hand-written header list does not drift from the source directory."""
    on_disk = {p.name for p in torch_trace_fingerprint._SO_SOURCE_DIR.glob("*.h")}
    named = set(torch_trace_fingerprint._COLLECTOR_HEADER_NAMES)
    assert on_disk == named, (
        f"update _COLLECTOR_HEADER_NAMES; only on disk: {sorted(on_disk - named)}, "
        f"only named: {sorted(named - on_disk)}"
    )


def test_required_input_paths_cover_every_collector_source():
    """The hand-written source list does not drift from the source directory."""
    on_disk = {p.name for p in torch_trace_fingerprint._SO_SOURCE_DIR.glob("*.cpp")}
    named = set(torch_trace_fingerprint._COLLECTOR_SOURCE_NAMES)
    assert on_disk == named, (
        f"update _COLLECTOR_SOURCE_NAMES; only on disk: {sorted(on_disk - named)}, "
        f"only named: {sorted(named - on_disk)}"
    )


# ---------------------------------------------------------------------------
# artifact_tag
# ---------------------------------------------------------------------------


def test_artifact_tag_is_well_formed():
    """The tag follows ``py<X>.<Y>_torch<ver>_src<12-hex>``."""
    tag = torch_trace_fingerprint.artifact_tag()
    if tag is None:
        pytest.skip("torch not importable")
    parts = tag.split("_")

    assert parts[0] == f"py{sys.version_info.major}.{sys.version_info.minor}"

    torch_components = [p for p in parts if p.startswith("torch")]
    assert len(torch_components) == 1, (
        f"expected exactly one '_torch...' component in tag {tag!r}"
    )

    src_components = [p for p in parts if p.startswith("src")]
    assert len(src_components) == 1, (
        f"expected exactly one '_src...' component in tag {tag!r}"
    )
    src_value = src_components[0][len("src") :]
    assert src_value == torch_trace_fingerprint.MISSING_FINGERPRINT or (
        len(src_value) == 12 and all(c in "0123456789abcdef" for c in src_value)
    ), f"unexpected src component {src_value!r} in tag {tag!r}"


def test_artifact_tag_is_stable_across_calls():
    """The tag is the same for repeated calls in one process."""
    assert torch_trace_fingerprint.artifact_tag() == (
        torch_trace_fingerprint.artifact_tag()
    )


def test_torch_version_carries_no_local_segment():
    """CMake reads Torch_VERSION, which has no ``+rocm`` segment, so neither
    does this.
    """
    version = torch_trace_fingerprint.torch_version()
    if version is None:
        pytest.skip("torch not importable")
    assert "+" not in version


def test_artifact_tag_is_none_without_torch(monkeypatch):
    """No torch means no tag, which is what sends the loader to the Python tier."""
    monkeypatch.setattr(torch_trace_fingerprint, "torch_version", lambda: None)
    assert torch_trace_fingerprint.artifact_tag() is None


# ---------------------------------------------------------------------------
# CMake contract
# ---------------------------------------------------------------------------


def test_running_the_module_prints_the_tag():
    """The collector's CMakeLists.txt resolves the tag by running this module,
    so running it must print exactly what ``artifact_tag()`` returns.
    """
    result = subprocess.run(
        [sys.executable, torch_trace_fingerprint.__file__],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (torch_trace_fingerprint.artifact_tag() or "")


def test_importing_the_module_does_not_import_torch():
    """The loader imports this module where torch may be absent."""
    code = (
        "import sys\n"
        "from utils.inject_roctx._backends import torch_trace_fingerprint\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "PYTHONPATH": str(torch_trace_fingerprint._NATIVE_TOOL_ROOT),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
