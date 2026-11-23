# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks nccl4py's version metadata and the generated ``DevResourceRequirements.next`` link.

No GPU needed. The ``test/nccl4py`` build-smoke harness runs it from ``CPU_SMOKE_TESTS``. Without
pytest, run ``python3 <this file> version|codegen|control`` with a Python that has nccl4py
installed.
"""

import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION_PY = HERE.parent / "nccl" / "core" / "_version.py"


def _version_seen_with(installed):
    """``__version__`` of this tree's _version.py with only nccl4py ``installed`` (or none)."""
    with tempfile.TemporaryDirectory() as site:
        if installed:
            info = Path(site) / f"nccl4py-{installed}.dist-info"
            info.mkdir()
            metadata = f"Metadata-Version: 2.1\nName: nccl4py\nVersion: {installed}\n"
            info.joinpath("METADATA").write_text(metadata)
        code = "import runpy, sys; sys.path.insert(0, sys.argv[1]); "
        code += "print(runpy.run_path(sys.argv[2])['__version__'])"
        # -I -S: no site-packages, so an nccl4py installed on the host cannot answer.
        cmd = [sys.executable, "-I", "-S", "-c", code, site, str(VERSION_PY)]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{VERSION_PY} did not run:\n{proc.stderr}")
    return proc.stdout.strip()


def test_control_version_module_runs():
    assert _version_seen_with(None)


def test_version_from_distribution_metadata():
    for installed in ("9.8.7", None):
        seen = _version_seen_with(installed)
        assert seen == (installed or "0+unknown"), f"installed: {installed}, __version__: {seen}"


def test_package_version_is_core_version():
    import nccl
    from nccl.core import _version

    assert nccl.__version__ == _version.__version__


def test_control_bindings_load():
    from nccl.bindings import nccl as bindings

    req = bindings.DevResourceRequirements()
    req.buffer_size = 4096
    assert req.buffer_size == 4096 and req.ptr != 0


def test_resource_requirements_link_struct_objects():
    from nccl.bindings import nccl as bindings

    head, tail, other = (bindings.DevResourceRequirements() for _ in range(3))
    head.next = tail  # TypeError while next is a raw address
    assert head.next is tail
    # The getter hands back the stored node only while the C field holds its address, so a
    # second view of the same struct that rewrites the field changes what head.next returns.
    view = bindings.DevResourceRequirements.from_ptr(head.ptr, owner=head)
    view.next = other
    assert head.next is not tail and head.next.ptr == other.ptr
    head.next = None
    assert head.next is None and view.next is None


def _with_nccl4py(*tests):
    """Runs ``tests`` in this Python; 77 when it cannot import nccl4py's bindings."""
    try:
        import nccl.bindings  # noqa: F401
    except ImportError as exc:
        print(f"SKIP: nccl4py is not importable here: {exc}", file=sys.stderr)
        return 77
    for test in tests:
        test()
    return 0


def main(case):
    """Exits 0 on a pass, 1 on a failed check, 77 without an importable nccl4py, else 2."""
    if case == "version":
        test_version_from_distribution_metadata()  # standard library only
        return _with_nccl4py(test_package_version_is_core_version)
    if case == "codegen":
        return _with_nccl4py(test_resource_requirements_link_struct_objects)
    if case == "control":
        test_control_version_module_runs()
        return _with_nccl4py(test_control_bindings_load)
    print(f"usage: {sys.argv[0]} version|codegen|control", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
    except Exception as exc:  # noqa: BLE001 - a broken harness is an error, not a failed check
        traceback.print_exc()
        sys.exit(1 if isinstance(exc, AssertionError) else 2)
