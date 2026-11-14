# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Outputs of the nccl4py build tooling (upstream nccl4py commits 59a784c3 and 76c74740).

``nccl.core._version`` reports the installed ``nccl4py`` distribution's version, which
setuptools-scm stamps at build time, and ``0+unknown`` when none is installed. The bindings
generated from the real NCCL headers type ``ncclDevResourceRequirements_t.next`` as a
``DevResourceRequirements`` link, which nccl.core uses to chain resource requirements, where it
used to be a raw address.

No GPU is needed. The version tests need only the standard library, the bindings tests the
extensions that run_inplace.sh builds. Without pytest: ``python3 <this file> CASE`` (see main).
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


def test_control_bindings_load():
    from nccl.bindings import nccl as bindings

    req = bindings.DevResourceRequirements()
    req.buffer_size = 4096
    assert req.buffer_size == 4096 and req.ptr != 0


def test_resource_requirements_link_struct_objects():
    from nccl.bindings import nccl as bindings

    head, tail = bindings.DevResourceRequirements(), bindings.DevResourceRequirements()
    head.next = tail  # TypeError while next is a raw address
    # The getter hands back the stored node only while the C field holds its address.
    assert head.next is tail
    head.next = None
    assert head.next is None


def _in_place(test):
    """Runs one bindings test under run_inplace.sh, which exits 77 when it cannot build them."""
    cmd = ["bash", str(HERE / "run_inplace.sh"), "-q", "-rs", f"{Path(__file__).name}::{test}"]
    return subprocess.run(cmd, check=False, cwd=HERE).returncode


def main(case):
    """Exits 0 on a pass, 1 on a failed check, 77 when the bindings cannot be built, else 2."""
    if case == "version":
        test_version_from_distribution_metadata()
        return 0
    if case == "codegen":
        return _in_place("test_resource_requirements_link_struct_objects")
    if case == "control":
        test_control_version_module_runs()
        status = _in_place("test_control_bindings_load")
        if status == 77:  # the codegen case skips then too
            print("control: no bindings build here, checked _version.py only", file=sys.stderr)
        return 0 if status in (0, 77) else status
    print(f"usage: {sys.argv[0]} version|codegen|control", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
    except Exception as exc:  # noqa: BLE001 - a broken harness is an error, not a failed check
        traceback.print_exc()
        sys.exit(1 if isinstance(exc, AssertionError) else 2)
