# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************

"""Tests for the LLVM IR support added in NCCL 2.29.2.

The release notes describe:
  "Added support to use LLVM intermediate representation (IR) to generate
   the NCCL device library wrappers in compilation."

These tests verify that:
  1. The IR build target produces an .ll / .bc artifact.
  2. The produced IR is well-formed LLVM IR (can be parsed by llvm-as).
  3. Selected NCCL device wrapper symbols are present in the IR module.

The tests skip when the IR target hasn't been built (which is the default
in the current RCCL build until the wrapper feature is enabled).
"""

import os
import shutil
import subprocess

import pytest


_IR_CANDIDATE_DIRS = (
    "bindings/ir/build",
    "build/release/bindings/ir",
    "build/bindings/ir",
)

_IR_FILE_EXTENSIONS = (".ll", ".bc")


def _repo_root():
    here = os.path.abspath(os.path.dirname(__file__))
    # tests/ -> ir/ -> bindings/ -> repo root
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _find_ir_artifact():
    root = _repo_root()
    for d in _IR_CANDIDATE_DIRS:
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            continue
        for dirpath, _, files in os.walk(full):
            for f in files:
                if f.endswith(_IR_FILE_EXTENSIONS):
                    return os.path.join(dirpath, f)
    return None


@pytest.mark.bindings_ir
def test_ir_artifact_exists():
    """At least one .ll or .bc IR artifact must exist after the IR target
    is built."""
    artifact = _find_ir_artifact()
    if artifact is None:
        pytest.skip("No IR artifact found under bindings/ir/build or any "
                    "RCCL build directory. Build the IR target before "
                    "running this test.")
    assert os.path.getsize(artifact) > 0, (
        f"IR artifact {artifact} exists but is empty."
    )


@pytest.mark.bindings_ir
def test_ir_artifact_is_well_formed():
    """The artifact must parse as valid LLVM IR (llvm-as for .ll,
    llvm-dis for .bc)."""
    artifact = _find_ir_artifact()
    if artifact is None:
        pytest.skip("No IR artifact found; nothing to validate.")

    if artifact.endswith(".ll"):
        tool = "llvm-as"
    else:
        tool = "llvm-dis"

    if shutil.which(tool) is None:
        pytest.skip(f"{tool} not installed; cannot validate IR shape.")

    result = subprocess.run([tool, artifact, "-o", os.devnull],
                            capture_output=True, text=True)
    assert result.returncode == 0, (
        f"{tool} rejected {artifact}:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.bindings_ir
def test_ir_contains_device_wrapper_symbols():
    """Selected NCCL device wrapper symbols must appear in the IR module.

    We look for any of a small set of canonical symbol-name fragments;
    a single hit is enough to confirm the wrapper generation actually
    produced device code.
    """
    artifact = _find_ir_artifact()
    if artifact is None:
        pytest.skip("No IR artifact found; nothing to inspect.")

    # For .bc we'd need llvm-dis first. For simplicity, only inspect .ll.
    if not artifact.endswith(".ll"):
        pytest.skip("IR artifact is bitcode (.bc); textual symbol check "
                    "is only run against .ll files.")

    with open(artifact, "r") as f:
        text = f.read()

    needles = (
        "ncclAllReduce", "ncclAllGather", "ncclReduceScatter",
        "ncclSendRecv",  "ncclBroadcast", "ncclDevComm",
        "nccl_device", "ncclReduceCopy", "ncclCopy",
    )
    hits = [n for n in needles if n in text]
    assert hits, (
        f"None of the expected NCCL device-wrapper symbols were found in "
        f"the IR module {artifact}. Searched for: {needles}"
    )
