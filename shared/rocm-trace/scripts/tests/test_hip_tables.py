# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""The in-tree HIP dispatch tables parse, and every entry gets a probe."""

import json
import re

import pytest

from conftest import DATA, REPO, generate_files, resource_dir
from rocm_trace_gen.parse import parse_tables
from test_schema import assert_schema_matches_source

HIP_INCLUDE = REPO / "projects" / "hip" / "include"
CLR_INCLUDE = REPO / "projects" / "clr" / "hipamd" / "include"
CLR_TRACE_SOURCE = REPO / "projects" / "clr" / "hipamd" / "src" / "hip_api_trace.cpp"

pytestmark = pytest.mark.skipif(
    not (HIP_INCLUDE.is_dir() and CLR_INCLUDE.is_dir() and CLR_TRACE_SOURCE.is_file()),
    reason="not in a rocm-systems tree with projects/hip and projects/clr",
)


@pytest.fixture(scope="module")
def clang_args():
    rd = resource_dir()
    if rd is None:
        pytest.skip("no clang resource directory; set ROCM_TRACE_GEN_RESOURCE_DIR")
    return [
        "-resource-dir",
        rd,
        f"-I{DATA / 'hip_version_stub'}",
        f"-I{CLR_INCLUDE}",
        f"-I{HIP_INCLUDE}",
        "-D__HIP_PLATFORM_AMD__",
    ]


def _abi_slot_counts():
    """{table: function slot count} from CLR's HIP_ENFORCE_ABI_VERSIONING checks."""
    text = CLR_TRACE_SOURCE.read_text()
    return {
        name: int(count)
        for name, count in re.findall(
            r"HIP_ENFORCE_ABI_VERSIONING\((\w+),\s*(\d+)\)", text
        )
    }


def test_slot_counts_match_clr(clang_args):
    expected = _abi_slot_counts()
    names = ["HipDispatchTable", "HipCompilerDispatchTable"]
    tables = parse_tables(
        [
            "hip/hip_runtime.h",
            "hip/hip_deprecated.h",
            "hip/amd_detail/amd_hip_gl_interop.h",
            "hip/amd_detail/hip_api_trace.hpp",
        ],
        [(n, "size") for n in names],
        clang_args,
    )
    for table in tables:
        assert table.name in expected, f"no HIP_ENFORCE_ABI_VERSIONING for {table.name}"
        assert len(table.slots) + len(table.reserved) == expected[table.name]


def test_every_entry_gets_a_probe(clang_args):
    source = generate_files(DATA / "hip_minimal.yaml", clang_args)[
        "hip_trace_probes.cpp"
    ]
    expected = _abi_slot_counts()
    installed = len(re.findall(r"table->\w+ = probe_", source))
    assert (
        installed == expected["HipDispatchTable"] + expected["HipCompilerDispatchTable"]
    )


def test_hip_schema_matches_its_probes(clang_args):
    files = generate_files(DATA / "hip_minimal.yaml", clang_args)
    assert_schema_matches_source(files, "hip_trace_probes")
    schema = json.loads(files["hip_trace_probes.json"])
    kinds = schema["enums"]["hipMemcpyKind"]["enumerators"]
    assert kinds["hipMemcpyHostToDevice"] == 1
    memcpy = next(e for e in schema["events"] if e["name"] == "hipMemcpyAsync_enter")
    kind = next(f for f in memcpy["fields"] if f["name"] == "kind")
    assert kind["enum"] == "hipMemcpyKind"
