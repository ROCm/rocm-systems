#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Verify the generic Torch collector's ELF and plain-C ABI contract."""

import argparse
import re
import subprocess
from pathlib import Path
from typing import FrozenSet, List

EXPECTED_EXPORTS = frozenset({
    "torch_trace_collector_abi_revision",
    "torch_trace_collector_install",
    "torch_trace_collector_pop_launcher_tid",
    "torch_trace_collector_push_launcher_tid",
})
EXPECTED_TORCH_IMPORTS = frozenset({
    "_ZN2at14RecordFunction15currentThreadIdEv",
    "_ZN2at17addGlobalCallbackENS_22RecordFunctionCallbackE",
    "_ZN3c1010Dispatcher13realSingletonEv",
    "_ZN3c1010Dispatcher17findSchemaOrThrowEPKcS2_",
    "_ZN3c1014OperatorHandleD1Ev",
    "_ZN3c1020ThreadLocalDebugInfo3getENS_13DebugInfoKindE",
    "_ZN3c1020ThreadLocalDebugInfo4_popENS_13DebugInfoKindE",
    "_ZN3c1020ThreadLocalDebugInfo5_pushENS_13DebugInfoKindESt10shared_ptrINS_13DebugInfoBaseEE",
    "_ZNK2at14RecordFunction13operator_nameEv",
    "_ZNK2at14RecordFunction4nameEv",
    "_ZNK3c104impl13OperatorEntry9dumpStateB5cxx11Ev",
    "_ZNK3c106IValue12isTensorListEv",
    "_ZNK3c106IValue5visitERKSt8functionIFbRKS0_EE",
    "aoti_torch_abi_version",
    "aoti_torch_dtype_bfloat16",
    "aoti_torch_dtype_bool",
    "aoti_torch_dtype_complex128",
    "aoti_torch_dtype_complex32",
    "aoti_torch_dtype_complex64",
    "aoti_torch_dtype_float16",
    "aoti_torch_dtype_float32",
    "aoti_torch_dtype_float64",
    "aoti_torch_dtype_int16",
    "aoti_torch_dtype_int32",
    "aoti_torch_dtype_int64",
    "aoti_torch_dtype_int8",
    "aoti_torch_dtype_uint8",
    "aoti_torch_get_dim",
    "aoti_torch_get_dtype",
    "aoti_torch_get_sizes",
    "aoti_torch_is_defined",
})
EXPECTED_ROCTX_IMPORTS = frozenset({"roctxRangePop", "roctxRangePushA"})
PYTORCH_C_IMPORT_PREFIXES = ("aoti_torch_", "at_", "c10_", "torch_", "TH")
PYTHON_C_IMPORT_PREFIXES = ("Py", "_Py")
PYTORCH_MANGLED_NAMESPACE_MARKERS = (
    "N2at",
    "NK2at",
    "N3c10",
    "NK3c10",
    "N4c10d",
    "NK4c10d",
    "N5torch",
    "NK5torch",
    "N6caffe2",
    "NK6caffe2",
)
FORBIDDEN_NEEDED_PREFIXES = ("libtorch", "libc10", "libpython")


def parse_args() -> argparse.Namespace:
    """Parse paths supplied by CMake."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collector", type=Path)
    parser.add_argument("--nm", type=Path, required=True)
    parser.add_argument("--readelf", type=Path, required=True)
    return parser.parse_args()


def run_tool(command: List[str]) -> str:
    """Run one ELF inspection tool and return its standard output."""
    try:
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ELF inspection command failed: {error}") from None
    return process.stdout


def parse_dynamic_symbols(output: str) -> FrozenSet[str]:
    """Parse and normalize dynamic symbol names from nm output."""
    return frozenset(
        line.split(maxsplit=1)[0].split("@", maxsplit=1)[0]
        for line in output.splitlines()
        if line
    )


def defined_dynamic_symbols(nm: Path, collector: Path) -> FrozenSet[str]:
    """Return symbols defined by the collector."""
    return parse_dynamic_symbols(
        run_tool([str(nm), "-D", "--defined-only", "--format=posix", str(collector)])
    )


def undefined_dynamic_symbols(nm: Path, collector: Path) -> FrozenSet[str]:
    """Return symbols imported by the collector."""
    return parse_dynamic_symbols(
        run_tool([
            str(nm),
            "-D",
            "--undefined-only",
            "--format=posix",
            str(collector),
        ])
    )


def dynamic_section(readelf: Path, collector: Path) -> str:
    """Return the collector's dynamic section."""
    return run_tool([str(readelf), "--dynamic", "--wide", str(collector)])


def needed_libraries(section: str) -> List[str]:
    """Extract DT_NEEDED library names from readelf output."""
    return re.findall(r"\(NEEDED\).*Shared library: \[([^]]+)]", section)


def require_equal(actual: FrozenSet[str], expected: FrozenSet[str], label: str) -> None:
    """Raise a useful error when an ELF symbol contract changes."""
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SystemExit(f"{label} mismatch: missing={missing}, extra={extra}")


def is_torch_import(symbol: str) -> bool:
    """Return whether a raw ELF symbol belongs to a PyTorch ABI namespace."""
    return symbol.startswith(PYTORCH_C_IMPORT_PREFIXES) or any(
        marker in symbol for marker in PYTORCH_MANGLED_NAMESPACE_MARKERS
    )


def main() -> None:
    """Validate the built collector without loading PyTorch."""
    args = parse_args()
    collector = args.collector.resolve()
    if collector.name != "torch_trace_collector.so":
        raise SystemExit(f"unexpected collector filename: {collector.name}")

    validate_symbols(args.nm, collector)
    validate_dynamic_section(args.readelf, collector)


def validate_symbols(nm: Path, collector: Path) -> None:
    """Validate the collector's exported and imported symbol contracts."""
    require_equal(
        defined_dynamic_symbols(nm, collector),
        EXPECTED_EXPORTS,
        "exported symbols",
    )

    undefined_symbols = undefined_dynamic_symbols(nm, collector)
    torch_imports = frozenset(
        symbol for symbol in undefined_symbols if is_torch_import(symbol)
    )
    require_equal(torch_imports, EXPECTED_TORCH_IMPORTS, "Torch imports")
    roctx_imports = frozenset(
        symbol for symbol in undefined_symbols if symbol.startswith("roctx")
    )
    require_equal(roctx_imports, EXPECTED_ROCTX_IMPORTS, "ROCTX imports")

    python_imports = sorted(
        symbol
        for symbol in undefined_symbols
        if symbol.startswith(PYTHON_C_IMPORT_PREFIXES)
    )
    if python_imports:
        raise SystemExit(f"collector has Python C-API imports: {python_imports}")


def validate_dynamic_section(readelf: Path, collector: Path) -> None:
    """Reject linked Torch/Python dependencies and embedded search paths."""
    section = dynamic_section(readelf, collector)
    forbidden_dependencies = [
        library
        for library in needed_libraries(section)
        if library.lower().startswith(FORBIDDEN_NEEDED_PREFIXES)
    ]
    if forbidden_dependencies:
        raise SystemExit(
            f"collector has forbidden DT_NEEDED entries: {forbidden_dependencies}"
        )
    if "(RPATH)" in section or "(RUNPATH)" in section:
        raise SystemExit("collector must not contain DT_RPATH or DT_RUNPATH")


if __name__ == "__main__":
    main()
