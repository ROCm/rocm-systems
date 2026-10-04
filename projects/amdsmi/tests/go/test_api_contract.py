# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import re
import sys
from pathlib import Path

COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
ENUM_BLOCKS = re.compile(r"typedef\s+enum\s*\{([^{}]*)\}\s*(\w+)\s*;", re.S)
ENUM_MEMBERS = re.compile(r"(?:^|,)\s*(AMDSMI_[A-Z0-9_]+)\b")
SOURCE_TOKENS = re.compile(
    r"/\*.*?\*/|//[^\n]*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`[^`]*`", re.S
)
CONST_DECLARATIONS = re.compile(r"\bconst\s+(?:\((.*?)\)|([^\n;]+))", re.S)
VALUE_DECLARATIONS = re.compile(r"\b(?:const|var)\s+(?:\((.*?)\)|([^\n;]+))", re.S)
CONSTANT_TYPES = re.compile(r"(?:^|[;\n])\s*(AMDSMI_[A-Z0-9_]+)\b(?:[ \t]+(\w+))?")
DIRECT_BINDINGS = re.compile(
    r"(?:^|[;\n])\s*(AMDSMI_[A-Z0-9_]+)\s+(\w+)\s*=\s*"
    r"C\.(AMDSMI_[A-Z0-9_]+)[ \t]*(?=[;\n]|$)"
)
C_IMPORT = re.compile(r'\s*import\s+"C"')
GO_CALLS = re.compile(r"\bC\s*\.\s*(amdsmi_[a-z0-9_]+)\s*\(")
C_CALLS = re.compile(r"\b(amdsmi_[a-z0-9_]+)\s*\(")
MODULE_DEPENDENCIES = re.compile(r"^\s*(require|replace|exclude|toolchain)\b", re.M)
REQUIRED_ENUMS = {
    "amdsmi_status_t": "Status",
    "amdsmi_vram_type_t": "VramType",
    "amdsmi_fw_block_t": "FwBlock",
    "amdsmi_temperature_type_t": "TemperatureType",
    "amdsmi_temperature_metric_t": "TemperatureMetric",
    "amdsmi_clk_type_t": "ClkType",
    "amdsmi_memory_type_t": "MemoryType",
    "amdsmi_memory_partition_type_t": "MemoryPartitionType",
    "amdsmi_accelerator_partition_type_t": "AcceleratorPartitionType",
    "amdsmi_gpu_block_t": "GpuBlock",
    "amdsmi_ras_err_state_t": "RASState",
}
ALLOWED_NATIVE_CALLS = {
    "amdsmi_init",
    "amdsmi_shut_down",
    "amdsmi_status_code_to_string",
    "amdsmi_get_lib_version",
    "amdsmi_get_socket_handles",
    "amdsmi_get_processor_handles",
    "amdsmi_get_processor_type",
    "amdsmi_get_processor_handle_from_bdf",
    "amdsmi_get_gpu_device_bdf",
    "amdsmi_get_gpu_device_uuid",
    "amdsmi_get_gpu_asic_info",
    "amdsmi_get_gpu_driver_info",
    "amdsmi_get_gpu_board_info",
    "amdsmi_get_fw_info",
    "amdsmi_get_gpu_vbios_info",
    "amdsmi_get_temp_metric",
    "amdsmi_get_power_info",
    "amdsmi_get_power_cap_info",
    "amdsmi_get_clock_info",
    "amdsmi_get_clk_freq",
    "amdsmi_get_gpu_activity",
    "amdsmi_get_gpu_memory_total",
    "amdsmi_get_gpu_memory_usage",
    "amdsmi_get_gpu_vram_info",
    "amdsmi_get_gpu_kfd_info",
    "amdsmi_get_gpu_memory_partition_config",
    "amdsmi_get_gpu_accelerator_partition_profile",
    "amdsmi_get_gpu_ecc_enabled",
    "amdsmi_get_gpu_ecc_count",
    "amdsmi_get_gpu_total_ecc_count",
    "amdsmi_get_gpu_ras_block_features_enabled",
    "amdsmi_get_gpu_ras_feature_info",
}


def source_code(source: str) -> str:
    return SOURCE_TOKENS.sub(lambda match: " " + "\n" * match.group().count("\n"), source)


def enum_members(header: str, type_name: str) -> list:
    for body, name in ENUM_BLOCKS.findall(COMMENTS.sub(" ", header)):
        if name == type_name:
            return ENUM_MEMBERS.findall(body)
    raise ValueError("native enum not found: " + type_name)


def check_constants(*, header: str, source: str, native_type: str, go_type: str) -> None:
    members = enum_members(header, native_type)
    if not members:
        raise ValueError("empty native enum: " + native_type)
    bindings = set()
    for block, single in CONST_DECLARATIONS.findall(source_code(source)):
        bindings.update(DIRECT_BINDINGS.findall(block or single))
    for name in members:
        if (name, go_type, name) not in bindings:
            raise ValueError("missing direct constant binding: " + name + " (" + go_type + ")")


def check_enums(*, header: str, source: str, available_only: bool) -> None:
    declared = {}
    for block, single in VALUE_DECLARATIONS.findall(source_code(source)):
        for name, go_type in CONSTANT_TYPES.findall(block or single):
            declared.setdefault(name, set()).add(go_type)
    for native_type, go_type in REQUIRED_ENUMS.items():
        if not available_only or declared.keys() & set(enum_members(header, native_type)):
            check_constants(header=header, source=source, native_type=native_type, go_type=go_type)
    for body, native_type in ENUM_BLOCKS.findall(COMMENTS.sub(" ", header)):
        members = ENUM_MEMBERS.findall(body)
        mirrored = declared.keys() & set(members)
        if not mirrored or native_type in REQUIRED_ENUMS:
            continue
        if native_type == "amdsmi_init_flags_t":
            # The Go lifecycle accepts AMD GPUs only.
            members = ["AMDSMI_INIT_AMD_GPUS"]
            go_type = "InitFlags"
            if mirrored - set(members):
                raise ValueError("only AMDSMI_INIT_AMD_GPUS may be exported")
        else:
            types = {go_type for name in mirrored for go_type in declared[name]}
            if len(types) != 1 or "" in types:
                raise ValueError("inconsistent constant binding types: " + native_type)
            go_type = types.pop()
        subset = "typedef enum { " + ", ".join(members) + " } " + native_type + ";"
        check_constants(header=subset, source=source, native_type=native_type, go_type=go_type)


def check_module(source: str) -> None:
    source = COMMENTS.sub(" ", source)
    if MODULE_DEPENDENCIES.search(source):
        raise ValueError("Go module must not declare dependencies or a toolchain")
    if source.split() != [
        "module",
        "github.com/ROCm/rocm-systems/projects/amdsmi/go",
        "go",
        "1.20",
    ]:
        raise ValueError("Go module must declare only the planned module and go 1.20")


def check_project(*, project: Path, available_only: bool) -> None:
    check_module((project / "go" / "go.mod").read_text(encoding="utf-8"))
    header = (project / "include" / "amd_smi" / "amdsmi.h").read_text(encoding="utf-8")
    package = project / "go" / "amdsmi"
    sources = {
        path: path.read_text(encoding="utf-8")
        for path in sorted((project / "go").rglob("*.go"))
        if not (
            path.name.endswith("_test.go")
            or path.name == "mock_bridge_linux.go"
            or "testdata" in path.relative_to(project / "go").parts
        )
    }
    source = "\n".join(text for path, text in sources.items() if path.parent == package)
    check_enums(header=header, source=source, available_only=available_only)
    calls = set()
    for text in sources.values():
        calls.update(native_calls(text))
    check_native_calls(calls, available_only=available_only)


def check_native_calls(calls: set, *, available_only: bool) -> None:
    extra = calls - ALLOWED_NATIVE_CALLS
    if extra:
        raise ValueError("prohibited native calls: " + ", ".join(sorted(extra)))
    missing = ALLOWED_NATIVE_CALLS - calls
    if missing and not available_only:
        raise ValueError("missing native calls: " + ", ".join(sorted(missing)))


def native_calls(source: str) -> set:
    calls = set(GO_CALLS.findall(source_code(source)))
    for token in SOURCE_TOKENS.finditer(source):
        if token.group().startswith("/*") and C_IMPORT.match(source, token.end()):
            calls.update(C_CALLS.findall(source_code(token.group()[2:-2])))
    return {
        name
        for name in calls
        if not name.endswith("_t")
        and name not in {"amdsmi_processor_handle", "amdsmi_socket_handle"}
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the AMD SMI Go API contract")
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="check existing enum owners and allow missing native calls",
    )
    args = parser.parse_args()
    try:
        check_project(
            project=Path(__file__).resolve().parents[2], available_only=args.available_only
        )
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print("Go API contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
