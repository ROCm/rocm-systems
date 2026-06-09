#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from api_diff import diff_manifests, load_manifest, render_diff
from clang_extract import ExtractedApi, ExtractedFunction, extract_header_api


@dataclass(frozen=True)
class Param:
    type: str
    name: str


@dataclass(frozen=True)
class Signature:
    return_type: str
    params: tuple[Param, ...]

    def declaration_params(self) -> str:
        if not self.params:
            return "void"
        return ", ".join(format_param(param) for param in self.params)

    def call_args(self) -> str:
        return ", ".join(param.name for param in self.params)


@dataclass(frozen=True)
class VersionNode:
    name: str
    parent: str | None
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class ApiEntry:
    public_symbol: str
    signature_symbol: str
    signature: Signature
    version_node: str
    backend_symbol: str
    backend_signature_symbol: str
    backend_signature: Signature
    loader_owned: bool
    stale: bool
    compat_function: str | None
    compatibility_reason: str | None


@dataclass(frozen=True)
class GeneratedPaths:
    include_dir: Path
    manifest: Path
    loader_v6: Path
    loader_v7: Path
    backend: Path
    wrong_major_backend: Path
    missing_handshake_backend: Path
    bad_public_export_backend: Path
    bad_compiler_export_backend: Path
    preload_interposer: Path
    v6_map: Path
    v7_map: Path
    v6_def: Path
    v7_def: Path


def format_param(param: Param) -> str:
    if "(*)" in param.type:
        return param.type.replace("(*)", f"(*{param.name})")
    return f"{param.type} {param.name}"


def split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "(<[":
            depth += 1
        elif char in ")>]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def parse_param(text: str, index: int) -> Param:
    text = remove_default_argument(text)
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s*__dparm\s*\([^)]*\)", "", text)
    if text == "void":
        raise ValueError("void is not a parameter")
    function_pointer = re.match(r"(.+\(\*)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\)\s*\(.*\))$", text)
    if function_pointer:
        return Param(type=f"{function_pointer.group(1)}{function_pointer.group(3)}",
                     name=function_pointer.group(2))
    match = re.match(r"(.+?)([A-Za-z_][A-Za-z0-9_]*)$", text)
    if not match:
        return Param(type=text, name=f"arg{index}")
    param_type = match.group(1).strip()
    name = match.group(2)
    return Param(type=param_type, name=name)


def parse_signature(return_type: str, args: str) -> Signature:
    args = re.sub(r"\s+", " ", args.strip())
    if not args or args == "void":
        return Signature(return_type=return_type.strip(), params=())
    params = tuple(parse_param(part, index) for index, part in enumerate(split_top_level_commas(args)))
    return Signature(return_type=return_type.strip(), params=params)


def signature_from_json(value: dict[str, object]) -> Signature:
    params_value = value.get("params", [])
    if not isinstance(params_value, list):
        raise ValueError("signature params must be a list")
    params: list[Param] = []
    for param_value in params_value:
        if not isinstance(param_value, dict):
            raise ValueError("signature param must be an object")
        param_type = param_value.get("type")
        name = param_value.get("name")
        if not isinstance(param_type, str) or not isinstance(name, str):
            raise ValueError("signature param requires string type/name")
        params.append(Param(param_type, name))
    return_type = value.get("return_type")
    if not isinstance(return_type, str):
        raise ValueError("signature requires string return_type")
    return Signature(return_type=return_type, params=tuple(params))


def signature_from_extracted(function: ExtractedFunction) -> Signature:
    return Signature(
        return_type=function.return_type,
        params=tuple(Param(param.type, param.name) for param in function.params),
    )


def signatures_from_extracted(api: ExtractedApi) -> dict[str, Signature]:
    return {
        name: signature_from_extracted(function)
        for name, function in api.functions.items()
    }


def semantic_manifest_from_api(api: ExtractedApi) -> dict[str, object]:
    return {
        "schema_version": 1,
        "generator": "clang-ast",
        "functions": {
            name: {
                "return_type": function.return_type,
                "params": [
                    {"type": param.type, "name": param.name}
                    for param in function.params
                ],
                "location": function.location,
            }
            for name, function in sorted(api.functions.items())
        },
        "records": {
            name: {
                "fields": [
                    {"type": field.type, "name": field.name}
                    for field in record.fields
                ],
                "location": record.location,
            }
            for name, record in sorted(api.records.items())
        },
        "enums": {
            name: {
                "constants": [
                    {"name": constant_name, "value": constant_value}
                    for constant_name, constant_value in enum.constants
                ],
                "location": enum.location,
            }
            for name, enum in sorted(api.enums.items())
        },
    }


def parse_trace_typedefs(trace_header: Path) -> dict[str, Signature]:
    text = trace_header.read_text()
    pattern = re.compile(r"typedef\s+(.+?)\(\*t_([A-Za-z_][A-Za-z0-9_]*)\)\s*\((.*?)\);", re.S)
    signatures: dict[str, Signature] = {}
    for match in pattern.finditer(text):
        return_type = re.sub(r"\s+", " ", match.group(1).strip())
        symbol = match.group(2)
        args = match.group(3)
        signatures[symbol] = parse_signature(return_type, args)
    return signatures


def remove_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def remove_default_argument(text: str) -> str:
    depth = 0
    for index, char in enumerate(text):
        if char in "(<[":
            depth += 1
        elif char in ")>]":
            depth -= 1
        elif char == "=" and depth == 0:
            return text[:index].strip()
    return text


def canonical_decl_symbol(raw_name: str) -> str:
    macro_match = re.fullmatch(r"(HIP_API_SYMBOL|HIP_PRIVATE_SYMBOL|HIP_COMPILER_API_SYMBOL)\(([A-Za-z_][A-Za-z0-9_]*)\)",
                               raw_name.strip())
    if macro_match is None:
        return raw_name.strip()
    macro = macro_match.group(1)
    suffix = macro_match.group(2)
    if macro == "HIP_API_SYMBOL":
        return f"hip{suffix}"
    return f"__hip{suffix}"


def normalize_return_type(return_type: str) -> str:
    return_type = re.sub(r"\bHIP_DEPRECATED\s*\([^)]*\)\s*", "", return_type)
    return_type = re.sub(r"\b__attribute__\s*\(\([^)]*\)\)\s*", "", return_type)
    return re.sub(r"\s+", " ", return_type).strip()


def parse_header_declarations(header: Path) -> dict[str, Signature]:
    text = remove_c_comments(header.read_text())
    text = re.sub(r"\\\n", " ", text)
    text = re.sub(r'extern\s+"C"\s*\{', "", text)
    text = re.sub(r"^\s*#.*$", "", text, flags=re.M)
    signatures: dict[str, Signature] = {}
    for statement in text.split(";"):
        if "(" not in statement or ")" not in statement or "{" in statement or "}" in statement:
            continue
        end = statement.rfind(")")
        depth = 0
        open_index: int | None = None
        for index in range(end, -1, -1):
            char = statement[index]
            if char == ")":
                depth += 1
            elif char == "(":
                depth -= 1
                if depth == 0:
                    open_index = index
                    break
        if open_index is None:
            continue
        before = statement[:open_index].strip()
        args = statement[open_index + 1:end]
        name_match = re.search(
            r"((?:HIP_API_SYMBOL|HIP_PRIVATE_SYMBOL|HIP_COMPILER_API_SYMBOL)"
            r"\([A-Za-z_][A-Za-z0-9_]*\)|[A-Za-z_][A-Za-z0-9_]*)\s*$",
            before,
        )
        if name_match is None:
            continue
        return_type = normalize_return_type(before[:name_match.start()].strip())
        if not return_type or return_type.startswith(("typedef ", "using ", "template ")):
            continue
        symbol = canonical_decl_symbol(name_match.group(1))
        if not (
            symbol.startswith("hip")
            or symbol.startswith("__hip")
            or symbol.startswith("__gnu_")
            or symbol.startswith("amd_dbgapi_")
        ):
            continue
        signatures.setdefault(symbol, parse_signature(return_type, args))
    return signatures


def parse_hip_header_signatures(source_root: Path) -> dict[str, Signature]:
    headers = [
        source_root / "projects/hip/include/hip/hip_runtime_api.h",
        source_root / "projects/hip/include/hip/hip_deprecated.h",
        source_root / "projects/hip/include/hip/hip_gl_interop.h",
        source_root / "projects/clr/hipamd/include/hip/amd_detail/hip_runtime_prof.h",
        source_root / "projects/clr/hipamd/include/hip/amd_detail/hip_profiler_ext.h",
        source_root / "projects/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h",
    ]
    signatures: dict[str, Signature] = {}
    for header in headers:
        if header.exists():
            signatures.update(parse_header_declarations(header))
    return signatures


def parse_version_script(version_script: Path) -> list[VersionNode]:
    text = version_script.read_text()
    pattern = re.compile(r"([A-Za-z0-9_.]+)\s*\{(.*?)\}\s*(?:([A-Za-z0-9_.]+))?\s*;", re.S)
    nodes: list[VersionNode] = []
    for match in pattern.finditer(text):
        name = match.group(1)
        body = match.group(2)
        parent = match.group(3)
        symbols: list[str] = []
        in_global = False
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if line == "global:":
                in_global = True
                continue
            if line == "local:":
                in_global = False
                continue
            if not in_global or not line or line == "*;":
                continue
            if line.endswith(";"):
                symbol = line[:-1].strip()
                if symbol.endswith("*"):
                    symbol = symbol[:-1]
                symbols.append(symbol)
        nodes.append(VersionNode(name=name, parent=parent, symbols=tuple(symbols)))
    return nodes


def backend_symbol_for(
    public_symbol: str,
    backend_api_major: int,
    compiler_api_symbols: set[str],
    private_api_symbols: set[str],
) -> str:
    if public_symbol.startswith("__hip"):
        namespace = "Compiler" if public_symbol in compiler_api_symbols else "Private"
        if public_symbol not in compiler_api_symbols and public_symbol not in private_api_symbols:
            namespace = "Private"
        return f"hipBackendV{backend_api_major}{namespace}{public_symbol[5:]}"
    if public_symbol.startswith("hip"):
        return f"hipBackendV{backend_api_major}{public_symbol[3:]}"
    encoded = re.sub(r"[^A-Za-z0-9]", "_", public_symbol)
    return f"hipBackendV{backend_api_major}PrivateExport_{encoded}"


def node_allowed_for_major(node: VersionNode, max_node: str, seen_max: bool) -> tuple[bool, bool]:
    if seen_max:
        return False, True
    if node.name == max_node:
        return True, True
    return True, False


def load_annotations(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError("annotations must be a JSON object")
    return loaded


def build_entries_for_major(
    public_major: int,
    nodes: list[VersionNode],
    signatures: dict[str, Signature],
    annotations: dict[str, object],
) -> tuple[list[VersionNode], list[ApiEntry]]:
    stale_exports = set(read_str_list(annotations, "stale_exports"))
    loader_owned_symbols = set(read_str_list(annotations, "loader_owned_symbols"))
    max_nodes = read_str_dict(annotations, "public_major_max_version_node")
    signature_overrides = read_object_dict(annotations, "signature_overrides")
    compatibility = read_nested_object_dict(annotations, "compatibility").get(str(public_major), {})
    backend_api_major = read_int(annotations, "backend_api_major")
    compiler_api_symbols = set(read_str_list(annotations, "compiler_api_symbols"))
    private_api_symbols = set(read_str_list(annotations, "private_api_symbols"))
    max_node = max_nodes[str(public_major)]

    selected_nodes: list[VersionNode] = []
    entries: list[ApiEntry] = []
    seen_public_symbols: set[str] = set()
    stopped = False
    for node in nodes:
        allowed, stopped = node_allowed_for_major(node, max_node, stopped)
        if not allowed:
            continue
        selected_symbols: list[str] = []
        for public_symbol in node.symbols:
            if public_symbol in seen_public_symbols:
                continue
            stale = public_symbol in stale_exports
            if stale:
                continue
            seen_public_symbols.add(public_symbol)
            selected_symbols.append(public_symbol)
            signature_symbol = public_symbol
            compat_info = compatibility.get(public_symbol)
            compat_function: str | None = None
            compatibility_reason: str | None = None
            backend_signature_symbol = public_symbol
            backend_symbol = backend_symbol_for(
                public_symbol,
                backend_api_major,
                compiler_api_symbols,
                private_api_symbols,
            )
            if isinstance(compat_info, dict):
                compat_function_value = compat_info.get("compat_function")
                backend_symbol_value = compat_info.get("backend_symbol")
                backend_signature_value = compat_info.get("backend_signature")
                reason_value = compat_info.get("reason")
                if not isinstance(compat_function_value, str):
                    raise ValueError(f"{public_symbol} compat metadata needs compat_function")
                if not isinstance(backend_symbol_value, str):
                    raise ValueError(f"{public_symbol} compat metadata needs backend_symbol")
                if not isinstance(backend_signature_value, str):
                    raise ValueError(f"{public_symbol} compat metadata needs backend_signature")
                if not isinstance(reason_value, str):
                    raise ValueError(f"{public_symbol} compat metadata needs reason")
                compat_function = compat_function_value
                backend_symbol = backend_symbol_value
                backend_signature_symbol = backend_signature_value
                compatibility_reason = reason_value
                public_signature_value = compat_info.get("public_signature")
                if isinstance(public_signature_value, dict):
                    signature = signature_from_json(public_signature_value)
                else:
                    signature = None
            else:
                signature = None

            if signature is None:
                override = signature_overrides.get(signature_symbol)
                if isinstance(override, dict):
                    signature = signature_from_json(override)
                    override_backend = override.get("backend_symbol")
                    if isinstance(override_backend, str) and compat_function is None:
                        backend_symbol = override_backend
                else:
                    signature = signatures.get(signature_symbol)
                    if signature is None:
                        raise ValueError(f"{public_symbol} has no parsed signature or override")

            backend_override = signature_overrides.get(backend_signature_symbol)
            if isinstance(backend_override, dict):
                backend_signature = signature_from_json(backend_override)
            else:
                backend_signature = signatures.get(backend_signature_symbol)
                if backend_signature is None:
                    raise ValueError(f"{public_symbol} backend signature {backend_signature_symbol} missing")

            entries.append(ApiEntry(
                public_symbol=public_symbol,
                signature_symbol=signature_symbol,
                signature=signature,
                version_node=node.name,
                backend_symbol=backend_symbol,
                backend_signature_symbol=backend_signature_symbol,
                backend_signature=backend_signature,
                loader_owned=public_symbol in loader_owned_symbols,
                stale=False,
                compat_function=compat_function,
                compatibility_reason=compatibility_reason,
            ))
        selected_nodes.append(VersionNode(node.name, node.parent, tuple(selected_symbols)))
    return selected_nodes, entries


def read_str_list(obj: dict[str, object], key: str) -> list[str]:
    value = obj.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return list(value)


def read_str_dict(obj: dict[str, object], key: str) -> dict[str, str]:
    value = obj.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    result: dict[str, str] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_key, str) or not isinstance(item_value, str):
            raise ValueError(f"{key} must contain string keys and values")
        result[item_key] = item_value
    return result


def read_object_dict(obj: dict[str, object], key: str) -> dict[str, object]:
    value = obj.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    result: dict[str, object] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_key, str):
            raise ValueError(f"{key} must contain string keys")
        result[item_key] = item_value
    return result


def read_nested_object_dict(obj: dict[str, object], key: str) -> dict[str, dict[str, object]]:
    value = obj.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    result: dict[str, dict[str, object]] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_key, str) or not isinstance(item_value, dict):
            raise ValueError(f"{key} must contain object values")
        result[item_key] = dict(item_value)
    return result


def read_int(obj: dict[str, object], key: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def prelude(rename_current_graph_memset: bool = False) -> str:
    graph_renames = ""
    if rename_current_graph_memset:
        graph_renames = """#define hipDrvGraphAddMemsetNode hipDrvGraphAddMemsetNode_HIP_LOADER_CURRENT_DECL
#define hipDrvGraphExecMemsetNodeSetParams hipDrvGraphExecMemsetNodeSetParams_HIP_LOADER_CURRENT_DECL
"""
    return f"""#define __HIP_PLATFORM_AMD__ 1
#define __HIP_DISABLE_CPP_FUNCTIONS__ 1
{graph_renames}typedef unsigned int GLenum;
typedef unsigned int GLuint;
#include <hip/hip_runtime.h>
#include <hip/hip_gl_interop.h>
#include <hip/hip_deprecated.h>
#include <hip/amd_detail/hip_profiler_ext.h>
#include <hip/amd_detail/hip_api_trace.hpp>
#include <hip_loader/hip_loader_abi.h>
#ifdef hipGetDeviceProperties
#undef hipGetDeviceProperties
#endif
#ifdef hipChooseDevice
#undef hipChooseDevice
#endif
#ifdef hipDeviceProp_t
#undef hipDeviceProp_t
#endif
#ifdef hipDrvGraphAddMemsetNode
#undef hipDrvGraphAddMemsetNode
#endif
#ifdef hipDrvGraphExecMemsetNodeSetParams
#undef hipDrvGraphExecMemsetNodeSetParams
#endif
"""


def emit_generated_api_header(entries_by_major: dict[int, list[ApiEntry]]) -> str:
    backend_entries = unique_backend_entries(entries_by_major)
    lines = [
        "/* Generated by hip_api_manifest/generate.py. Do not edit. */",
        "#ifndef HIP_LOADER_GENERATED_API_H",
        "#define HIP_LOADER_GENERATED_API_H",
        prelude(),
        "#include <stddef.h>",
        "namespace hip_loader_generated {",
    ]
    for member_name, entry in backend_entries:
        lines.append(f"using Fn_{member_name} = {entry.backend_signature.return_type} (*)("
                     f"{entry.backend_signature.declaration_params()});")
    lines.append("}  // namespace hip_loader_generated")
    lines.append("#endif")
    return "\n".join(lines) + "\n"


def safe_member_name(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", symbol)


def unique_backend_entries(entries_by_major: dict[int, list[ApiEntry]]) -> list[tuple[str, ApiEntry]]:
    by_symbol: dict[str, ApiEntry] = {}
    for entries in entries_by_major.values():
        for entry in entries:
            if entry.loader_owned:
                continue
            by_symbol.setdefault(entry.backend_symbol, entry)
    return [(safe_member_name(symbol), entry) for symbol, entry in sorted(by_symbol.items())]


def return_load_failure(return_type: str, status_expr: str = "status") -> str:
    if return_type == "void":
        return "return;"
    if return_type == "hipError_t":
        return f"return {status_expr};"
    if return_type.endswith("*") or return_type in {"const char*", "char*"}:
        return "return nullptr;"
    return "return {};"


def emit_public_prototypes(entries: list[ApiEntry]) -> list[str]:
    lines = ['extern "C" {']
    for entry in entries:
        lines.append(f"HIP_LOADER_EXPORT {entry.signature.return_type} {entry.public_symbol}("
                     f"{entry.signature.declaration_params()});")
    lines.append("}")
    return lines


def param_types(signature: Signature) -> str:
    if not signature.params:
        return "void"
    return ", ".join(param.type for param in signature.params)


def function_address_expr(entry: ApiEntry) -> str:
    function_type = f"{entry.signature.return_type} (*)({param_types(entry.signature)})"
    return f"reinterpret_cast<void*>(static_cast<{function_type}>(&{entry.public_symbol}))"


def symbol_to_entry(entries: list[ApiEntry], symbol: str) -> ApiEntry:
    for entry in entries:
        if entry.public_symbol == symbol:
            return entry
    raise ValueError(f"public symbol not present in generated loader: {symbol}")


def emit_loader_source(public_major: int, nodes: list[VersionNode], entries: list[ApiEntry],
                       annotations: dict[str, object]) -> str:
    backend_entries = unique_backend_entries({public_major: entries})
    aliases = read_object_dict(annotations, "hip_get_proc_address_aliases")
    lines = [
        "/* Generated by hip_api_manifest/generate.py. Do not edit. */",
        prelude(rename_current_graph_memset=(public_major == 6)),
        '#include "generated_api.hpp"',
        '#include "hip_loader_platform.h"',
        '#include "loader_v6_compat.h"',
    ]
    lines.extend([
        "#include <cstdlib>",
        "#include <cstring>",
        "#include <mutex>",
        "#include <string>",
        "",
    ])
    lines.extend(emit_public_prototypes(entries))
    lines.extend([
        "",
        "namespace {",
        "using FnBackendGetInterface = hipError_t (*)(hip_loader_backend_info_v1*);",
        "struct BackendApi {",
    ])
    for member_name, _entry in backend_entries:
        lines.append(f"  hip_loader_generated::Fn_{member_name} {member_name} = nullptr;")
    lines.extend([
        "};",
        "struct BackendState {",
        "  std::mutex mutex;",
        "  bool load_attempted = false;",
        "  hipError_t load_status = hipSuccess;",
        "  hip_loader::DynamicLibrary library;",
        "  BackendApi api;",
        "};",
        "BackendState& backend_state_instance() { static BackendState instance; return instance; }",
        "bool has_forbidden_public_export(BackendState& backend_state) {",
    ])
    for entry in entries:
        lines.append(f'  if (backend_state.library.symbol("{entry.public_symbol}") != nullptr) return true;')
    lines.extend([
        "  return false;",
        "}",
        "template <typename Fn>",
        "bool resolve_required(hip_loader::DynamicLibrary& library, Fn* slot, const char* symbol) {",
        "  *slot = reinterpret_cast<Fn>(library.symbol(symbol));",
        "  if (*slot == nullptr) { hip_loader::log(\"backend is missing required symbol %s\", symbol); }",
        "  return *slot != nullptr;",
        "}",
        "hipError_t load_backend_locked(BackendState& backend_state) {",
        '  const char* backend_path = std::getenv("HIP_LOADER_BACKEND_PATH");',
        "  if (backend_path == nullptr || backend_path[0] == '\\0') return hipErrorNotInitialized;",
        "  std::string error;",
        "  if (!backend_state.library.open(backend_path, &error)) {",
        "    hip_loader::log(\"failed to load backend %s: %s\", backend_path, error.c_str());",
        "    return hipErrorNotInitialized;",
        "  }",
        "  auto get_interface = reinterpret_cast<FnBackendGetInterface>("
        'backend_state.library.symbol("hipBackendV7GetInterface"));',
        "  if (get_interface == nullptr) return hipErrorInvalidValue;",
        "  hip_loader_backend_info_v1 info = {};",
        "  info.struct_size = sizeof(info);",
        "  hipError_t status = get_interface(&info);",
        "  if (status != hipSuccess) return status;",
        "  if (info.backend_api_major != HIP_LOADER_BACKEND_API_MAJOR || "
        "info.loader_backend_abi_version != HIP_LOADER_BACKEND_ABI_VERSION) return hipErrorInvalidValue;",
        "  if (has_forbidden_public_export(backend_state)) return hipErrorInvalidValue;",
        "  bool ok = true;",
    ])
    for member_name, entry in backend_entries:
        lines.append(f'  ok &= resolve_required(backend_state.library, &backend_state.api.{member_name}, '
                     f'"{entry.backend_symbol}");')
    lines.extend([
        "  return ok ? hipSuccess : hipErrorInvalidValue;",
        "}",
        "hipError_t ensure_backend_loaded() {",
        "  BackendState& backend_state = backend_state_instance();",
        "  std::lock_guard<std::mutex> lock(backend_state.mutex);",
        "  if (backend_state.load_attempted) return backend_state.load_status;",
        "  backend_state.load_attempted = true;",
        "  backend_state.load_status = load_backend_locked(backend_state);",
        "  return backend_state.load_status;",
        "}",
        "void set_proc_status(hipDriverProcAddressQueryResult* symbolStatus, "
        "hipDriverProcAddressQueryResult value) {",
        "  if (symbolStatus != nullptr) *symbolStatus = value;",
        "}",
        "void* lookup_public_symbol(const char* symbol, int hipVersion) {",
        "  if (symbol == nullptr) return nullptr;",
    ])
    for alias_symbol, alias_value in sorted(aliases.items()):
        if not isinstance(alias_value, dict):
            continue
        before_version = alias_value.get("before_version")
        legacy_symbol = alias_value.get("legacy_symbol")
        current_symbol = alias_value.get("current_symbol")
        if isinstance(before_version, int) and isinstance(legacy_symbol, str) and isinstance(current_symbol, str):
            lines.extend([
                f'  if (std::strcmp(symbol, "{alias_symbol}") == 0) {{',
                f"    if (hipVersion < {before_version}) return {function_address_expr(symbol_to_entry(entries, legacy_symbol))};",
                f"    return {function_address_expr(symbol_to_entry(entries, current_symbol))};",
                "  }",
            ])
    for entry in entries:
        lines.append(f'  if (std::strcmp(symbol, "{entry.public_symbol}") == 0) '
                     f"return {function_address_expr(entry)};")
    lines.extend([
        "  return nullptr;",
        "}",
        "}  // namespace",
        "",
        'extern "C" {',
    ])
    for entry in entries:
        lines.extend(emit_public_function(public_major, entry))
    lines.append("}  // extern \"C\"")
    return "\n".join(lines) + "\n"


def emit_public_function(public_major: int, entry: ApiEntry) -> list[str]:
    sig = entry.signature
    lines = [
        f"HIP_LOADER_EXPORT {sig.return_type} {entry.public_symbol}({sig.declaration_params()}) {{"
    ]
    if entry.public_symbol in {"hipGetProcAddress", "hipGetProcAddress_spt"}:
        lines.extend([
            "  if (pfn == nullptr) {",
            "    set_proc_status(symbolStatus, HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND);",
            "    return hipErrorInvalidValue;",
            "  }",
            "  *pfn = lookup_public_symbol(symbol, hipVersion);",
            "  if (*pfn == nullptr) {",
            "    set_proc_status(symbolStatus, HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND);",
            "    return hipErrorInvalidValue;",
            "  }",
            "  set_proc_status(symbolStatus, HIP_GET_PROC_ADDRESS_SUCCESS);",
            "  return hipSuccess;",
            "}",
        ])
        return lines
    if entry.public_symbol in {"hipGetDriverEntryPoint", "hipGetDriverEntryPoint_spt"}:
        pfn_name = sig.params[1].name if len(sig.params) > 1 else "funcPtr"
        flags_name = sig.params[2].name if len(sig.params) > 2 else "flags"
        status_name = sig.params[3].name if len(sig.params) > 3 else "status"
        lines.extend([
            f"  (void){status_name};",
            f"  return hipGetProcAddress(symbol, {pfn_name}, {public_major * 100}, {flags_name}, nullptr);",
            "}",
        ])
        return lines

    lines.extend([
        "  hipError_t status = ensure_backend_loaded();",
        f"  if (status != hipSuccess) {{ {return_load_failure(sig.return_type)} }}",
    ])
    if entry.compat_function is not None:
        member = safe_member_name(entry.backend_symbol)
        lines.append(f"  return {entry.compat_function}(backend_state_instance().api.{member}, {sig.call_args()});")
    else:
        member = safe_member_name(entry.backend_symbol)
        if sig.return_type == "void":
            lines.append(f"  backend_state_instance().api.{member}({sig.call_args()});")
            lines.append("  return;")
        else:
            lines.append(f"  return backend_state_instance().api.{member}({sig.call_args()});")
    lines.append("}")
    return lines


def emit_version_script(nodes: list[VersionNode], entries: list[ApiEntry]) -> str:
    symbols_by_node: dict[str, list[str]] = {}
    for entry in entries:
        symbols_by_node.setdefault(entry.version_node, []).append(entry.public_symbol)
    lines: list[str] = []
    for node in nodes:
        lines.append(f"{node.name} {{")
        node_symbols = sorted(symbols_by_node.get(node.name, []))
        if node_symbols:
            lines.append("global:")
            for symbol in node_symbols:
                lines.append(f"    {symbol};")
            lines.append("local:")
            lines.append("    *;")
        suffix = f" {node.parent}" if node.parent else ""
        lines.append(f"}}{suffix};")
        lines.append("")
    return "\n".join(lines)


def emit_def(entries: list[ApiEntry]) -> str:
    lines = ["EXPORTS"]
    for entry in sorted(entries, key=lambda item: item.public_symbol):
        lines.append(entry.public_symbol)
    return "\n".join(lines) + "\n"


def emit_backend_source(entries_by_major: dict[int, list[ApiEntry]]) -> str:
    backend_entries = unique_backend_entries(entries_by_major)
    lines = [
        "/* Generated by hip_api_manifest/generate.py. Do not edit. */",
        prelude(),
        '#include "generated_api.hpp"',
        "#include <cstring>",
        "#include <map>",
        "#include <mutex>",
        "#include <string>",
        "namespace {",
        "std::mutex g_mutex;",
        "hip_loader_test_api_callback_t g_callback = nullptr;",
        "std::map<std::string, unsigned int> g_call_counts;",
        "hipMemsetParams g_last_memset_params = {};",
        "void* g_fat_binary_handle = reinterpret_cast<void*>(static_cast<uintptr_t>(0xBEEFu));",
        "hipError_t record_call(const char* symbol, hipError_t default_result) {",
        "  hip_loader_test_api_callback_t callback = nullptr;",
        "  { std::lock_guard<std::mutex> lock(g_mutex); ++g_call_counts[symbol]; callback = g_callback; }",
        "  if (callback == nullptr) return default_result;",
        "  hip_loader_test_call_v1 call = {};",
        "  call.struct_size = sizeof(call); call.symbol = symbol; call.default_result = default_result;",
        "  hipError_t override_result = callback(&call);",
        "  return override_result == hipSuccess ? default_result : override_result;",
        "}",
        "void fill_device_properties(hipDeviceProp_tR0600* prop) {",
        "  if (prop == nullptr) return;",
        "  std::memset(prop, 0, sizeof(*prop));",
        '  std::strncpy(prop->name, "example_backend generated test device", sizeof(prop->name) - 1);',
        "  prop->totalGlobalMem = 0x123456789ULL;",
        "  prop->sharedMemPerBlock = 49152;",
        "  prop->maxThreadsPerBlock = 1024;",
        "  prop->major = 9; prop->minor = 4;",
        "  prop->multiProcessorCount = 120;",
        "  prop->memoryClockRate = 1600000;",
        "  prop->memoryBusWidth = 4096;",
        "  prop->l2CacheSize = 8388608;",
        '  std::strncpy(prop->gcnArchName, "gfx942", sizeof(prop->gcnArchName) - 1);',
        "}",
        "template <typename T> T default_value() { return T{}; }",
        "template <> const char* default_value<const char*>() { return \"example_backend\"; }",
        "}  // namespace",
        'extern "C" {',
        "HIP_LOADER_EXPORT hipError_t hipBackendV7GetInterface(hip_loader_backend_info_v1* info) {",
        "  if (info == nullptr || info->struct_size < sizeof(*info)) return hipErrorInvalidValue;",
        "  info->loader_backend_abi_version = HIP_LOADER_BACKEND_ABI_VERSION;",
        "  info->backend_api_major = HIP_LOADER_BACKEND_API_MAJOR;",
        "  info->backend_api_minor = HIP_LOADER_BACKEND_API_MINOR;",
        "  info->flags = 0;",
        "  info->backend_name = \"example_backend\";",
        "  return hipSuccess;",
        "}",
    ]
    for _member_name, entry in backend_entries:
        lines.extend(emit_backend_function(entry))
    lines.extend([
        "HIP_LOADER_EXPORT void __testBackendSetAPICallback(hip_loader_test_api_callback_t callback) {",
        "  std::lock_guard<std::mutex> lock(g_mutex); g_callback = callback;",
        "}",
        "HIP_LOADER_EXPORT void __testBackendReset(void) {",
        "  std::lock_guard<std::mutex> lock(g_mutex); g_callback = nullptr; g_call_counts.clear();",
        "  g_last_memset_params = {};",
        "}",
        "HIP_LOADER_EXPORT unsigned int __testBackendGetCallCount(const char* symbol) {",
        "  std::lock_guard<std::mutex> lock(g_mutex);",
        "  auto it = g_call_counts.find(symbol == nullptr ? std::string() : std::string(symbol));",
        "  return it == g_call_counts.end() ? 0u : it->second;",
        "}",
        "HIP_LOADER_EXPORT const hipMemsetParams* __testBackendGetLastMemsetParams(void) {",
        "  return &g_last_memset_params;",
        "}",
        "}  // extern \"C\"",
    ])
    return "\n".join(lines) + "\n"


def emit_backend_function(entry: ApiEntry) -> list[str]:
    sig = entry.backend_signature
    lines = [
        f"HIP_LOADER_EXPORT {sig.return_type} {entry.backend_symbol}({sig.declaration_params()}) {{"
    ]
    symbol = entry.backend_signature_symbol
    if symbol in {"hipRuntimeGetVersion", "hipDriverGetVersion"} and sig.params:
        version_param = sig.params[0].name
        lines.append(f"  if ({version_param} != nullptr) *{version_param} = 70000000;")
    if symbol == "hipGetDeviceCount":
        lines.append("  if (count != nullptr) *count = 1;")
    if symbol == "hipGetDevice":
        lines.append("  if (deviceId != nullptr) *deviceId = 0;")
    if symbol == "hipGetDevicePropertiesR0600":
        lines.append("  fill_device_properties(prop);")
    if symbol == "hipChooseDeviceR0600":
        lines.append("  if (device != nullptr) *device = 0;")
    if symbol in {"hipDrvGraphAddMemsetNode", "hipDrvGraphExecMemsetNodeSetParams"}:
        lines.append("  if (memsetParams != nullptr) g_last_memset_params = *memsetParams;")
        if symbol == "hipDrvGraphAddMemsetNode":
            lines.append("  if (phGraphNode != nullptr) *phGraphNode = reinterpret_cast<hipGraphNode_t>(0xCAFE);")
    if entry.backend_symbol == "hipBackendV7CompilerRegisterFatBinary":
        lines.append(f'  (void)record_call("{entry.backend_symbol}", hipSuccess);')
        lines.append("  return reinterpret_cast<void**>(&g_fat_binary_handle);")
        lines.append("}")
        return lines
    if sig.return_type == "void":
        lines.append(f'  (void)record_call("{entry.backend_symbol}", hipSuccess);')
        lines.append("  return;")
    elif sig.return_type == "hipError_t":
        lines.append(f'  return record_call("{entry.backend_symbol}", hipSuccess);')
    elif sig.return_type in {"const char*", "char*"}:
        lines.append(f'  (void)record_call("{entry.backend_symbol}", hipSuccess);')
        lines.append('  return "example_backend";')
    else:
        lines.append(f'  (void)record_call("{entry.backend_symbol}", hipSuccess);')
        lines.append(f"  return default_value<{sig.return_type}>();")
    lines.append("}")
    return lines


def emit_bad_backend(kind: str) -> str:
    common = [
        prelude(),
        'extern "C" {',
    ]
    if kind == "wrong_major":
        common.extend([
            "HIP_LOADER_EXPORT hipError_t hipBackendV7GetInterface(hip_loader_backend_info_v1* info) {",
            "  if (info == nullptr || info->struct_size < sizeof(*info)) return hipErrorInvalidValue;",
            "  info->loader_backend_abi_version = HIP_LOADER_BACKEND_ABI_VERSION;",
            "  info->backend_api_major = 6;",
            "  info->backend_api_minor = 0;",
            "  info->backend_name = \"wrong_major_backend\";",
            "  return hipSuccess;",
            "}",
        ])
    elif kind == "missing_handshake":
        common.append("HIP_LOADER_EXPORT hipError_t hipBackendV7Init(unsigned int) { return hipSuccess; }")
    elif kind == "bad_public_export":
        common.extend([
            "HIP_LOADER_EXPORT hipError_t hipBackendV7GetInterface(hip_loader_backend_info_v1* info) {",
            "  if (info == nullptr || info->struct_size < sizeof(*info)) return hipErrorInvalidValue;",
            "  info->loader_backend_abi_version = HIP_LOADER_BACKEND_ABI_VERSION;",
            "  info->backend_api_major = HIP_LOADER_BACKEND_API_MAJOR;",
            "  info->backend_api_minor = HIP_LOADER_BACKEND_API_MINOR;",
            "  info->backend_name = \"bad_public_export_backend\";",
            "  return hipSuccess;",
            "}",
            "HIP_LOADER_EXPORT hipError_t hipInit(unsigned int) { return hipSuccess; }",
        ])
    elif kind == "bad_compiler_export":
        common.extend([
            "HIP_LOADER_EXPORT hipError_t hipBackendV7GetInterface(hip_loader_backend_info_v1* info) {",
            "  if (info == nullptr || info->struct_size < sizeof(*info)) return hipErrorInvalidValue;",
            "  info->loader_backend_abi_version = HIP_LOADER_BACKEND_ABI_VERSION;",
            "  info->backend_api_major = HIP_LOADER_BACKEND_API_MAJOR;",
            "  info->backend_api_minor = HIP_LOADER_BACKEND_API_MINOR;",
            "  info->backend_name = \"bad_compiler_export_backend\";",
            "  return hipSuccess;",
            "}",
            "HIP_LOADER_EXPORT void** __hipRegisterFatBinary(const void*) { return nullptr; }",
        ])
    elif kind == "preload_interposer":
        common.append("HIP_LOADER_EXPORT hipError_t hipBackendV7RuntimeGetVersion(int* version) { "
                      "if (version) *version = 123; return hipSuccess; }")
    common.append("}  // extern \"C\"")
    return "\n".join(common) + "\n"


def write_manifest(paths: GeneratedPaths, nodes_by_major: dict[int, list[VersionNode]],
                   entries_by_major: dict[int, list[ApiEntry]], annotations: dict[str, object],
                   signature_sources: dict[str, list[str]],
                   public_header_api: ExtractedApi,
                   backend_header_api: ExtractedApi) -> None:
    manifest: dict[str, object] = {
        "backend_api_major": read_int(annotations, "backend_api_major"),
        "header_api": semantic_manifest_from_api(public_header_api),
        "backend_header_api": semantic_manifest_from_api(backend_header_api),
        "public_abi": {},
        "signature_sources": {
            key: sorted(value) for key, value in signature_sources.items()
        },
        "stale_exports": read_str_list(annotations, "stale_exports"),
    }
    public_abi: dict[str, object] = {}
    for major, entries in entries_by_major.items():
        public_abi[str(major)] = {
            "export_count": len(entries),
            "nodes": [
                {
                    "name": node.name,
                    "parent": node.parent,
                    "symbols": sorted(entry.public_symbol for entry in entries if entry.version_node == node.name),
                }
                for node in nodes_by_major[major]
            ],
            "compat_symbols": [
                {
                    "symbol": entry.public_symbol,
                    "backend_symbol": entry.backend_symbol,
                    "compat_function": entry.compat_function,
                    "reason": entry.compatibility_reason,
                }
                for entry in entries if entry.compat_function is not None
            ],
            "loader_owned_symbols": sorted(entry.public_symbol for entry in entries if entry.loader_owned),
        }
    manifest["public_abi"] = public_abi
    paths.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def generate_hip_version(source_root: Path, include_dir: Path) -> None:
    version_values = [
        int(line.strip())
        for line in (source_root / "projects/hip/VERSION").read_text().splitlines()
        if line.strip().isdigit()
    ]
    major, minor, patch = (version_values[index] for index in range(3))
    hip_include = include_dir / "hip"
    hip_include.mkdir(parents=True, exist_ok=True)
    (hip_include / "hip_version.h").write_text(
        "#ifndef HIP_INCLUDE_HIP_HIP_VERSION_H\n"
        "#define HIP_INCLUDE_HIP_HIP_VERSION_H\n"
        f"#define HIP_VERSION_MAJOR {major}\n"
        f"#define HIP_VERSION_MINOR {minor}\n"
        f"#define HIP_VERSION_PATCH {patch}\n"
        f"#define HIP_VERSION ({major} * 100 + {minor})\n"
        "#endif\n"
    )


def output_paths(build_dir: Path) -> GeneratedPaths:
    generated = build_dir / "generated"
    include_dir = generated / "include"
    return GeneratedPaths(
        include_dir=include_dir,
        manifest=generated / "hip_api_manifest.json",
        loader_v6=generated / "loader_v6.cpp",
        loader_v7=generated / "loader_v7.cpp",
        backend=generated / "test_backend.cpp",
        wrong_major_backend=generated / "test_backend_wrong_major.cpp",
        missing_handshake_backend=generated / "test_backend_missing_handshake.cpp",
        bad_public_export_backend=generated / "test_backend_bad_public_export.cpp",
        bad_compiler_export_backend=generated / "test_backend_bad_compiler_export.cpp",
        preload_interposer=generated / "test_preload_interposer.cpp",
        v6_map=generated / "hip_loader_v6.map",
        v7_map=generated / "hip_loader_v7.map",
        v6_def=generated / "amdhip-6.def",
        v7_def=generated / "amdhip-7.def",
    )


def validate_backend_macro_coverage(
    entries_by_major: dict[int, list[ApiEntry]],
    backend_header_api: ExtractedApi,
) -> None:
    backend_symbols = set(backend_header_api.functions)
    missing: list[str] = []
    checked: set[tuple[str, str]] = set()
    for entries in entries_by_major.values():
        for entry in entries:
            if not (entry.public_symbol.startswith("hip") or entry.public_symbol.startswith("__hip")):
                continue
            key = (entry.public_symbol, entry.backend_symbol)
            if key in checked:
                continue
            checked.add(key)
            if entry.backend_symbol not in backend_symbols:
                missing.append(f"{entry.public_symbol} -> {entry.backend_symbol}")
    if missing:
        details = "\n  ".join(missing[:50])
        if len(missing) > 50:
            details += f"\n  ... {len(missing) - 50} more"
        raise RuntimeError(f"HIP backend-mode header extraction missed ABI symbols:\n  {details}")


def generate(source_root: Path, build_dir: Path, annotations_path: Path) -> None:
    paths = output_paths(build_dir)
    paths.include_dir.mkdir(parents=True, exist_ok=True)
    paths.loader_v6.parent.mkdir(parents=True, exist_ok=True)
    generate_hip_version(source_root, paths.include_dir)

    annotations = load_annotations(annotations_path)
    backend_api_major = read_int(annotations, "backend_api_major")
    public_header_api = extract_header_api(
        source_root,
        build_dir,
        paths.include_dir,
        "public",
        backend_api_major,
    )
    backend_header_api = extract_header_api(
        source_root,
        build_dir,
        paths.include_dir,
        "backend",
        backend_api_major,
    )
    trace_header = source_root / "projects/clr/hipamd/include/hip/amd_detail/hip_api_trace.hpp"
    version_script = source_root / "projects/clr/hipamd/src/hip_hcc.map.in"
    header_signatures = signatures_from_extracted(public_header_api)
    trace_signatures = parse_trace_typedefs(trace_header)
    signatures = dict(trace_signatures)
    signatures.update(header_signatures)
    signature_overrides = read_object_dict(annotations, "signature_overrides")
    for symbol, value in signature_overrides.items():
        if isinstance(value, dict):
            signatures[symbol] = signature_from_json(value)
    nodes = parse_version_script(version_script)

    nodes_by_major: dict[int, list[VersionNode]] = {}
    entries_by_major: dict[int, list[ApiEntry]] = {}
    for major in (6, 7):
        selected_nodes, entries = build_entries_for_major(major, nodes, signatures, annotations)
        nodes_by_major[major] = selected_nodes
        entries_by_major[major] = entries
    validate_backend_macro_coverage(entries_by_major, backend_header_api)

    (paths.include_dir / "generated_api.hpp").write_text(emit_generated_api_header(entries_by_major))
    paths.loader_v6.write_text(emit_loader_source(6, nodes_by_major[6], entries_by_major[6], annotations))
    paths.loader_v7.write_text(emit_loader_source(7, nodes_by_major[7], entries_by_major[7], annotations))
    paths.backend.write_text(emit_backend_source(entries_by_major))
    paths.wrong_major_backend.write_text(emit_bad_backend("wrong_major"))
    paths.missing_handshake_backend.write_text(emit_bad_backend("missing_handshake"))
    paths.bad_public_export_backend.write_text(emit_bad_backend("bad_public_export"))
    paths.bad_compiler_export_backend.write_text(emit_bad_backend("bad_compiler_export"))
    paths.preload_interposer.write_text(emit_bad_backend("preload_interposer"))
    paths.v6_map.write_text(emit_version_script(nodes_by_major[6], entries_by_major[6]))
    paths.v7_map.write_text(emit_version_script(nodes_by_major[7], entries_by_major[7]))
    paths.v6_def.write_text(emit_def(entries_by_major[6]))
    paths.v7_def.write_text(emit_def(entries_by_major[7]))
    write_manifest(paths, nodes_by_major, entries_by_major, annotations, {
        "clang_headers": list(header_signatures),
        "trace_typedefs": list(trace_signatures),
        "annotations": [
            symbol for symbol, value in signature_overrides.items()
            if isinstance(value, dict)
        ],
    }, public_header_api, backend_header_api)


def run_ast_probe(source_root: Path, build_dir: Path) -> None:
    del source_root
    for mode in ("public", "backend"):
        ast = build_dir / "generated" / "clang_extract" / mode / "hip_header_probe.ast.json"
        preprocessed = build_dir / "generated" / "clang_extract" / mode / "hip_header_probe.ii"
        for path in (preprocessed, ast):
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"clang extraction artifact is missing or empty: {path}")


def check(source_root: Path, build_dir: Path, annotations_path: Path, ast: bool) -> None:
    paths = output_paths(build_dir)
    if not paths.manifest.exists():
        raise RuntimeError(f"missing manifest: {paths.manifest}")
    manifest = json.loads(paths.manifest.read_text())
    public_abi = manifest.get("public_abi")
    if not isinstance(public_abi, dict):
        raise RuntimeError("manifest missing public_abi")
    for major in ("6", "7"):
        major_data = public_abi.get(major)
        if not isinstance(major_data, dict):
            raise RuntimeError(f"manifest missing public ABI {major}")
        export_count = major_data.get("export_count")
        minimum = 450 if major == "6" else 550
        if not isinstance(export_count, int) or export_count < minimum:
            raise RuntimeError(f"public ABI {major} export count is too small: {export_count}")
    stale = manifest.get("stale_exports")
    if not isinstance(stale, list) or "hiprtcCompileProgram" not in stale:
        raise RuntimeError("manifest did not classify stale hiprtc exports")
    signature_sources = manifest.get("signature_sources")
    if not isinstance(signature_sources, dict):
        raise RuntimeError("manifest missing signature source metadata")
    header_symbols = signature_sources.get("clang_headers")
    trace_symbols = signature_sources.get("trace_typedefs")
    if not isinstance(header_symbols, list) or len(header_symbols) < 450:
        raise RuntimeError("manifest did not clang-introspect enough HIP header declarations")
    if not isinstance(trace_symbols, list) or len(trace_symbols) < 450:
        raise RuntimeError("manifest did not record trace typedef fallback declarations")
    for symbol in ("hipInit", "hipGetDeviceCount", "__hipPushCallConfiguration"):
        if symbol not in header_symbols:
            raise RuntimeError(f"manifest did not see header declaration for {symbol}")
    backend_header_api = manifest.get("backend_header_api")
    if not isinstance(backend_header_api, dict):
        raise RuntimeError("manifest missing backend_header_api")
    backend_functions = backend_header_api.get("functions")
    if not isinstance(backend_functions, dict):
        raise RuntimeError("manifest missing backend header function metadata")
    for symbol in ("hipBackendV7Init", "hipBackendV7GetDeviceCount",
                   "hipBackendV7CompilerPushCallConfiguration",
                   "hipBackendV7CompilerRegisterFatBinary",
                   "hipBackendV7PrivateGetPCH"):
        if symbol not in backend_functions:
            raise RuntimeError(f"backend-mode header extraction did not expose {symbol}")
    compat_v6 = public_abi["6"].get("compat_symbols")
    if not isinstance(compat_v6, list) or len(compat_v6) < 6:
        raise RuntimeError("HIP 6 compatibility metadata is incomplete")
    annotations = load_annotations(annotations_path)
    stale_exports = set(read_str_list(annotations, "stale_exports"))
    nodes = parse_version_script(source_root / "projects/clr/hipamd/src/hip_hcc.map.in")
    map_symbols = {symbol for node in nodes for symbol in node.symbols}
    for stale_symbol in stale_exports:
        if stale_symbol not in map_symbols:
            raise RuntimeError(f"stale export annotation is not in source map: {stale_symbol}")
    if ast:
        run_ast_probe(source_root, build_dir)


def extract_semantic_manifest(
    source_root: Path,
    build_dir: Path,
    annotations_path: Path,
) -> dict[str, object]:
    paths = output_paths(build_dir)
    paths.include_dir.mkdir(parents=True, exist_ok=True)
    generate_hip_version(source_root, paths.include_dir)
    annotations = load_annotations(annotations_path)
    backend_api_major = read_int(annotations, "backend_api_major")
    public_header_api = extract_header_api(
        source_root,
        build_dir,
        paths.include_dir,
        "public",
        backend_api_major,
    )
    return semantic_manifest_from_api(public_header_api)


def write_extracted_manifest(
    source_root: Path,
    build_dir: Path,
    annotations_path: Path,
    output: Path,
) -> None:
    manifest = extract_semantic_manifest(source_root, build_dir, annotations_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"failed to write semantic manifest: {output}")


def compare_inputs(args: argparse.Namespace) -> int:
    if args.old_manifest is not None and args.new_manifest is not None:
        old_manifest = load_manifest(args.old_manifest)
        new_manifest = load_manifest(args.new_manifest)
    elif args.old_source_root is not None and args.new_source_root is not None:
        if args.build_dir is None or args.annotations is None:
            raise ValueError("source-root comparison requires --build-dir and --annotations")
        old_manifest = extract_semantic_manifest(
            args.old_source_root.resolve(),
            (args.build_dir / "old").resolve(),
            args.annotations.resolve(),
        )
        new_manifest = extract_semantic_manifest(
            args.new_source_root.resolve(),
            (args.build_dir / "new").resolve(),
            args.annotations.resolve(),
        )
    else:
        raise ValueError(
            "compare requires either --old-manifest/--new-manifest or "
            "--old-source-root/--new-source-root"
        )
    diff = diff_manifests(old_manifest, new_manifest)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(diff.to_jsonable(), indent=2, sort_keys=True) + "\n")
    sys.stdout.write(render_diff(diff))
    if args.fail_on_major and diff.status == "MAJOR":
        return 2
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "check"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--source-root", required=True, type=Path)
        subparser.add_argument("--build-dir", required=True, type=Path)
        subparser.add_argument("--annotations", required=True, type=Path)
    subparsers.choices["check"].add_argument("--ast", action="store_true")
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--source-root", required=True, type=Path)
    extract_parser.add_argument("--build-dir", required=True, type=Path)
    extract_parser.add_argument("--annotations", required=True, type=Path)
    extract_parser.add_argument("--output", required=True, type=Path)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--old-manifest", type=Path)
    compare_parser.add_argument("--new-manifest", type=Path)
    compare_parser.add_argument("--old-source-root", type=Path)
    compare_parser.add_argument("--new-source-root", type=Path)
    compare_parser.add_argument("--build-dir", type=Path)
    compare_parser.add_argument("--annotations", type=Path)
    compare_parser.add_argument("--json-output", type=Path)
    compare_parser.add_argument("--fail-on-major", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "generate":
        generate(args.source_root.resolve(), args.build_dir.resolve(), args.annotations.resolve())
    elif args.command == "check":
        check(args.source_root.resolve(), args.build_dir.resolve(), args.annotations.resolve(), args.ast)
    elif args.command == "extract":
        write_extracted_manifest(
            args.source_root.resolve(),
            args.build_dir.resolve(),
            args.annotations.resolve(),
            args.output.resolve(),
        )
    elif args.command == "compare":
        return compare_inputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
