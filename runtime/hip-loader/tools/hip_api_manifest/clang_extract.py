#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess


@dataclass(frozen=True)
class ExtractedParam:
    type: str
    name: str


@dataclass(frozen=True)
class ExtractedFunction:
    name: str
    return_type: str
    params: tuple[ExtractedParam, ...]
    qual_type: str
    location: str | None


@dataclass(frozen=True)
class ExtractedRecord:
    name: str
    fields: tuple[ExtractedParam, ...]
    location: str | None


@dataclass(frozen=True)
class ExtractedEnum:
    name: str
    constants: tuple[tuple[str, str | None], ...]
    location: str | None


@dataclass(frozen=True)
class ExtractedApi:
    functions: dict[str, ExtractedFunction]
    records: dict[str, ExtractedRecord]
    enums: dict[str, ExtractedEnum]
    preprocessed_path: Path
    ast_path: Path


def find_clang(source_root: Path) -> Path:
    workspace_root = source_root.parents[2]
    bundled = workspace_root / "rocm/lib/llvm/bin/clang++"
    if bundled.exists():
        return bundled
    resolved = shutil.which("clang++")
    if resolved is None:
        raise RuntimeError("clang++ is required for HIP header extraction")
    return Path(resolved)


def extract_header_api(
    source_root: Path,
    build_dir: Path,
    generated_include_dir: Path,
    mode: str,
    backend_api_major: int,
) -> ExtractedApi:
    if mode not in {"public", "backend"}:
        raise ValueError(f"unsupported HIP ABI extraction mode: {mode}")
    clang = find_clang(source_root)
    extract_dir = build_dir / "generated" / "clang_extract" / mode
    extract_dir.mkdir(parents=True, exist_ok=True)
    probe = extract_dir / "hip_header_probe.cpp"
    preprocessed = extract_dir / "hip_header_probe.ii"
    ast = extract_dir / "hip_header_probe.ast.json"
    probe.write_text(umbrella_source(mode, backend_api_major))

    include_args = [
        f"-I{generated_include_dir}",
        f"-I{source_root / 'runtime/hip-loader/include'}",
        f"-I{source_root / 'projects/clr/hipamd/include'}",
        f"-I{source_root / 'projects/hip/include'}",
    ]
    definitions = [
        "-D__HIP_PLATFORM_AMD__=1",
        "-D__HIP_DISABLE_CPP_FUNCTIONS__=1",
    ]
    preprocess_cmd = [
        str(clang),
        "-std=c++17",
        "-E",
        *definitions,
        *include_args,
        str(probe),
    ]
    with preprocessed.open("w") as output:
        subprocess.run(preprocess_cmd, stdout=output, check=True)
    if preprocessed.stat().st_size == 0:
        raise RuntimeError(f"clang produced an empty preprocessed file: {preprocessed}")

    ast_cmd = [
        str(clang),
        "-std=c++17",
        "-Wno-c++20-extensions",
        "-x",
        "c++-cpp-output",
        "-fsyntax-only",
        "-Xclang",
        "-ast-dump=json",
        str(preprocessed),
    ]
    with ast.open("w") as output:
        subprocess.run(ast_cmd, stdout=output, check=True)
    if ast.stat().st_size == 0:
        raise RuntimeError(f"clang produced an empty AST dump: {ast}")

    root_value = json.loads(ast.read_text())
    if not isinstance(root_value, dict):
        raise RuntimeError("clang AST dump root is not a JSON object")
    return parse_ast(root_value, preprocessed, ast)


def umbrella_source(mode: str, backend_api_major: int) -> str:
    abi_mode = ""
    if mode == "backend":
        abi_mode = (
            "#define HIP_ABI_MODE HIP_ABI_MODE_BACKEND\n"
            f"#define HIP_API_VERSION {backend_api_major}\n"
        )
    return f"""#define __HIP_PLATFORM_AMD__ 1
#define __HIP_DISABLE_CPP_FUNCTIONS__ 1
#define __HIP_HAS_GET_PCH 1
{abi_mode}typedef unsigned int GLenum;
typedef unsigned int GLuint;
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hip/hip_gl_interop.h>
#include <hip/hip_deprecated.h>
#include <hip/amd_detail/amd_channel_descriptor.h>
#include <hip/amd_detail/hip_profiler_ext.h>
#include <hip/amd_detail/hip_runtime_prof.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/amd_hip_runtime_pt_api.h>
#include <hip/amd_detail/amd_hip_gl_interop.h>
"""


def parse_ast(root: dict[str, object], preprocessed: Path, ast: Path) -> ExtractedApi:
    functions: dict[str, ExtractedFunction] = {}
    records: dict[str, ExtractedRecord] = {}
    enums: dict[str, ExtractedEnum] = {}
    for node in walk_nodes(root):
        kind = read_optional_str(node, "kind")
        if kind == "FunctionDecl":
            function = parse_function_decl(node)
            if function is not None and is_interesting_function(function.name):
                functions.setdefault(function.name, function)
        elif kind == "RecordDecl":
            record = parse_record_decl(node)
            if record is not None:
                records.setdefault(record.name, record)
        elif kind == "EnumDecl":
            enum = parse_enum_decl(node)
            if enum is not None:
                enums.setdefault(enum.name, enum)
    return ExtractedApi(functions, records, enums, preprocessed, ast)


def walk_nodes(node: object) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    stack = [node]
    while stack:
        value = stack.pop()
        if not isinstance(value, dict):
            continue
        result.append(value)
        inner = value.get("inner", [])
        if isinstance(inner, list):
            stack.extend(reversed(inner))
    return result


def parse_function_decl(node: dict[str, object]) -> ExtractedFunction | None:
    name = read_optional_str(node, "name")
    type_value = node.get("type")
    if name is None or not isinstance(type_value, dict):
        return None
    qual_type = read_optional_str(type_value, "qualType")
    if qual_type is None:
        return None
    return_type = function_return_type(qual_type)
    params: list[ExtractedParam] = []
    for child in read_inner(node):
        if read_optional_str(child, "kind") != "ParmVarDecl":
            continue
        param_type_value = child.get("type")
        param_type = None
        if isinstance(param_type_value, dict):
            param_type = read_optional_str(param_type_value, "qualType")
        if param_type is None:
            param_type = "void*"
        param_name = read_optional_str(child, "name")
        if param_name is None:
            param_name = f"arg{len(params)}"
        params.append(ExtractedParam(normalize_type(param_type), param_name))
    return ExtractedFunction(
        name=name,
        return_type=normalize_type(return_type),
        params=tuple(params),
        qual_type=normalize_type(qual_type),
        location=format_location(node),
    )


def parse_record_decl(node: dict[str, object]) -> ExtractedRecord | None:
    name = read_optional_str(node, "name")
    if name is None or not is_interesting_record(name):
        return None
    fields: list[ExtractedParam] = []
    for child in read_inner(node):
        if read_optional_str(child, "kind") != "FieldDecl":
            continue
        field_type_value = child.get("type")
        field_type = None
        if isinstance(field_type_value, dict):
            field_type = read_optional_str(field_type_value, "qualType")
        if field_type is None:
            continue
        field_name = read_optional_str(child, "name") or f"field{len(fields)}"
        fields.append(ExtractedParam(normalize_type(field_type), field_name))
    if not fields:
        return None
    return ExtractedRecord(name=name, fields=tuple(fields), location=format_location(node))


def parse_enum_decl(node: dict[str, object]) -> ExtractedEnum | None:
    name = read_optional_str(node, "name")
    if name is None or not is_interesting_record(name):
        return None
    constants: list[tuple[str, str | None]] = []
    for child in read_inner(node):
        if read_optional_str(child, "kind") != "EnumConstantDecl":
            continue
        constant_name = read_optional_str(child, "name")
        if constant_name is None:
            continue
        constants.append((constant_name, enum_constant_value(child)))
    if not constants:
        return None
    return ExtractedEnum(name=name, constants=tuple(constants), location=format_location(node))


def enum_constant_value(node: dict[str, object]) -> str | None:
    inner = read_inner(node)
    if not inner:
        return None
    value_node = inner[0]
    value = value_node.get("value")
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return None


def function_return_type(qual_type: str) -> str:
    match = re.match(r"^(.*?)\s*\(", qual_type)
    if match is None:
        return qual_type
    return match.group(1)


def normalize_type(value: str) -> str:
    value = value.replace(" *", "*").replace("* ", "* ")
    value = value.replace(" &", "&").replace("& ", "& ")
    value = re.sub(r"\bstruct\s+(hip[A-Za-z_][A-Za-z0-9_]*)", r"struct \1", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def is_interesting_function(name: str) -> bool:
    return (
        name.startswith("hip")
        or name.startswith("__hip")
        or name.startswith("__gnu_")
        or name.startswith("amd_dbgapi_")
    )


def is_interesting_record(name: str) -> bool:
    return (
        name.startswith("hip")
        or name.startswith("HIP_")
        or name.startswith("ihip")
        or name.startswith("amd")
        or name.startswith("Hip")
    )


def read_inner(node: dict[str, object]) -> list[dict[str, object]]:
    inner = node.get("inner", [])
    if not isinstance(inner, list):
        return []
    return [item for item in inner if isinstance(item, dict)]


def read_optional_str(node: dict[str, object], key: str) -> str | None:
    value = node.get(key)
    if isinstance(value, str):
        return value
    return None


def format_location(node: dict[str, object]) -> str | None:
    loc = node.get("loc")
    if not isinstance(loc, dict):
        return None
    file_value = loc.get("presumedFile") or loc.get("file")
    line_value = loc.get("presumedLine") or loc.get("line")
    included_from = loc.get("includedFrom")
    if not isinstance(file_value, str) and isinstance(included_from, dict):
        file_value = included_from.get("file")
    if not isinstance(line_value, int) and isinstance(loc.get("spellingLoc"), dict):
        spelling_loc = loc["spellingLoc"]
        line_value = spelling_loc.get("line")
        if not isinstance(file_value, str):
            file_value = spelling_loc.get("file")
    if isinstance(file_value, str) and isinstance(line_value, int):
        return f"{file_value}:{line_value}"
    if isinstance(file_value, str):
        return file_value
    return None
