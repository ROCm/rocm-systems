#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify and inventory the pinned shipping rocBLAS M2 SGEMM artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any

PACKAGE_IDENTITY_FIELDS = ("name", "version", "architecture", "filename", "sha256")


class InventoryError(ValueError):
    """A shipping workload artifact violates the pinned M2 contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InventoryError(f"{name} must be a JSON object")
    return value


def require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise InventoryError(f"{name} must be a nonempty string")
    return value


def require_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise InventoryError(f"{name} must be an integer")
    return value


def require_sha256(value: Any, name: str) -> str:
    digest = require_string(value, name)
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise InventoryError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def package_records(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise InventoryError("packages must be a nonempty list")
    records: list[dict[str, str]] = []
    for index, item in enumerate(value):
        package = require_object(item, f"packages[{index}]")
        record = {
            field: (
                require_sha256(package.get(field), f"packages[{index}].{field}")
                if field == "sha256"
                else require_string(package.get(field), f"packages[{index}].{field}")
            )
            for field in PACKAGE_IDENTITY_FIELDS
        }
        if set(package) != set(PACKAGE_IDENTITY_FIELDS):
            raise InventoryError(
                f"packages[{index}] must contain only verified artifact identity fields"
            )
        records.append(record)
    if records != sorted(records, key=lambda package: package["name"]):
        raise InventoryError("packages must be sorted by package name")
    if len({package["name"] for package in records}) != len(records):
        raise InventoryError("packages contains duplicate package names")
    return records


def regular_file(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise InventoryError(f"{name} is not a regular non-symlink file: {path}")
    return path.resolve()


def verify_hash(path: Path, expected: str, name: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise InventoryError(
            f"{name} hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


class MsgpackReader:
    """Decode the MessagePack subset used by rocBLAS Tensile metadata."""

    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def take(self, size: int) -> bytes:
        result = self.payload[self.offset : self.offset + size]
        if len(result) != size:
            raise InventoryError(
                f"truncated MessagePack value at byte {self.offset}, need {size}"
            )
        self.offset += size
        return result

    def unsigned(self, size: int) -> int:
        return int.from_bytes(self.take(size), "big")

    def signed(self, size: int) -> int:
        return int.from_bytes(self.take(size), "big", signed=True)

    def string(self, size: int) -> str:
        try:
            return self.take(size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise InventoryError("invalid UTF-8 in MessagePack string") from error

    def array(self, size: int) -> list[Any]:
        return [self.read() for _ in range(size)]

    def mapping(self, size: int) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        for _ in range(size):
            key = self.read()
            if key in result:
                raise InventoryError(f"duplicate MessagePack map key: {key!r}")
            result[key] = self.read()
        return result

    def read(self) -> Any:
        marker = self.unsigned(1)
        if marker <= 0x7F:
            return marker
        if marker >= 0xE0:
            return marker - 0x100
        if 0xA0 <= marker <= 0xBF:
            return self.string(marker & 0x1F)
        if 0x90 <= marker <= 0x9F:
            return self.array(marker & 0x0F)
        if 0x80 <= marker <= 0x8F:
            return self.mapping(marker & 0x0F)
        if marker == 0xC0:
            return None
        if marker == 0xC2:
            return False
        if marker == 0xC3:
            return True
        if marker == 0xC4:
            return self.take(self.unsigned(1))
        if marker == 0xC5:
            return self.take(self.unsigned(2))
        if marker == 0xC6:
            return self.take(self.unsigned(4))
        if marker == 0xCA:
            return struct.unpack(">f", self.take(4))[0]
        if marker == 0xCB:
            return struct.unpack(">d", self.take(8))[0]
        if marker == 0xCC:
            return self.unsigned(1)
        if marker == 0xCD:
            return self.unsigned(2)
        if marker == 0xCE:
            return self.unsigned(4)
        if marker == 0xCF:
            return self.unsigned(8)
        if marker == 0xD0:
            return self.signed(1)
        if marker == 0xD1:
            return self.signed(2)
        if marker == 0xD2:
            return self.signed(4)
        if marker == 0xD3:
            return self.signed(8)
        if marker == 0xD9:
            return self.string(self.unsigned(1))
        if marker == 0xDA:
            return self.string(self.unsigned(2))
        if marker == 0xDB:
            return self.string(self.unsigned(4))
        if marker == 0xDC:
            return self.array(self.unsigned(2))
        if marker == 0xDD:
            return self.array(self.unsigned(4))
        if marker == 0xDE:
            return self.mapping(self.unsigned(2))
        if marker == 0xDF:
            return self.mapping(self.unsigned(4))
        raise InventoryError(
            f"unsupported MessagePack marker 0x{marker:02x} at byte {self.offset - 1}"
        )


def decode_msgpack(path: Path) -> Any:
    reader = MsgpackReader(path.read_bytes())
    value = reader.read()
    if reader.offset != len(reader.payload):
        raise InventoryError(
            f"trailing bytes in {path}: decoded {reader.offset} of {len(reader.payload)}"
        )
    return value


def contains_placeholder(value: Any, placeholder: str) -> bool:
    if isinstance(value, dict):
        if value.get("type") == "Placeholder" and value.get("value") == placeholder:
            return True
        return any(contains_placeholder(item, placeholder) for item in value.values())
    if isinstance(value, list):
        return any(contains_placeholder(item, placeholder) for item in value)
    return False


def matching_entries(value: Any, key: list[int], mode: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("type") == "Matching":
            table = value.get("table")
            if not isinstance(table, list):
                raise InventoryError("Tensile Matching library has no table")
            entries = [
                require_object(entry, "Tensile matching entry") for entry in table
            ]
            if mode == "exact":
                matches.extend(entry for entry in entries if entry.get("key") == key)
            elif mode == "nearest_euclidean":
                distances: list[tuple[int, dict[str, Any]]] = []
                for entry in entries:
                    entry_key = entry.get("key")
                    if (
                        not isinstance(entry_key, list)
                        or len(entry_key) != len(key)
                        or not all(
                            isinstance(item, int) and not isinstance(item, bool)
                            for item in entry_key
                        )
                    ):
                        raise InventoryError(
                            "Tensile matching entry key has the wrong shape"
                        )
                    distance = sum(
                        (entry_item - requested_item) ** 2
                        for entry_item, requested_item in zip(entry_key, key)
                    )
                    distances.append((distance, entry))
                if distances:
                    nearest = min(distance for distance, _entry in distances)
                    matches.extend(
                        entry for distance, entry in distances if distance == nearest
                    )
            else:
                raise InventoryError(f"unsupported selection.matching_mode {mode!r}")
        for item in value.values():
            matches.extend(matching_entries(item, key, mode))
    elif isinstance(value, list):
        for item in value:
            matches.extend(matching_entries(item, key, mode))
    return matches


def select_solution(metadata: Any, selection: dict[str, Any]) -> dict[str, Any]:
    root = require_object(metadata, "Tensile metadata")
    solutions = root.get("solutions")
    library = root.get("library")
    if not isinstance(solutions, list) or library is None:
        raise InventoryError("Tensile metadata must contain solutions and library")
    key = selection.get("matching_key")
    if (
        not isinstance(key, list)
        or len(key) != 3
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in key)
    ):
        raise InventoryError("selection.matching_key must contain three integers")
    mode = require_string(selection.get("matching_mode"), "selection.matching_mode")
    entries = matching_entries(library, key, mode)
    if not entries:
        raise InventoryError(
            f"Tensile metadata has no {mode} matching entry for key {key}"
        )
    indices = {
        require_integer(entry.get("index"), "matching entry index") for entry in entries
    }
    candidates: list[dict[str, Any]] = []
    expected_strided = selection.get("strided_batched")
    if not isinstance(expected_strided, bool):
        raise InventoryError("selection.strided_batched must be boolean")
    for solution in solutions:
        item = require_object(solution, "Tensile solution")
        if item.get("index") not in indices:
            continue
        problem_type = require_object(item.get("problemType"), "solution.problemType")
        if problem_type.get("stridedBatched") == expected_strided:
            candidates.append(item)
    if not candidates:
        raise InventoryError(
            f"matching key {key} resolved to no solutions for "
            f"stridedBatched={expected_strided}"
        )
    expected_indices_value = selection.get("expected_solution_indices")
    if (
        not isinstance(expected_indices_value, list)
        or not expected_indices_value
        or not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in expected_indices_value
        )
    ):
        raise InventoryError(
            "selection.expected_solution_indices must contain integers"
        )
    expected_indices = sorted(expected_indices_value)
    if expected_indices != sorted(set(expected_indices)):
        raise InventoryError("selection.expected_solution_indices contains duplicates")
    candidate_indices = sorted(
        require_integer(candidate.get("index"), "solution.index")
        for candidate in candidates
    )
    if candidate_indices != expected_indices:
        raise InventoryError(
            "selected solution indices changed: expected "
            f"{expected_indices}, got {candidate_indices}"
        )
    expected_symbol = require_string(
        selection.get("expected_symbol"), "selection.expected_symbol"
    )
    if {candidate.get("name") for candidate in candidates} != {expected_symbol}:
        raise InventoryError("selected solution symbol changed")
    solution = candidates[0]
    problem_type = require_object(solution.get("problemType"), "solution.problemType")
    expected_operation = require_string(
        selection.get("operation_identifier"), "selection.operation_identifier"
    )
    if problem_type.get("operationIdentifier") != expected_operation:
        raise InventoryError("selected solution operation identifier changed")
    if [problem_type.get(name) for name in ("aType", "bType", "cType", "dType")] != [
        "Float",
        "Float",
        "Float",
        "Float",
    ]:
        raise InventoryError("selected solution is not FP32 SGEMM")
    selected_entries = [
        entry for entry in entries if entry.get("index") in expected_indices
    ]
    matched_keys = sorted({tuple(entry["key"]) for entry in selected_entries})
    return {
        "candidate_indices": sorted(indices),
        "indices": expected_indices,
        "matched_keys": [list(matched_key) for matched_key in matched_keys],
        "matching_key": key,
        "matching_mode": mode,
        "name": expected_symbol,
        "problem_type": problem_type,
        "size_mapping": require_object(
            solution.get("sizeMapping"), "solution.sizeMapping"
        ),
        "speeds": [entry.get("speed") for entry in selected_entries],
    }


def run_tool(arguments: list[str], name: str) -> str:
    result = subprocess.run(arguments, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        diagnostics = result.stderr.strip() or result.stdout.strip()
        raise InventoryError(f"{name} failed: {diagnostics}")
    return result.stdout


def tool_record(path: Path) -> dict[str, str]:
    output = run_tool([str(path), "--version"], f"{path.name} --version").strip()
    if not output:
        raise InventoryError(f"{path.name} --version produced no output")
    return {"path": str(path), "sha256": sha256_file(path), "version": output}


def runtime_dependencies(binary: Path, ldd: Path) -> list[dict[str, str]]:
    output = run_tool([str(ldd), str(binary)], "ldd")
    dependencies: dict[str, dict[str, str]] = {}
    for line in output.splitlines():
        if "not found" in line:
            raise InventoryError(f"unresolved runtime dependency: {line.strip()}")
        match = re.search(r"=>\s+(/\S+)", line)
        if match is None:
            match = re.match(r"\s*(/\S+)\s+\(", line)
        if match is None:
            continue
        guest_path = match.group(1).removeprefix("/")
        try:
            source = Path(match.group(1)).resolve(strict=True)
        except OSError as error:
            raise InventoryError(
                f"cannot resolve runtime dependency {guest_path}: {error}"
            ) from error
        if not source.is_file():
            raise InventoryError(f"runtime dependency is not a regular file: {source}")
        dependencies[guest_path] = {
            "guest_path": guest_path,
            "sha256": sha256_file(source),
            "source": str(source),
        }
    if not dependencies:
        raise InventoryError("ldd reported no runtime dependencies")
    return [dependencies[name] for name in sorted(dependencies)]


def parse_kernel_metadata(readobj_output: str, symbol: str) -> dict[str, Any]:
    chunks = readobj_output.split("AMDGPU Metadata: ---")
    matching = [chunk for chunk in chunks if f".name:           {symbol}\n" in chunk]
    if len(matching) != 1:
        raise InventoryError(
            f"llvm-readobj reported {len(matching)} metadata records for {symbol}"
        )
    metadata = matching[0].split("...", 1)[0]
    name_marker = f".name:           {symbol}\n"
    name_offset = metadata.index(name_marker)
    kernel_start = metadata.rfind("\n  - .args:", 0, name_offset)
    if kernel_start < 0:
        raise InventoryError("selected kernel metadata has no record start")
    next_kernel = metadata.find("\n  - .args:", name_offset)
    target_offset = metadata.find("\namdhsa.target:", name_offset)
    kernel_end_candidates = [
        offset for offset in (next_kernel, target_offset) if offset >= 0
    ]
    if not kernel_end_candidates:
        raise InventoryError("selected kernel metadata has no record end")
    kernel = metadata[kernel_start : min(kernel_end_candidates)]

    def integer_field(name: str) -> int:
        match = re.search(
            rf"^\s*\.{re.escape(name)}:\s+(\d+)\s*$", kernel, re.MULTILINE
        )
        if match is None:
            raise InventoryError(f"selected kernel metadata has no .{name}")
        return int(match.group(1))

    target = re.search(r"^amdhsa\.target:\s+(\S+)\s*$", metadata, re.MULTILINE)
    if target is None:
        raise InventoryError("selected kernel metadata has no amdhsa.target")
    metadata_version = re.search(
        r"^amdhsa\.version:\s*\n\s*-\s*(\d+)\s*\n\s*-\s*(\d+)",
        metadata,
        re.MULTILINE,
    )
    if metadata_version is None:
        raise InventoryError("selected kernel metadata has no amdhsa.version")
    result = {
        "group_segment_fixed_size": integer_field("group_segment_fixed_size"),
        "kernarg_segment_align": integer_field("kernarg_segment_align"),
        "kernarg_segment_size": integer_field("kernarg_segment_size"),
        "metadata_version": [
            int(metadata_version.group(1)),
            int(metadata_version.group(2)),
        ],
        "private_segment_fixed_size": integer_field("private_segment_fixed_size"),
        "sgpr_count": integer_field("sgpr_count"),
        "sgpr_spill_count": integer_field("sgpr_spill_count"),
        "target": target.group(1),
        "vgpr_count": integer_field("vgpr_count"),
        "vgpr_spill_count": integer_field("vgpr_spill_count"),
        "wavefront_size": integer_field("wavefront_size"),
    }
    revision = re.search(r"^\s*\.gfx\d+_revision:\s+(\S+)\s*$", kernel, re.MULTILINE)
    if revision is not None:
        result["silicon_revision"] = revision.group(1)
    return result


def parse_instructions(disassembly: str) -> dict[str, Any]:
    opcodes = sorted(
        {
            match.group(1)
            for line in disassembly.splitlines()
            if (match := re.match(r"^\s+([a-z][a-z0-9_.]+)(?:\s|$)", line))
        }
    )
    if not opcodes:
        raise InventoryError("selected kernel disassembly contains no instructions")
    coherency_flags = sorted(
        {
            flag
            for flag in ("dlc", "glc", "sc0", "sc1", "slc")
            if re.search(rf"\b{flag}\b", disassembly)
        }
    )
    return {
        "atomics": [opcode for opcode in opcodes if "atomic" in opcode],
        "barriers": [opcode for opcode in opcodes if "barrier" in opcode],
        "coherency_flags": coherency_flags,
        "instruction_count": sum(
            1
            for line in disassembly.splitlines()
            if re.match(r"^\s+[a-z][a-z0-9_.]+(?:\s|$)", line)
        ),
        "lds": [opcode for opcode in opcodes if opcode.startswith("ds_")],
        "mfma": [opcode for opcode in opcodes if opcode.startswith("v_mfma")],
        "opcodes": opcodes,
        "waits": [opcode for opcode in opcodes if opcode.startswith("s_wait")],
    }


def verify_elf_identity(
    readobj_output: str, expected: dict[str, Any]
) -> dict[str, Any]:
    abi_version = require_integer(
        expected.get("abi_version"), "artifacts.code_object.abi_version"
    )
    processor = require_string(
        expected.get("processor"), "artifacts.code_object.processor"
    )
    required = (
        "Format: elf64-amdgpu",
        "Arch: amdgcn",
        "OS/ABI: AMDGPU_HSA",
        f"ABIVersion: {abi_version}",
        "Machine: EM_AMDGPU",
        f"EF_AMDGPU_MACH_AMDGCN_{processor.upper()}",
    )
    missing = [text for text in required if text not in readobj_output]
    if missing:
        raise InventoryError(
            "code object identity changed; missing " + ", ".join(missing)
        )
    return {
        "abi_version": abi_version,
        "architecture": "amdgcn",
        "elf_class": 64,
        "machine": "EM_AMDGPU",
        "os_abi": "AMDGPU_HSA",
        "processor": processor,
        "sramecc": "any",
        "xnack": "any",
    }


def inventory(
    contract_path: Path,
    workload_binary: Path,
    code_object: Path,
    metadata: Path,
    lazy_metadata: Path,
    llvm_readobj: Path,
    llvm_nm: Path,
    llvm_objdump: Path,
    ldd: Path,
    guest_init: Path | None = None,
) -> dict[str, Any]:
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InventoryError(f"cannot read workload contract: {error}") from error
    root = require_object(contract, "workload contract")
    if root.get("schema_version") != 2:
        raise InventoryError("workload contract schema_version must be 2")
    artifacts = require_object(root.get("artifacts"), "artifacts")
    selection = require_object(root.get("selection"), "selection")
    source = require_object(root.get("source"), "source")
    workload = require_object(root.get("workload"), "workload")
    runtime = require_object(root.get("runtime"), "runtime")
    guest_init_contract = require_object(root.get("guest_init"), "guest_init")
    packages = package_records(root.get("packages"))
    if [workload.get(name) for name in ("m", "n", "k")] != [128, 128, 128]:
        raise InventoryError("M2 workload dimensions must remain 128 x 128 x 128")
    if workload.get("api") != "rocblas_sgemm":
        raise InventoryError("M2 workload must call rocblas_sgemm")
    if runtime.get("backend") != "Tensile":
        raise InventoryError("M2 workload must select the rocBLAS Tensile backend")
    environment = require_object(runtime.get("environment"), "runtime.environment")
    if set(environment) != {"ROCBLAS_TENSILE_LIBPATH", "ROCBLAS_USE_HIPBLASLT"}:
        raise InventoryError("M2 runtime environment has unsupported entries")
    require_string(
        environment.get("ROCBLAS_TENSILE_LIBPATH"),
        "runtime.environment.ROCBLAS_TENSILE_LIBPATH",
    )
    if environment.get("ROCBLAS_USE_HIPBLASLT") != "0":
        raise InventoryError("M2 runtime environment must disable hipBLASLt")
    if set(guest_init_contract) != {"guest_path", "sha256"}:
        raise InventoryError("guest_init must contain only guest_path and sha256")
    if (
        require_string(guest_init_contract.get("guest_path"), "guest_init.guest_path")
        != "init"
    ):
        raise InventoryError("M2 guest init must be installed at init")
    guest_init = regular_file(
        guest_init or contract_path.parent / "init", "M2 guest init"
    )
    verify_hash(
        guest_init,
        require_sha256(guest_init_contract.get("sha256"), "guest_init.sha256"),
        "M2 guest init",
    )

    checked_paths = {
        "code_object": regular_file(code_object, "code object"),
        "lazy_metadata": regular_file(lazy_metadata, "lazy metadata"),
        "metadata": regular_file(metadata, "solution metadata"),
    }
    for name, path in checked_paths.items():
        artifact = require_object(artifacts.get(name), f"artifacts.{name}")
        verify_hash(
            path,
            require_sha256(artifact.get("sha256"), f"artifacts.{name}.sha256"),
            name,
        )

    contract_path = regular_file(contract_path, "workload contract")
    source_path = regular_file(
        contract_path.parent / require_string(source.get("path"), "source.path"),
        "workload source",
    )
    workload_binary = regular_file(workload_binary, "workload binary")
    llvm_readobj = regular_file(llvm_readobj, "llvm-readobj")
    llvm_nm = regular_file(llvm_nm, "llvm-nm")
    llvm_objdump = regular_file(llvm_objdump, "llvm-objdump")
    ldd = regular_file(ldd, "ldd")

    try:
        described_workload = json.loads(
            run_tool([str(workload_binary), "--describe"], "workload binary --describe")
        )
    except json.JSONDecodeError as error:
        raise InventoryError(
            "workload binary --describe returned invalid JSON"
        ) from error
    if described_workload != workload:
        raise InventoryError("workload binary does not match the pinned contract")

    lazy = decode_msgpack(checked_paths["lazy_metadata"])
    placeholder = require_string(selection.get("placeholder"), "selection.placeholder")
    if not contains_placeholder(lazy, placeholder):
        raise InventoryError(f"lazy metadata does not route to {placeholder}")
    selected = select_solution(decode_msgpack(checked_paths["metadata"]), selection)

    code_object_contract = require_object(
        artifacts.get("code_object"), "artifacts.code_object"
    )
    target = require_string(
        code_object_contract.get("target"), "artifacts.code_object.target"
    )

    readobj_output = run_tool(
        [
            str(llvm_readobj),
            "--file-headers",
            "--notes",
            str(checked_paths["code_object"]),
        ],
        "llvm-readobj",
    )
    symbol = selected["name"]
    nm_output = run_tool(
        [str(llvm_nm), "--defined-only", str(checked_paths["code_object"])],
        "llvm-nm",
    )
    text_symbols = [
        (int(match.group(1), 16), match.group(2))
        for line in nm_output.splitlines()
        if (match := re.match(r"^([0-9a-fA-F]+)\s+T\s+(\S+)$", line))
    ]
    symbol_addresses = [address for address, name in text_symbols if name == symbol]
    if len(symbol_addresses) != 1:
        raise InventoryError("selected Tensile solution is not a defined kernel symbol")
    start_address = symbol_addresses[0]
    following_addresses = sorted(
        {address for address, _ in text_symbols if address > start_address}
    )
    if not following_addresses:
        raise InventoryError("selected Tensile solution has no bounded text range")
    stop_address = following_addresses[0]
    disassembly = run_tool(
        [
            str(llvm_objdump),
            "--disassemble",
            f"--start-address={start_address}",
            f"--stop-address={stop_address}",
            str(checked_paths["code_object"]),
        ],
        "llvm-objdump",
    )
    kernel_metadata = parse_kernel_metadata(readobj_output, symbol)
    if kernel_metadata["target"] != target:
        raise InventoryError(
            "selected kernel target changed: expected "
            f"{target}, got {kernel_metadata['target']}"
        )
    expected_revision = code_object_contract.get("silicon_revision")
    if expected_revision is not None:
        expected_revision = require_string(
            expected_revision, "artifacts.code_object.silicon_revision"
        )
        if kernel_metadata.get("silicon_revision") != expected_revision:
            raise InventoryError(
                "selected kernel silicon revision changed: expected "
                f"{expected_revision}, got {kernel_metadata.get('silicon_revision')}"
            )

    required_guest_files = {
        require_string(source.get("guest_path"), "source.guest_path"): sha256_file(
            source_path
        ),
        require_string(
            workload.get("guest_executable"), "workload.guest_executable"
        ): sha256_file(workload_binary),
    }
    for name in ("code_object", "lazy_metadata", "metadata"):
        artifact = require_object(artifacts.get(name), f"artifacts.{name}")
        required_guest_files[
            require_string(artifact.get("guest_path"), f"artifacts.{name}.guest_path")
        ] = sha256_file(checked_paths[name])
    dependencies = runtime_dependencies(workload_binary, ldd)
    for dependency in dependencies:
        required_guest_files[dependency["guest_path"]] = dependency["sha256"]

    return {
        "artifacts": {
            "code_object": {
                "path": str(checked_paths["code_object"]),
                "sha256": sha256_file(checked_paths["code_object"]),
                "target": target,
            },
            "lazy_metadata": {
                "path": str(checked_paths["lazy_metadata"]),
                "sha256": sha256_file(checked_paths["lazy_metadata"]),
            },
            "metadata": {
                "path": str(checked_paths["metadata"]),
                "sha256": sha256_file(checked_paths["metadata"]),
            },
            "workload_binary": {
                "path": str(workload_binary),
                "sha256": sha256_file(workload_binary),
            },
            "workload_source": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
            },
            "runtime_dependencies": dependencies,
        },
        "code_object": {
            "identity": verify_elf_identity(readobj_output, code_object_contract),
            "instructions": parse_instructions(disassembly),
            "kernel_count": len(text_symbols),
            "selected_kernel_range": {
                "size": stop_address - start_address,
                "start": start_address,
                "stop": stop_address,
            },
            "selected_kernel_metadata": kernel_metadata,
        },
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "feature_tasks": require_object(root.get("feature_tasks"), "feature_tasks"),
        "guest_init": {
            "guest_path": "init",
            "sha256": sha256_file(guest_init),
        },
        "packages": packages,
        "required_guest_files": required_guest_files,
        "runtime": {
            "backend": "Tensile",
            "environment": environment,
        },
        "schema_version": 2,
        "selected_solution": selected,
        "tools": {
            "llvm_nm": tool_record(llvm_nm),
            "llvm_objdump": tool_record(llvm_objdump),
            "llvm_readobj": tool_record(llvm_readobj),
            "ldd": tool_record(ldd),
        },
        "workload": workload,
    }


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--workload-binary", required=True, type=Path)
    parser.add_argument("--code-object", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--lazy-metadata", required=True, type=Path)
    parser.add_argument("--llvm-readobj", required=True, type=Path)
    parser.add_argument("--llvm-nm", required=True, type=Path)
    parser.add_argument("--llvm-objdump", required=True, type=Path)
    parser.add_argument("--ldd", default=Path("/usr/bin/ldd"), type=Path)
    parser.add_argument(
        "--guest-init",
        type=Path,
        help="repository-owned M2 init (defaults to the contract directory's init)",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_arguments(arguments)
    try:
        result = inventory(
            args.contract,
            args.workload_binary,
            args.code_object,
            args.metadata,
            args.lazy_metadata,
            args.llvm_readobj,
            args.llvm_nm,
            args.llvm_objdump,
            args.ldd,
            args.guest_init,
        )
        args.output.write_text(canonical_json(result), encoding="utf-8")
    except (InventoryError, OSError, subprocess.SubprocessError) as error:
        print(f"vfio guest SGEMM inventory failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
