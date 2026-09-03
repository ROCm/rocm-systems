#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build a deterministic vfio-user initramfs from an explicit input lock."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO, Iterator

SCHEMA_VERSION = 2
M2_IP_BLOCK_MASK = "0x3f"
M2_FIRMWARE_DIRECTORY = PurePosixPath("lib/firmware/amdgpu")
M2_SYNTHETIC_FIRMWARE = (
    "gc_12_1_0_imu.bin",
    "gc_12_1_0_mec.bin",
    "gc_12_1_0_rlc_1.bin",
    "gc_12_1_0_uni_mes.bin",
    "sdma_7_1_0.bin",
)
M2_IP_DISCOVERY_FIRMWARE = "ip_discovery.bin"
M2_FIRMWARE_PATHS = frozenset(
    M2_FIRMWARE_DIRECTORY / name
    for name in (*M2_SYNTHETIC_FIRMWARE, M2_IP_DISCOVERY_FIRMWARE)
)
NEWC_HEADER_BYTES = 110
FIRMWARE_DIRECTORIES = (
    PurePosixPath("lib/firmware/amdgpu"),
    PurePosixPath("lib/firmware/updates/amdgpu"),
)
RUNTIME_DIRECTORIES = (
    (PurePosixPath("dev"), 0o755),
    (PurePosixPath("proc"), 0o755),
    (PurePosixPath("sys"), 0o755),
    (PurePosixPath("tmp"), 0o1777),
)
HOST_FIRMWARE_DIRECTORIES = (
    Path("/lib/firmware/amdgpu"),
    Path("/lib/firmware/updates/amdgpu"),
)
PACKAGE_IDENTITY_FIELDS = ("name", "version", "architecture", "filename", "sha256")
AMDGPU_ZERO_INSTANCE_GUARD = re.compile(
    r"if\s*\(\s*!max_res\[i\]\s*\)\s*continue\s*;", re.MULTILINE
)
AMDGPU_ZERO_INSTANCE_GUARD_SOURCES = (
    "soc_v1_0.c",
    "aqua_vanjaram.c",
)
AMDGPU_NO_VCN_DISCOVERY_GUARD = re.compile(
    r"vcn_version\s*=\s*amdgpu_ip_version\s*\(\s*adev\s*,\s*UVD_HWIP\s*,\s*0\s*\)\s*;"
    r".{0,512}?if\s*\(\s*!vcn_version\s*\)\s*return\s+0\s*;",
    re.DOTALL,
)


class ImageBuildError(ValueError):
    """An image lock or input violates the hermetic guest contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ImageBuildError(f"{name} must be a JSON object")
    return value


def require_m2_runtime(inventory: dict[str, Any]) -> dict[str, Any]:
    runtime = require_object(inventory.get("runtime"), "workload.inventory.runtime")
    if runtime.get("backend") != "Tensile":
        raise ImageBuildError(
            "workload inventory must select the rocBLAS Tensile backend"
        )
    environment = require_object(runtime.get("environment"), "runtime.environment")
    if set(environment) != {"ROCBLAS_TENSILE_LIBPATH", "ROCBLAS_USE_HIPBLASLT"}:
        raise ImageBuildError(
            "workload inventory has an unsupported runtime environment"
        )
    require_string(
        environment.get("ROCBLAS_TENSILE_LIBPATH"),
        "runtime.environment.ROCBLAS_TENSILE_LIBPATH",
    )
    if environment.get("ROCBLAS_USE_HIPBLASLT") != "0":
        raise ImageBuildError("workload inventory must disable the hipBLASLt backend")
    return runtime


def require_m2_init(inventory: dict[str, Any], init_path: Path) -> dict[str, str]:
    record = require_object(
        inventory.get("guest_init"), "workload.inventory.guest_init"
    )
    if set(record) != {"guest_path", "sha256"}:
        raise ImageBuildError(
            "workload.inventory.guest_init must contain only guest_path and sha256"
        )
    if guest_path(
        record.get("guest_path"), "workload.inventory.guest_init.guest_path"
    ) != PurePosixPath("init"):
        raise ImageBuildError("M2 guest init must be installed at init")
    expected = require_sha256(
        record.get("sha256"), "workload.inventory.guest_init.sha256"
    )
    actual = sha256_file(init_path)
    if actual != expected:
        raise ImageBuildError(
            f"M2 guest init hash mismatch: expected {expected}, got {actual}"
        )
    return {"guest_path": "init", "sha256": expected}


def require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ImageBuildError(f"{name} must be a nonempty string")
    return value


def require_sha256(value: Any, name: str) -> str:
    digest = require_string(value, name).lower()
    if len(digest) != 64:
        raise ImageBuildError(f"{name} must be a SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ImageBuildError(f"{name} must be a SHA-256 digest") from error
    return digest


def guest_path(value: Any, name: str) -> PurePosixPath:
    raw = require_string(value, name)
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or raw.startswith("./")
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ImageBuildError(f"{name} is not a canonical guest path: {raw!r}")
    return path


def is_firmware(path: PurePosixPath) -> bool:
    return any(
        path != directory and path.is_relative_to(directory)
        for directory in FIRMWARE_DIRECTORIES
    )


def source_path(lock_path: Path, value: Any, name: str) -> Path:
    raw = Path(require_string(value, name))
    path = raw if raw.is_absolute() else lock_path.parent / raw
    if path.is_symlink():
        raise ImageBuildError(f"{name} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ImageBuildError(f"{name} cannot be resolved: {path}: {error}") from error
    if not resolved.is_file():
        raise ImageBuildError(f"{name} is not a regular file: {resolved}")
    for directory in HOST_FIRMWARE_DIRECTORIES:
        if resolved.is_relative_to(directory):
            raise ImageBuildError(
                f"{name} imports host AMD firmware, which is forbidden: {resolved}"
            )
    return resolved


def run_tool(command: list[str], description: str) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as error:
        raise ImageBuildError(f"cannot run {description}: {error}") from error
    if result.returncode != 0:
        diagnostics = result.stderr.strip() or result.stdout.strip()
        raise ImageBuildError(f"{description} failed: {diagnostics}")
    return result.stdout


def package_identity(record: dict[str, Any]) -> dict[str, str]:
    return {field: str(record[field]) for field in PACKAGE_IDENTITY_FIELDS}


def debian_upstream_version(version: str) -> str:
    without_epoch = version.split(":", 1)[-1]
    return without_epoch.rsplit("-", 1)[0]


def debian_package_record(path: Path, dpkg_deb: Path) -> dict[str, str]:
    output = run_tool(
        [
            str(dpkg_deb),
            "-W",
            "--showformat=${Package}\\n${Version}\\n${Architecture}\\n",
            str(path),
        ],
        f"dpkg-deb metadata inspection for {path}",
    )
    fields = output.splitlines()
    if len(fields) != 3 or any(not field for field in fields):
        raise ImageBuildError(f"dpkg-deb returned incomplete metadata for {path}")
    return {
        "name": fields[0],
        "version": fields[1],
        "architecture": fields[2],
        "filename": path.name,
        "sha256": sha256_file(path),
        "source": str(path),
    }


def validate_package_record(
    lock_path: Path, value: Any, name: str, dpkg_deb: Path
) -> dict[str, str]:
    record = require_object(value, name)
    expected_fields = set(PACKAGE_IDENTITY_FIELDS) | {"source"}
    if set(record) != expected_fields:
        raise ImageBuildError(
            f"{name} must contain only verified artifact identity and source fields"
        )
    source = source_path(lock_path, record.get("source"), f"{name}.source")
    for field in PACKAGE_IDENTITY_FIELDS:
        if field == "sha256":
            require_sha256(record.get(field), f"{name}.{field}")
        else:
            require_string(record.get(field), f"{name}.{field}")
    actual = debian_package_record(source, dpkg_deb)
    if package_identity(record) != package_identity(actual):
        raise ImageBuildError(
            f"{name} metadata or digest does not match package artifact {source}"
        )
    return actual


def require_amdgpu_compute_only_guards(path: Path, dpkg_deb: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="rocjitsu-amdgpu-package-") as temporary:
        extracted = Path(temporary)
        run_tool(
            [str(dpkg_deb), "-x", str(path), str(extracted)],
            f"dpkg-deb extraction for {path}",
        )
        for filename in AMDGPU_ZERO_INSTANCE_GUARD_SOURCES:
            candidates = sorted(extracted.glob(f"usr/src/*/amd/amdgpu/{filename}"))
            if len(candidates) != 1:
                raise ImageBuildError(
                    "amdgpu package must contain exactly one " f"amd/amdgpu/{filename}"
                )
            try:
                source = candidates[0].read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                raise ImageBuildError(
                    f"cannot inspect amdgpu partition source {filename} in {path}: "
                    f"{error}"
                ) from error
            if not AMDGPU_ZERO_INSTANCE_GUARD.search(source):
                raise ImageBuildError(
                    "amdgpu package lacks the zero-instance partition guard in "
                    f"{filename} required for compute-only devices"
                )
        candidates = sorted(extracted.glob("usr/src/*/amd/amdgpu/amdgpu_discovery.c"))
        if len(candidates) != 1:
            raise ImageBuildError(
                "amdgpu package must contain exactly one "
                "amd/amdgpu/amdgpu_discovery.c"
            )
        try:
            source = candidates[0].read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ImageBuildError(
                "cannot inspect amdgpu discovery source amdgpu_discovery.c in "
                f"{path}: {error}"
            ) from error
        if not AMDGPU_NO_VCN_DISCOVERY_GUARD.search(source):
            raise ImageBuildError(
                "amdgpu package lacks the no-VCN discovery guard in "
                "amdgpu_discovery.c required for compute-only devices"
            )


def module_identity(path: Path, modinfo: Path) -> dict[str, str]:
    identity: dict[str, str] = {}
    for field in ("version", "srcversion", "vermagic"):
        value = run_tool(
            [str(modinfo), "-F", field, str(path)],
            f"modinfo {field} inspection for {path}",
        ).strip()
        if not value:
            raise ImageBuildError(f"amdgpu module has no {field}: {path}")
        identity[field] = value
    return identity


def require_string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ImageBuildError(f"{name} must be a nonempty string list")
    return [
        require_string(item, f"{name}[{index}]") for index, item in enumerate(value)
    ]


def validate_build_attestation(
    lock_path: Path,
    value: Any,
    package: dict[str, str],
    kernel_release: str,
    kernel_sha256: str,
    module_sha256: str,
    identity: dict[str, str],
) -> dict[str, Any]:
    wrapper = require_object(value, "amdgpu.build_attestation")
    if set(wrapper) != {"record", "sha256", "source"}:
        raise ImageBuildError(
            "amdgpu.build_attestation must contain only record, sha256, and source"
        )
    source = source_path(
        lock_path, wrapper.get("source"), "amdgpu.build_attestation.source"
    )
    expected_digest = require_sha256(
        wrapper.get("sha256"), "amdgpu.build_attestation.sha256"
    )
    if sha256_file(source) != expected_digest:
        raise ImageBuildError("amdgpu build-attestation artifact hash mismatch")
    try:
        source_record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImageBuildError(
            f"invalid amdgpu build-attestation artifact: {error}"
        ) from error
    record = require_object(wrapper.get("record"), "amdgpu.build_attestation.record")
    if source_record != record:
        raise ImageBuildError(
            "embedded amdgpu build attestation differs from its source artifact"
        )
    if record.get("schema_version") != 1:
        raise ImageBuildError("amdgpu build attestation schema_version must be 1")
    if set(record) != {
        "amdgpu_package_sha256",
        "build",
        "kernel",
        "module",
        "schema_version",
    }:
        raise ImageBuildError("amdgpu build attestation has unsupported fields")
    if (
        require_sha256(
            record.get("amdgpu_package_sha256"),
            "amdgpu build attestation package digest",
        )
        != package["sha256"]
    ):
        raise ImageBuildError("amdgpu build attestation names a different package")
    kernel = require_object(record.get("kernel"), "amdgpu build attestation kernel")
    if kernel != {"release": kernel_release, "sha256": kernel_sha256}:
        raise ImageBuildError("amdgpu build attestation names a different kernel")
    module = require_object(record.get("module"), "amdgpu build attestation module")
    if module != {"sha256": module_sha256, **identity}:
        raise ImageBuildError("amdgpu build attestation names a different module")
    vermagic_release = identity["vermagic"].split(maxsplit=1)[0]
    if vermagic_release != kernel_release:
        raise ImageBuildError(
            "amdgpu module vermagic release does not match the guest kernel release"
        )
    build = require_object(record.get("build"), "amdgpu build attestation build")
    if set(build) != {"command", "inputs", "toolchain", "working_directory"}:
        raise ImageBuildError(
            "amdgpu build attestation build must contain command, inputs, "
            "toolchain, and working_directory"
        )
    require_string_list(build.get("command"), "amdgpu build attestation command")
    require_string(
        build.get("working_directory"),
        "amdgpu build attestation working_directory",
    )
    for collection in ("inputs", "toolchain"):
        entries = build.get(collection)
        if not isinstance(entries, list) or not entries:
            raise ImageBuildError(
                f"amdgpu build attestation {collection} must be a nonempty list"
            )
        for index, item in enumerate(entries):
            entry = require_object(
                item, f"amdgpu build attestation {collection}[{index}]"
            )
            required = {"path", "sha256"}
            if collection == "toolchain":
                required.add("version")
            if set(entry) != required:
                raise ImageBuildError(
                    f"amdgpu build attestation {collection}[{index}] has unsupported fields"
                )
            artifact = source_path(
                source.parent,
                entry.get("path"),
                f"amdgpu build attestation {collection}[{index}].path",
            )
            artifact_digest = require_sha256(
                entry.get("sha256"), f"{collection}[{index}].sha256"
            )
            if sha256_file(artifact) != artifact_digest:
                raise ImageBuildError(
                    f"amdgpu build attestation {collection}[{index}] hash mismatch"
                )
            if collection == "toolchain":
                require_string(entry.get("version"), f"{collection}[{index}].version")
    return {"record": record, "sha256": expected_digest, "source": str(source)}


def _archive_path(value: str, name: str) -> PurePosixPath:
    normalized = value.removeprefix("./").lstrip("/")
    path = PurePosixPath(normalized)
    if not normalized or normalized == "." or ".." in path.parts:
        raise ImageBuildError(f"unsafe path in {name}: {value!r}")
    return path


def package_payload_entries(
    package: dict[str, str], dpkg_deb: Path
) -> dict[PurePosixPath, tuple[str, str]]:
    process = subprocess.Popen(
        [str(dpkg_deb), "--fsys-tarfile", package["source"]],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    entries: dict[PurePosixPath, tuple[str, str]] = {}
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|*") as archive:
            for member in archive:
                if member.name in (".", "./"):
                    continue
                path = _archive_path(member.name, f"package {package['filename']}")
                if member.isdir():
                    continue
                if path in entries:
                    raise ImageBuildError(
                        f"package {package['filename']} contains duplicate path {path}"
                    )
                if member.isfile():
                    payload = archive.extractfile(member)
                    if payload is None:
                        raise ImageBuildError(
                            f"cannot read {path} from package {package['filename']}"
                        )
                    digest = hashlib.sha256()
                    while chunk := payload.read(1024 * 1024):
                        digest.update(chunk)
                    entries[path] = ("file", digest.hexdigest())
                elif member.issym():
                    entries[path] = ("symlink", member.linkname)
                elif member.islnk():
                    entries[path] = ("hardlink", member.linkname)
                else:
                    entries[path] = ("special", member.type.decode("ascii", "replace"))
    except (tarfile.TarError, OSError) as error:
        process.kill()
        process.wait()
        raise ImageBuildError(
            f"cannot inspect payload of package {package['filename']}: {error}"
        ) from error
    except BaseException:
        process.kill()
        process.wait()
        raise
    process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    if process.stderr is not None:
        process.stderr.close()
    if process.wait() != 0:
        raise ImageBuildError(
            f"dpkg-deb payload inspection for {package['filename']} failed: {stderr.strip()}"
        )
    return entries


def resolve_package_payload(
    path: PurePosixPath,
    entries: dict[PurePosixPath, tuple[str, str]],
) -> tuple[str, list[str]] | None:
    current = path
    chain: list[str] = []
    seen: set[PurePosixPath] = set()
    while True:
        if current in seen:
            raise ImageBuildError(f"package payload symlink loop at {current}")
        seen.add(current)
        entry = entries.get(current)
        if entry is None:
            linked_prefix = None
            for length in range(len(current.parts) - 1, 0, -1):
                prefix = PurePosixPath(*current.parts[:length])
                candidate_entry = entries.get(prefix)
                if candidate_entry is not None and candidate_entry[0] in (
                    "symlink",
                    "hardlink",
                ):
                    linked_prefix = prefix
                    entry = candidate_entry
                    break
            if linked_prefix is None:
                return None
            suffix = current.parts[len(linked_prefix.parts) :]
        else:
            linked_prefix = current
            suffix = ()
        kind, value = entry
        if kind == "file":
            return value, chain
        if kind not in ("symlink", "hardlink"):
            raise ImageBuildError(f"package payload path is not a file: {current}")
        chain.append(f"{linked_prefix}->{value}")
        target = PurePosixPath(value)
        if kind == "hardlink":
            candidate = _archive_path(value, f"hardlink at {linked_prefix}")
        elif target.is_absolute():
            candidate = target.relative_to("/")
        else:
            parts: list[str] = []
            for part in (*linked_prefix.parent.parts, *target.parts):
                if part in ("", "."):
                    continue
                if part == "..":
                    if not parts:
                        raise ImageBuildError(
                            f"package payload symlink escapes archive root: {current}->{value}"
                        )
                    parts.pop()
                else:
                    parts.append(part)
            candidate = PurePosixPath(*parts)
        current = PurePosixPath(*candidate.parts, *suffix)


def rocm_payload_attestations(
    required_files: dict[str, Any],
    staged: dict[str, str],
    packages: list[dict[str, str]],
    dpkg_deb: Path,
) -> list[dict[str, Any]]:
    package_entries = [
        (package, package_payload_entries(package, dpkg_deb)) for package in packages
    ]
    attestations: list[dict[str, Any]] = []
    for destination, expected_value in sorted(required_files.items()):
        path = guest_path(destination, "workload required guest path")
        if not path.is_relative_to(PurePosixPath("opt/rocm")):
            continue
        expected = require_sha256(
            expected_value, f"workload required digest for {path}"
        )
        if staged.get(str(path)) != expected:
            raise ImageBuildError(f"workload file is missing or changed: {path}")
        matches: list[dict[str, Any]] = []
        for package, entries in package_entries:
            resolved = resolve_package_payload(path, entries)
            if resolved is None:
                continue
            payload_digest, chain = resolved
            if payload_digest == expected:
                matches.append(
                    {
                        "guest_path": str(path),
                        "link_chain": chain,
                        "package": package_identity(package),
                        "package_path": str(path),
                        "sha256": expected,
                    }
                )
        if not matches:
            raise ImageBuildError(
                f"ROCm workload file cannot be attributed to a pinned package: {path}"
            )
        matches.sort(
            key=lambda item: (item["package"]["name"], item["package"]["sha256"])
        )
        attestations.append(matches[0])
    return attestations


def verify_m2_firmware(records: list[dict[str, Any]]) -> None:
    firmware = {
        record["destination"]: record
        for record in records
        if is_firmware(record["destination"])
    }
    if set(firmware) != M2_FIRMWARE_PATHS:
        missing = sorted(str(path) for path in M2_FIRMWARE_PATHS - set(firmware))
        extra = sorted(str(path) for path in set(firmware) - M2_FIRMWARE_PATHS)
        raise ImageBuildError(
            "M2 firmware set must contain exactly the six compute-parser fixtures"
            + (f"; missing: {', '.join(missing)}" if missing else "")
            + (f"; extra: {', '.join(extra)}" if extra else "")
        )
    with tempfile.TemporaryDirectory(prefix="rocjitsu-m2-firmware-") as temporary:
        generated = Path(temporary)
        synthetic = firmware[M2_FIRMWARE_DIRECTORY / M2_SYNTHETIC_FIRMWARE[0]][
            "fixture"
        ]
        assert synthetic is not None
        script = source_path(
            Path("/"), synthetic.get("generator_source"), "firmware generator source"
        )
        if synthetic.get("generator") != "tools/vfio_guest_firmware.py":
            raise ImageBuildError("M2 synthetic firmware uses an unexpected generator")
        if require_sha256(
            synthetic.get("generator_sha256"), "firmware generator digest"
        ) != sha256_file(script):
            raise ImageBuildError("M2 synthetic firmware generator hash mismatch")
        expected_argv = ["python3", str(script), "--output", "{output}"]
        if synthetic.get("generator_argv") != expected_argv:
            raise ImageBuildError("M2 synthetic firmware generator argv mismatch")
        run_tool(
            [sys.executable, str(script), "--output", str(generated)],
            "M2 synthetic firmware regeneration",
        )
        for name in M2_SYNTHETIC_FIRMWARE:
            record = firmware[M2_FIRMWARE_DIRECTORY / name]
            if record["fixture"] != synthetic:
                raise ImageBuildError("M2 synthetic firmware generator records differ")
            if sha256_file(generated / name) != record["sha256"]:
                raise ImageBuildError(
                    f"M2 synthetic firmware does not reproduce: {name}"
                )

        discovery = firmware[M2_FIRMWARE_DIRECTORY / M2_IP_DISCOVERY_FIRMWARE][
            "fixture"
        ]
        assert discovery is not None
        generator = source_path(
            Path("/"),
            discovery.get("generator_source"),
            "IP-discovery generator source",
        )
        if discovery.get("generator") != "rj-ip-discovery":
            raise ImageBuildError("M2 IP discovery uses an unexpected generator")
        if require_sha256(
            discovery.get("generator_sha256"), "IP-discovery generator digest"
        ) != sha256_file(generator):
            raise ImageBuildError("M2 IP-discovery generator hash mismatch")
        expected_argv = [str(generator), "gfx1250", "{output}"]
        if discovery.get("generator_argv") != expected_argv:
            raise ImageBuildError("M2 IP-discovery generator argv mismatch")
        output = generated / M2_IP_DISCOVERY_FIRMWARE
        run_tool(
            [str(generator), "gfx1250", str(output)], "M2 IP discovery regeneration"
        )
        if (
            sha256_file(output)
            != firmware[M2_FIRMWARE_DIRECTORY / M2_IP_DISCOVERY_FIRMWARE]["sha256"]
        ):
            raise ImageBuildError("M2 IP-discovery firmware does not reproduce")


def parse_mode(value: Any, name: str) -> int:
    if not isinstance(value, str) or len(value) != 4 or value[0] != "0":
        raise ImageBuildError(f"{name} must be an octal string such as 0755")
    try:
        mode = int(value, 8)
    except ValueError as error:
        raise ImageBuildError(f"{name} must be an octal string such as 0755") from error
    if mode & ~0o777:
        raise ImageBuildError(f"{name} contains unsupported mode bits")
    return mode


def checked_input(
    lock_path: Path, record: Any, index: int
) -> tuple[Path, PurePosixPath, int, str, dict[str, Any] | None]:
    item = require_object(record, f"files[{index}]")
    source = source_path(lock_path, item.get("source"), f"files[{index}].source")
    destination = guest_path(item.get("destination"), f"files[{index}].destination")
    mode = parse_mode(item.get("mode"), f"files[{index}].mode")
    expected = require_sha256(item.get("sha256"), f"files[{index}].sha256")
    actual = sha256_file(source)
    if actual != expected:
        raise ImageBuildError(
            f"files[{index}] hash mismatch for {source}: "
            f"expected {expected}, got {actual}"
        )

    fixture: dict[str, Any] | None = None
    if is_firmware(destination):
        kind = item.get("kind")
        if kind != "generated-fixture":
            raise ImageBuildError(
                f"firmware input {destination} must declare kind generated-fixture"
            )
        generator = require_string(item.get("generator"), f"files[{index}].generator")
        fixture = {"generator": generator}
        if "generator_revision" in item:
            fixture["generator_revision"] = require_string(
                item.get("generator_revision"),
                f"files[{index}].generator_revision",
            )
        provenance_fields = {
            "generator_source",
            "generator_sha256",
            "generator_argv",
        }
        present = provenance_fields & set(item)
        if present and present != provenance_fields:
            raise ImageBuildError(
                f"files[{index}] has an incomplete firmware generator provenance record"
            )
        if present:
            generator_source = source_path(
                lock_path,
                item.get("generator_source"),
                f"files[{index}].generator_source",
            )
            fixture.update(
                {
                    "generator_source": str(generator_source),
                    "generator_sha256": require_sha256(
                        item.get("generator_sha256"),
                        f"files[{index}].generator_sha256",
                    ),
                    "generator_argv": require_string_list(
                        item.get("generator_argv"),
                        f"files[{index}].generator_argv",
                    ),
                }
            )
            if fixture["generator_sha256"] != sha256_file(generator_source):
                raise ImageBuildError(
                    f"files[{index}] firmware generator hash mismatch"
                )
    elif any(
        field in item
        for field in (
            "kind",
            "generator",
            "generator_revision",
            "generator_source",
            "generator_sha256",
            "generator_argv",
        )
    ):
        raise ImageBuildError(
            f"files[{index}] uses firmware metadata for non-firmware path {destination}"
        )
    return source, destination, mode, actual, fixture


def validate_lock(
    lock_path: Path, dpkg_deb: Path, modinfo: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImageBuildError(f"cannot read image lock {lock_path}: {error}") from error
    root = require_object(lock, "image lock")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ImageBuildError(f"schema_version must be {SCHEMA_VERSION}")

    guest = require_object(root.get("guest"), "guest")
    require_string(guest.get("distribution"), "guest.distribution")
    require_string(guest.get("kernel_release"), "guest.kernel_release")
    role = require_string(guest.get("role"), "guest.role")
    if role not in ("driver-discovery", "m2-workload"):
        raise ImageBuildError("guest.role must be driver-discovery or m2-workload")
    kernel = require_object(guest.get("kernel"), "guest.kernel")
    kernel_source = source_path(lock_path, kernel.get("source"), "guest.kernel.source")
    kernel_sha256 = require_sha256(kernel.get("sha256"), "guest.kernel.sha256")
    if sha256_file(kernel_source) != kernel_sha256:
        raise ImageBuildError("guest.kernel hash does not match its source artifact")
    if role == "m2-workload":
        policy = require_object(guest.get("policy"), "guest.policy")
        if policy != {"ip_block_mask": M2_IP_BLOCK_MASK}:
            raise ImageBuildError(
                f"M2 guest policy must pin ip_block_mask={M2_IP_BLOCK_MASK}"
            )
    amdgpu = require_object(root.get("amdgpu"), "amdgpu")
    require_string(amdgpu.get("repository"), "amdgpu.repository")
    has_git_source = "branch" in amdgpu or "commit" in amdgpu
    has_package_source = "package" in amdgpu
    if has_git_source == has_package_source:
        raise ImageBuildError("amdgpu must identify exactly one Git or package source")
    if role == "m2-workload" and not has_package_source:
        raise ImageBuildError(
            "m2-workload images require a verified amdgpu-dkms package artifact"
        )
    if has_git_source:
        require_string(amdgpu.get("build_id"), "amdgpu.build_id")
        require_string(amdgpu.get("branch"), "amdgpu.branch")
        commit = require_string(amdgpu.get("commit"), "amdgpu.commit")
        if len(commit) != 40 or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise ImageBuildError(
                "amdgpu.commit must be a lowercase 40-digit Git commit"
            )
    else:
        package = validate_package_record(
            lock_path, amdgpu.get("package"), "amdgpu.package", dpkg_deb
        )
        if package["name"] != "amdgpu-dkms":
            raise ImageBuildError("amdgpu.package must be an amdgpu-dkms artifact")
        require_amdgpu_compute_only_guards(Path(package["source"]), dpkg_deb)
        compatibility = require_object(
            amdgpu.get("compatibility"), "amdgpu.compatibility"
        )
        if compatibility != {
            "no_vcn_discovery_guard": True,
            "zero_instance_partition_guard": True,
        }:
            raise ImageBuildError(
                "amdgpu.compatibility must attest the verified no-VCN discovery "
                "and zero-instance partition guards"
            )
    rocm = require_object(root.get("rocm"), "rocm")
    packages = rocm.get("packages")
    if not isinstance(packages, list):
        raise ImageBuildError("rocm.packages must be a list")
    if role == "m2-workload" and not packages:
        raise ImageBuildError("m2-workload images require pinned ROCm packages")
    checked_packages = [
        validate_package_record(lock_path, package, f"rocm.packages[{index}]", dpkg_deb)
        for index, package in enumerate(packages)
    ]
    if checked_packages != sorted(
        checked_packages, key=lambda package: package["name"]
    ):
        raise ImageBuildError("rocm.packages must be sorted by package name")
    if len({package["name"] for package in checked_packages}) != len(checked_packages):
        raise ImageBuildError("rocm.packages contains duplicate package names")
    build_tools = require_object(root.get("build_tools"), "build_tools")
    if not build_tools:
        raise ImageBuildError("build_tools must not be empty")
    for name, version in build_tools.items():
        require_string(name, "build_tools key")
        require_string(version, f"build_tools.{name}")

    records = root.get("files")
    if not isinstance(records, list) or not records:
        raise ImageBuildError("files must be a nonempty list")
    checked: list[dict[str, Any]] = []
    destinations: set[PurePosixPath] = set()
    for index, record in enumerate(records):
        source, destination, mode, digest, fixture = checked_input(
            lock_path, record, index
        )
        if destination in destinations:
            raise ImageBuildError(f"duplicate guest destination: {destination}")
        destinations.add(destination)
        checked.append(
            {
                "source": source,
                "destination": destination,
                "mode": mode,
                "sha256": digest,
                "fixture": fixture,
            }
        )

    required = {PurePosixPath("init"), PurePosixPath("bin/busybox")}
    missing = required - destinations
    if missing:
        raise ImageBuildError(
            "files is missing required guest path(s): "
            + ", ".join(str(path) for path in sorted(missing, key=str))
        )

    if has_package_source:
        module = require_object(amdgpu.get("module"), "amdgpu.module")
        module_destination = guest_path(
            module.get("destination"), "amdgpu.module.destination"
        )
        module_digest = require_sha256(module.get("sha256"), "amdgpu.module.sha256")
        locked_module = next(
            (
                record
                for record in checked
                if record["destination"] == module_destination
            ),
            None,
        )
        if locked_module is None:
            raise ImageBuildError(
                f"amdgpu module is not staged at {module_destination}"
            )
        if locked_module["sha256"] != module_digest:
            raise ImageBuildError(
                "amdgpu.module.sha256 does not match the staged module"
            )
        expected_identity = {
            field: require_string(module.get(field), f"amdgpu.module.{field}")
            for field in ("version", "srcversion", "vermagic")
        }
        actual_identity = module_identity(locked_module["source"], modinfo)
        if actual_identity != expected_identity:
            raise ImageBuildError(
                "amdgpu.module identity does not match the staged module"
            )
        package_version = require_string(
            amdgpu["package"].get("version"), "amdgpu.package.version"
        )
        if actual_identity["version"] != debian_upstream_version(package_version):
            raise ImageBuildError(
                "amdgpu module version does not match the amdgpu-dkms package"
            )
        if role == "m2-workload":
            validate_build_attestation(
                lock_path,
                amdgpu.get("build_attestation"),
                package,
                guest["kernel_release"],
                kernel_sha256,
                module_digest,
                actual_identity,
            )

    workload_value = root.get("workload")
    if role == "m2-workload":
        workload = require_object(workload_value, "workload")
        inventory = require_object(workload.get("inventory"), "workload.inventory")
        if inventory.get("schema_version") != 2:
            raise ImageBuildError("workload.inventory has an unsupported schema")
        inventory_source = source_path(
            lock_path, workload.get("source"), "workload.source"
        )
        inventory_expected = require_sha256(workload.get("sha256"), "workload.sha256")
        inventory_actual = sha256_file(inventory_source)
        if inventory_actual != inventory_expected:
            raise ImageBuildError(
                "workload inventory hash mismatch: "
                f"expected {inventory_expected}, got {inventory_actual}"
            )
        try:
            source_inventory = json.loads(inventory_source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ImageBuildError(
                "workload inventory source is invalid JSON"
            ) from error
        if source_inventory != inventory:
            raise ImageBuildError(
                "embedded workload inventory differs from its locked source"
            )
        inventory_packages = inventory.get("packages")
        expected_packages = [package_identity(package) for package in checked_packages]
        if inventory_packages != expected_packages:
            raise ImageBuildError(
                "ROCm package artifacts do not match the workload inventory"
            )
        runtime = require_m2_runtime(inventory)
        init_record = next(
            record
            for record in checked
            if record["destination"] == PurePosixPath("init")
        )
        require_m2_init(inventory, init_record["source"])
        required_files = require_object(
            inventory.get("required_guest_files"),
            "workload.inventory.required_guest_files",
        )
        staged = {str(record["destination"]): record["sha256"] for record in checked}
        for destination, expected_digest in required_files.items():
            path = guest_path(destination, "workload required guest path")
            digest = require_sha256(
                expected_digest, f"workload required digest for {path}"
            )
            if staged.get(str(path)) != digest:
                raise ImageBuildError(f"workload file is missing or changed: {path}")
        payloads = rocm_payload_attestations(
            required_files, staged, checked_packages, dpkg_deb
        )
        if root["rocm"].get("payloads") != payloads:
            raise ImageBuildError(
                "locked ROCm payload attestations do not match pinned package contents"
            )
        verify_m2_firmware(checked)
    elif workload_value is not None:
        raise ImageBuildError("workload metadata is valid only for m2-workload")
    return root, checked


def write_newc_header(
    stream: BinaryIO, inode: int, path: PurePosixPath, mode: int, size: int
) -> None:
    name = str(path).encode("utf-8") + b"\0"
    fields = (
        inode,
        mode,
        0,
        0,
        1,
        0,
        size,
        0,
        0,
        0,
        0,
        len(name),
        0,
    )
    header = b"070701" + b"".join(f"{field:08x}".encode("ascii") for field in fields)
    if len(header) != NEWC_HEADER_BYTES:
        raise AssertionError("invalid newc header size")
    stream.write(header)
    stream.write(name)
    stream.write(bytes(-(len(header) + len(name)) % 4))


def write_newc_bytes(
    stream: BinaryIO, inode: int, path: PurePosixPath, mode: int, payload: bytes
) -> None:
    write_newc_header(stream, inode, path, mode, len(payload))
    stream.write(payload)
    stream.write(bytes(-len(payload) % 4))


def archive_entries(root: Path) -> Iterator[tuple[PurePosixPath, int, Path | None]]:
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if path.is_symlink():
            raise ImageBuildError(f"staged image contains a symlink: {relative}")
        if path.is_dir():
            yield relative, stat.S_IFDIR | stat.S_IMODE(path.stat().st_mode), None
        elif path.is_file():
            yield relative, stat.S_IFREG | stat.S_IMODE(path.stat().st_mode), path
        else:
            raise ImageBuildError(
                f"staged image contains a non-regular entry: {relative}"
            )


def write_initramfs(root: Path, output: Path) -> None:
    with output.open("wb") as compressed:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=1, fileobj=compressed, mtime=0
        ) as archive:
            inode = 1
            for path, mode, source in archive_entries(root):
                if source is None:
                    write_newc_bytes(archive, inode, path, mode, b"")
                else:
                    size = source.stat().st_size
                    write_newc_header(archive, inode, path, mode, size)
                    with source.open("rb") as payload:
                        shutil.copyfileobj(payload, archive, 1024 * 1024)
                    archive.write(bytes(-size % 4))
                inode += 1
            write_newc_bytes(
                archive,
                inode,
                PurePosixPath("TRAILER!!!"),
                stat.S_IFREG,
                b"",
            )


def build_image(
    lock_path: Path,
    output: Path,
    provenance_tool: Path,
    dpkg_deb: Path = Path("/usr/bin/dpkg-deb"),
    modinfo: Path = Path("/usr/sbin/modinfo"),
) -> dict[str, Any]:
    lock, records = validate_lock(lock_path, dpkg_deb, modinfo)
    output = output.resolve()
    if output.exists():
        raise ImageBuildError(f"output already exists: {output}")
    output_parent = output.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output_parent
    ) as temporary:
        temporary_path = Path(temporary)
        root = temporary_path / "root"
        root.mkdir()
        for directory, mode in RUNTIME_DIRECTORIES:
            destination = root.joinpath(*directory.parts)
            destination.mkdir(parents=True)
            destination.chmod(mode)
        firmware_lines: list[str] = []
        file_manifest: list[dict[str, Any]] = []
        for record in records:
            destination = root.joinpath(*record["destination"].parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record["source"], destination)
            destination.chmod(record["mode"])
            copied_digest = sha256_file(destination)
            if copied_digest != record["sha256"]:
                raise ImageBuildError(
                    f"input changed while copying {record['source']}: "
                    f"expected {record['sha256']}, got {copied_digest}"
                )
            manifest_record: dict[str, Any] = {
                "destination": str(record["destination"]),
                "mode": f"0{record['mode']:03o}",
                "sha256": record["sha256"],
                "source": str(record["source"]),
            }
            if record["fixture"] is not None:
                manifest_record.update(record["fixture"])
                firmware_lines.append(f"{record['sha256']}  {record['destination']}\n")
            file_manifest.append(manifest_record)

        metadata_directory = root / "etc/rocjitsu"
        metadata_directory.mkdir(parents=True, exist_ok=True)
        normalized_lock = canonical_json(lock)
        (metadata_directory / "image-lock.json").write_bytes(normalized_lock)
        workload_inventory = lock.get("workload", {}).get("inventory")
        if workload_inventory is not None:
            workload_inventory_bytes = canonical_json(workload_inventory)
            (metadata_directory / "workload-inventory.json").write_bytes(
                workload_inventory_bytes
            )
            (temporary_path / "workload-inventory.json").write_bytes(
                workload_inventory_bytes
            )
        allowlist = temporary_path / "firmware-SHA256SUMS"
        allowlist.write_text("".join(sorted(firmware_lines)), encoding="ascii")
        shutil.copyfile(allowlist, metadata_directory / "firmware-SHA256SUMS")

        initramfs = temporary_path / "initramfs.gz"
        write_initramfs(root, initramfs)
        command = [
            sys.executable,
            str(provenance_tool),
            "--root",
            str(root),
            "--initramfs",
            str(initramfs),
            "--allowlist",
            str(allowlist),
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            diagnostics = result.stderr.strip() or result.stdout.strip()
            raise ImageBuildError(f"provenance check failed: {diagnostics}")

        kernel = require_object(lock["guest"].get("kernel"), "guest.kernel")
        kernel_source = source_path(
            lock_path, kernel.get("source"), "guest.kernel.source"
        )
        kernel_expected = require_sha256(kernel.get("sha256"), "guest.kernel.sha256")
        kernel_actual = sha256_file(kernel_source)
        if kernel_actual != kernel_expected:
            raise ImageBuildError(
                "guest.kernel hash mismatch: "
                f"expected {kernel_expected}, got {kernel_actual}"
            )
        shutil.copyfile(kernel_source, temporary_path / "vmlinuz")
        copied_kernel = sha256_file(temporary_path / "vmlinuz")
        if copied_kernel != kernel_actual:
            raise ImageBuildError(
                "guest.kernel changed while copying: "
                f"expected {kernel_actual}, got {copied_kernel}"
            )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "amdgpu": lock["amdgpu"],
            "build_tools": lock["build_tools"],
            "files": file_manifest,
            "firmware_allowlist": [
                line.rstrip("\n") for line in sorted(firmware_lines)
            ],
            "guest": {
                "distribution": lock["guest"]["distribution"],
                "kernel_release": lock["guest"]["kernel_release"],
                "role": lock["guest"]["role"],
                **(
                    {"policy": lock["guest"]["policy"]}
                    if "policy" in lock["guest"]
                    else {}
                ),
            },
            "outputs": {
                "firmware-SHA256SUMS": sha256_file(allowlist),
                "initramfs.gz": sha256_file(initramfs),
                "vmlinuz": kernel_actual,
            },
            "rocm": lock["rocm"],
            "source_lock_sha256": hashlib.sha256(normalized_lock).hexdigest(),
        }
        if workload_inventory is not None:
            manifest["outputs"]["workload-inventory.json"] = sha256_file(
                temporary_path / "workload-inventory.json"
            )
            manifest["workload"] = {
                "guest_init": workload_inventory["guest_init"],
                "inventory_sha256": hashlib.sha256(
                    canonical_json(workload_inventory)
                ).hexdigest(),
                "selected_solution": workload_inventory["selected_solution"],
                "workload": workload_inventory["workload"],
            }
        (temporary_path / "manifest.json").write_bytes(canonical_json(manifest))
        os.rename(temporary_path, output)
        return manifest


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--provenance-tool",
        type=Path,
        default=Path(__file__).with_name("vfio_guest_provenance.py"),
    )
    parser.add_argument("--dpkg-deb", type=Path, default=Path("/usr/bin/dpkg-deb"))
    parser.add_argument("--modinfo", type=Path, default=Path("/usr/sbin/modinfo"))
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_arguments(arguments)
    try:
        manifest = build_image(
            args.lock.resolve(),
            args.output,
            args.provenance_tool.resolve(),
            args.dpkg_deb.absolute(),
            args.modinfo.absolute(),
        )
    except (ImageBuildError, OSError) as error:
        print(f"vfio guest image build failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest["outputs"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
