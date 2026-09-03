#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Snapshot an explicitly prepared vfio-user guest root into an input lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
from typing import Any

from vfio_guest_image import (
    ImageBuildError,
    M2_FIRMWARE_DIRECTORY,
    M2_FIRMWARE_PATHS,
    M2_IP_BLOCK_MASK,
    M2_IP_DISCOVERY_FIRMWARE,
    M2_SYNTHETIC_FIRMWARE,
    canonical_json,
    debian_package_record,
    debian_upstream_version,
    is_firmware,
    module_identity,
    package_identity,
    require_amdgpu_compute_only_guards,
    require_m2_init,
    require_m2_runtime,
    rocm_payload_attestations,
    run_tool,
    sha256_file,
    validate_build_attestation,
)


class LockError(ValueError):
    """A staged input cannot be represented by the hermetic image lock."""


def key_value(value: str, name: str) -> tuple[str, str]:
    key, separator, item = value.partition("=")
    if not separator or not key or not item:
        raise LockError(f"{name} must use NAME=VALUE syntax")
    return key, item


def regular_file(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise LockError(f"{name} is not a regular non-symlink file: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise LockError(f"cannot resolve {name} {path}: {error}") from error
    if not resolved.is_file():
        raise LockError(f"{name} is not a regular non-symlink file: {resolved}")
    return resolved


def tool_file(path: Path, name: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise LockError(f"cannot resolve {name} {path}: {error}") from error
    if not resolved.is_file():
        raise LockError(f"{name} is not a file: {resolved}")
    return path.absolute()


def fixture_value(value: str) -> tuple[PurePosixPath, str, str]:
    destination_text, separator, provenance = value.partition("=")
    generator, revision_separator, revision = provenance.rpartition("@")
    if not separator or not revision_separator or not generator or not revision:
        raise LockError(
            "--fixture must use DESTINATION=GENERATOR@GENERATOR_REVISION syntax"
        )
    destination = PurePosixPath(destination_text)
    if (
        destination.is_absolute()
        or ".." in destination.parts
        or "." in destination.parts
        or not is_firmware(destination)
    ):
        raise LockError(
            f"fixture destination is not an AMD firmware path: {destination}"
        )
    return destination, generator, revision


def snapshot_root(
    root: Path, fixtures: dict[PurePosixPath, dict[str, Any]]
) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise LockError(f"guest root is not a non-symlink directory: {root}")
    records: list[dict[str, Any]] = []
    seen_fixtures: set[PurePosixPath] = set()
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if path.is_symlink():
            raise LockError(f"guest root contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise LockError(f"guest root contains a non-regular entry: {relative}")
        record: dict[str, Any] = {
            "destination": str(relative),
            "mode": f"0{stat.S_IMODE(path.stat().st_mode):03o}",
            "sha256": sha256_file(path),
            "source": str(path.resolve()),
        }
        if is_firmware(relative):
            if relative not in fixtures:
                raise LockError(
                    f"AMD firmware is not a declared generated fixture: {relative}"
                )
            record.update({"kind": "generated-fixture", **fixtures[relative]})
            seen_fixtures.add(relative)
        records.append(record)
    missing = fixtures.keys() - seen_fixtures
    if missing:
        raise LockError(
            "declared fixture is missing from the guest root: "
            + ", ".join(str(path) for path in sorted(missing, key=str))
        )
    return records


def m2_firmware_fixtures(
    root: Path, ip_discovery_generator: Path
) -> dict[PurePosixPath, dict[str, Any]]:
    firmware_generator = regular_file(
        Path(__file__).with_name("vfio_guest_firmware.py"),
        "repository firmware generator",
    )
    ip_discovery_generator = regular_file(
        ip_discovery_generator, "rj-ip-discovery generator"
    )
    synthetic_record = {
        "generator": "tools/vfio_guest_firmware.py",
        "generator_argv": [
            "python3",
            str(firmware_generator),
            "--output",
            "{output}",
        ],
        "generator_sha256": sha256_file(firmware_generator),
        "generator_source": str(firmware_generator),
    }
    discovery_record = {
        "generator": "rj-ip-discovery",
        "generator_argv": [
            str(ip_discovery_generator),
            "gfx1250",
            "{output}",
        ],
        "generator_sha256": sha256_file(ip_discovery_generator),
        "generator_source": str(ip_discovery_generator),
    }
    fixtures = {
        M2_FIRMWARE_DIRECTORY / name: dict(synthetic_record)
        for name in M2_SYNTHETIC_FIRMWARE
    }
    fixtures[M2_FIRMWARE_DIRECTORY / M2_IP_DISCOVERY_FIRMWARE] = discovery_record
    present = {
        PurePosixPath(path.relative_to(root).as_posix())
        for directory in (
            root / "lib/firmware/amdgpu",
            root / "lib/firmware/updates/amdgpu",
        )
        if directory.exists()
        for path in directory.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if present != M2_FIRMWARE_PATHS:
        raise LockError("M2 guest must stage exactly the six compute-parser fixtures")
    with tempfile.TemporaryDirectory(prefix="rocjitsu-m2-firmware-lock-") as temporary:
        generated = Path(temporary)
        run_tool(
            [sys.executable, str(firmware_generator), "--output", str(generated)],
            "M2 synthetic firmware regeneration",
        )
        discovery = generated / M2_IP_DISCOVERY_FIRMWARE
        run_tool(
            [str(ip_discovery_generator), "gfx1250", str(discovery)],
            "M2 IP discovery regeneration",
        )
        for destination in sorted(M2_FIRMWARE_PATHS, key=str):
            staged = root.joinpath(*destination.parts)
            expected = generated / destination.name
            if staged.is_symlink() or not staged.is_file():
                raise LockError(
                    f"M2 firmware fixture is not a regular file: {destination}"
                )
            if sha256_file(staged) != sha256_file(expected):
                raise LockError(
                    f"M2 firmware fixture does not reproduce: {destination}"
                )
    return fixtures


def load_workload_inventory(
    path: Path,
    packages: list[dict[str, str]],
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LockError(f"workload inventory is not a regular non-symlink file: {path}")
    try:
        inventory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LockError(f"cannot read workload inventory {path}: {error}") from error
    if not isinstance(inventory, dict) or inventory.get("schema_version") != 2:
        raise LockError("workload inventory has an unsupported schema")
    inventory_packages = inventory.get("packages")
    expected_packages = [package_identity(package) for package in packages]
    if inventory_packages != expected_packages:
        raise LockError("ROCm package pins do not match the workload inventory")
    require_m2_runtime(inventory)
    init_records = [record for record in files if record["destination"] == "init"]
    if len(init_records) != 1:
        raise LockError("M2 workload requires exactly one guest init")
    require_m2_init(inventory, Path(init_records[0]["source"]))
    required = inventory.get("required_guest_files")
    if not isinstance(required, dict) or not required:
        raise LockError("workload inventory has no required_guest_files")
    staged = {record["destination"]: record["sha256"] for record in files}
    for destination, expected_digest in required.items():
        if not isinstance(destination, str) or not isinstance(expected_digest, str):
            raise LockError("workload required_guest_files must map paths to digests")
        if staged.get(destination) != expected_digest:
            raise LockError(
                f"staged workload file is missing or changed: {destination}"
            )
    return {
        "inventory": inventory,
        "sha256": sha256_file(path),
        "source": str(path.resolve()),
    }


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--kernel", required=True, type=Path)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--kernel-release", required=True)
    parser.add_argument(
        "--role",
        choices=("driver-discovery", "m2-workload"),
        default="driver-discovery",
    )
    parser.add_argument("--amdgpu-repository", required=True)
    parser.add_argument("--amdgpu-branch")
    parser.add_argument("--amdgpu-commit")
    parser.add_argument(
        "--amdgpu-package",
        type=Path,
        metavar="DEB",
        help="pin a packaged driver artifact instead of a Git source revision",
    )
    parser.add_argument("--amdgpu-build-id")
    parser.add_argument(
        "--rocm-package", action="append", default=[], type=Path, metavar="DEB"
    )
    parser.add_argument("--dpkg-deb", type=Path, default=Path("/usr/bin/dpkg-deb"))
    parser.add_argument("--modinfo", type=Path, default=Path("/usr/sbin/modinfo"))
    parser.add_argument(
        "--amdgpu-build-attestation",
        type=Path,
        metavar="JSON",
        help="module build attestation required by the m2-workload role",
    )
    parser.add_argument(
        "--ip-discovery-generator",
        type=Path,
        metavar="RJ_IP_DISCOVERY",
        help="pinned rj-ip-discovery binary required by the m2-workload role",
    )
    parser.add_argument(
        "--build-tool", action="append", default=[], metavar="NAME=VERSION"
    )
    parser.add_argument(
        "--fixture",
        action="append",
        default=[],
        metavar="DESTINATION=GENERATOR@GENERATOR_REVISION",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workload-inventory", type=Path)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_arguments(arguments)
    try:
        if args.root.is_symlink():
            raise LockError(f"guest root is not a non-symlink directory: {args.root}")
        root = args.root.resolve(strict=True)
        output = args.output.resolve()
        if output.is_relative_to(root):
            raise LockError("lock output must be outside the staged guest root")
        if args.kernel.is_symlink():
            raise LockError(f"kernel is not a regular non-symlink file: {args.kernel}")
        kernel = args.kernel.resolve(strict=True)
        if not kernel.is_file():
            raise LockError(f"kernel is not a regular non-symlink file: {kernel}")
        dpkg_deb = tool_file(args.dpkg_deb, "dpkg-deb")
        modinfo = tool_file(args.modinfo, "modinfo")
        packages = [
            debian_package_record(regular_file(path, f"ROCm package {path}"), dpkg_deb)
            for path in args.rocm_package
        ]
        packages.sort(key=lambda package: package["name"])
        if args.role == "m2-workload" and not packages:
            raise LockError("m2-workload requires at least one --rocm-package")
        git_source = args.amdgpu_branch is not None or args.amdgpu_commit is not None
        package_source = args.amdgpu_package is not None
        if git_source == package_source:
            raise LockError(
                "select exactly one amdgpu source: --amdgpu-branch/--amdgpu-commit "
                "or --amdgpu-package"
            )
        if args.role == "m2-workload" and not package_source:
            raise LockError("m2-workload requires a verified --amdgpu-package artifact")
        if args.role == "m2-workload" and args.workload_inventory is None:
            raise LockError("m2-workload requires --workload-inventory")
        if git_source and (args.amdgpu_branch is None or args.amdgpu_commit is None):
            raise LockError("Git amdgpu provenance requires both branch and commit")
        if git_source and args.amdgpu_build_id is None:
            raise LockError("Git amdgpu provenance requires --amdgpu-build-id")
        if package_source and args.amdgpu_build_id is not None:
            raise LockError(
                "packaged amdgpu provenance cannot assert --amdgpu-build-id"
            )
        amdgpu_package = None
        if package_source:
            artifact = regular_file(args.amdgpu_package, "amdgpu package")
            amdgpu_package = debian_package_record(artifact, dpkg_deb)
            if amdgpu_package["name"] != "amdgpu-dkms":
                raise LockError("--amdgpu-package must name an amdgpu-dkms artifact")
            require_amdgpu_compute_only_guards(artifact, dpkg_deb)
        tools = dict(key_value(value, "--build-tool") for value in args.build_tool)
        if not tools:
            raise LockError("at least one --build-tool is required")
        fixtures: dict[PurePosixPath, dict[str, Any]] = {}
        if args.role == "m2-workload":
            if args.fixture:
                raise LockError(
                    "m2-workload firmware fixtures are fixed, not caller-declared"
                )
            if args.ip_discovery_generator is None:
                raise LockError("m2-workload requires --ip-discovery-generator")
            fixtures = m2_firmware_fixtures(root, args.ip_discovery_generator)
        else:
            if args.ip_discovery_generator is not None:
                raise LockError(
                    "--ip-discovery-generator is valid only for role m2-workload"
                )
            for value in args.fixture:
                destination, generator, revision = fixture_value(value)
                if destination in fixtures:
                    raise LockError(f"fixture is declared twice: {destination}")
                fixtures[destination] = {
                    "generator": generator,
                    "generator_revision": revision,
                }

        files = snapshot_root(root, fixtures)
        workload = None
        if args.role == "m2-workload":
            if args.workload_inventory is None:
                raise LockError("m2-workload requires --workload-inventory")
            workload = load_workload_inventory(
                args.workload_inventory.resolve(), packages, files
            )
        elif args.workload_inventory is not None:
            raise LockError("--workload-inventory is valid only for role m2-workload")

        amdgpu = {
            "repository": args.amdgpu_repository,
        }
        if amdgpu_package is not None:
            amdgpu["package"] = amdgpu_package
            module_records = [
                record
                for record in files
                if record["destination"] == "modules/amdgpu.ko"
            ]
            if len(module_records) != 1:
                raise LockError("packaged amdgpu provenance requires modules/amdgpu.ko")
            module_record = module_records[0]
            identity = module_identity(Path(module_record["source"]), modinfo)
            if identity["version"] != debian_upstream_version(
                amdgpu_package["version"]
            ):
                raise LockError(
                    "staged amdgpu module version does not match amdgpu-dkms package"
                )
            amdgpu["module"] = {
                "destination": module_record["destination"],
                "sha256": module_record["sha256"],
                **identity,
            }
            if args.role == "m2-workload":
                if args.amdgpu_build_attestation is None:
                    raise LockError("m2-workload requires --amdgpu-build-attestation")
                attestation_source = regular_file(
                    args.amdgpu_build_attestation, "amdgpu build attestation"
                )
                try:
                    attestation_record = json.loads(
                        attestation_source.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as error:
                    raise LockError(
                        f"cannot read amdgpu build attestation: {error}"
                    ) from error
                wrapper = {
                    "record": attestation_record,
                    "sha256": sha256_file(attestation_source),
                    "source": str(attestation_source),
                }
                amdgpu["build_attestation"] = validate_build_attestation(
                    Path("/"),
                    wrapper,
                    amdgpu_package,
                    args.kernel_release,
                    sha256_file(kernel),
                    module_record["sha256"],
                    identity,
                )
            amdgpu["compatibility"] = {
                "no_vcn_discovery_guard": True,
                "zero_instance_partition_guard": True,
            }
        else:
            amdgpu["build_id"] = args.amdgpu_build_id
            amdgpu["branch"] = args.amdgpu_branch
            amdgpu["commit"] = args.amdgpu_commit

        lock = {
            "amdgpu": amdgpu,
            "build_tools": tools,
            "files": files,
            "guest": {
                "distribution": args.distribution,
                "kernel": {
                    "sha256": sha256_file(kernel),
                    "source": str(kernel),
                },
                "kernel_release": args.kernel_release,
                "role": args.role,
                **(
                    {"policy": {"ip_block_mask": M2_IP_BLOCK_MASK}}
                    if args.role == "m2-workload"
                    else {}
                ),
            },
            "rocm": {"packages": packages},
            "schema_version": 2,
        }
        if workload is not None:
            lock["workload"] = workload
            required_files = workload["inventory"]["required_guest_files"]
            staged = {record["destination"]: record["sha256"] for record in files}
            lock["rocm"]["payloads"] = rocm_payload_attestations(
                required_files, staged, packages, dpkg_deb
            )
        args.output.write_bytes(canonical_json(lock))
    except (ImageBuildError, LockError, OSError) as error:
        print(f"vfio guest lock failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
