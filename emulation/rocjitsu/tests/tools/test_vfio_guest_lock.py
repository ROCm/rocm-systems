#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the hermetic vfio-user guest lock generator."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
SCRIPT = TOOLS / "vfio_guest_lock.py"
SPEC = importlib.util.spec_from_file_location("vfio_guest_lock", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
lock_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lock_tool)


def write_file(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)


def write_deb(
    path: Path,
    name: str,
    version: str,
    architecture: str = "amd64",
    files: dict[str, bytes] | None = None,
    symlinks: dict[str, str] | None = None,
) -> None:
    root = path.parent / f"package-{path.stem}"
    control = root / "DEBIAN/control"
    control.parent.mkdir(parents=True)
    control.write_text(
        f"Package: {name}\nVersion: {version}\nArchitecture: {architecture}\n"
        "Maintainer: rocjitsu test\nDescription: test package\n",
        encoding="utf-8",
    )
    for relative, payload in (files or {}).items():
        write_file(root / relative, payload)
    for relative, target in (symlinks or {}).items():
        link = root / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
    subprocess.run(
        ["dpkg-deb", "--build", "--root-owner-group", str(root), str(path)],
        check=True,
        capture_output=True,
    )


class GuestLockTest(unittest.TestCase):
    def common_arguments(self, work: Path) -> list[str]:
        return [
            "--root",
            str(work / "root"),
            "--kernel",
            str(work / "vmlinuz"),
            "--distribution",
            "test-linux 1",
            "--kernel-release",
            "6.1.2-test",
            "--amdgpu-repository",
            "https://github.com/ROCm/amdgpu.git",
            "--amdgpu-branch",
            "roc-7.1.x",
            "--amdgpu-commit",
            "01cee31f91e55dc913933217d10341360f4a6f9e",
            "--amdgpu-build-id",
            "test-build",
            "--rocm-package",
            str(work / "rocblas.deb"),
            "--modinfo",
            str(work / "modinfo"),
            "--build-tool",
            "python=3-test",
            "--output",
            str(work / "lock.json"),
        ]

    def make_base(self, work: Path) -> None:
        write_file(
            work / "root/init",
            b"#!/bin/busybox sh\n"
            b"  export ROCBLAS_TENSILE_LIBPATH=\"/opt/rocm/core-7.14/lib/rocblas/library\"\n"
            b"  export ROCBLAS_USE_HIPBLASLT=\"0\"\n",
            0o755,
        )
        write_file(work / "root/bin/busybox", b"busybox", 0o755)
        write_file(work / "root/modules/amdgpu.ko", b"module")
        write_file(work / "vmlinuz", b"kernel")
        write_deb(work / "rocblas.deb", "rocblas", "7.14-test")
        write_file(
            work / "modinfo",
            b"#!/bin/sh\ncase \"$2\" in\n"
            b"version) echo 7.1.3.31500000 ;;\n"
            b"srcversion) echo TESTSRC ;;\n"
            b"vermagic) echo '6.1.2-test SMP' ;;\n"
            b"esac\n",
            0o755,
        )

    def use_packaged_amdgpu(self, work: Path, arguments: list[str]) -> Path:
        branch_index = arguments.index("--amdgpu-branch")
        del arguments[branch_index : branch_index + 4]
        build_id_index = arguments.index("--amdgpu-build-id")
        del arguments[build_id_index : build_id_index + 2]
        package = work / "amdgpu-dkms.deb"
        write_deb(
            package,
            "amdgpu-dkms",
            "1:7.1.3.31500000-2390945.24.04",
            "all",
            {
                "usr/src/amdgpu-test/amd/amdgpu/soc_v1_0.c": b"if (!max_res[i])\n    continue;\n",
                "usr/src/amdgpu-test/amd/amdgpu/aqua_vanjaram.c": b"if (!max_res[i])\n    continue;\n",
                "usr/src/amdgpu-test/amd/amdgpu/amdgpu_discovery.c": b"uint32_t vcn_version = "
                b"amdgpu_ip_version(adev, UVD_HWIP, 0);\n"
                b"if (!vcn_version)\n    return 0;\n",
            },
        )
        arguments[0:0] = ["--amdgpu-package", str(package)]
        return package

    def prepare_m2_inputs(
        self, work: Path, arguments: list[str], package: Path
    ) -> None:
        firmware = work / "m2-firmware"
        subprocess.run(
            [
                sys.executable,
                str(TOOLS / "vfio_guest_firmware.py"),
                "--output",
                str(firmware),
            ],
            check=True,
        )
        destination = work / "root/lib/firmware/amdgpu"
        destination.mkdir(parents=True)
        for path in firmware.iterdir():
            write_file(destination / path.name, path.read_bytes())
        ip_generator = work / "rj-ip-discovery"
        write_file(
            ip_generator,
            b"#!/bin/sh\n[ \"$1\" = gfx1250 ] || exit 2\nprintf discovery > \"$2\"\n",
            0o755,
        )
        subprocess.run(
            [str(ip_generator), "gfx1250", str(destination / "ip_discovery.bin")],
            check=True,
        )
        arguments[0:0] = ["--ip-discovery-generator", str(ip_generator)]

        module_sha256 = lock_tool.sha256_file(work / "root/modules/amdgpu.ko")
        build_input = work / "kernel-config"
        toolchain = work / "cc"
        write_file(build_input, b"config")
        write_file(toolchain, b"compiler", 0o755)
        attestation = {
            "schema_version": 1,
            "amdgpu_package_sha256": lock_tool.sha256_file(package),
            "kernel": {
                "release": "6.1.2-test",
                "sha256": lock_tool.sha256_file(work / "vmlinuz"),
            },
            "module": {
                "sha256": module_sha256,
                "srcversion": "TESTSRC",
                "vermagic": "6.1.2-test SMP",
                "version": "7.1.3.31500000",
            },
            "build": {
                "command": ["make", "modules"],
                "working_directory": "/build/amdgpu",
                "inputs": [
                    {
                        "path": str(build_input),
                        "sha256": lock_tool.sha256_file(build_input),
                    }
                ],
                "toolchain": [
                    {
                        "path": str(toolchain),
                        "sha256": lock_tool.sha256_file(toolchain),
                        "version": "test",
                    }
                ],
            },
        }
        attestation_path = work / "amdgpu-build-attestation.json"
        attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
        arguments[0:0] = [
            "--amdgpu-build-attestation",
            str(attestation_path),
        ]

    def test_snapshots_only_explicit_regular_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            self.make_base(work)
            status = lock_tool.main(self.common_arguments(work))

            self.assertEqual(status, 0)
            lock = json.loads((work / "lock.json").read_text(encoding="utf-8"))
            destinations = {record["destination"] for record in lock["files"]}
            self.assertEqual(destinations, {"bin/busybox", "init", "modules/amdgpu.ko"})

    def test_records_packaged_amdgpu_without_a_git_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            self.make_base(work)
            arguments = self.common_arguments(work)
            package = self.use_packaged_amdgpu(work, arguments)

            status = lock_tool.main(arguments)

            self.assertEqual(status, 0)
            lock = json.loads((work / "lock.json").read_text(encoding="utf-8"))
            self.assertNotIn("branch", lock["amdgpu"])
            self.assertNotIn("commit", lock["amdgpu"])
            self.assertEqual(
                lock["amdgpu"]["package"],
                {
                    "architecture": "all",
                    "filename": "amdgpu-dkms.deb",
                    "name": "amdgpu-dkms",
                    "sha256": lock_tool.sha256_file(package),
                    "source": str(package.resolve()),
                    "version": "1:7.1.3.31500000-2390945.24.04",
                },
            )
            self.assertEqual(
                lock["amdgpu"]["module"],
                {
                    "destination": "modules/amdgpu.ko",
                    "sha256": lock_tool.sha256_file(work / "root/modules/amdgpu.ko"),
                    "srcversion": "TESTSRC",
                    "vermagic": "6.1.2-test SMP",
                    "version": "7.1.3.31500000",
                },
            )
            self.assertEqual(
                lock["amdgpu"]["compatibility"],
                {
                    "no_vcn_discovery_guard": True,
                    "zero_instance_partition_guard": True,
                },
            )

    def test_rejects_ambiguous_amdgpu_git_and_package_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            self.make_base(work)
            arguments = self.common_arguments(work)
            arguments[0:0] = [
                "--amdgpu-package",
                str(work / "amdgpu-dkms.deb"),
            ]
            diagnostics = io.StringIO()

            with contextlib.redirect_stderr(diagnostics):
                status = lock_tool.main(arguments)

            self.assertEqual(status, 1)
            self.assertIn("select exactly one amdgpu source", diagnostics.getvalue())

    def test_rejects_asserted_package_metadata_instead_of_an_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            self.make_base(work)
            arguments = self.common_arguments(work)
            branch_index = arguments.index("--amdgpu-branch")
            del arguments[branch_index : branch_index + 4]
            build_id_index = arguments.index("--amdgpu-build-id")
            del arguments[build_id_index : build_id_index + 2]
            arguments[0:0] = ["--amdgpu-package", "amdgpu-dkms=nightly"]
            diagnostics = io.StringIO()

            with contextlib.redirect_stderr(diagnostics):
                status = lock_tool.main(arguments)

            self.assertEqual(status, 1)
            self.assertIn("cannot resolve amdgpu package", diagnostics.getvalue())

    def test_rejects_amdgpu_package_without_compute_only_driver_fix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            self.make_base(work)
            package = work / "amdgpu-dkms.deb"
            write_deb(
                package,
                "amdgpu-dkms",
                "1:7.1-test",
                "all",
                {"usr/src/amdgpu-test/amd/amdgpu/soc_v1_0.c": b"divide();\n"},
            )
            arguments = self.common_arguments(work)
            branch_index = arguments.index("--amdgpu-branch")
            del arguments[branch_index : branch_index + 4]
            build_id_index = arguments.index("--amdgpu-build-id")
            del arguments[build_id_index : build_id_index + 2]
            arguments[0:0] = ["--amdgpu-package", str(package)]
            diagnostics = io.StringIO()

            with contextlib.redirect_stderr(diagnostics):
                status = lock_tool.main(arguments)

            self.assertEqual(status, 1)
            self.assertIn("zero-instance partition guard", diagnostics.getvalue())

    def test_rejects_amdgpu_package_without_no_vcn_discovery_fix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            self.make_base(work)
            package = work / "amdgpu-dkms.deb"
            write_deb(
                package,
                "amdgpu-dkms",
                "1:7.1-test",
                "all",
                {
                    "usr/src/amdgpu-test/amd/amdgpu/soc_v1_0.c": b"if (!max_res[i])\n    continue;\n",
                    "usr/src/amdgpu-test/amd/amdgpu/aqua_vanjaram.c": b"if (!max_res[i])\n    continue;\n",
                    "usr/src/amdgpu-test/amd/amdgpu/amdgpu_discovery.c": b"switch (vcn_version) {}\n",
                },
            )
            arguments = self.common_arguments(work)
            branch_index = arguments.index("--amdgpu-branch")
            del arguments[branch_index : branch_index + 4]
            build_id_index = arguments.index("--amdgpu-build-id")
            del arguments[build_id_index : build_id_index + 2]
            arguments[0:0] = ["--amdgpu-package", str(package)]
            diagnostics = io.StringIO()

            with contextlib.redirect_stderr(diagnostics):
                status = lock_tool.main(arguments)

            self.assertEqual(status, 1)
            self.assertIn("no-VCN discovery guard", diagnostics.getvalue())

    def test_rejects_recursive_host_firmware_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            self.make_base(work)
            write_file(work / "root/lib/firmware/amdgpu/vendor.bin", b"vendor firmware")
            diagnostics = io.StringIO()

            with contextlib.redirect_stderr(diagnostics):
                status = lock_tool.main(self.common_arguments(work))

        self.assertEqual(status, 1)
        self.assertIn("not a declared generated fixture", diagnostics.getvalue())

    def test_records_declared_generated_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            self.make_base(work)
            write_file(work / "root/lib/firmware/amdgpu/ip_discovery.bin", b"generated")
            arguments = self.common_arguments(work)
            arguments[0:0] = [
                "--fixture",
                "lib/firmware/amdgpu/ip_discovery.bin="
                "tools/rj-ip-discovery@0123456789abcdef",
            ]

            status = lock_tool.main(arguments)

            self.assertEqual(status, 0)
            lock = json.loads((work / "lock.json").read_text(encoding="utf-8"))
            firmware = next(
                record
                for record in lock["files"]
                if record["destination"].endswith("ip_discovery.bin")
            )
            self.assertEqual(firmware["kind"], "generated-fixture")
            self.assertEqual(firmware["generator"], "tools/rj-ip-discovery")

    def test_m2_workload_requires_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            self.make_base(work)
            arguments = self.common_arguments(work)
            arguments[0:0] = ["--role", "m2-workload"]
            self.use_packaged_amdgpu(work, arguments)
            diagnostics = io.StringIO()

            with contextlib.redirect_stderr(diagnostics):
                status = lock_tool.main(arguments)

        self.assertEqual(status, 1)
        self.assertIn("requires --workload-inventory", diagnostics.getvalue())

    def test_m2_workload_rejects_unverified_git_driver_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            self.make_base(work)
            arguments = self.common_arguments(work)
            arguments[0:0] = ["--role", "m2-workload"]
            diagnostics = io.StringIO()

            with contextlib.redirect_stderr(diagnostics):
                status = lock_tool.main(arguments)

            self.assertEqual(status, 1)
            self.assertIn("verified --amdgpu-package", diagnostics.getvalue())

    def test_m2_firmware_rejects_any_extra_media_or_vendor_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            self.make_base(work)
            arguments = self.common_arguments(work)
            package = self.use_packaged_amdgpu(work, arguments)
            self.prepare_m2_inputs(work, arguments, package)
            write_file(
                work / "root/lib/firmware/amdgpu/vcn_5_0_0.bin",
                b"forbidden media firmware",
            )

            with self.assertRaisesRegex(
                lock_tool.LockError,
                "exactly the six compute-parser fixtures",
            ):
                lock_tool.m2_firmware_fixtures(work / "root", work / "rj-ip-discovery")

    def test_m2_workload_locks_verified_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            self.make_base(work)
            workload = b"workload"
            write_file(work / "root/opt/rocjitsu/bin/m2-rocblas-sgemm", workload, 0o755)
            rocblas_payload = b"package-owned rocBLAS"
            rocblas_package = work / "rocblas-payload.deb"
            write_deb(
                rocblas_package,
                "rocblas",
                "7.14-test",
                files={"opt/rocm/lib/librocblas.so.1": rocblas_payload},
                symlinks={"opt/rocm/lib/librocblas.so": "librocblas.so.1"},
            )
            write_file(work / "root/opt/rocm/lib/librocblas.so", rocblas_payload)
            inventory = {
                "schema_version": 2,
                "packages": [
                    lock_tool.package_identity(
                        lock_tool.debian_package_record(
                            rocblas_package, Path("/usr/bin/dpkg-deb")
                        )
                    )
                ],
                "required_guest_files": {
                    "opt/rocjitsu/bin/m2-rocblas-sgemm": lock_tool.sha256_file(
                        work / "root/opt/rocjitsu/bin/m2-rocblas-sgemm"
                    ),
                    "opt/rocm/lib/librocblas.so": lock_tool.sha256_file(
                        work / "root/opt/rocm/lib/librocblas.so"
                    ),
                },
                "guest_init": {
                    "guest_path": "init",
                    "sha256": lock_tool.sha256_file(work / "root/init"),
                },
                "runtime": {
                    "backend": "Tensile",
                    "environment": {
                        "ROCBLAS_TENSILE_LIBPATH": "/opt/rocm/core-7.14/lib/rocblas/library",
                        "ROCBLAS_USE_HIPBLASLT": "0",
                    },
                },
            }
            inventory_path = work / "workload-inventory.json"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            arguments = self.common_arguments(work)
            arguments[arguments.index("--rocm-package") + 1] = str(rocblas_package)
            arguments[0:0] = [
                "--role",
                "m2-workload",
                "--workload-inventory",
                str(inventory_path),
            ]
            package = self.use_packaged_amdgpu(work, arguments)
            self.prepare_m2_inputs(work, arguments, package)

            status = lock_tool.main(arguments)

            self.assertEqual(status, 0)
            lock = json.loads((work / "lock.json").read_text(encoding="utf-8"))
            self.assertEqual(lock["workload"]["inventory"], inventory)
            self.assertEqual(lock["guest"]["policy"], {"ip_block_mask": "0x3f"})
            self.assertEqual(
                lock["rocm"]["payloads"][0]["link_chain"],
                ["opt/rocm/lib/librocblas.so->librocblas.so.1"],
            )
            self.assertEqual(
                lock["workload"]["sha256"], lock_tool.sha256_file(inventory_path)
            )


if __name__ == "__main__":
    unittest.main()
