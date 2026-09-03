#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the hermetic vfio-user guest image builder."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest

TOOLS = Path(__file__).parents[2] / "tools"
SCRIPT = TOOLS / "vfio_guest_image.py"
SPEC = importlib.util.spec_from_file_location("vfio_guest_image", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
image_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(image_builder)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_input(path: Path, payload: bytes, mode: int = 0o644) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return {"source": str(path), "sha256": digest(payload)}


def write_deb(
    path: Path,
    name: str,
    version: str,
    architecture: str = "amd64",
    files: dict[str, bytes] | None = None,
) -> dict[str, str]:
    root = path.parent / f"package-{path.stem}"
    control = root / "DEBIAN/control"
    control.parent.mkdir(parents=True)
    control.write_text(
        f"Package: {name}\nVersion: {version}\nArchitecture: {architecture}\n"
        "Maintainer: rocjitsu test\nDescription: test package\n",
        encoding="utf-8",
    )
    for relative, payload in (files or {}).items():
        write_input(root / relative, payload)
    subprocess.run(
        ["dpkg-deb", "--build", "--root-owner-group", str(root), str(path)],
        check=True,
        capture_output=True,
    )
    return image_builder.debian_package_record(path, Path("/usr/bin/dpkg-deb"))


def write_modinfo(path: Path) -> None:
    write_input(
        path,
        b"#!/bin/sh\ncase \"$2\" in\n"
        b"version) echo 7.1.3.31500000 ;;\n"
        b"srcversion) echo TESTSRC ;;\n"
        b"vermagic) echo '6.1.2-test SMP' ;;\n"
        b"esac\n",
        0o755,
    )


def base_lock(work: Path) -> dict[str, object]:
    init = write_input(
        work / "inputs/init",
        b"#!/bin/busybox sh\n"
        b"  export ROCBLAS_TENSILE_LIBPATH=\"/opt/rocm/core-7.14/lib/rocblas/library\"\n"
        b"  export ROCBLAS_USE_HIPBLASLT=\"0\"\n"
        b"poweroff -f\n",
        0o755,
    )
    busybox = write_input(work / "inputs/busybox", b"fake static busybox", 0o755)
    kernel = write_input(work / "inputs/vmlinuz", b"fake pinned kernel")
    return {
        "schema_version": 2,
        "guest": {
            "distribution": "test-linux 1",
            "kernel_release": "6.1.2-test",
            "role": "driver-discovery",
            "kernel": kernel,
        },
        "amdgpu": {
            "repository": "https://github.com/ROCm/amdgpu.git",
            "branch": "roc-7.1.x",
            "build_id": "test-build",
            "commit": "01cee31f91e55dc913933217d10341360f4a6f9e",
        },
        "rocm": {
            "packages": [
                write_deb(work / "rocblas.deb", "rocblas", "4.0.0-test"),
                write_deb(work / "rocm-core.deb", "rocm-core", "7.1.0-test"),
            ]
        },
        "build_tools": {"busybox": "1.36-test", "python": "3-test"},
        "files": [
            {
                **init,
                "destination": "init",
                "mode": "0755",
            },
            {
                **busybox,
                "destination": "bin/busybox",
                "mode": "0755",
            },
        ],
    }


def packaged_lock(work: Path) -> tuple[dict[str, object], Path]:
    lock = base_lock(work)
    package = write_deb(
        work / "amdgpu-dkms.deb",
        "amdgpu-dkms",
        "1:7.1.3.31500000-2390945.24.04",
        "all",
        {
            "usr/src/amdgpu-test/amd/amdgpu/soc_v1_0.c": b"if (!max_res[i])\n    continue;\n",
            "usr/src/amdgpu-test/amd/amdgpu/aqua_vanjaram.c": b"if (!max_res[i])\n    continue;\n",
            "usr/src/amdgpu-test/amd/amdgpu/amdgpu_discovery.c": b"uint32_t vcn_version = amdgpu_ip_version(adev, UVD_HWIP, 0);\n"
            b"if (!vcn_version)\n    return 0;\n",
        },
    )
    module = write_input(work / "inputs/amdgpu.ko", b"module")
    files = lock["files"]
    assert isinstance(files, list)
    files.append(
        {
            **module,
            "destination": "modules/amdgpu.ko",
            "mode": "0644",
        }
    )
    modinfo = work / "modinfo"
    write_modinfo(modinfo)
    lock["amdgpu"] = {
        "repository": "https://repo.radeon.com/amdgpu/nightly/ubuntu",
        "compatibility": {
            "no_vcn_discovery_guard": True,
            "zero_instance_partition_guard": True,
        },
        "module": {
            "destination": "modules/amdgpu.ko",
            "sha256": module["sha256"],
            "srcversion": "TESTSRC",
            "vermagic": "6.1.2-test SMP",
            "version": "7.1.3.31500000",
        },
        "package": package,
    }
    return lock, modinfo


def write_lock(path: Path, lock: dict[str, object]) -> None:
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def prepare_m2_lock(work: Path, lock: dict[str, object]) -> None:
    guest = lock["guest"]
    assert isinstance(guest, dict)
    guest["role"] = "m2-workload"
    guest["policy"] = {"ip_block_mask": "0x3f"}
    files = lock["files"]
    assert isinstance(files, list)
    firmware_directory = work / "generated-firmware"
    subprocess.run(
        [
            "python3",
            str(TOOLS / "vfio_guest_firmware.py"),
            "--output",
            str(firmware_directory),
        ],
        check=True,
    )
    ip_generator = work / "rj-ip-discovery"
    write_input(
        ip_generator,
        b"#!/bin/sh\n[ \"$1\" = gfx1250 ] || exit 2\nprintf discovery > \"$2\"\n",
        0o755,
    )
    subprocess.run(
        [str(ip_generator), "gfx1250", str(firmware_directory / "ip_discovery.bin")],
        check=True,
    )
    synthetic = {
        "generator": "tools/vfio_guest_firmware.py",
        "generator_argv": [
            "python3",
            str(TOOLS / "vfio_guest_firmware.py"),
            "--output",
            "{output}",
        ],
        "generator_sha256": image_builder.sha256_file(TOOLS / "vfio_guest_firmware.py"),
        "generator_source": str(TOOLS / "vfio_guest_firmware.py"),
        "kind": "generated-fixture",
    }
    discovery = {
        "generator": "rj-ip-discovery",
        "generator_argv": [str(ip_generator.resolve()), "gfx1250", "{output}"],
        "generator_sha256": image_builder.sha256_file(ip_generator),
        "generator_source": str(ip_generator.resolve()),
        "kind": "generated-fixture",
    }
    for name in (*image_builder.M2_SYNTHETIC_FIRMWARE, "ip_discovery.bin"):
        source = firmware_directory / name
        files.append(
            {
                "destination": f"lib/firmware/amdgpu/{name}",
                "mode": "0644",
                "sha256": image_builder.sha256_file(source),
                "source": str(source),
                **(discovery if name == "ip_discovery.bin" else synthetic),
            }
        )
    amdgpu = lock["amdgpu"]
    assert isinstance(amdgpu, dict)
    module = amdgpu["module"]
    package = amdgpu["package"]
    kernel = guest["kernel"]
    assert (
        isinstance(module, dict)
        and isinstance(package, dict)
        and isinstance(kernel, dict)
    )
    build_input = work / "kernel-config"
    toolchain = work / "cc"
    write_input(build_input, b"config")
    write_input(toolchain, b"compiler", 0o755)
    attestation = {
        "schema_version": 1,
        "amdgpu_package_sha256": package["sha256"],
        "kernel": {
            "release": guest["kernel_release"],
            "sha256": kernel["sha256"],
        },
        "module": {
            key: module[key] for key in ("sha256", "version", "srcversion", "vermagic")
        },
        "build": {
            "command": ["make", "modules"],
            "working_directory": "/build/amdgpu",
            "inputs": [
                {
                    "path": str(build_input),
                    "sha256": image_builder.sha256_file(build_input),
                }
            ],
            "toolchain": [
                {
                    "path": str(toolchain),
                    "sha256": image_builder.sha256_file(toolchain),
                    "version": "test",
                }
            ],
        },
    }
    attestation_path = work / "amdgpu-build-attestation.json"
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    amdgpu["build_attestation"] = {
        "record": attestation,
        "sha256": image_builder.sha256_file(attestation_path),
        "source": str(attestation_path),
    }


class GuestImageTest(unittest.TestCase):
    def run_builder(self, arguments: list[str]) -> tuple[int, str]:
        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            status = image_builder.main(arguments)
        return status, diagnostics.getvalue()

    def test_rejects_m2_runtime_that_enables_hipblaslt(self) -> None:
        with self.assertRaisesRegex(
            image_builder.ImageBuildError, "must disable the hipBLASLt backend"
        ):
            image_builder.require_m2_runtime(
                {
                    "runtime": {
                        "backend": "Tensile",
                        "environment": {
                            "ROCBLAS_TENSILE_LIBPATH": "/opt/rocm/lib/rocblas/library",
                            "ROCBLAS_USE_HIPBLASLT": "1",
                        },
                    }
                }
            )

    def test_rejects_m2_init_that_differs_from_the_contracted_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            init = Path(temporary) / "init"
            init.write_text("#!/bin/busybox sh\n", encoding="utf-8")
            inventory = {
                "guest_init": {
                    "guest_path": "init",
                    "sha256": digest(b"different"),
                }
            }

            with self.assertRaisesRegex(
                image_builder.ImageBuildError,
                "M2 guest init hash mismatch",
            ):
                image_builder.require_m2_init(inventory, init)

    def test_accepts_packaged_amdgpu_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock, modinfo = packaged_lock(work)
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)
            image = work / "image"

            image_builder.build_image(
                lock_path,
                image,
                TOOLS / "vfio_guest_provenance.py",
                modinfo=modinfo,
            )

            manifest = json.loads((image / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["amdgpu"], lock["amdgpu"])

    def test_rejects_package_missing_a_zero_instance_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock, modinfo = packaged_lock(work)
            package = write_deb(
                work / "amdgpu-dkms-missing-companion.deb",
                "amdgpu-dkms",
                "1:7.1.3.31500000-2390945.24.04",
                "all",
                {
                    "usr/src/amdgpu-test/amd/amdgpu/soc_v1_0.c": b"if (!max_res[i])\n    continue;\n",
                    "usr/src/amdgpu-test/amd/amdgpu/aqua_vanjaram.c": b"res_lt_xcp = max_res[i] < num_xcp;\n",
                    "usr/src/amdgpu-test/amd/amdgpu/amdgpu_discovery.c": b"uint32_t vcn_version = "
                    b"amdgpu_ip_version(adev, UVD_HWIP, 0);\n"
                    b"if (!vcn_version)\n    return 0;\n",
                },
            )
            lock["amdgpu"]["package"] = package
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)

            status, diagnostics = self.run_builder(
                [
                    "--lock",
                    str(lock_path),
                    "--output",
                    str(work / "image"),
                    "--modinfo",
                    str(modinfo),
                ]
            )

            self.assertEqual(status, 1)
            self.assertIn("aqua_vanjaram.c", diagnostics)

    def test_rejects_package_missing_no_vcn_discovery_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock, modinfo = packaged_lock(work)
            package = write_deb(
                work / "amdgpu-dkms-missing-no-vcn-guard.deb",
                "amdgpu-dkms",
                "1:7.1.3.31500000-2390945.24.04",
                "all",
                {
                    "usr/src/amdgpu-test/amd/amdgpu/soc_v1_0.c": b"if (!max_res[i])\n    continue;\n",
                    "usr/src/amdgpu-test/amd/amdgpu/aqua_vanjaram.c": b"if (!max_res[i])\n    continue;\n",
                    "usr/src/amdgpu-test/amd/amdgpu/amdgpu_discovery.c": b"uint32_t vcn_version = "
                    b"amdgpu_ip_version(adev, UVD_HWIP, 0);\n"
                    b"switch (vcn_version) {}\n",
                },
            )
            lock["amdgpu"]["package"] = package
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)

            status, diagnostics = self.run_builder(
                [
                    "--lock",
                    str(lock_path),
                    "--output",
                    str(work / "image"),
                    "--modinfo",
                    str(modinfo),
                ]
            )

            self.assertEqual(status, 1)
            self.assertIn("no-VCN discovery guard", diagnostics)

    def test_rejects_staged_amdgpu_module_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock, modinfo = packaged_lock(work)
            lock["amdgpu"]["module"]["srcversion"] = "WRONG"
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)
            diagnostics = io.StringIO()

            with contextlib.redirect_stderr(diagnostics):
                status = image_builder.main(
                    [
                        "--lock",
                        str(lock_path),
                        "--output",
                        str(work / "image"),
                        "--modinfo",
                        str(modinfo),
                    ]
                )

            self.assertEqual(status, 1)
            self.assertIn("identity does not match", diagnostics.getvalue())

    def test_rejects_changed_staged_amdgpu_module_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock, modinfo = packaged_lock(work)
            module = next(
                item
                for item in lock["files"]
                if item["destination"] == "modules/amdgpu.ko"
            )
            Path(module["source"]).write_bytes(b"different module")
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)

            status, diagnostics = self.run_builder(
                [
                    "--lock",
                    str(lock_path),
                    "--output",
                    str(work / "image"),
                    "--modinfo",
                    str(modinfo),
                ]
            )

            self.assertEqual(status, 1)
            self.assertIn("hash mismatch", diagnostics)

    def test_rejects_changed_package_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock, modinfo = packaged_lock(work)
            package = Path(str(lock["rocm"]["packages"][0]["source"]))
            package.write_bytes(package.read_bytes() + b"changed")
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)

            status, diagnostics = self.run_builder(
                ["--lock", str(lock_path), "--output", str(work / "image")]
            )

            self.assertEqual(status, 1)
            self.assertIn("metadata or digest does not match", diagnostics)

    def test_rejects_asserted_rocm_package_without_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock = base_lock(work)
            lock["rocm"] = {"packages": [{"name": "rocblas", "version": "nightly"}]}
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)

            status, diagnostics = self.run_builder(
                ["--lock", str(lock_path), "--output", str(work / "image")]
            )

            self.assertEqual(status, 1)
            self.assertIn("verified artifact identity and source fields", diagnostics)

    def test_rejects_rocm_package_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock = base_lock(work)
            lock["rocm"]["packages"][0]["version"] = "forged"
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)

            status, diagnostics = self.run_builder(
                ["--lock", str(lock_path), "--output", str(work / "image")]
            )

            self.assertEqual(status, 1)
            self.assertIn("metadata or digest does not match", diagnostics)

    def test_rejects_a_required_rocm_library_not_owned_by_a_pinned_package(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            package = write_deb(
                work / "rocblas.deb",
                "rocblas",
                "7.14-nightly",
                files={"opt/rocm/lib/librocblas.so": b"package payload"},
            )
            required = {"opt/rocm/lib/librocblas.so": digest(b"substituted")}
            staged = {"opt/rocm/lib/librocblas.so": digest(b"substituted")}

            with self.assertRaisesRegex(
                image_builder.ImageBuildError,
                "cannot be attributed to a pinned package",
            ):
                image_builder.rocm_payload_attestations(
                    required, staged, [package], Path("/usr/bin/dpkg-deb")
                )

    def test_rejects_module_build_attestation_for_a_different_module(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            artifact = {
                "schema_version": 1,
                "amdgpu_package_sha256": "1" * 64,
                "kernel": {"release": "6.1-test", "sha256": "2" * 64},
                "module": {
                    "sha256": "3" * 64,
                    "version": "7.1",
                    "srcversion": "SRC",
                    "vermagic": "6.1-test SMP",
                },
                "build": {
                    "command": ["make", "modules"],
                    "working_directory": "/build/amdgpu",
                    "inputs": [
                        {
                            "path": str(work / "config"),
                            "sha256": digest(b"config"),
                        }
                    ],
                    "toolchain": [
                        {
                            "path": str(work / "cc"),
                            "version": "test",
                            "sha256": digest(b"compiler"),
                        }
                    ],
                },
            }
            source = work / "attestation.json"
            write_input(work / "config", b"config")
            write_input(work / "cc", b"compiler", 0o755)
            source.write_text(json.dumps(artifact), encoding="utf-8")
            wrapper = {
                "record": artifact,
                "sha256": image_builder.sha256_file(source),
                "source": str(source),
            }
            with self.assertRaisesRegex(
                image_builder.ImageBuildError,
                "names a different module",
            ):
                image_builder.validate_build_attestation(
                    Path("/"),
                    wrapper,
                    {"sha256": "1" * 64},
                    "6.1-test",
                    "2" * 64,
                    "9" * 64,
                    {
                        "version": "7.1",
                        "srcversion": "SRC",
                        "vermagic": "6.1-test SMP",
                    },
                )

    def test_rejects_module_vermagic_for_a_different_guest_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock, _ = packaged_lock(work)
            prepare_m2_lock(work, lock)
            guest = lock["guest"]
            amdgpu = lock["amdgpu"]
            assert isinstance(guest, dict) and isinstance(amdgpu, dict)
            wrapper = amdgpu["build_attestation"]
            package = amdgpu["package"]
            module = amdgpu["module"]
            kernel = guest["kernel"]
            assert all(
                isinstance(value, dict) for value in (wrapper, package, module, kernel)
            )
            record = wrapper["record"]
            assert isinstance(record, dict)
            record_kernel = record["kernel"]
            assert isinstance(record_kernel, dict)
            record_kernel["release"] = "6.2.0-other"
            source = Path(wrapper["source"])
            source.write_text(json.dumps(record), encoding="utf-8")
            wrapper["sha256"] = image_builder.sha256_file(source)

            with self.assertRaisesRegex(
                image_builder.ImageBuildError,
                "vermagic release does not match the guest kernel release",
            ):
                image_builder.validate_build_attestation(
                    Path("/"),
                    wrapper,
                    package,
                    "6.2.0-other",
                    kernel["sha256"],
                    module["sha256"],
                    {key: module[key] for key in ("version", "srcversion", "vermagic")},
                )

    def test_rejects_symlinked_package_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock = base_lock(work)
            package = lock["rocm"]["packages"][0]
            original = Path(str(package["source"]))
            link = work / "package-link.deb"
            link.symlink_to(original)
            package["source"] = str(link)
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)

            status, diagnostics = self.run_builder(
                ["--lock", str(lock_path), "--output", str(work / "image")]
            )

            self.assertEqual(status, 1)
            self.assertIn("must not be a symlink", diagnostics)

    def test_builds_reproducible_image_and_generated_fixture_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock = base_lock(work)
            fixture = write_input(work / "generated/ip_discovery.bin", b"discovery")
            files = lock["files"]
            assert isinstance(files, list)
            files.append(
                {
                    **fixture,
                    "destination": "lib/firmware/amdgpu/ip_discovery.bin",
                    "mode": "0644",
                    "kind": "generated-fixture",
                    "generator": "tools/rj-ip-discovery",
                    "generator_revision": "0123456789abcdef",
                }
            )
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)
            first = work / "first"
            second = work / "second"

            first_manifest = image_builder.build_image(
                lock_path, first, TOOLS / "vfio_guest_provenance.py"
            )
            second_manifest = image_builder.build_image(
                lock_path, second, TOOLS / "vfio_guest_provenance.py"
            )

            self.assertEqual(first_manifest["outputs"], second_manifest["outputs"])
            self.assertEqual(
                (first / "initramfs.gz").read_bytes(),
                (second / "initramfs.gz").read_bytes(),
            )
            self.assertEqual(
                (first / "firmware-SHA256SUMS").read_text(encoding="ascii"),
                f"{digest(b'discovery')}  lib/firmware/amdgpu/ip_discovery.bin\n",
            )
            self.assertEqual(
                (first / "root/etc/rocjitsu/image-lock.json").read_bytes(),
                image_builder.canonical_json(lock),
            )
            for directory in ("dev", "proc", "sys", "tmp"):
                self.assertTrue((first / "root" / directory).is_dir())
            self.assertEqual(
                stat.S_IMODE((first / "root/tmp").stat().st_mode),
                0o1777,
            )
            archive_modes = {
                path.as_posix(): mode
                for path, mode, _source in image_builder.archive_entries(first / "root")
            }
            self.assertEqual(archive_modes["tmp"], stat.S_IFDIR | 0o1777)

    def test_rejects_old_host_style_firmware_directory_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock = base_lock(work)
            firmware_directory = work / "host/lib/firmware/amdgpu"
            firmware_directory.mkdir(parents=True)
            (firmware_directory / "vendor.bin").write_bytes(b"vendor firmware")
            files = lock["files"]
            assert isinstance(files, list)
            files.append(
                {
                    "source": str(firmware_directory),
                    "destination": "lib/firmware/amdgpu",
                    "mode": "0755",
                    "sha256": "0" * 64,
                }
            )
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)

            status, diagnostics = self.run_builder(
                [
                    "--lock",
                    str(lock_path),
                    "--output",
                    str(work / "image"),
                ]
            )

        self.assertEqual(status, 1)
        self.assertIn("is not a regular file", diagnostics)

    def test_rejects_unclassified_firmware_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock = base_lock(work)
            firmware = write_input(work / "inputs/vendor.bin", b"vendor firmware")
            files = lock["files"]
            assert isinstance(files, list)
            files.append(
                {
                    **firmware,
                    "destination": "lib/firmware/amdgpu/vendor.bin",
                    "mode": "0644",
                }
            )
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)

            status, diagnostics = self.run_builder(
                [
                    "--lock",
                    str(lock_path),
                    "--output",
                    str(work / "image"),
                ]
            )

        self.assertEqual(status, 1)
        self.assertIn("must declare kind generated-fixture", diagnostics)

    def test_rejects_changed_locked_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock = base_lock(work)
            files = lock["files"]
            assert isinstance(files, list)
            first = files[0]
            assert isinstance(first, dict)
            Path(str(first["source"])).write_bytes(b"changed")
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)

            status, diagnostics = self.run_builder(
                [
                    "--lock",
                    str(lock_path),
                    "--output",
                    str(work / "image"),
                ]
            )

        self.assertEqual(status, 1)
        self.assertIn("hash mismatch", diagnostics)

    def test_m2_workload_requires_and_embeds_verified_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock, modinfo = packaged_lock(work)
            guest = lock["guest"]
            assert isinstance(guest, dict)
            prepare_m2_lock(work, lock)
            binary = write_input(work / "inputs/m2-rocblas-sgemm", b"workload", 0o755)
            files = lock["files"]
            assert isinstance(files, list)
            files.append(
                {
                    **binary,
                    "destination": "opt/rocjitsu/bin/m2-rocblas-sgemm",
                    "mode": "0755",
                }
            )
            inventory = {
                "schema_version": 2,
                "packages": [
                    image_builder.package_identity(package)
                    for package in lock["rocm"]["packages"]
                ],
                "required_guest_files": {
                    "opt/rocjitsu/bin/m2-rocblas-sgemm": binary["sha256"]
                },
                "guest_init": {
                    "guest_path": "init",
                    "sha256": next(
                        item["sha256"]
                        for item in files
                        if item["destination"] == "init"
                    ),
                },
                "runtime": {
                    "backend": "Tensile",
                    "environment": {
                        "ROCBLAS_TENSILE_LIBPATH": "/opt/rocm/core-7.14/lib/rocblas/library",
                        "ROCBLAS_USE_HIPBLASLT": "0",
                    },
                },
                "selected_solution": {"index": 7, "name": "test-kernel"},
                "workload": {"api": "rocblas_sgemm", "m": 128, "n": 128, "k": 128},
            }
            inventory_path = work / "workload-inventory.json"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            lock["workload"] = {
                "inventory": inventory,
                "sha256": image_builder.sha256_file(inventory_path),
                "source": str(inventory_path),
            }
            lock["rocm"]["payloads"] = []
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)

            manifest = image_builder.build_image(
                lock_path,
                work / "image",
                TOOLS / "vfio_guest_provenance.py",
                modinfo=modinfo,
            )

            self.assertEqual(manifest["workload"]["selected_solution"]["index"], 7)
            self.assertEqual(
                (work / "image/workload-inventory.json").read_text(encoding="utf-8"),
                image_builder.canonical_json(inventory).decode("utf-8"),
            )

            firmware_record = next(
                item
                for item in files
                if item["destination"].endswith("gc_12_1_0_imu.bin")
            )
            firmware_record["generator"] = "trusted-looking-label"
            write_lock(lock_path, lock)
            with self.assertRaisesRegex(
                image_builder.ImageBuildError, "unexpected generator"
            ):
                image_builder.build_image(
                    lock_path,
                    work / "bad-firmware-image",
                    TOOLS / "vfio_guest_provenance.py",
                    modinfo=modinfo,
                )

    def test_m2_workload_rejects_missing_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock, modinfo = packaged_lock(work)
            guest = lock["guest"]
            assert isinstance(guest, dict)
            prepare_m2_lock(work, lock)
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)

            status, diagnostics = self.run_builder(
                [
                    "--lock",
                    str(lock_path),
                    "--output",
                    str(work / "image"),
                    "--modinfo",
                    str(modinfo),
                ]
            )

        self.assertEqual(status, 1)
        self.assertIn("workload must be a JSON object", diagnostics)

    def test_m2_workload_rejects_unverified_git_driver_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            lock = base_lock(work)
            guest = lock["guest"]
            assert isinstance(guest, dict)
            guest["role"] = "m2-workload"
            guest["policy"] = {"ip_block_mask": "0x3f"}
            lock_path = work / "image-lock.json"
            write_lock(lock_path, lock)

            status, diagnostics = self.run_builder(
                ["--lock", str(lock_path), "--output", str(work / "image")]
            )

        self.assertEqual(status, 1)
        self.assertIn("verified amdgpu-dkms package", diagnostics)


if __name__ == "__main__":
    unittest.main()
