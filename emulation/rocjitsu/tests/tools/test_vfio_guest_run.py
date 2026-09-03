#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the vfio-user guest launcher."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

TOOLS = Path(__file__).parents[2] / "tools"


def load_tool(name: str):
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


image_builder = load_tool("vfio_guest_image")
guest_runner = load_tool("vfio_guest_run")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_file(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)


def build_test_image(work: Path) -> Path:
    inputs = work / "inputs"
    write_file(inputs / "init", b"#!/bin/busybox sh\npoweroff -f\n", 0o755)
    write_file(inputs / "busybox", b"busybox", 0o755)
    write_file(inputs / "vmlinuz", b"kernel")
    lock = {
        "schema_version": 2,
        "guest": {
            "distribution": "test-linux 1",
            "kernel_release": "6.1.2-test",
            "role": "driver-discovery",
            "kernel": {
                "source": str(inputs / "vmlinuz"),
                "sha256": digest(b"kernel"),
            },
        },
        "amdgpu": {
            "repository": "https://github.com/ROCm/amdgpu.git",
            "branch": "roc-7.1.x",
            "build_id": "test-build",
            "commit": "01cee31f91e55dc913933217d10341360f4a6f9e",
        },
        "rocm": {"packages": []},
        "build_tools": {"python": "test"},
        "files": [
            {
                "source": str(inputs / "init"),
                "destination": "init",
                "mode": "0755",
                "sha256": digest(b"#!/bin/busybox sh\npoweroff -f\n"),
            },
            {
                "source": str(inputs / "busybox"),
                "destination": "bin/busybox",
                "mode": "0755",
                "sha256": digest(b"busybox"),
            },
        ],
    }
    lock_path = work / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    image = work / "image"
    image_builder.build_image(lock_path, image, TOOLS / "vfio_guest_provenance.py")
    return image


class GuestRunTest(unittest.TestCase):
    def test_auto_acceleration_falls_back_to_tcg(self) -> None:
        self.assertEqual(
            guest_runner.select_accelerator(
                "auto", {"kvm", "tcg"}, Path("/definitely/no/kvm")
            ),
            "tcg",
        )

    def test_qemu_arguments_use_shared_memory_hypervisor_and_no_rom(self) -> None:
        arguments = guest_runner.build_qemu_arguments(
            Path("/qemu"),
            Path("/image"),
            Path("/tmp/device.sock"),
            "tcg",
            "2G",
            "0x123",
            "ignore_loglevel",
        )

        self.assertIn("qemu64,+hypervisor", arguments)
        self.assertIn("memory-backend-memfd,id=mem,size=2G,share=on", arguments)
        self.assertEqual(
            arguments[arguments.index("-machine") + 1], "q35,memory-backend=mem"
        )
        append = arguments[arguments.index("-append") + 1]
        self.assertIn("amdgpu.fw_load_type=0", append)
        self.assertIn("amdgpu.discovery=2", append)
        self.assertIn("amdgpu.vm_update_mode=3", append)
        self.assertIn("amdgpu.vramlimit=256", append)
        self.assertIn("amdgpu.ip_block_mask=0x123", append)
        self.assertIn("panic=-1", append)
        self.assertFalse(any("romfile" in argument for argument in arguments))
        self.assertNotIn("-option-rom", arguments)
        self.assertIn("-nodefaults", arguments)
        device = json.loads(arguments[arguments.index("-device") + 1])
        self.assertEqual(device["rombar"], 0)

    def test_m2_qemu_arguments_select_the_sgemm_workload(self) -> None:
        manifest = {"guest": {"role": "m2-workload"}}
        arguments = guest_runner.build_qemu_arguments(
            Path("/qemu"),
            Path("/image"),
            Path("/tmp/device.sock"),
            "tcg",
            guest_runner.select_guest_memory(manifest, None),
            "0x3f",
            "",
            "sgemm",
        )

        append = arguments[arguments.index("-append") + 1]
        self.assertIn("rocjitsu.workload=sgemm", append)
        self.assertEqual(arguments[arguments.index("-m") + 1], "4G")
        self.assertIn("memory-backend-memfd,id=mem,size=4G,share=on", arguments)

    def test_m2_rejects_a_caller_ip_block_mask_override(self) -> None:
        manifest = {
            "guest": {
                "role": "m2-workload",
                "policy": {"ip_block_mask": "0x3f"},
            }
        }
        self.assertEqual(guest_runner.select_ip_block_mask(manifest, None), "0x3f")
        self.assertEqual(guest_runner.select_ip_block_mask(manifest, "63"), "0x3f")
        with self.assertRaisesRegex(
            guest_runner.GuestRunError, "M2 IP-block mask is locked to 0x3f"
        ):
            guest_runner.select_ip_block_mask(manifest, "0x7f")

    def test_guest_memory_default_follows_the_image_role(self) -> None:
        discovery = {"guest": {"role": "driver-discovery"}}
        m2 = {"guest": {"role": "m2-workload"}}

        self.assertEqual(guest_runner.select_guest_memory(discovery, None), "2G")
        self.assertEqual(guest_runner.select_guest_memory(m2, None), "4G")
        self.assertEqual(guest_runner.select_guest_memory(m2, "6G"), "6G")
        self.assertEqual(guest_runner.select_guest_memory(discovery, "3G"), "3G")

    def test_extra_append_cannot_override_required_policy(self) -> None:
        with self.assertRaisesRegex(guest_runner.GuestRunError, "may not override"):
            guest_runner.build_qemu_arguments(
                Path("/qemu"),
                Path("/image"),
                Path("/tmp/device.sock"),
                "tcg",
                "2G",
                "0x123",
                "amdgpu.fw_load_type=2",
            )

        with self.assertRaisesRegex(guest_runner.GuestRunError, "may not override"):
            guest_runner.build_qemu_arguments(
                Path("/qemu"),
                Path("/image"),
                Path("/tmp/device.sock"),
                "tcg",
                "2G",
                "0x123",
                "amdgpu.vramlimit=512",
            )

        with self.assertRaisesRegex(guest_runner.GuestRunError, "may not override"):
            guest_runner.build_qemu_arguments(
                Path("/qemu"),
                Path("/image"),
                Path("/tmp/device.sock"),
                "tcg",
                "2G",
                "0x123",
                "rocjitsu.workload=none",
            )

    def test_guest_log_must_prove_hypervisor_bit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            log = work / "guest.log"
            log.write_text("guest: cpuid hypervisor bit: clear\n", encoding="utf-8")
            with self.assertRaisesRegex(guest_runner.GuestRunError, "did not prove"):
                guest_runner.record_guest_assertions(log, work, [], [])

    def test_guest_log_records_bounded_firmware_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            log = work / "guest.log"
            log.write_text(
                "guest: cpuid hypervisor bit: set\n"
                "guest: === firmware request inventory ===\n"
                "amdgpu: can't load firmware \"amdgpu/ip_discovery.bin\"\n"
                "guest: === KFD ===\n"
                "guest: done\n",
                encoding="utf-8",
            )

            guest_runner.record_guest_assertions(
                log, work, ["can't load firmware"], ["loaded successfully"]
            )

            self.assertEqual(
                (work / "firmware-requests.log").read_text(encoding="utf-8"),
                "amdgpu: can't load firmware \"amdgpu/ip_discovery.bin\"\n",
            )

    def test_m2_guest_log_intrinsically_rejects_media_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            log = work / "guest.log"
            log.write_text(
                "guest: cpuid hypervisor bit: set\n"
                "guest: === firmware request inventory ===\n"
                "amdgpu: loading amdgpu/vcn_5_0_0.bin\n"
                "guest: === KFD ===\n"
                "guest: done\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                guest_runner.GuestRunError, "forbidden media evidence: vcn"
            ):
                guest_runner.record_guest_assertions(log, work, [], [], "m2-workload")

    def test_non_m2_guest_log_does_not_apply_the_media_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            log = work / "guest.log"
            log.write_text(
                "guest: cpuid hypervisor bit: set\n"
                "guest: === firmware request inventory ===\n"
                "amdgpu: VCN diagnostic\n"
                "guest: === KFD ===\n"
                "guest: done\n",
                encoding="utf-8",
            )

            guest_runner.record_guest_assertions(log, work, [], [], "driver-discovery")

    def test_dry_run_records_exact_arguments_and_rechecks_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            image = build_test_image(work)
            qemu = work / "qemu"
            write_file(
                qemu,
                b"#!/bin/sh\n"
                b"if [ \"$1 $2\" = \"-accel help\" ]; then\n"
                b"  printf 'Accelerators supported in QEMU binary:\\ntcg\\n'\n"
                b"elif [ \"$1\" = \"--version\" ]; then\n"
                b"  printf 'QEMU test version\\n'\n"
                b"else\n"
                b"  exit 1\n"
                b"fi\n",
                0o755,
            )
            rocjitsu = work / "rocjitsu"
            write_file(
                rocjitsu,
                b"#!/bin/sh\n"
                b"[ \"$1\" = \"--version\" ] && printf 'rocjitsu test\\n'\n",
                0o755,
            )
            config = work / "config.json"
            write_file(config, b"{}\n")
            output = work / "run"
            diagnostics = io.StringIO()

            with contextlib.redirect_stderr(diagnostics), contextlib.redirect_stdout(
                io.StringIO()
            ):
                status = guest_runner.main(
                    [
                        "--image",
                        str(image),
                        "--qemu",
                        str(qemu),
                        "--rocjitsu",
                        str(rocjitsu),
                        "--config",
                        str(config),
                        "--output",
                        str(output),
                        "--ip-block-mask",
                        "0x123",
                        "--dry-run",
                    ]
                )

            self.assertEqual(status, 0, diagnostics.getvalue())
            manifest = json.loads(
                (output / "run-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["accelerator"], "tcg")
            self.assertEqual(manifest["qemu_argv"][0], str(qemu.resolve()))
            self.assertEqual(
                manifest["qemu_argv"][manifest["qemu_argv"].index("-m") + 1],
                "2G",
            )
            self.assertIn(
                "memory-backend-memfd,id=mem,size=2G,share=on",
                manifest["qemu_argv"],
            )
            self.assertFalse(
                any("romfile" in argument for argument in manifest["qemu_argv"])
            )

            (image / "root/lib/firmware/amdgpu").mkdir(parents=True)
            (image / "root/lib/firmware/amdgpu/vendor.bin").write_bytes(b"vendor")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                status = guest_runner.main(
                    [
                        "--image",
                        str(image),
                        "--qemu",
                        str(qemu),
                        "--rocjitsu",
                        str(rocjitsu),
                        "--config",
                        str(config),
                        "--output",
                        str(work / "rejected-run"),
                        "--ip-block-mask",
                        "0x123",
                        "--dry-run",
                    ]
                )
            self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
