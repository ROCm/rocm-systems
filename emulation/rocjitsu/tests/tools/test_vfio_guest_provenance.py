#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the vfio-user guest provenance gate."""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import importlib.util
import io
from pathlib import Path
import stat
import tempfile
import unittest

SCRIPT = Path(__file__).parents[2] / "tools" / "vfio_guest_provenance.py"
SPEC = importlib.util.spec_from_file_location("vfio_guest_provenance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)


def write_allowlist(path: Path, entries: dict[str, bytes]) -> None:
    path.write_text(
        "".join(
            f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
            for name, payload in entries.items()
        ),
        encoding="ascii",
    )


def newc_entry(name: str, payload: bytes, mode: int = stat.S_IFREG | 0o644) -> bytes:
    name_bytes = name.encode("utf-8") + b"\0"
    fields = [
        1,
        mode,
        0,
        0,
        1,
        0,
        len(payload),
        0,
        0,
        0,
        0,
        len(name_bytes),
        0,
    ]
    header = b"070701" + b"".join(f"{field:08x}".encode("ascii") for field in fields)
    name_padding = bytes(-(len(header) + len(name_bytes)) % 4)
    data_padding = bytes(-len(payload) % 4)
    return header + name_bytes + name_padding + payload + data_padding


def write_initramfs(path: Path, entries: dict[str, bytes]) -> None:
    archive = b"".join(newc_entry(name, payload) for name, payload in entries.items())
    archive += newc_entry("TRAILER!!!", b"")
    with gzip.GzipFile(filename=path, mode="wb", mtime=0) as output:
        output.write(archive)


class ProvenanceTest(unittest.TestCase):
    def run_gate(self, arguments: list[str]) -> tuple[int, str]:
        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            status = provenance.main(arguments)
        return status, diagnostics.getvalue()

    def test_accepts_only_allowlisted_firmware_in_root_and_initramfs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            root = work / "root"
            firmware = root / "lib/firmware/amdgpu/ip_discovery.bin"
            firmware.parent.mkdir(parents=True)
            firmware.write_bytes(b"generated discovery")
            allowlist = work / "SHA256SUMS"
            write_allowlist(
                allowlist,
                {"lib/firmware/amdgpu/ip_discovery.bin": firmware.read_bytes()},
            )
            initramfs = work / "initramfs.gz"
            write_initramfs(
                initramfs,
                {
                    "init": b"#!/bin/sh\n",
                    "lib/firmware/amdgpu/ip_discovery.bin": firmware.read_bytes(),
                },
            )

            status, diagnostics = self.run_gate(
                [
                    "--root",
                    str(root),
                    "--allowlist",
                    str(allowlist),
                    "--initramfs",
                    str(initramfs),
                    "--",
                    "qemu-system-x86_64",
                    "-device",
                    '{"driver":"vfio-user-pci"}',
                ]
            )

        self.assertEqual(status, 0, diagnostics)

    def test_rejects_unmanifested_firmware_in_both_search_paths(self) -> None:
        for relative in (
            "lib/firmware/amdgpu/vendor.bin",
            "lib/firmware/updates/amdgpu/vendor.bin",
        ):
            with self.subTest(
                relative=relative
            ), tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary)
                root = work / "root"
                firmware = root / relative
                firmware.parent.mkdir(parents=True)
                firmware.write_bytes(b"vendor payload")
                allowlist = work / "SHA256SUMS"
                allowlist.write_text("", encoding="ascii")

                status, diagnostics = self.run_gate(
                    ["--root", str(root), "--allowlist", str(allowlist)]
                )

                self.assertEqual(status, 1)
                self.assertIn(f"unmanifested firmware: {relative}", diagnostics)

    def test_rejects_hash_mismatch_and_missing_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            root = work / "root"
            firmware = root / "lib/firmware/amdgpu/fixture.bin"
            firmware.parent.mkdir(parents=True)
            firmware.write_bytes(b"changed")
            allowlist = work / "SHA256SUMS"
            write_allowlist(
                allowlist,
                {
                    "lib/firmware/amdgpu/fixture.bin": b"expected",
                    "lib/firmware/amdgpu/missing.bin": b"missing",
                },
            )

            status, diagnostics = self.run_gate(
                ["--root", str(root), "--allowlist", str(allowlist)]
            )

        self.assertEqual(status, 1)
        self.assertIn(
            "firmware hash mismatch: lib/firmware/amdgpu/fixture.bin", diagnostics
        )
        self.assertIn(
            "manifested firmware is missing: lib/firmware/amdgpu/missing.bin",
            diagnostics,
        )

    def test_rejects_firmware_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            root = work / "root"
            firmware = root / "lib/firmware/amdgpu/fixture.bin"
            firmware.parent.mkdir(parents=True)
            target = work / "payload"
            target.write_bytes(b"fixture")
            firmware.symlink_to(target)
            allowlist = work / "SHA256SUMS"
            write_allowlist(allowlist, {"lib/firmware/amdgpu/fixture.bin": b"fixture"})

            status, diagnostics = self.run_gate(
                ["--root", str(root), "--allowlist", str(allowlist)]
            )

        self.assertEqual(status, 1)
        self.assertIn("firmware symlink is forbidden", diagnostics)

    def test_rejects_a_symlinked_firmware_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            root = work / "root"
            (root / "lib").mkdir(parents=True)
            outside = work / "outside"
            firmware = outside / "amdgpu/fixture.bin"
            firmware.parent.mkdir(parents=True)
            firmware.write_bytes(b"fixture")
            (root / "lib/firmware").symlink_to(outside)
            allowlist = work / "SHA256SUMS"
            write_allowlist(allowlist, {"lib/firmware/amdgpu/fixture.bin": b"fixture"})

            status, diagnostics = self.run_gate(
                ["--root", str(root), "--allowlist", str(allowlist)]
            )

        self.assertEqual(status, 1)
        self.assertIn("firmware symlink is forbidden: lib/firmware", diagnostics)

    def test_rejects_unmanifested_firmware_inside_initramfs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            root = work / "root"
            root.mkdir()
            allowlist = work / "SHA256SUMS"
            allowlist.write_text("", encoding="ascii")
            initramfs = work / "initramfs.gz"
            write_initramfs(
                initramfs, {"lib/firmware/amdgpu/vendor.bin": b"vendor payload"}
            )

            status, diagnostics = self.run_gate(
                [
                    "--root",
                    str(root),
                    "--allowlist",
                    str(allowlist),
                    "--initramfs",
                    str(initramfs),
                ]
            )

        self.assertEqual(status, 1)
        self.assertIn(
            "initramfs: unmanifested firmware: lib/firmware/amdgpu/vendor.bin",
            diagnostics,
        )

    def test_rejects_an_archive_hidden_after_the_trailer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            root = work / "root"
            root.mkdir()
            allowlist = work / "SHA256SUMS"
            allowlist.write_text("", encoding="ascii")
            initramfs = work / "initramfs.gz"
            archive = newc_entry("TRAILER!!!", b"")
            archive += newc_entry("lib/firmware/amdgpu/vendor.bin", b"hidden")
            archive += newc_entry("TRAILER!!!", b"")
            with gzip.GzipFile(filename=initramfs, mode="wb", mtime=0) as output:
                output.write(archive)

            status, diagnostics = self.run_gate(
                [
                    "--root",
                    str(root),
                    "--allowlist",
                    str(allowlist),
                    "--initramfs",
                    str(initramfs),
                ]
            )

        self.assertEqual(status, 2)
        self.assertIn("initramfs has nonzero data after TRAILER!!!", diagnostics)

    def test_rejects_pci_expansion_rom_arguments(self) -> None:
        cases = (
            ["--", "qemu-system-x86_64", "-option-rom", "gpu.rom"],
            ["--", "qemu-system-x86_64", "-device", "vfio-user-pci,romfile=gpu.rom"],
            [
                "--",
                "qemu-system-x86_64",
                "-device",
                '{"driver":"vfio-user-pci","romfile":"gpu.rom"}',
            ],
        )
        for qemu_arguments in cases:
            with self.subTest(
                arguments=qemu_arguments
            ), tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary)
                root = work / "root"
                root.mkdir()
                allowlist = work / "SHA256SUMS"
                allowlist.write_text("", encoding="ascii")

                status, diagnostics = self.run_gate(
                    [
                        "--root",
                        str(root),
                        "--allowlist",
                        str(allowlist),
                        *qemu_arguments,
                    ]
                )

                self.assertEqual(status, 1)
                self.assertIn("PCI expansion ROM argument is forbidden", diagnostics)


if __name__ == "__main__":
    unittest.main()
