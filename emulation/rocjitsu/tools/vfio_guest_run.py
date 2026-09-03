#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Launch a hermetic vfio-user guest and record its exact command provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

# The product configuration retains its full HBM capacity, but registering a
# struct page for every emulated VRAM page would consume more memory than the
# deliberately small hermetic guest owns. The driver's supported limit keeps
# the guest-visible allocation heap and HMM metadata bounded for M2.
DEFAULT_APPEND = (
    "console=ttyS0 rdinit=/init panic=-1 "
    "amdgpu.discovery=2 amdgpu.emu_mode=1 amdgpu.fw_load_type=0 "
    "amdgpu.vm_update_mode=3 amdgpu.gpu_recovery=0 amdgpu.vramlimit=256"
)
M2_IP_BLOCK_MASK = "0x3f"
M2_FORBIDDEN_MEDIA_LOG_TOKENS = ("vcn", "jpeg")
DISCOVERY_DEFAULT_MEMORY = "2G"
M2_DEFAULT_MEMORY = "4G"


class GuestRunError(ValueError):
    """A launch input violates the vfio-user guest contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def regular_file(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise GuestRunError(f"{name} is not a regular non-symlink file: {path}")
    return path.resolve()


def read_image_manifest(image: Path) -> dict[str, Any]:
    manifest_path = regular_file(image / "manifest.json", "image manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GuestRunError(f"invalid image manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise GuestRunError("image manifest has an unsupported schema")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise GuestRunError("image manifest is missing outputs")
    required_outputs = {"firmware-SHA256SUMS", "initramfs.gz", "vmlinuz"}
    if not required_outputs.issubset(outputs):
        raise GuestRunError("image manifest is missing required outputs")
    for name, expected in outputs.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise GuestRunError(f"image manifest has invalid output name: {name!r}")
        if not isinstance(expected, str) or len(expected) != 64:
            raise GuestRunError(f"image manifest has no valid digest for {name}")
        path = regular_file(image / name, name)
        actual = sha256_file(path)
        if actual != expected:
            raise GuestRunError(
                f"image output hash mismatch for {name}: "
                f"expected {expected}, got {actual}"
            )
    root = image / "root"
    if not root.is_dir() or root.is_symlink():
        raise GuestRunError(f"image root is not a directory: {root}")
    guest = manifest.get("guest")
    if isinstance(guest, dict) and guest.get("role") == "m2-workload":
        if not isinstance(manifest.get("workload"), dict):
            raise GuestRunError("m2-workload image manifest has no workload record")
        if "workload-inventory.json" not in outputs:
            raise GuestRunError("m2-workload image has no workload inventory output")
        if guest.get("policy") != {"ip_block_mask": M2_IP_BLOCK_MASK}:
            raise GuestRunError(
                f"m2-workload image must pin ip_block_mask={M2_IP_BLOCK_MASK}"
            )
    return manifest


def available_accelerators(qemu: Path) -> set[str]:
    result = subprocess.run(
        [str(qemu), "-accel", "help"],
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if result.returncode != 0:
        raise GuestRunError(
            "cannot query QEMU accelerators: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    return {
        line.strip().split()[0]
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("Accelerators supported")
    }


def tool_version(binary: Path) -> str:
    result = subprocess.run(
        [str(binary), "--version"],
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    output = result.stdout.strip() or result.stderr.strip()
    if result.returncode != 0 or not output:
        raise GuestRunError(f"cannot identify tool version: {binary}")
    return output


def select_accelerator(
    requested: str, accelerators: set[str], kvm_device: Path = Path("/dev/kvm")
) -> str:
    if requested == "tcg":
        if "tcg" not in accelerators:
            raise GuestRunError("QEMU does not support requested TCG acceleration")
        return "tcg"
    if requested == "kvm":
        if "kvm" not in accelerators:
            raise GuestRunError("QEMU does not support requested KVM acceleration")
        if not os.access(kvm_device, os.R_OK | os.W_OK):
            raise GuestRunError(f"KVM is not usable: {kvm_device}")
        return "kvm"
    if requested != "auto":
        raise GuestRunError(f"unknown accelerator selection: {requested}")
    if "kvm" in accelerators and os.access(kvm_device, os.R_OK | os.W_OK):
        return "kvm"
    if "tcg" not in accelerators:
        raise GuestRunError("KVM is unavailable and QEMU does not support TCG")
    return "tcg"


def build_qemu_arguments(
    qemu: Path,
    image: Path,
    socket_path: Path,
    accelerator: str,
    memory: str,
    ip_block_mask: str,
    extra_append: str,
    workload: str | None = None,
) -> list[str]:
    try:
        parsed_mask = int(ip_block_mask, 0)
    except ValueError as error:
        raise GuestRunError(
            f"IP-block mask is not an integer: {ip_block_mask}"
        ) from error
    if parsed_mask < 0 or parsed_mask > 0xFFFFFFFF:
        raise GuestRunError(
            f"IP-block mask is outside the 32-bit range: {ip_block_mask}"
        )
    protected_parameters = {
        "console",
        "panic",
        "rdinit",
        "amdgpu.emu_mode",
        "amdgpu.discovery",
        "amdgpu.fw_load_type",
        "amdgpu.gpu_recovery",
        "amdgpu.ip_block_mask",
        "amdgpu.vm_update_mode",
        "amdgpu.vramlimit",
        "rocjitsu.workload",
    }
    for word in extra_append.split():
        if word.split("=", 1)[0] in protected_parameters:
            raise GuestRunError(f"--append may not override required policy: {word}")
    cpu = "host,+hypervisor" if accelerator == "kvm" else "qemu64,+hypervisor"
    append = f"{DEFAULT_APPEND} amdgpu.ip_block_mask={parsed_mask:#x}"
    if workload is not None:
        append += f" rocjitsu.workload={workload}"
    if extra_append:
        append += f" {extra_append}"
    return [
        str(qemu),
        "-accel",
        accelerator,
        "-cpu",
        cpu,
        "-m",
        memory,
        "-object",
        f"memory-backend-memfd,id=mem,size={memory},share=on",
        "-machine",
        "q35,memory-backend=mem",
        "-kernel",
        str(image / "vmlinuz"),
        "-initrd",
        str(image / "initramfs.gz"),
        "-append",
        append,
        "-device",
        json.dumps(
            {
                "driver": "vfio-user-pci",
                "rombar": 0,
                "socket": {"path": str(socket_path), "type": "unix"},
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        "-nodefaults",
        "-no-user-config",
        "-display",
        "none",
        "-monitor",
        "none",
        "-serial",
        "stdio",
        "-no-reboot",
    ]


def select_ip_block_mask(image_manifest: dict[str, Any], requested: str | None) -> str:
    role = image_manifest.get("guest", {}).get("role")
    if role == "m2-workload":
        locked_mask = image_manifest["guest"]["policy"]["ip_block_mask"]
        if requested is not None:
            try:
                requested_mask = int(requested, 0)
            except ValueError as error:
                raise GuestRunError(
                    f"IP-block mask is not an integer: {requested}"
                ) from error
            if requested_mask != int(locked_mask, 0):
                raise GuestRunError(
                    f"M2 IP-block mask is locked to {locked_mask}; "
                    f"caller requested {requested}"
                )
        return locked_mask
    if requested is None:
        raise GuestRunError("--ip-block-mask is required for discovery images")
    return requested


def select_guest_memory(image_manifest: dict[str, Any], requested: str | None) -> str:
    if requested is not None:
        return requested
    role = image_manifest.get("guest", {}).get("role")
    if role == "m2-workload":
        return M2_DEFAULT_MEMORY
    return DISCOVERY_DEFAULT_MEMORY


def run_provenance_gate(tool: Path, image: Path, qemu_arguments: list[str]) -> None:
    command = [
        sys.executable,
        str(tool),
        "--root",
        str(image / "root"),
        "--initramfs",
        str(image / "initramfs.gz"),
        "--allowlist",
        str(image / "firmware-SHA256SUMS"),
        "--",
        *qemu_arguments,
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise GuestRunError(
            "guest provenance check failed: "
            + (result.stderr.strip() or result.stdout.strip())
        )


def wait_for_socket(path: Path, process: subprocess.Popen[Any], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise GuestRunError(
                "rocjitsu exited before creating the vfio-user socket "
                f"({process.returncode})"
            )
        try:
            mode = path.stat().st_mode
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        if stat_is_socket(mode):
            return
        raise GuestRunError(f"vfio-user socket path is not a socket: {path}")
    raise GuestRunError(f"rocjitsu did not create {path} within {timeout:g} seconds")


def stat_is_socket(mode: int) -> bool:
    return (mode & 0o170000) == 0o140000


def terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def record_guest_assertions(
    guest_log_path: Path,
    output: Path,
    expected: list[str],
    rejected: list[str],
    role: str | None = None,
) -> None:
    log = guest_log_path.read_text(encoding="utf-8", errors="replace")
    if "guest: cpuid hypervisor bit: set" not in log:
        raise GuestRunError("guest did not prove that the hypervisor CPUID bit was set")
    if "guest: FATAL:" in log:
        raise GuestRunError("guest reported a fatal assertion")
    if "guest: done" not in log:
        raise GuestRunError("guest did not reach its completion marker")
    for text in expected:
        if text not in log:
            raise GuestRunError(f"guest log is missing expected frontier: {text}")
    for text in rejected:
        if text in log:
            raise GuestRunError(f"guest log contains rejected frontier: {text}")
    if role == "m2-workload":
        folded_log = log.casefold()
        for token in M2_FORBIDDEN_MEDIA_LOG_TOKENS:
            if token in folded_log:
                raise GuestRunError(
                    f"M2 guest log contains forbidden media evidence: {token}"
                )
    start_marker = "guest: === firmware request inventory ==="
    end_marker = "guest: === KFD ==="
    if start_marker not in log or end_marker not in log:
        raise GuestRunError("guest did not emit the firmware-request inventory markers")
    inventory_text = log.split(start_marker, 1)[1].split(end_marker, 1)[0]
    inventory = [line for line in inventory_text.splitlines() if line]
    (output / "firmware-requests.log").write_text(
        "\n".join(inventory) + ("\n" if inventory else ""), encoding="utf-8"
    )


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--qemu", required=True, type=Path)
    parser.add_argument("--rocjitsu", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--accel", choices=("auto", "kvm", "tcg"), default="auto")
    parser.add_argument(
        "--memory",
        help="guest RAM size (defaults to 4G for M2 workloads and 2G otherwise)",
    )
    parser.add_argument(
        "--ip-block-mask",
        help="required for discovery images; M2 is locked to 0x3f",
    )
    parser.add_argument("--append", default="")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--socket-timeout", type=float, default=10)
    parser.add_argument("--expect-log", action="append", default=[])
    parser.add_argument("--reject-log", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--provenance-tool",
        type=Path,
        default=Path(__file__).with_name("vfio_guest_provenance.py"),
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_arguments(arguments)
    try:
        image = args.image.resolve()
        image_manifest = read_image_manifest(image)
        qemu = regular_file(args.qemu, "QEMU")
        rocjitsu = regular_file(args.rocjitsu, "rocjitsu")
        config = regular_file(args.config, "rocjitsu config")
        provenance_tool = regular_file(args.provenance_tool, "provenance tool")
        accelerators = available_accelerators(qemu)
        accelerator = select_accelerator(args.accel, accelerators)
        output = args.output.resolve()
        if output.exists() and any(output.iterdir()):
            raise GuestRunError(f"run output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        socket_path = (args.socket or output / "vfio-user.sock").resolve()
        if len(str(socket_path).encode("utf-8")) >= 108:
            raise GuestRunError(f"vfio-user socket path is too long: {socket_path}")
        if socket_path.exists():
            if not socket_path.is_socket():
                raise GuestRunError(
                    f"refusing to replace non-socket path: {socket_path}"
                )
            socket_path.unlink()

        role = image_manifest.get("guest", {}).get("role")
        memory = select_guest_memory(image_manifest, args.memory)
        ip_block_mask = select_ip_block_mask(image_manifest, args.ip_block_mask)

        qemu_arguments = build_qemu_arguments(
            qemu,
            image,
            socket_path,
            accelerator,
            memory,
            ip_block_mask,
            args.append,
            "sgemm" if role == "m2-workload" else None,
        )
        server_arguments = [
            str(rocjitsu),
            "--config",
            str(config),
            "--vfio-socket",
            str(socket_path),
        ]
        run_provenance_gate(provenance_tool, image, qemu_arguments)
        record = {
            "accelerator": accelerator,
            "config_sha256": sha256_file(config),
            "image_manifest_sha256": sha256_file(image / "manifest.json"),
            "image_outputs": image_manifest["outputs"],
            "image_amdgpu": image_manifest["amdgpu"],
            "image_guest": image_manifest["guest"],
            "image_rocm": image_manifest["rocm"],
            "expected_guest_log": args.expect_log,
            "intrinsic_rejected_guest_log": (
                list(M2_FORBIDDEN_MEDIA_LOG_TOKENS) if role == "m2-workload" else []
            ),
            "qemu_argv": qemu_arguments,
            "qemu_sha256": sha256_file(qemu),
            "qemu_version": tool_version(qemu),
            "rocjitsu_argv": server_arguments,
            "rocjitsu_sha256": sha256_file(rocjitsu),
            "rocjitsu_version": tool_version(rocjitsu),
            "rejected_guest_log": args.reject_log,
            "schema_version": 1,
        }
        (output / "run-manifest.json").write_text(
            canonical_json(record), encoding="utf-8"
        )
        if args.dry_run:
            print(canonical_json(record), end="")
            return 0

        server_log_path = output / "server.log"
        guest_log_path = output / "guest.log"
        with server_log_path.open("wb") as server_log:
            server = subprocess.Popen(
                server_arguments,
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )
        try:
            wait_for_socket(socket_path, server, args.socket_timeout)
            with guest_log_path.open("wb") as guest_log:
                try:
                    guest = subprocess.run(
                        qemu_arguments,
                        stdout=guest_log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=args.timeout,
                    )
                except subprocess.TimeoutExpired as error:
                    raise GuestRunError(
                        f"QEMU did not exit within {args.timeout:g} seconds"
                    ) from error
            if guest.returncode != 0:
                raise GuestRunError(f"QEMU exited with status {guest.returncode}")
            record_guest_assertions(
                guest_log_path, output, args.expect_log, args.reject_log, role
            )
        finally:
            terminate(server)
            if socket_path.exists() and socket_path.is_socket():
                socket_path.unlink()
        record["result"] = {
            "firmware_requests_sha256": sha256_file(output / "firmware-requests.log"),
            "guest_log_sha256": sha256_file(guest_log_path),
            "qemu_exit_status": guest.returncode,
            "server_log_sha256": sha256_file(server_log_path),
        }
        (output / "run-manifest.json").write_text(
            canonical_json(record), encoding="utf-8"
        )
        return 0
    except (GuestRunError, OSError, subprocess.SubprocessError) as error:
        print(f"vfio guest launch failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
