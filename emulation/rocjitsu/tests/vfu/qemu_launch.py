#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
qemu_launch.py: Launch QEMU with a rocjitsu vfio-user GPU device.

Wraps the QEMU command line construction for reproducible test runs.
Can be imported as a module or run directly.

Usage:
  python3 qemu_launch.py [--socket SOCK] [--image IMAGE] [--ssh-port PORT]
"""

import argparse
import os
import subprocess
import sys
import time
import socket


def wait_for_ssh(port: int, timeout: int = 120) -> bool:
    """Poll until SSH port is open or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(2)
    return False


def build_qemu_args(
    socket_path: str,
    guest_image: str,
    ssh_port: int = 2222,
    memory_gb: int = 16,
    vcpus: int = 4,
) -> list[str]:
    vfio_device = (
        '{"driver":"vfio-user-pci",'
        f'"socket":{{"path":"{socket_path}","type":"unix"}}}}'
    )
    return [
        "qemu-system-x86_64",
        "-accel", "kvm",
        "-m", f"{memory_gb}G",
        "-smp", str(vcpus),
        "-drive", f"file={guest_image},format=qcow2,if=virtio",
        "-netdev", f"user,id=net0,hostfwd=tcp::{ssh_port}-:22",
        "-device", "virtio-net-pci,netdev=net0",
        "-device", vfio_device,
        "-display", "none",
        "-serial", "mon:stdio",
    ]


def launch(
    socket_path: str,
    guest_image: str,
    ssh_port: int = 2222,
    memory_gb: int = 16,
    vcpus: int = 4,
) -> subprocess.Popen:
    args = build_qemu_args(socket_path, guest_image, ssh_port, memory_gb, vcpus)
    print(f"[qemu_launch] {' '.join(args)}", file=sys.stderr)
    return subprocess.Popen(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch QEMU with rocjitsu vfio-user GPU")
    parser.add_argument("--socket",    default="/tmp/rocjitsu-vfu-0.sock")
    parser.add_argument("--image",     required=True, help="Guest QCOW2 image path")
    parser.add_argument("--ssh-port",  type=int, default=2222)
    parser.add_argument("--memory-gb", type=int, default=16)
    parser.add_argument("--vcpus",     type=int, default=4)
    parser.add_argument("--wait-ssh",  action="store_true",
                        help="Wait until SSH port is reachable")
    args = parser.parse_args()

    proc = launch(args.socket, args.image, args.ssh_port, args.memory_gb, args.vcpus)

    if args.wait_ssh:
        print(f"[qemu_launch] waiting for SSH on port {args.ssh_port}...", file=sys.stderr)
        if wait_for_ssh(args.ssh_port):
            print("[qemu_launch] SSH ready", file=sys.stderr)
        else:
            print("[qemu_launch] SSH timed out", file=sys.stderr)
            proc.terminate()
            sys.exit(1)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
