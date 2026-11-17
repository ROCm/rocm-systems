#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run and verify four two-GPU VFIO guests over one NIC multicast segment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import time

CASES = ("AllReduce", "Broadcast", "AllGather", "ReduceScatter", "SendRecv")
SIZES = (64, 262144)
ITERATIONS = 3
RANK_HEADER = re.compile(r"^pod: === rank=(\d+) size=(\d+) log ===$", re.M)
PAYLOAD = re.compile(r"RCCL payload case=(\w+) rank=(\d+) iter=(\d+) elements=(\d+)")
CHANNEL = re.compile(
    r"NCCL INFO Channel \d+/\d+\s*: (\d+)\[[0-9a-f]+\] -> "
    r"(\d+)\[[0-9a-f]+\](?: \[(?:send|receive)\])? "
    r"via (P2P/IPC|NET/Socket/\S+)",
    re.I,
)
ROUTE = re.compile(
    r"committed fabric effects (\d+) -> (\d+): reads=(\d+), "
    r"writes=(\d+), read_bytes=(\d+), write_bytes=(\d+)"
)
EFFECT = re.compile(
    r"committed fabric effect request=(\d+) (\d+) -> (\d+): "
    r"op=(read|write), backing_id=(\d+), offset=(\d+), bytes=(\d+)"
)
TRACE = re.compile(r"fabric effect trace captured=(\d+), dropped=(\d+)")


class GateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def guest_sections(
    path: Path, node: int, driver_hash: str | None = None
) -> dict[tuple[int, int], str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    clean = re.sub(r"\x1b\[[0-9;]*m", "", raw).replace("\r", "")
    if f"pod: node={node} gpu_count=2" not in clean:
        raise GateError(f"node {node}: two stock-driver GPUs were not reported")
    if driver_hash and f"driver_sha256={driver_hash}" not in clean:
        raise GateError(f"node {node}: guest driver hash differs from staged module")
    if f"pod: node={node} result=PASS" not in clean:
        raise GateError(f"node {node}: guest did not complete")
    if "pod: FATAL:" in clean or "RCCL_COLLECTIVE_TIMEOUT" in clean:
        raise GateError(f"node {node}: guest reported a failure or deadline")
    headers = list(RANK_HEADER.finditer(clean))
    sections = {}
    for index, header in enumerate(headers):
        rank, size = map(int, header.groups())
        end = headers[index + 1].start() if index + 1 < len(headers) else len(clean)
        key = (rank, size)
        if key in sections:
            raise GateError(f"node {node}: duplicate rank section {key}")
        sections[key] = clean[header.end() : end]
    expected = {(2 * node + local, size) for local in (0, 1) for size in SIZES}
    if set(sections) != expected:
        raise GateError(f"node {node}: rank sections {set(sections)} != {expected}")
    return sections


def verify_rank(section: str, rank: int, size: int, node: int) -> dict:
    ran = re.findall(r"^\[\s*RUN\s*\]\s+RcclTest\.(\w+)\s*$", section, re.M)
    passed = re.findall(r"^\[\s*OK\s*\]\s+RcclTest\.(\w+)", section, re.M)
    if ran != list(CASES) or passed != list(CASES):
        raise GateError(f"rank {rank} size {size}: missing numerical case success")
    if not re.search(r"^\[\s*PASSED\s*\]\s+5 tests?\.", section, re.M):
        raise GateError(f"rank {rank} size {size}: missing passing summary")
    if re.search(r"^\[\s*FAILED\s*\]", section, re.M):
        raise GateError(f"rank {rank} size {size}: GoogleTest failure")
    payloads = [
        (case, int(log_rank), int(iteration), int(elements))
        for case, log_rank, iteration, elements in PAYLOAD.findall(section)
    ]
    expected_payloads = [
        (case, rank, iteration, size)
        for case in CASES
        for iteration in range(ITERATIONS)
    ]
    if payloads != expected_payloads:
        raise GateError(
            f"rank {rank} size {size}: incorrect case/iteration/size record"
        )
    if re.search(r"\b(?:h2d|copy|sentinel|result)?_?mismatches=[1-9]", section):
        raise GateError(f"rank {rank} size {size}: numerical mismatch")

    local_channels = []
    network_channels = []
    for source_text, target_text, transport in CHANNEL.findall(section):
        source, target = int(source_text), int(target_text)
        same_node = source // 2 == target // 2
        if same_node and transport == "P2P/IPC":
            local_channels.append((source, target))
        elif not same_node and transport.startswith("NET/Socket/"):
            network_channels.append((source, target))
    if not local_channels or not network_channels:
        raise GateError(
            f"rank {rank} size {size}: requires local P2P/IPC and cross-node NET"
        )
    if not any(source // 2 == node for source, _ in local_channels):
        raise GateError(f"rank {rank} size {size}: no P2P/IPC channel from its node")
    if not any(source // 2 == node for source, _ in network_channels):
        raise GateError(f"rank {rank} size {size}: no NET channel from its node")
    return {
        "rank": rank,
        "node": node,
        "size_elements": size,
        "iterations_per_case": ITERATIONS,
        "cases_passed": list(CASES),
        "numerical_mismatches": 0,
        "local_p2p_channels": local_channels,
        "cross_node_net_channels": network_channels,
    }


def verify_fabric(path: Path, node: int, prefix: str = "vfu") -> dict:
    route_totals: dict[str, dict[str, int]] = {}
    traced: dict[str, dict[str, int]] = {}
    captured = []
    request_ids = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = f"{prefix}: "
        if marker not in line:
            continue
        line = line.split(marker, 1)[1]
        if match := ROUTE.search(line):
            source, target, reads, writes, read_bytes, write_bytes = map(
                int, match.groups()
            )
            route = f"{source}->{target}"
            if route in route_totals:
                raise GateError(f"node {node}: duplicate route total {route}")
            route_totals[route] = dict(
                reads=reads,
                writes=writes,
                read_bytes=read_bytes,
                write_bytes=write_bytes,
            )
        if match := EFFECT.search(line):
            request, source, target, op, backing, _offset, length = match.groups()
            request = int(request)
            if (
                request in request_ids
                or (prefix == "vfu" and int(backing) == 0)
                or int(length) <= 0
            ):
                raise GateError(f"node {node}: invalid committed effect {request}")
            request_ids.add(request)
            route = f"{source}->{target}"
            totals = traced.setdefault(
                route, dict(reads=0, writes=0, read_bytes=0, write_bytes=0)
            )
            totals["writes" if op == "write" else "reads"] += 1
            totals["write_bytes" if op == "write" else "read_bytes"] += int(length)
        if match := TRACE.search(line):
            captured.append(tuple(map(int, match.groups())))
    expected = {
        f"{1250 + 2 * node}->{1251 + 2 * node}",
        f"{1251 + 2 * node}->{1250 + 2 * node}",
    }
    if set(route_totals) != expected or set(traced) != expected:
        raise GateError(f"node {node}: incorrect directed GPU routes")
    if any(route_totals[route] != traced[route] for route in expected):
        raise GateError(f"node {node}: directed effect trace disagrees with totals")
    if any(sum(route_totals[route].values()) == 0 for route in expected):
        raise GateError(f"node {node}: expected directed route has no committed bytes")
    if captured != [(len(request_ids), 0)]:
        raise GateError(f"node {node}: effect trace missing or dropped")
    return {
        "routes": route_totals,
        "trace_captured": len(request_ids),
        "trace_dropped": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "qemu",
        "kernel",
        "initramfs",
        "rocjitsu",
        "config",
        "driver-module",
        "driver-package",
        "rank-binary",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=5400)
    parser.add_argument("--mcast-group", default="239.192.14.1")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    inputs = {
        key: getattr(args, key).resolve(strict=True)
        for key in (
            "qemu",
            "kernel",
            "initramfs",
            "rocjitsu",
            "config",
            "driver_module",
            "driver_package",
            "rank_binary",
        )
    }
    output = args.output.resolve()
    if output.exists():
        raise GateError(f"output already exists: {output}")
    manifest = json.loads(inputs["config"].read_text())
    driver_hash = sha256(inputs["driver_module"])
    nodes = manifest["system"]["nodes"]
    if len(nodes) != 4 or any(len(node["gpus"]) != 2 for node in nodes):
        raise GateError("manifest must have four nodes with two GPUs each")
    for index, node in enumerate(nodes):
        if node["id"] != f"node{index}":
            raise GateError("manifest node IDs must be node0 through node3")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    segment = f"{args.mcast_group}:{port}"
    run_id = secrets.token_hex(16)
    output.mkdir(parents=True, mode=0o700)
    records = []
    processes = []
    start = time.monotonic()
    failure = None
    try:
        for node in range(4):
            node_output = output / f"node{node}"
            kernel_args = (
                "console=ttyS0 rdinit=/init panic=-1 amdgpu.discovery=2 "
                "amdgpu.emu_mode=1 amdgpu.fw_load_type=0 amdgpu.vm_update_mode=3 "
                "amdgpu.gpu_recovery=0 amdgpu.vramlimit=1024 amdgpu.ip_block_mask=0x3f "
                f"rocjitsu.pod.node={node} rocjitsu.pod.run_id={run_id}"
            )
            command = [
                sys.executable,
                "scripts/run-vfio-guest.py",
                "--qemu",
                str(inputs["qemu"]),
                "--kernel",
                str(inputs["kernel"]),
                "--initramfs",
                str(inputs["initramfs"]),
                "--rocjitsu",
                str(inputs["rocjitsu"]),
                "--config",
                str(inputs["config"]),
                "--node",
                f"node{node}",
                "--fabric-effect-trace",
                "--output",
                str(node_output),
                "--socket",
                str(node_output / "gpu0.sock"),
                "--socket",
                str(node_output / "gpu1.sock"),
                "--net-socket-mcast",
                segment,
                "--net-mac",
                f"02:00:00:00:14:{node + 1:02x}",
                "--p2v-switch-topology",
                "--p2v-atomic-root",
                "--accel",
                "tcg",
                "--smp",
                "4",
                "--memory",
                "12G",
                "--append",
                kernel_args,
                "--expect-log",
                f"pod: node={node} result=PASS",
                "--reject-log",
                "pod: FATAL:",
                "--reject-log",
                "[  FAILED  ]",
            ]
            launcher_log = output / f"node{node}-launcher.log"
            environment = os.environ.copy()
            environment["ROCJITSU_RUNTIME_DIR"] = str(output / f"runtime-node{node}")
            environment["XDG_RUNTIME_DIR"] = str(output / f"xdg-node{node}")
            (output / f"runtime-node{node}").mkdir(mode=0o700)
            (output / f"xdg-node{node}").mkdir(mode=0o700)
            for key in tuple(environment):
                if key.startswith("GTEST_") or key in (
                    "HIP_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                ):
                    environment.pop(key)
            with launcher_log.open("wb") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
            processes.append(process)
            records.append(
                {
                    "node": node,
                    "command": command,
                    "launcher_log": str(launcher_log),
                    "environment": {
                        key: environment[key]
                        for key in (
                            "ROCJITSU_RUNTIME_DIR",
                            "XDG_RUNTIME_DIR",
                        )
                    },
                }
            )
        deadline = time.monotonic() + args.timeout
        while True:
            statuses = [process.poll() for process in processes]
            if any(status is not None and status != 0 for status in statuses):
                raise GateError(f"node launcher failed: {statuses}")
            if all(status == 0 for status in statuses):
                break
            if time.monotonic() >= deadline:
                raise GateError(f"four-guest deadline exceeded: {statuses}")
            for node in range(4):
                for log_name in ("guest.log", "server.log"):
                    path = output / f"node{node}" / log_name
                    if path.exists() and path.stat().st_size > 1 << 30:
                        raise GateError(f"log cap exceeded: {path}")
            time.sleep(0.5)
    except Exception as error:
        failure = str(error)
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
                failure = (failure or "launcher cleanup failed") + "; forced SIGKILL"
    statuses = [process.returncode for process in processes]
    rank_results = []
    fabric_results = []
    if failure is None:
        try:
            for node in range(4):
                node_output = output / f"node{node}"
                sections = guest_sections(node_output / "guest.log", node, driver_hash)
                for (rank, size), section in sorted(sections.items()):
                    rank_results.append(verify_rank(section, rank, size, node))
                fabric_results.append(verify_fabric(node_output / "server.log", node))
        except (GateError, OSError) as error:
            failure = str(error)
    leftover = [str(path) for path in output.glob("node*/gpu*.sock")]
    if leftover:
        failure = (
            failure + "; " if failure else ""
        ) + f"VFIO sockets remain: {leftover}"
    artifact_paths = [inputs[key] for key in inputs]
    artifact_paths += [
        output / f"node{node}" / name
        for node in range(4)
        for name in ("guest.log", "server.log", "run-manifest.json")
    ]
    artifact_paths += [output / f"node{node}-launcher.log" for node in range(4)]
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "status": "PASS" if failure is None else "FAIL",
        "failure": failure,
        "mode": "qemu_vfio",
        "run_id": run_id,
        "nic_segment": segment,
        "node_commands": records,
        "node_statuses": statuses,
        "rank_results": rank_results,
        "fabric_results": fabric_results,
        "remaining_vfio_sockets": leftover,
        "sha256": {
            str(path): sha256(path) for path in artifact_paths if path.is_file()
        },
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except (GateError, OSError, ValueError) as error:
        print(f"pod gate setup failed: {error}", file=sys.stderr)
        return 1
    print(
        f"same-host VFIO pod gate: {result['status']}; record={args.output / 'result.json'}"
    )
    if result["failure"]:
        print(result["failure"], file=sys.stderr)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
