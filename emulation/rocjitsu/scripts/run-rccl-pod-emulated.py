#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run four two-GPU emulated KFD nodes in one eight-rank RCCL communicator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time

from rccl_pod_vfio import GateError, SIZES, sha256, verify_fabric, verify_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocjitsu", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rank-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=5400)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    binary = args.rocjitsu.resolve(strict=True)
    config = args.config.resolve(strict=True)
    rank_binary = args.rank_binary.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise GateError(f"output already exists: {output}")
    manifest = json.loads(config.read_text())
    nodes = manifest["system"]["nodes"]
    if len(nodes) != 4 or any(len(node["gpus"]) != 2 for node in nodes):
        raise GateError("manifest must have four nodes with two GPUs each")
    output.mkdir(parents=True, mode=0o700)
    run_id = secrets.token_hex(16)
    processes = []
    records = []
    failure = None
    start = time.monotonic()
    try:
        for node in range(4):
            node_dir = output / f"node{node}"
            node_dir.mkdir(mode=0o700)
            runtime = output / f"runtime-node{node}"
            xdg = output / f"xdg-node{node}"
            runtime.mkdir(mode=0o700)
            xdg.mkdir(mode=0o700)
            command = [
                str(binary),
                "--daemon",
                "--config",
                str(config),
                "--node",
                f"node{node}",
                "--fabric-effect-trace",
                "--",
                sys.executable,
                "scripts/run-rccl-pod-emulated-node.py",
                "--node",
                str(node),
                "--run-id",
                run_id,
                "--rank-binary",
                str(rank_binary),
                "--output",
                str(node_dir),
            ]
            environment = os.environ.copy()
            for key in tuple(environment):
                if key.startswith("GTEST_") or key in (
                    "HIP_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                ):
                    environment.pop(key)
            environment.update(
                {
                    "ROCJITSU_RUNTIME_DIR": str(runtime),
                    "XDG_RUNTIME_DIR": str(xdg),
                    "HSA_ENABLE_IPC_MODE_LEGACY": "1",
                }
            )
            launcher_log = node_dir / "launcher.log"
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
                    "environment": {
                        key: environment[key]
                        for key in (
                            "ROCJITSU_RUNTIME_DIR",
                            "XDG_RUNTIME_DIR",
                            "HSA_ENABLE_IPC_MODE_LEGACY",
                        )
                    },
                }
            )
        deadline = time.monotonic() + args.timeout
        while True:
            statuses = [process.poll() for process in processes]
            if any(status is not None and status != 0 for status in statuses):
                raise GateError(f"node helper failed: {statuses}")
            if all(status == 0 for status in statuses):
                break
            if time.monotonic() >= deadline:
                raise GateError(f"emulated pod deadline exceeded: {statuses}")
            for node in range(4):
                for path in (output / f"node{node}").glob("*.log"):
                    if path.stat().st_size > 1 << 30:
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
                failure = (failure or "node cleanup failed") + "; forced SIGKILL"

    rank_results = []
    fabric_results = []
    if failure is None:
        try:
            # The child launcher execs the helper. Its daemon receives
            # PDEATHSIG and writes the final trace after the helper exits.
            trace_deadline = time.monotonic() + 30
            while time.monotonic() < trace_deadline:
                if all(
                    "kfd: fabric effect trace captured="
                    in (output / f"node{node}" / "launcher.log").read_text(
                        errors="replace"
                    )
                    for node in range(4)
                ):
                    break
                time.sleep(0.1)
            for node in range(4):
                node_dir = output / f"node{node}"
                log = (node_dir / "launcher.log").read_text(errors="replace")
                if f"pod-emulated: node={node} result=PASS" not in log:
                    raise GateError(f"node {node}: helper did not report success")
                if "kfd: fabric effect trace captured=" not in log:
                    raise GateError(
                        f"node {node}: daemon did not finish its fabric trace"
                    )
                for local_gpu in range(2):
                    rank = 2 * node + local_gpu
                    for size in SIZES:
                        section = (node_dir / f"rank{rank}-size{size}.log").read_text(
                            encoding="utf-8", errors="replace"
                        )
                        rank_results.append(verify_rank(section, rank, size, node))
                fabric_results.append(
                    verify_fabric(node_dir / "launcher.log", node, "kfd")
                )
        except (GateError, OSError) as error:
            failure = str(error)
    socket_deadline = time.monotonic() + 30
    while True:
        leftover = [str(path) for path in output.glob("runtime-node*/**/daemon.sock")]
        if not leftover or time.monotonic() >= socket_deadline:
            break
        time.sleep(0.1)
    if leftover:
        failure = (
            failure + "; " if failure else ""
        ) + f"daemon sockets remain: {leftover}"
    artifact_paths = [
        binary,
        config,
        rank_binary,
        Path(__file__).resolve(),
        Path("scripts/run-rccl-pod-emulated-node.py").resolve(),
    ]
    artifact_paths += [
        path for node in range(4) for path in (output / f"node{node}").glob("*.log")
    ]
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "status": "PASS" if failure is None else "FAIL",
        "failure": failure,
        "mode": "emulated_kfd",
        "run_id": run_id,
        "node_commands": records,
        "node_statuses": [process.returncode for process in processes],
        "rank_results": rank_results,
        "fabric_results": fabric_results,
        "remaining_daemon_sockets": leftover,
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
        print(f"emulated pod gate setup failed: {error}", file=sys.stderr)
        return 1
    print(
        f"same-host emulated KFD pod gate: {result['status']}; record={args.output / 'result.json'}"
    )
    if result["failure"]:
        print(result["failure"], file=sys.stderr)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
