#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Launch two RCCL ranks through one emulated KFD node daemon."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CASES = "RcclTest.AllReduce:RcclTest.Broadcast:RcclTest.AllGather:RcclTest.ReduceScatter:RcclTest.SendRecv"


def signal_process_group(
    process: subprocess.Popen[bytes], signum: signal.Signals
) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        # A rank may exit between poll() and killpg(); keep cleaning its peers.
        pass


def cleanup_ranks(running: list[subprocess.Popen[bytes]]) -> None:
    for process in running:
        if process.poll() is None:
            signal_process_group(process, signal.SIGTERM)
    for process in running:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            signal_process_group(process, signal.SIGKILL)
            process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", type=int, choices=range(4), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rank-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    running: list[subprocess.Popen[bytes]] = []
    interrupted = False

    def terminate(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"pod-emulated: node={args.node} daemon_pid={os.environ.get('ROCJITSU_DAEMON_PID', '')}",
        flush=True,
    )
    try:
        for size, port in ((64, 24317), (262144, 24319)):
            running.clear()
            for local_gpu in range(2):
                rank = 2 * args.node + local_gpu
                bootstrap = (
                    f"--bootstrap-listen=127.0.0.1:{port}"
                    if rank == 0
                    else f"--bootstrap-connect=127.0.0.1:{port}"
                )
                command = [
                    str(args.rank_binary),
                    f"--rank={rank}",
                    "--world-size=8",
                    bootstrap,
                    f"--bootstrap-run-id={args.run_id}",
                    "--bootstrap-timeout-ms=180000",
                    "--collective-timeout-ms=180000",
                    "--expected-devices=1",
                    f"--payload-elements={size}",
                    "--seed=79",
                    f"--gtest_filter={CASES}",
                ]
                environment = os.environ.copy()
                for key in (
                    "HIP_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                    "GPU_DEVICE_ORDINAL",
                    "HSA_VISIBLE_DEVICES",
                ):
                    environment.pop(key, None)
                environment.update(
                    {
                        "ROCR_VISIBLE_DEVICES": str(local_gpu),
                        "NCCL_HOSTID": f"pod-node-{args.node}",
                        "NCCL_NET": "Socket",
                        "NCCL_SOCKET_IFNAME": "lo",
                        "NCCL_IB_DISABLE": "1",
                        "NCCL_P2P_DISABLE": "0",
                        "NCCL_P2P_LEVEL": "SYS",
                        "NCCL_SHM_DISABLE": "1",
                        "NCCL_DEBUG": "INFO",
                        "NCCL_DEBUG_SUBSYS": "INIT,NET,GRAPH",
                        "HSA_NO_SCRATCH_RECLAIM": "1",
                    }
                )
                log_path = args.output / f"rank{rank}-size{size}.log"
                with log_path.open("wb") as log:
                    process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        start_new_session=True,
                    )
                running.append(process)
                print(
                    f"pod-emulated: rank={rank} gpu={local_gpu} size={size} pid={process.pid}",
                    flush=True,
                )
            deadline = time.monotonic() + 1200
            while True:
                statuses = [process.poll() for process in running]
                if interrupted:
                    raise RuntimeError("node helper interrupted")
                if any(status is not None and status != 0 for status in statuses):
                    raise RuntimeError(
                        f"node {args.node} size {size}: rank failure {statuses}"
                    )
                if all(status == 0 for status in statuses):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"node {args.node} size {size}: rank deadline {statuses}"
                    )
                time.sleep(0.1)
            print(
                f"pod-emulated: node={args.node} size={size} ranks=2 PASS", flush=True
            )
        print(f"pod-emulated: node={args.node} result=PASS", flush=True)
        return 0
    except (OSError, RuntimeError) as error:
        print(
            f"pod-emulated: node={args.node} FATAL: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        cleanup_ranks(running)


if __name__ == "__main__":
    raise SystemExit(main())
