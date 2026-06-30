#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
nic_simulator.py — simulate live AI NIC traffic by incrementing hw_counter
files at a fixed interval.

Usage:
    python3 nic_simulator.py <hw_counters_dir> [--interval SECONDS]

Where <hw_counters_dir> is the path to the hw_counters directory created by
fake_sysfs.create().

The process runs until it receives SIGTERM or SIGINT (Ctrl-C).
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

INCREMENTS: dict[str, int] = {
    "rx_rdma_ucast_bytes": 1_000_000,
    "tx_rdma_ucast_bytes": 800_000,
    "rx_rdma_ucast_pkts": 1_000,
    "tx_rdma_ucast_pkts": 800,
    "rx_rdma_cnp_pkts": 5,
    "tx_rdma_cnp_pkts": 3,
    "tx_rdma_ack_timeout": 1,
    "resp_tx_pkt_seq_err": 1,
    "req_rx_pkt_seq_err": 1,
    "req_rx_impl_nak_seq_err": 1,
}


def _read_counter(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except (ValueError, OSError):
        return 0


def _write_counter(path: Path, value: int) -> None:
    try:
        path.write_text(str(value) + "\n")
    except OSError as exc:
        print(f"[nic_simulator] Warning: could not write {path}: {exc}", file=sys.stderr)


def run(hw_counters_dir: Path, interval: float) -> None:
    running = True

    def _stop(sig, frame):  # noqa: ANN001
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    print(f"[nic_simulator] started (dir={hw_counters_dir}, interval={interval}s)", flush=True)

    while running:
        for counter_name, delta in INCREMENTS.items():
            path = hw_counters_dir / counter_name
            if path.exists() and delta > 0:
                _write_counter(path, _read_counter(path) + delta)
        time.sleep(interval)

    print("[nic_simulator] stopped", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate AI NIC hw_counters")
    parser.add_argument("hw_counters_dir", type=Path, help="Path to the hw_counters directory")
    parser.add_argument(
        "--interval", type=float, default=0.05, help="Update interval in seconds (default: 0.05)"
    )
    args = parser.parse_args()

    if not args.hw_counters_dir.is_dir():
        print(f"ERROR: {args.hw_counters_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    run(args.hw_counters_dir, args.interval)


if __name__ == "__main__":
    main()
