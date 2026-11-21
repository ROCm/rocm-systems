#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write a rocJITsu launch config with optional dispatch and quantum overrides.

ctest --parallel is a test_runner.py flag and is not applied here.
"""

import argparse
import json
from pathlib import Path


def apply_overrides(
    config: dict,
    functional_quantum: int | None = None,
    cpu_dispatch_threads: int | None = None,
) -> dict:
    if cpu_dispatch_threads is not None:
        config["cpu_dispatch_threads"] = cpu_dispatch_threads
    if functional_quantum is not None:
        set_functional_quantum(config.get("topology", {}), functional_quantum)
    return config


def set_functional_quantum(node, quantum: int) -> None:
    if isinstance(node, dict):
        if node.get("type") == "compute_unit":
            entries = [item for item in node.get("config", []) if item.get("key") != "functional_quantum"]
            entries.append({"key": "functional_quantum", "value": str(quantum)})
            node["config"] = entries
        for value in list(node.values()):
            set_functional_quantum(value, quantum)
    elif isinstance(node, list):
        for item in node:
            set_functional_quantum(item, quantum)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--functional-quantum", type=int)
    parser.add_argument("--cpu-dispatch-threads", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.src).read_text(encoding="utf-8"))
    apply_overrides(
        config,
        functional_quantum=args.functional_quantum,
        cpu_dispatch_threads=args.cpu_dispatch_threads,
    )
    destination = Path(args.out)
    destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
