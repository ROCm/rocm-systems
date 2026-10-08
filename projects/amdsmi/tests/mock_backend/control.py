#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Create and atomically update CPU-only AMD SMI test fixtures."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import uuid

FIELDS = (
    "temperature_c",
    "power_w",
    "power_cap_w",
    "utilization",
    "used_mib",
    "total_mib",
    "correctable",
    "uncorrectable",
)
DEFAULTS = (45, 100, 750, 0, 0, 196608, 0, 0)


def validate(state):
    for key in FIELDS:
        value = state[key]
        if not isinstance(value, int) or not 0 <= value < 2**64:
            raise ValueError(f"{key} must be an unsigned 64-bit integer")
    if state["temperature_c"] > 65535 or state["utilization"] > 100:
        raise ValueError("temperature or utilization out of range")
    if any(state[key] >= 2**32 - 1 for key in ("power_w", "power_cap_w")):
        raise ValueError("power must be below UINT32_MAX")
    if not 0 < state["total_mib"] < 2**32 or state["used_mib"] > state["total_mib"]:
        raise ValueError("memory must satisfy 0 <= used <= total <= UINT32_MAX")
    if not state["name"] or len(state["name"].encode()) >= 256:
        raise ValueError("name must contain 1..255 bytes")
    if any(character in state["name"] for character in "\n\r\0"):
        raise ValueError("name must be a single text line")


def write_atomic(path, text):
    # Rename in the same directory lets readers see an entire old or new record.
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as output:
            output.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save(path, state):
    validate(state)
    write_atomic(
        path,
        "AMDSMI_MOCK_V1\n"
        + " ".join(str(state[key]) for key in FIELDS)
        + f"\n{state['name']}\n{state['uuid']}\n{state['bdf']}\n",
    )


def load(path):
    lines = path.read_text().splitlines()
    if len(lines) != 5 or lines[0] != "AMDSMI_MOCK_V1":
        raise ValueError(f"invalid state file: {path}")
    values = lines[1].split()
    if len(values) != len(FIELDS):
        raise ValueError(f"invalid telemetry record: {path}")
    state = dict(zip(FIELDS, map(int, values)))
    state.update(name=lines[2], uuid=lines[3], bdf=lines[4])
    validate(state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="create a new fixture directory")
    initialize.add_argument("--gpus", type=int, default=1)
    initialize.add_argument("--name", default="AMD Instinct MI300X (mock)")
    initialize.add_argument("--total-mib", type=int, default=196608)
    change = commands.add_parser("set", help="update selected telemetry/ECC values")
    for key in FIELDS:
        change.add_argument(f"--{key.replace('_', '-')}", type=int)
    for command in (change, commands.add_parser("reset", help="restore baseline telemetry")):
        command.add_argument("--gpu", type=int, required=True)
    commands.add_parser("show", help="print all fixture records as JSON")
    args = parser.parse_args()
    try:
        directory = args.state_dir
        if args.command == "init":
            if not 0 <= args.gpus <= 32:
                raise ValueError("gpus must be in 0..32 (AMDSMI_MAX_DEVICES)")
            states = []
            for index in range(args.gpus):
                state = dict(zip(FIELDS, DEFAULTS))
                state.update(
                    name=args.name,
                    total_mib=args.total_mib,
                    uuid=str(uuid.uuid5(uuid.NAMESPACE_OID, f"amdsmi-mock-gpu-{index}")),
                    bdf=f"0000:{index + 1:02x}:00.0",
                )
                validate(state)
                states.append(state)
            directory.mkdir(parents=True, exist_ok=True)
            if any(directory.iterdir()):
                raise ValueError("init requires an empty state directory")
            for index, state in enumerate(states):
                save(directory / f"gpu{index}", state)
            # Publish discovery last, after all records exist.
            write_atomic(directory / "count", f"{args.gpus}\n")
            return
        count = int((directory / "count").read_text())
        if args.command == "show":
            print(json.dumps([load(directory / f"gpu{i}") for i in range(count)], indent=2))
            return
        if not 0 <= args.gpu < count:
            raise ValueError("GPU index is outside the fixture")
        path = directory / f"gpu{args.gpu}"
        state = load(path)
        if args.command == "reset":
            total = state["total_mib"]
            state.update(zip(FIELDS, DEFAULTS))
            state["total_mib"] = total
        else:
            changes = {key: getattr(args, key) for key in FIELDS if getattr(args, key) is not None}
            if not changes:
                raise ValueError("set requires at least one telemetry value")
            state.update(changes)
        save(path, state)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
