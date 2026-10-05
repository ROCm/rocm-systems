#!/usr/bin/env python3

import argparse
import os
import shlex
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def default_config_path() -> tuple[Path, str]:
    for repository_root in Path(__file__).resolve().parents:
        config_path = repository_root / "source/share/rocprofiler-sdk/config.yaml"
        if config_path.is_file():
            return config_path, "source tree"

    metrics_path = os.environ.get("ROCPROFILER_METRICS_PATH")
    if metrics_path:
        config_path = Path(metrics_path) / "config.yaml"
        if config_path.is_file():
            return config_path, "ROCPROFILER_METRICS_PATH"

    rocm_path_value = os.environ.get("ROCM_PATH")
    rocm_path = Path(rocm_path_value or "/opt/rocm")
    config_path = rocm_path / "share/rocprofiler-sdk/config.yaml"
    if config_path.is_file():
        return config_path, "ROCM_PATH" if rocm_path_value else "/opt/rocm"

    raise FileNotFoundError("could not locate rocprofiler-sdk config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve rocprofiler-sdk metrics into packet-test arguments"
    )
    parser.add_argument("--kind", choices=("pmc",), default="pmc")
    parser.add_argument("--gfx", required=True, help="architecture name, e.g. gfx1201")
    parser.add_argument("--block", help="optional PMC block name, e.g. SQ")
    parser.add_argument(
        "--events",
        help="optional comma-separated metric names or numeric event IDs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="path to config.yaml",
    )
    return parser.parse_args()


def load_counters(config_path: Path) -> list[dict]:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return config["rocprofiler-sdk"]["counters"]


def matching_definitions(counter: dict, gfx: str) -> list[dict]:
    return [
        definition
        for definition in counter.get("definitions", [])
        if gfx in definition.get("architectures", [])
        and "block" in definition
        and "event" in definition
    ]


def resolve_cases(args: argparse.Namespace, counters: list[dict]) -> dict[str, set[int]]:
    requested_block = args.block.upper() if args.block else None
    cases: dict[str, set[int]] = defaultdict(set)

    if args.events:
        counters_by_name = {counter["name"]: counter for counter in counters}
        for token in args.events.split(","):
            token = token.strip()
            if not token:
                raise ValueError("event list contains an empty entry")

            try:
                event_id = int(token, 0)
            except ValueError:
                event_id = None

            if event_id is not None:
                if requested_block is None:
                    raise ValueError(f"numeric event {token} requires --block")
                if not 0 <= event_id <= 0xFFFFFFFF:
                    raise ValueError(f"event ID is outside uint32_t: {token}")
                cases[requested_block].add(event_id)
                continue

            counter = counters_by_name.get(token)
            if counter is None:
                raise ValueError(f"unknown metric name: {token}")

            definitions = matching_definitions(counter, args.gfx)
            if not definitions:
                raise ValueError(
                    f"metric {token} has no hardware event for architecture {args.gfx}"
                )
            for definition in definitions:
                block = str(definition["block"]).upper()
                if requested_block and block != requested_block:
                    raise ValueError(
                        f"metric {token} belongs to {block}, not {requested_block}"
                    )
                cases[block].add(int(definition["event"]))
    else:
        for counter in counters:
            for definition in matching_definitions(counter, args.gfx):
                block = str(definition["block"]).upper()
                if requested_block and block != requested_block:
                    continue
                cases[block].add(int(definition["event"]))

    if not cases:
        scope = f"block {requested_block}" if requested_block else args.gfx
        raise ValueError(f"no PMC events found for {scope}")
    return cases


def format_cpp_arguments(gfx: str, cases: dict[str, set[int]]) -> str:
    arguments = ["--gfx", gfx]
    for block in sorted(cases):
        events = ",".join(str(event) for event in sorted(cases[block]))
        arguments.extend(("--block", block, "--events", events))
    return shlex.join(arguments)


def main() -> None:
    args = parse_args()
    if args.config is None:
        args.config, config_source = default_config_path()
    else:
        config_source = "--config"
    args.config = args.config.expanduser().resolve()
    print(f"resolved config path ({config_source}): {args.config}", file=sys.stderr)
    try:
        cases = resolve_cases(args, load_counters(args.config))
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"packet argument error: {error}") from error
    print(format_cpp_arguments(args.gfx, cases))


if __name__ == "__main__":
    main()
