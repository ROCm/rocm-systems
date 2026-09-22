#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Validate and list target-specific test selectors.

import argparse
import json
from pathlib import Path
import re
from typing import Any

TARGET_PATTERN = re.compile(r"gfx[0-9A-Za-z]+")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--target")
    parser.add_argument("--suite")
    return parser


def load_selector_config(
    path: Path,
) -> dict[str, dict[str, tuple[str, ...]]]:
    try:
        with path.open(encoding="utf-8") as config_file:
            payload = json.load(config_file)
    except OSError as error:
        raise ValueError(f"could not read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error

    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} must contain a non-empty object keyed by target")

    config: dict[str, dict[str, tuple[str, ...]]] = {}
    for target, suites in payload.items():
        _validate_target(path, target, suites)
        config[target] = {}
        for suite, selectors in suites.items():
            config[target][suite] = _validate_selectors(path, target, suite, selectors)
    return config


def _validate_target(path: Path, target: Any, suites: Any) -> None:
    if not isinstance(target, str) or TARGET_PATTERN.fullmatch(target) is None:
        raise ValueError(f"{path} has invalid target {target!r}")
    if not isinstance(suites, dict) or not suites:
        raise ValueError(f"{path} target {target!r} must contain a suite object")


def _validate_selectors(
    path: Path, target: str, suite: Any, selectors: Any
) -> tuple[str, ...]:
    if not isinstance(suite, str) or not suite:
        raise ValueError(f"{path} target {target!r} has an invalid suite name")
    if not isinstance(selectors, list) or not selectors:
        raise ValueError(
            f"{path} target {target!r} suite {suite!r} " "must contain a non-empty list"
        )

    validated: list[str] = []
    for selector in selectors:
        if (
            not isinstance(selector, str)
            or not selector
            or selector != selector.strip()
            or "," in selector
        ):
            raise ValueError(
                f"{path} target {target!r} suite {suite!r} has invalid "
                f"selector {selector!r}"
            )
        if selector in validated:
            raise ValueError(
                f"{path} target {target!r} suite {suite!r} repeats "
                f"selector {selector!r}"
            )
        validated.append(selector)
    return tuple(validated)


def selectors_for(
    config: dict[str, dict[str, tuple[str, ...]]],
    target: str,
    suite: str,
) -> tuple[str, ...]:
    return config.get(target, {}).get(suite, ())


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()
    if args.target is not None and args.suite is None:
        parser.error("--suite is required with --target")
    if args.validate and args.suite is not None:
        parser.error("--suite cannot be used with --validate")

    try:
        config = load_selector_config(args.config)
    except ValueError as error:
        parser.error(str(error))

    if args.target is not None:
        print(",".join(selectors_for(config, args.target, args.suite)))


if __name__ == "__main__":
    main()
