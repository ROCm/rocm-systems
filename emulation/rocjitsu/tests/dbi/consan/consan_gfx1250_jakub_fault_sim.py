#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import consan_validation as validation


def _existing_directory(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path


def _existing_file(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the gfx1250 Jakub producer-skew barrier fault contract through "
            "the supported simulator launcher"
        )
    )
    parser.add_argument("--workspace", required=True, type=_existing_directory)
    parser.add_argument("--rocjitsu", required=True, type=_existing_file)
    parser.add_argument("--config", required=True, type=_existing_file)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spec = Path(__file__).with_name("consan_validation_faults_gfx1250.json")
    launcher = json.dumps([str(args.rocjitsu), "--config", str(args.config), "--"])
    previous_workspace = os.environ.get(validation.WORKSPACE_ENV)
    os.environ[validation.WORKSPACE_ENV] = str(args.workspace)
    try:
        with tempfile.TemporaryDirectory(prefix="consan-gfx1250-jakub-") as root:
            return validation.main(
                [
                    "--target",
                    "gfx1250",
                    "fault",
                    "--workload",
                    "jakub-attention",
                    "--profile",
                    "supercollider",
                    "--spec",
                    str(spec),
                    "--fault",
                    "barrier-drop",
                    "--artifact-root",
                    str(Path(root) / "artifacts"),
                    "--timeout",
                    "60",
                    "--health-timeout",
                    "30",
                    "--allow-destructive",
                    "--launcher-json",
                    launcher,
                ]
            )
    finally:
        if previous_workspace is None:
            os.environ.pop(validation.WORKSPACE_ENV, None)
        else:
            os.environ[validation.WORKSPACE_ENV] = previous_workspace


if __name__ == "__main__":
    raise SystemExit(main())
