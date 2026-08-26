#!/usr/bin/env python3
"""Flatten the GIN test matrix JSON into tab-separated rows for bash.

`run-gin-ci.sh` reads the emitted rows with `IFS=$'\t' read`, so the column
layout below is a stable contract:

  mca\t<flags>
  debug_env\t<-x flags>   (appended to every test only when RCCL_CI_DEBUG=1)
  test\t<name>\t<kind>\t<bin>\t<-x env flags>\t<args>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple


class GinTest(NamedTuple):
    """A single GIN test entry parsed from the matrix JSON."""

    name: str
    kind: str
    bin: str
    env: list[str]
    args: str


class GinConfig(NamedTuple):
    """The parsed GIN test matrix."""

    mca: str
    debug_env: list[str]
    tests: list[GinTest]


def parse_config(path: Path) -> GinConfig:
    """Read and validate the GIN test-matrix JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e
    except OSError as e:
        raise RuntimeError(f"Cannot read {path}: {e}") from e

    tests: list[GinTest] = []
    for entry in data.get("tests", []) or []:
        tests.append(
            GinTest(
                name=entry.get("name", ""),
                kind=entry.get("kind", ""),
                bin=entry.get("bin", ""),
                env=list(entry.get("env", []) or []),
                args=entry.get("args", "") or "",
            )
        )

    return GinConfig(
        mca=data.get("mca", "") or "",
        debug_env=list(data.get("debug_env", []) or []),
        tests=tests,
    )


def format_rows(config: GinConfig) -> list[str]:
    """Render the parsed config as tab-separated rows for the bash runner."""
    rows = ["mca\t" + config.mca]
    rows.append("debug_env\t" + " ".join("-x " + e for e in config.debug_env))
    for test in config.tests:
        env_flags = " ".join("-x " + e for e in test.env)
        rows.append(
            "\t".join(["test", test.name, test.kind, test.bin, env_flags, test.args])
        )
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Flatten the GIN test matrix JSON into TSV rows for bash."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the GIN test-matrix JSON file (e.g. lib/gin-tests.json)",
    )
    args = parser.parse_args(argv)

    if not args.config.exists():
        parser.error(f"config file not found: {args.config}")

    config = parse_config(args.config)
    for row in format_rows(config):
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
