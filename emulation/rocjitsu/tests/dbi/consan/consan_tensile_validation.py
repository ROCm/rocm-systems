#!/usr/bin/env python3
"""Runs one numerically validated gfx1250 Tensile corpus configuration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def _prepend(environment: dict[str, str], name: str, value: Path) -> None:
    previous = environment.get(name)
    environment[name] = f"{value}{os.pathsep}{previous}" if previous else str(value)


def _write_oracle_result(outcome: str, detail: object) -> None:
    result_path = os.environ.get("CONSAN_ROW_RESULT_PATH") or os.environ.get(
        "CONSAN_WORKLOAD_RESULT_PATH"
    )
    if not result_path:
        return
    payload = {
        "schema_version": 1,
        "oracle": outcome,
        "detail": detail,
        "source_diagnostics": {
            "outcome": "not_applicable",
            "count": None,
            "expectation": "not_applicable",
            "detail": "Tensile numeric validation has no separate source-diagnostic channel",
        },
    }
    path = Path(result_path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, choices=(1,), default=1)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    corpus = workspace / "rocjitsu-test-corpus"
    config = (corpus / args.config).resolve()
    if corpus.resolve() not in config.parents or not config.is_file():
        parser.error(f"invalid corpus config: {args.config}")

    tensilelite = (
        workspace
        / "TheRock"
        / "rocm-libraries"
        / "projects"
        / "hipblaslt"
        / "tensilelite"
    )
    rocm = workspace / "TheRock" / "build" / "dist" / "rocm"
    client = tensilelite / "build_tmp" / "tensilelite" / "client" / "tensilelite-client"
    for required in (tensilelite / "Tensile", rocm / "bin" / "amdclang++", client):
        if not required.exists():
            parser.error(f"missing Tensile validation prerequisite: {required}")

    os.environ["ROCM_PATH"] = str(rocm)
    os.environ["HIP_PATH"] = str(rocm)
    _prepend(os.environ, "PATH", rocm / "bin")
    _prepend(os.environ, "LD_LIBRARY_PATH", rocm / "lib")
    sys.path.insert(0, str(tensilelite))

    from Tensile import Tensile as tensile_mod

    started = time.monotonic()
    try:
        tensile_mod.Tensile(
            [
                str(config),
                str(args.output_dir),
                "--gpu-targets",
                "gfx1250",
                "--prebuilt-client",
                str(client),
                "--global-parameters",
                "NumBenchmarks=1",
                "SyncsPerBenchmark=1",
                "EnqueuesPerSync=1",
                "NumWarmups=0",
            ]
        )
    except BaseException as exc:
        _write_oracle_result("fail", {"config": args.config, "reason": str(exc)})
        raise
    elapsed_ms = (time.monotonic() - started) * 1000.0
    print(
        json.dumps(
            {
                args.label: {
                    "median_ms": elapsed_ms,
                    "oracle_passed": True,
                    "repetitions": args.repetitions,
                }
            },
            sort_keys=True,
        )
    )
    _write_oracle_result("pass", {"config": args.config, "label": args.label})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
