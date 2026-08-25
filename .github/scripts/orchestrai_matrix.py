#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build the OrchestrAI test matrix and batches.

Turns the configured suites plus the resolved TheRock build into:

  * matrix  - one entry per (suite, platform, device); each becomes one GitHub
              Actions job that waits for its pipeline build and reports a verdict.
  * batches - the same entries grouped by (platform, device); each batch becomes
              ONE OrchestrAI pipeline build whose suites share one machine.

Everything environment-specific (suites, device -> broker tags, device -> AMDGPU
family, skips) comes from .github/orchestrai-config.yml so this stays generic.

A (suite, platform, device) is dropped — with a warning, never silently — when:
  * the device is in skip_devices;
  * the device has no device_to_family entry;
  * the device has no broker-tag mapping (the MAAS broker could never satisfy the
    batch; it would queue and time out hours later);
  * the resolved build for that platform did not include the device's family.

Usage:
    orchestrai_matrix.py [--config .github/orchestrai-config.yml] [--print]

Env:
    BUILD_SOURCE             JSON from orchestrai_resolve_build.py
    ORCHESTRAI_DEVICE_TAGS   JSON {device: [maas tags]} (overrides device_to_tags)
    SUITES / DEVICES / PLATFORMS   comma-separated workflow_dispatch filters

Outputs (to $GITHUB_OUTPUT):
    matrix=[...]  batches={...}  has_entries=true|false
"""

import argparse
import json
import os
import sys
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_csv(value: str) -> set[str] | None:
    """None means "no filter"; that is different from an empty selection."""
    value = (value or "").strip()
    if not value or value.lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def build(
    cfg: dict[str, Any],
    build_source: dict[str, Any],
    suites: set[str] | None = None,
    devices: set[str] | None = None,
    platforms: set[str] | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    device_to_tags = cfg.get("device_to_tags") or {}
    device_to_family = cfg.get("device_to_family") or {}
    skip_devices = set(cfg.get("skip_devices") or [])

    matrix: list[dict] = []
    # Collect reasons and report once per (kind, subject) so a device that is
    # unusable for six suites produces one warning, not six.
    dropped: dict[str, set[str]] = {}

    def drop(reason: str, subject: str) -> None:
        dropped.setdefault(reason, set()).add(subject)

    for suite_id, suite in (cfg.get("suites") or {}).items():
        if suites is not None and suite_id not in suites:
            continue
        for platform in suite.get("platforms") or []:
            if platforms is not None and platform not in platforms:
                continue
            source = build_source.get(platform)
            if not source:
                drop("no resolved TheRock build for platform", platform)
                continue
            available = set(source.get("families") or [])
            for device in suite.get("devices") or []:
                if devices is not None and device not in devices:
                    continue
                if device in skip_devices:
                    drop("listed in skip_devices", device)
                    continue
                family = device_to_family.get(device)
                if not family:
                    drop("no device_to_family mapping", device)
                    continue
                if device not in device_to_tags:
                    drop("no broker-tag mapping (ORCHESTRAI_DEVICE_TAGS)", device)
                    continue
                if family not in available:
                    drop(
                        f"family not built by the resolved {platform} build "
                        f"(has: {','.join(sorted(available)) or 'none'})",
                        f"{device} ({family})",
                    )
                    continue
                matrix.append(
                    {
                        "suite": suite_id,
                        "platform": platform,
                        "device": device,
                        "family": family,
                        "batch_id": f"{platform}/{device}",
                        "required": bool(suite.get("required", False)),
                    }
                )

    for reason, subjects in sorted(dropped.items()):
        print(
            f"::warning::skipped {', '.join(sorted(subjects))}: {reason}",
            file=sys.stderr,
        )

    batches: dict[str, dict] = {}
    for entry in matrix:
        batch_id = entry["batch_id"]
        source = build_source[entry["platform"]]
        if batch_id not in batches:
            batches[batch_id] = {
                "platform": entry["platform"],
                "device": entry["device"],
                "family": entry["family"],
                "tags": sorted(device_to_tags[entry["device"]]),
                "suites": [],
                "run_repo": source.get("repository", ""),
                "run_id": source["run_id"],
                "run_url": source.get("run_url", ""),
                "sha": source.get("sha", ""),
                "artifact_base_url": source["artifact_base_url"],
            }
        if entry["suite"] not in batches[batch_id]["suites"]:
            batches[batch_id]["suites"].append(entry["suite"])

    return matrix, batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=".github/orchestrai-config.yml")
    ap.add_argument("--print", action="store_true", dest="do_print")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Internal MAAS broker tags are not committed to this public repo; they come
    # from the ORCHESTRAI_DEVICE_TAGS variable (a JSON device -> tags map).
    raw_tags = os.environ.get("ORCHESTRAI_DEVICE_TAGS")
    if raw_tags:
        try:
            cfg["device_to_tags"] = json.loads(raw_tags)
        except json.JSONDecodeError:
            print(
                "::warning::ORCHESTRAI_DEVICE_TAGS is not valid JSON — "
                "falling back to device_to_tags in the config",
                file=sys.stderr,
            )

    raw_source = (os.environ.get("BUILD_SOURCE") or "").strip()
    build_source = json.loads(raw_source) if raw_source else {}

    matrix, batches = build(
        cfg,
        build_source,
        suites=parse_csv(os.environ.get("SUITES", "")),
        devices=parse_csv(os.environ.get("DEVICES", "")),
        platforms=parse_csv(os.environ.get("PLATFORMS", "")),
    )

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"matrix={json.dumps(matrix)}\n")
            f.write(f"batches={json.dumps(batches)}\n")
            f.write(f"has_entries={'true' if matrix else 'false'}\n")

    print(f"Matrix: {len(matrix)} entries, Batches: {len(batches)}", file=sys.stderr)
    for batch_id, batch in batches.items():
        print(
            f"  {batch_id}: {', '.join(batch['suites'])}  "
            f"family={batch['family']} run={batch['run_id']} tags={batch['tags']}",
            file=sys.stderr,
        )

    if args.do_print:
        print(json.dumps({"matrix": matrix, "batches": batches}, indent=2))


if __name__ == "__main__":
    main()
