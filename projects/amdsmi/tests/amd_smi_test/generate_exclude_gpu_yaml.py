#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write the build-tree test_categories YAML the ctest parser reads.

test_categories.yaml holds the category definitions and the GPU-family to
FILTER[] key map. The per-ASIC patterns live only in amdsmitst.exclude.
This script sources that file and fills exclude_gpu_* so
parse_test_categories.py does not need a second hand-written list.

FILTER[virtualization] is not copied into any family. detect_asic_filter.sh
appends it only for SR-IOV or VF, which configure time cannot know.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_filters(exclude_path):
    """Return {FILTER key: value} from a sourced amdsmitst.exclude."""
    script = (
        'source "$1"\n'
        'for key in "${!FILTER[@]}"; do\n'
        '  printf "%s\\0%s\\0" "$key" "${FILTER[$key]}"\n'
        "done\n"
    )
    raw = subprocess.check_output(
        ["bash", "-c", script, "bash", str(exclude_path)],
    )
    filters = {}
    parts = raw.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    if len(parts) % 2:
        raise RuntimeError(f"undecodable FILTER dump from {exclude_path}")
    for key, value in zip(parts[0::2], parts[1::2]):
        filters[key.decode()] = value.decode()
    return filters


def patterns_of(value):
    return [part for part in value.split(":") if part]


def _string_map(config, name):
    raw = config.get(name)
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError(f"{name} must be a non-empty map in test_categories.yaml")
    return {str(key): str(value) for key, value in raw.items()}


def build_exclude_gpu(config, filters):
    """Return (exclude_gpu mapping, family names) or raise RuntimeError."""
    families = _string_map(config, "gpu_family_to_asic")
    aliases = _string_map(config, "filter_key_aliases")
    categories = list(config.get("test_categories", {}))
    if not categories:
        raise RuntimeError("test_categories.yaml has no test_categories")

    if "virtualization" in families.values():
        raise RuntimeError(
            "gpu_family_to_asic must not point at FILTER[virtualization]"
        )

    missing = [asic for asic in families.values() if asic not in filters]
    if missing:
        raise RuntimeError(f"gpu_family_to_asic keys missing from FILTER[]: {missing}")

    for alias, target in aliases.items():
        if alias not in filters:
            raise RuntimeError(f"filter_key_aliases key {alias} is not in FILTER[]")
        if target not in filters:
            raise RuntimeError(f"filter_key_aliases target {target} is not in FILTER[]")
        if patterns_of(filters[alias]) != patterns_of(filters[target]):
            raise RuntimeError(
                f"FILTER[{alias}] differs from FILTER[{target}]; aliases must match"
            )

    mapped = set(families.values()) | set(aliases)
    unexpected = sorted(
        key
        for key, value in filters.items()
        if patterns_of(value) and key not in mapped and key != "virtualization"
    )
    if unexpected:
        raise RuntimeError(
            "FILTER[] keys with patterns have no gpu_family_to_asic or "
            f"filter_key_aliases entry: {unexpected}"
        )

    exclude_gpu = {}
    archs = []
    for family, asic in families.items():
        patterns = patterns_of(filters[asic])
        if not patterns:
            continue
        archs.append(family)
        exclude_gpu[f"exclude_gpu_{family}"] = {
            "test_patterns": patterns,
            "labels": categories + [f"ex_gpu_{family}"],
        }
    return exclude_gpu, archs


def generate(categories_path, exclude_path, output_path, archs_path):
    with open(categories_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    filters = load_filters(exclude_path)
    exclude_gpu, archs = build_exclude_gpu(config, filters)
    config["exclude_gpu"] = exclude_gpu
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    archs_path.write_text("".join(f"{arch}\n" for arch in archs), encoding="utf-8")
    return archs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archs", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        archs = generate(args.categories, args.exclude, args.output, args.archs)
    except (
        OSError,
        subprocess.CalledProcessError,
        RuntimeError,
        yaml.YAMLError,
    ) as exc:
        print(f"amdsmi: failed to generate exclude_gpu YAML: {exc}", file=sys.stderr)
        return 1
    print(f"amdsmi: generated exclude_gpu for {', '.join(archs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
