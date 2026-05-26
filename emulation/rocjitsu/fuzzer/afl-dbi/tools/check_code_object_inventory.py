#!/usr/bin/env python3

import argparse
import collections
import subprocess
import sys


def parse_fields(line):
    fields = {}
    for part in line.strip().split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    return fields


def parse_bool(value):
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(value)


def parse_int(row, field):
    try:
        return int(row[field], 0)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid or missing {field!r} in row: {row}") from exc


def main():
    parser = argparse.ArgumentParser(
        description="Validate a code-object inventory through the rocfuzz inspector."
    )
    parser.add_argument("--inspector", required=True,
                        help="Path to rocjitsu_afl_inspect_code_object_image")
    parser.add_argument("--summary-label", default="code-object-inventory")
    parser.add_argument("--min-files", type=int, default=1)
    parser.add_argument("--require-top-level-ccob", action="store_true")
    parser.add_argument("--require-any-top-level-ccob", action="store_true")
    parser.add_argument("--require-device-images", action="store_true")
    parser.add_argument("--require-single-device-image", action="store_true")
    parser.add_argument("--require-any-multi-payload", action="store_true")
    parser.add_argument("--print-details", action="store_true")
    parser.add_argument("code_objects", nargs="+")
    args = parser.parse_args()

    if len(args.code_objects) < args.min_files:
        print(
            f"{args.summary_label}: expected at least {args.min_files} files, "
            f"got {len(args.code_objects)}",
            file=sys.stderr,
        )
        return 1

    proc = subprocess.run(
        [args.inspector, *args.code_objects],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if args.print_details and proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        print(
            f"{args.summary_label}: inspector failed with exit code {proc.returncode}",
            file=sys.stderr,
        )
        return proc.returncode

    rows = []
    targets = collections.Counter()
    current_path = None
    for line in proc.stdout.splitlines():
        if line.startswith("path="):
            row = parse_fields(line)
            rows.append(row)
            current_path = row.get("path")
            continue
        stripped = line.strip()
        if stripped.startswith("device[") and current_path is not None:
            fields = parse_fields(stripped)
            target = fields.get("target_id", "-")
            if target:
                targets[target] += 1

    if len(rows) != len(args.code_objects):
        print(
            f"{args.summary_label}: inspector produced {len(rows)} summary rows "
            f"for {len(args.code_objects)} inputs",
            file=sys.stderr,
        )
        return 1

    top_level_ccob = 0
    top_level_raw_elf = 0
    top_level_bundle = 0
    zero_payload = 0
    single_payload = 0
    multi_payload = 0
    device_images = 0
    for row in rows:
        try:
            row_is_raw_elf = parse_bool(row["top_level_raw_elf"])
            row_is_bundle = parse_bool(row["top_level_bundle"])
            row_is_ccob = parse_bool(row["top_level_ccob"])
            count = parse_int(row, "device_images")
        except (KeyError, ValueError) as exc:
            print(f"{args.summary_label}: malformed row: {exc}", file=sys.stderr)
            return 1

        if row_is_raw_elf:
            top_level_raw_elf += 1
        if row_is_bundle:
            top_level_bundle += 1
        if row_is_ccob:
            top_level_ccob += 1
        if count == 0:
            zero_payload += 1
        elif count == 1:
            single_payload += 1
        else:
            multi_payload += 1
        device_images += count

        if args.require_top_level_ccob and not row_is_ccob:
            print(
                f"{args.summary_label}: non-CCOB input: {row.get('path', '<unknown>')}",
                file=sys.stderr,
            )
            return 1
        if args.require_device_images and count == 0:
            print(
                f"{args.summary_label}: no device images: {row.get('path', '<unknown>')}",
                file=sys.stderr,
            )
            return 1
        if args.require_single_device_image and count != 1:
            print(
                f"{args.summary_label}: expected one device image, got {count}: "
                f"{row.get('path', '<unknown>')}",
                file=sys.stderr,
            )
            return 1

    target_summary = ",".join(f"{target}:{count}" for target, count in sorted(targets.items()))
    print(
        f"{args.summary_label}: files={len(rows)} top_level_raw_elf={top_level_raw_elf} "
        f"top_level_bundle={top_level_bundle} top_level_ccob={top_level_ccob} "
        f"device_images={device_images} single_payload={single_payload} "
        f"multi_payload={multi_payload} zero_payload={zero_payload} targets={target_summary}"
    )
    if multi_payload == 0:
        print(
            f"{args.summary_label}: no multi-payload CCOB found; real multi-payload/RDC "
            "rebuild validation remains open"
        )

    if args.require_any_multi_payload and multi_payload == 0:
        print(
            f"{args.summary_label}: expected at least one multi-payload CCOB",
            file=sys.stderr,
        )
        return 1
    if args.require_any_top_level_ccob and top_level_ccob == 0:
        print(
            f"{args.summary_label}: expected at least one top-level CCOB",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
