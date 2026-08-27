#!/usr/bin/env python3
"""Merge a timing-tuning overlay into a rocjitsu config.

rocjitsu reads one config and there is no search path, which is deliberate: a
run's timing has to be reproducible from a single artefact.  The calibrated
values are measurements against hardware and are not in this repository, so
something has to put the two together before the emulator sees them.  mirage
does it with `--timing-tuning`; this does it for a config handed to rocjitsu
directly.

The overlay only ever *sets* keys inside `timing.machine`.  It cannot add a
block, rename anything, or reach outside that object, so a bad overlay produces
a wrong number and never a different machine.
"""

from __future__ import annotations

import argparse
import json
import re
import sys


def strip_comments(text: str) -> str:
    """Drop whole-line `//` comments, which the config uses and JSON does not."""
    return re.sub(r'^\s*//.*$', '', text, flags=re.M)


def load(path: str) -> dict:
    return json.loads(strip_comments(open(path, encoding="utf-8").read()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--tuning", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    config = load(args.config)
    overlay = load(args.tuning).get("machine", {})
    if not overlay:
        print(f"{args.tuning}: no timing.machine values to merge", file=sys.stderr)
        return 2

    machine = config.setdefault("timing", {}).setdefault("machine", {})
    unknown = []
    for key, value in overlay.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            unknown.append(key)
            continue
        machine[key] = value
    if unknown:
        # Loudly, and without writing: an overlay entry the plane cannot read is
        # a calibration that silently did not apply, which is the one failure
        # mode that looks exactly like success.
        print(
            f"{args.tuning}: not numbers: {', '.join(sorted(unknown))}", file=sys.stderr
        )
        return 2

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=False)
        handle.write("\n")
    print(f"merged {len(overlay)} calibrated values into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
