#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Derive the total amd-smi CLI command surface by introspecting the parser.

The API-side KPI goals divide by a measured number (the APIs in amdsmi.h). The
CLI had no equivalent, so its 100% mark used to be a hand-typed constant. This
computes it instead, by walking the real argparse tree in amdsmi_parser.py.

    base    every subcommand invoked bare                        (19)
    valid   every command and feature option, once per
            output-format variant                                (base+feature) * formats
    invalid negative cases per command -- conflicting formats,
            sink flags without --file, bad enum values           commands * INVALID_PER_COMMAND

The multipliers below are the tunable part: they encode *how thoroughly* a
command should be exercised, which is a policy choice, not something the parser
can tell us. The command and option counts are measured.

    python3 projects/amdsmi/tests/cli_surface.py           # human summary
    python3 projects/amdsmi/tests/cli_surface.py --total   # just the number
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Output formats every command accepts: plain (human-readable), --json, --csv.
OUTPUT_FORMATS = 3
# Where that output can go: stdout, --file, --file --append, --file --overwrite.
OUTPUT_SINKS = 4
# Negative cases worth running per command:
#   --json --csv together, --append without --file, --overwrite without --file,
#   invalid --loglevel value, invalid device id.
INVALID_PER_COMMAND = 5

CLI_DIRS = (
    pathlib.Path("/opt/rocm/libexec/amdsmi_cli"),
    pathlib.Path(__file__).resolve().parents[1] / "amdsmi_cli",
)

# amdsmi_parser.py only registers an option group when the matching device is
# present: --cpu/--core and the CPU Arguments groups sit behind
# is_amd_hsmp_initialized(), the NIC/switch groups behind their own probes. On a
# GPU-only host that hides 134 of 412 options, so measuring the surface on the
# local machine would make the KPI denominator move from host to host. Forcing
# these on measures the product's full surface instead, which is what a goal
# should be measured against. OS/platform probes are left alone -- overriding
# those trips help strings the parser only assigns on the matching platform.
DEVICE_GATES = (
    "is_amdgpu_initialized",
    "is_amd_hsmp_initialized",
    "is_brcm_nic_initialized",
    "is_brcm_switch_initialized",
    "is_ainic_initialized",
)
# Option groups that scope a command to a device class, reported separately
# because they are what the local-host measurement silently drops.
SCOPE_GROUPS = ("Device Arguments", "CPU Arguments", "CPU Core Arguments")


class _AllDevicesPresent:
    """Helpers proxy reporting every supported device as present."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        if name in DEVICE_GATES:
            return lambda *args, **kwargs: True
        return getattr(self._real, name)


def load_parser(all_devices=True):
    """Build the full parser. '--help' in argv makes it register every subcommand."""
    for path in CLI_DIRS:
        if not (path / "amdsmi_parser.py").exists():
            continue
        sys.path.insert(0, str(path))
        try:
            from amdsmi_helpers import AMDSMIHelpers
            from amdsmi_parser import AMDSMIParser

            helpers = AMDSMIHelpers()
            if all_devices:
                helpers = _AllDevicesPresent(helpers)
            noop = [lambda *a, **k: None] * 20
            return AMDSMIParser(*noop, sys_argv=["amd-smi", "--help"], helpers=helpers)
        except Exception:  # noqa: BLE001 - try the next candidate directory
            sys.path.pop(0)
    return None


def measure(parser):
    """Return (options_per_command, global_modifiers, feature_counts, scope_counts)."""
    subparsers = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    if not subparsers:
        return {}, set(), {}, {}
    options = {}
    scopes = {}
    seen_parsers = set()
    for name, sub in subparsers[0].choices.items():
        # subparsers.add_parser(..., aliases=[...]) registers the SAME parser
        # object under each alias (e.g. "firmware"/"ucode", "monitor"/"dmon").
        # It is one command reachable by two names, not two commands with
        # independent option sets -- count it once, under whichever name
        # argparse saw first (the primary name, since aliases are added after).
        if id(sub) in seen_parsers:
            continue
        seen_parsers.add(id(sub))
        options[name] = {
            tuple(a.option_strings)
            for a in sub._actions
            if a.option_strings and not isinstance(a, argparse._HelpAction)
        }
        for group in sub._action_groups:
            if group.title not in SCOPE_GROUPS:
                continue
            count = len(
                [
                    a
                    for a in group._group_actions
                    if a.option_strings and not isinstance(a, argparse._HelpAction)
                ]
            )
            scopes[group.title] = scopes.get(group.title, 0) + count
    if not options:
        return {}, set(), {}, {}
    # Options present on every command are the global output modifiers; the rest
    # are the command's own feature surface.
    globals_ = set.intersection(*options.values())
    features = {name: len(opts - globals_) for name, opts in options.items()}
    return options, globals_, features, scopes


def compute(features):
    commands = len(features)
    feature_total = sum(features.values())
    variants = OUTPUT_FORMATS * OUTPUT_SINKS
    valid = (commands + feature_total) * variants
    invalid = commands * INVALID_PER_COMMAND
    return {
        "commands": commands,
        "feature_options": feature_total,
        "format_variants": variants,
        "valid": valid,
        "invalid": invalid,
        "total": valid + invalid,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--total", action="store_true", help="print only the total, for scripting")
    args = ap.parse_args()

    parser = load_parser(all_devices=True)
    if parser is None:
        print("error: could not import amdsmi_parser.py", file=sys.stderr)
        return 1
    _, globals_, features, scopes = measure(parser)
    if not features:
        print("error: no subcommands found in the parser", file=sys.stderr)
        return 1
    result = compute(features)

    if args.total:
        print(result["total"])
        return 0

    local = load_parser(all_devices=False)
    local_opts = 0
    if local is not None:
        _, _, local_features, _ = measure(local)
        local_opts = sum(local_features.values())

    print("amd-smi CLI surface (measured from amdsmi_parser.py, all devices present)\n")
    print(f"  base commands            {result['commands']:>6d}")
    print(f"  global output modifiers  {len(globals_):>6d}   {sorted(o[0] for o in globals_)}")
    print(f"  feature options          {result['feature_options']:>6d}")
    for title in SCOPE_GROUPS:
        if title in scopes:
            print(f"      {title:<22s} {scopes[title]:>4d}")
    if local_opts and local_opts != result["feature_options"]:
        print(
            f"\n  NOTE: this host exposes only {local_opts} feature options; "
            f"{result['feature_options'] - local_opts} are hidden because the\n"
            f"        matching device is absent (CPU/HSMP, NIC, switch). The goal is "
            f"measured\n        against the full product surface so it does not move "
            f"between hosts."
        )
    print()
    print("per-command feature options:")
    for name in sorted(features):
        print(f"  {name:<14s} {features[name]:>4d}")
    print()
    print(
        f"  format variants          {result['format_variants']:>6d}"
        f"   ({OUTPUT_FORMATS} formats x {OUTPUT_SINKS} sinks)"
    )
    print(
        f"  valid combinations       {result['valid']:>6d}"
        f"   ({result['commands']} + {result['feature_options']})"
        f" x {result['format_variants']}"
    )
    print(
        f"  invalid combinations     {result['invalid']:>6d}"
        f"   ({result['commands']} x {INVALID_PER_COMMAND})"
    )
    print(f"  {'TOTAL CLI SURFACE':<24s} {result['total']:>6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
