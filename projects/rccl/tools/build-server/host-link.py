#!/usr/bin/env python3
"""Post-device-compile host integration.

Runs the host-side steps after the build server produces combined.<arch>.o:
  1. SPLIT[cobj]  - ld.lld -shared -> combined.<arch>.so
  2. SPLIT[hipfb] - clang-offload-bundler -> combined.hipfb
  3. SPLIT[host]  - host compile with embedded fat binary -> common.host.o

Then optionally triggers ninja to relink librccl.so.

Usage:
    python3 host-link.py <build_dir> [--link]
"""

import subprocess
import sys
import time
from pathlib import Path


def parse_command_line(cmd_str):
    """Split a shell command string, handling simple quoting."""
    tokens = []
    current = []
    in_quote = None
    for ch in cmd_str:
        if in_quote:
            if ch == in_quote:
                in_quote = None
            else:
                current.append(ch)
        elif ch in ('"', "'"):
            in_quote = ch
        elif ch in (' ', '\t'):
            if current:
                tokens.append(''.join(current))
                current = []
        else:
            current.append(ch)
    if current:
        tokens.append(''.join(current))
    return tokens


def extract_command(ninja_path, desc_marker):
    """Find COMMAND for a build rule identified by its DESC marker."""
    with open(ninja_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if desc_marker in line:
            for j in range(i - 1, max(i - 5, -1), -1):
                if lines[j].strip().startswith('COMMAND ='):
                    return lines[j].strip()[len('COMMAND = '):]
    return None


def run_step(name, cmd_str, build_dir):
    """Run a shell command, printing timing info."""
    tokens = parse_command_line(cmd_str)
    if tokens and tokens[0] == 'cd':
        try:
            amp = tokens.index('&&')
            tokens = tokens[amp + 1:]
        except ValueError:
            pass

    print(f"=== {name} ===")
    print(f"  {' '.join(tokens[:4])} ...")
    t0 = time.time()
    r = subprocess.run(tokens, cwd=str(build_dir))
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"  FAILED (rc={r.returncode}) in {dt:.1f}s")
        sys.exit(r.returncode)
    print(f"  OK in {dt:.1f}s")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <build_dir> [--link]", file=sys.stderr)
        sys.exit(1)

    build_dir = Path(sys.argv[1]).resolve()
    do_link = '--link' in sys.argv

    ninja_path = build_dir / 'build.ninja'
    if not ninja_path.exists():
        print(f"Error: {ninja_path} not found", file=sys.stderr)
        sys.exit(1)

    steps = [
        ("SPLIT[cobj]",  "DESC = SPLIT[cobj]"),
        ("SPLIT[hipfb]", "DESC = SPLIT[hipfb]"),
        ("SPLIT[host]",  "DESC = SPLIT[host]"),
    ]

    if do_link:
        # Let ninja handle the full chain: cobj → hipfb → host → librccl.so
        print("=== Full rebuild via ninja ===")
        t0 = time.time()
        r = subprocess.run(['ninja', 'librccl.so.1.0'], cwd=str(build_dir))
        dt = time.time() - t0
        if r.returncode != 0:
            print(f"  FAILED (rc={r.returncode}) in {dt:.1f}s")
            sys.exit(r.returncode)
        print(f"  OK in {dt:.1f}s")
        so = build_dir / 'librccl.so.1.0'
        if so.exists():
            mb = so.stat().st_size / (1024 * 1024)
            print(f"  Output: {so} ({mb:.1f} MB)")
    else:
        # Run the three post-device steps explicitly (no final link)
        for name, marker in steps:
            cmd = extract_command(str(ninja_path), marker)
            if not cmd:
                print(f"Error: cannot find {name} command in build.ninja",
                      file=sys.stderr)
                sys.exit(1)
            run_step(name, cmd, build_dir)

        print("\nHost objects ready. Run with --link to relink librccl.so.")
        host_obj = build_dir / 'split_device' / 'host_obj' / 'common.host.o'
        if host_obj.exists():
            kb = host_obj.stat().st_size / 1024
            print(f"  {host_obj} ({kb:.0f} KB)")


if __name__ == '__main__':
    main()
