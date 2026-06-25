#!/usr/bin/env python3
"""
Normalize relocated PHDR layout (p_vaddr == p_offset) across a tree of ELFs.

The kpack split relocates the program header table to a trailing PT_LOAD and
``normalize_phdr_vaddr`` pins it to ``p_vaddr == p_offset`` so the binary execs
on old kernels (e.g. EL8 4.18). Any later ELF mutation that moves the trailing
table re-breaks this -- most notably ``strip`` during rpm/deb packaging, which
removes sections ahead of the table and shifts its file offset while leaving its
vaddr. This tool re-applies the same normalization as a final, post-strip step,
and (with ``--check``) doubles as a CI gate.

It reuses ``normalize_phdr_vaddr`` verbatim: that function only acts on
executables with a genuinely relocated PHDR and is a no-op otherwise, so it is
safe to run over an entire install tree.

Usage:
    python -m rocm_kpack.tools.normalize_phdr_tree <dir-or-file> [...]      # apply
    python -m rocm_kpack.tools.normalize_phdr_tree --check <dir-or-file>    # gate
"""

import argparse
import sys
from pathlib import Path
from typing import Iterator, List

from rocm_kpack.elf import ElfSurgery
from rocm_kpack.elf.phdr_manager import normalize_phdr_vaddr


def _iter_files(roots: List[str]) -> Iterator[Path]:
    for root in roots:
        p = Path(root)
        if p.is_file():
            yield p
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and not child.is_symlink():
                    yield child


def _is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Pin relocated PHDR to p_vaddr == p_offset across a tree of ELFs."
    )
    ap.add_argument("paths", nargs="+", help="ELF files or directories to scan")
    ap.add_argument(
        "--check",
        action="store_true",
        help="report non-compliant binaries and exit non-zero; do not modify",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="list compliant files too")
    args = ap.parse_args(argv)

    scanned = changed = 0
    for path in _iter_files(args.paths):
        if not _is_elf(path):
            continue
        scanned += 1
        try:
            surgery = ElfSurgery.load(path)
            # Returns True iff this binary had a relocated PHDR needing the fix.
            needs_fix = normalize_phdr_vaddr(surgery)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"WARN  {path}: {exc}", file=sys.stderr)
            continue

        if needs_fix:
            changed += 1
            if args.check:
                print(f"NONCOMPLIANT  {path}  (PHDR p_vaddr != p_offset)")
            else:
                surgery.save_preserving_mode(path, path.stat().st_mode)
                print(f"FIXED  {path}")
        elif args.verbose:
            print(f"ok    {path}")

    if args.check:
        print(f"\nScanned {scanned} ELF file(s); {changed} non-compliant.")
        return 1 if changed else 0
    print(f"\nScanned {scanned} ELF file(s); fixed {changed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
