#!/usr/bin/env python3
# Post-build sanity check for the .hip_fatbin tail-placement linker script.
#
# Reads `readelf -SW <so>` (or `llvm-readelf -SW <so>`) output and confirms
# that .hip_fatbin is the section with the highest file offset among allocated
# PROGBITS sections.  This is informational only — the script never exits
# non-zero, so a layout change in a future toolchain version won't break
# the build; the worst case is a WARNING line in the build log.
#
# Usage: check_hip_fatbin_tail.py <path-to-librccl.so>

import os
import re
import shutil
import subprocess
import sys

SECTION_RE = re.compile(
    r"\s*\[\s*\d+\]\s+"      # [Nr]
    r"(\S+)\s+"               # name
    r"(\S+)\s+"               # type
    r"\S+\s+"                 # address
    r"([0-9a-fA-F]+)\s+"      # file offset (hex)
    r"\S+\s+"                 # size
    r"\S+\s+"                 # ES
    r"([A-Za-z]*)"            # flags
)


def find_readelf():
    """Locate a readelf-compatible binary.

    Prefer ROCm's bundled llvm-readelf so the toolchain that produced
    librccl.so also inspects it (it always supports -SW with the same
    output schema we parse below).  Fall back to a PATH llvm-readelf,
    and only as a last resort to system readelf, which on older
    binutils may not understand wide-mode hex offsets the same way.
    """
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    candidates = [
        os.path.join(rocm, "llvm/bin/llvm-readelf"),
        shutil.which("llvm-readelf"),
        shutil.which("readelf"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def main(argv):
    if len(argv) != 2:
        print("usage: check_hip_fatbin_tail.py <librccl.so>", file=sys.stderr)
        return 0  # informational; do not fail

    so = argv[1]
    if not os.path.exists(so):
        print(f"[rccl] hip_fatbin tail check: SKIP ({so} not found)")
        return 0

    readelf = find_readelf()
    if not readelf:
        print("[rccl] hip_fatbin tail check: SKIP (no readelf in PATH or ROCM_PATH)")
        return 0

    try:
        out = subprocess.check_output(
            [readelf, "-SW", so], stderr=subprocess.STDOUT, timeout=30
        ).decode("utf-8", errors="replace")
    except subprocess.SubprocessError as e:
        print(f"[rccl] hip_fatbin tail check: SKIP (readelf failed: {e})")
        return 0

    max_off = -1
    max_name = None
    saw_fatbin = False
    fatbin_off = None
    for line in out.splitlines():
        m = SECTION_RE.match(line)
        if not m:
            continue
        name, sect_type, off_hex, flg = m.groups()
        if sect_type != "PROGBITS":
            continue
        if "A" not in flg:
            continue
        try:
            off = int(off_hex, 16)
        except ValueError:
            continue
        if name == ".hip_fatbin":
            saw_fatbin = True
            fatbin_off = off
        if off > max_off:
            max_off = off
            max_name = name

    if not saw_fatbin:
        print("[rccl] hip_fatbin tail check: SKIP (.hip_fatbin not present)")
        return 0

    if max_name == ".hip_fatbin":
        print(
            f"[rccl] hip_fatbin tail check: OK "
            f"(.hip_fatbin at file offset 0x{fatbin_off:x}, last alloc PROGBITS)"
        )
    else:
        print(
            f"[rccl] hip_fatbin tail check: WARNING — "
            f".hip_fatbin at 0x{fatbin_off:x} but a later alloc PROGBITS exists: "
            f"{max_name!r} at 0x{max_off:x}. "
            f"Host 32-bit relocations may overflow once .hip_fatbin grows. "
            f"Check that the linker honoured cmake/linker_scripts/hip_fatbin_tail.ld."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
