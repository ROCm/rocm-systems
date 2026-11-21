#!/usr/bin/env python3
# Post-build check that librccl.so carries exactly one device image.
#
# Each registered image has one 24-byte fat binary wrapper in
# .hipFatBinSegment. With the device linker, clang-linker-wrapper links RCCL's
# device code and any other -fgpu-rdc device code on the link line (rocSHMEM)
# into one image. A second wrapper means something on the link line brought its
# own image and registration, which gives the device globals in both a second,
# separately initialized copy.
#
# Exit status: 0 one image, 1 none or more than one (or unreadable), 2 bad usage.
#
# Usage: check_single_device_image.py <path-to-librccl.so>

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_hip_fatbin_tail import ElfError, parse_elf  # noqa: E402

PREFIX = "[rccl] device image check:"
SEGMENT = ".hipFatBinSegment"
WRAPPER_SIZE = 24


def main(argv):
    if len(argv) != 2:
        print("usage: check_single_device_image.py <librccl.so>", file=sys.stderr)
        return 2
    so = argv[1]
    try:
        with open(so, "rb") as f:
            sections, _ = parse_elf(f)
    except (OSError, ElfError, struct.error) as e:
        print(f"{PREFIX} FAIL (could not parse {so}: {e})", file=sys.stderr)
        return 1
    if sections is None:
        print(f"{PREFIX} SKIP (not a 64-bit ELF)")
        return 0

    size = sum(s["size"] for s in sections if s["name"] == SEGMENT)
    images = size // WRAPPER_SIZE
    if images == 1 and size == WRAPPER_SIZE:
        print(f"{PREFIX} OK (one device image)")
        return 0
    print(f"{PREFIX} FAIL: {SEGMENT} is {size} bytes, {images} fat binary "
          f"wrappers; expected exactly one device image.", file=sys.stderr)
    print("  Something on the link line brought its own device image, for example "
          "an object or library whose device code was not compiled with -fgpu-rdc.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
