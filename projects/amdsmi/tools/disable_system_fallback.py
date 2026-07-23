#!/usr/bin/env python3
"""Disable the loader's system-library fallback in a staged wheel wrapper.

The committed ``amdsmi_wrapper.py`` ships with
``_AMDSMI_ALLOW_SYSTEM_FALLBACK = True`` -- the value kept by the system
rpm/deb, which load ``libamd_smi.so`` via the SONAME. The pip wheel bundles its
own ``libamd_smi_python.so`` and must never fall back to a system library, so
the wheel build flips the flag to ``False`` in its staged copy before
``pip wheel`` zips it. Invoked from ``py-interface/CMakeLists.txt``'s
``python_wheel`` target.
"""

import pathlib
import sys
from typing import List

ANCHOR = "_AMDSMI_ALLOW_SYSTEM_FALLBACK = True"
REPLACEMENT = "_AMDSMI_ALLOW_SYSTEM_FALLBACK = False"


def main(argv: List[str]) -> None:
    if len(argv) != 2:
        sys.exit(f"usage: {argv[0]} <staged amdsmi_wrapper.py>")
    path = pathlib.Path(argv[1])
    text = path.read_text()
    count = text.count(ANCHOR)
    if count != 1:
        sys.exit(
            f"expected exactly one '{ANCHOR}' in {path}, found {count}; "
            "refusing to build a wheel with an ambiguous loader fallback flag"
        )
    path.write_text(text.replace(ANCHOR, REPLACEMENT, 1))
    print(f"[disable_system_fallback] {path}: system library fallback disabled for wheel")


if __name__ == "__main__":
    main(sys.argv)
