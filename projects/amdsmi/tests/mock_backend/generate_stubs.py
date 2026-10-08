#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Export unsupported APIs from the same preprocessed header used by clients."""

import argparse
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    header = subprocess.check_output(
        [args.compiler, "-E", "-P", "-x", "c", str(args.header)], text=True
    )
    prototypes = re.findall(r"amdsmi_status_t\s+(amdsmi_\w+)\s*\(([^;]*?)\)\s*;", header, re.S)
    implemented = set(
        re.findall(r"amdsmi_status_t\s+(amdsmi_\w+)\s*\(", args.implementation.read_text())
    )
    declared = {name for name, _ in prototypes}
    if not declared or not implemented <= declared:
        raise RuntimeError("AMD SMI declarations do not match the mock implementation")
    text = "#include <amd_smi/amdsmi.h>\n\n"
    for name, parameters in prototypes:
        if name not in implemented:
            text += (
                f"amdsmi_status_t {name}({parameters}) {{\n"
                "  return AMDSMI_STATUS_NOT_SUPPORTED;\n}\n"
            )
    args.output.write_text(text)


if __name__ == "__main__":
    main()
