"""Simple utility for unbundling all passed files.

This is presently mostly a debugging aid.

Usage:
  python -m rocm_kpack.tools.bulk_unbundle {fat binary files...}

This can also be used for decompressing compressed code object bundle files
(i.e. CCOB), since decompressing is implicit in unpacking.
"""

import argparse
from pathlib import Path
import sys

from rocm_kpack.binutils import Toolchain, BundledBinary


def run(args: argparse.Namespace, *, toolchain: Toolchain):
    for raw_file in args.files:
        file: Path = raw_file
        dest_dir = file.with_suffix(".unbundled")
        binary = BundledBinary(file, toolchain=toolchain)
        with binary.unbundle(dest_dir=dest_dir, delete_on_close=False) as ub:
            print(f"Unbundled {dest_dir}: {', '.join(ub.file_names)}")


def main(argv: list[str]):
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    Toolchain.configure_argparse(p)
    args = p.parse_args()
    toolchain = Toolchain.from_args(args)
    run(args, toolchain=toolchain)


if __name__ == "__main__":
    main(sys.argv[1:])
