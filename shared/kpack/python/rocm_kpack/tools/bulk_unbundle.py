"""Simple utility for unbundling all passed files.

This is presently mostly a debugging aid.

Usage:
  python -m rocm_kpack.tools.bulk_unbundle [--output-dir DIR]
      [--gfx-arch ARCH] {fat binary files...}

This can also be used for decompressing compressed code object bundle files
(i.e. CCOB), since decompressing is implicit in unpacking.
"""

import argparse
from pathlib import Path
import sys

from rocm_kpack.binutils import BundledBinary


def run(args: argparse.Namespace):
    for raw_file in args.files:
        file: Path = raw_file
        dest_dir = args.output_dir or file.with_suffix(".unbundled")
        binary = BundledBinary(file)
        with binary.unbundle(
            dest_dir=dest_dir,
            delete_on_close=False,
            gfx_arch=args.gfx_arch,
        ) as ub:
            print(f"Unbundled {dest_dir}: {', '.join(ub.file_names)}")


def main(argv: list[str]):
    p = argparse.ArgumentParser(
        description="Extract device code objects from fat binaries"
    )
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="write the contents of one input to this directory",
    )
    p.add_argument(
        "--gfx-arch",
        help="write only code objects for this base architecture (for example, gfx1250)",
    )
    args = p.parse_args(argv)
    if args.output_dir is not None and len(args.files) != 1:
        p.error("--output-dir requires exactly one input file")
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
