#!/usr/bin/env python3

"""
Command-line tool for splitting Python wheels into host and per-arch device wheels.

Processes fat Python wheels (e.g., PyTorch with embedded HIP device code) to separate
host code from GPU device code, producing:
- A host wheel with device code stripped and kpack references injected
- One device wheel per GPU architecture containing .kpack archives
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

from rocm_kpack.binutils import Toolchain
from rocm_kpack.database_handlers import (
    WHEEL_TYPE_PRESETS,
    get_database_handlers,
    list_available_handlers,
)

try:
    from rocm_kpack.wheel_splitter import WheelSplitter, WheelSplitError
except ModuleNotFoundError as e:
    raise SystemExit(
        f"Required package not found: {e.name}\n"
        "Wheel splitting requires: pip install rocm-bootstrap"
    ) from e


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split a Python wheel into host + per-arch device wheels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Split a .whl file into .whl outputs
  %(prog)s \\
      --input torch-2.10.0+rocm7.1-cp313-cp313-manylinux_2_28_x86_64.whl \\
      --output-dir dist/ \\
      --device-package-prefix amd-torch-device \\
      --overlay-root torch/

  # Split an exploded directory, output as directories (fast iteration)
  %(prog)s \\
      --input /path/to/torch-exploded/ \\
      --output-dir dist/ \\
      --device-package-prefix amd-torch-device \\
      --overlay-root torch/ \\
      --output-format directory

  # With device wheel dependencies
  %(prog)s \\
      --input torch.whl \\
      --output-dir dist/ \\
      --device-package-prefix amd-torch-device \\
      --overlay-root torch/ \\
      --device-requires-dist "rocm-sdk-device-@GFXARCH@ == 7.1"
""",
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input .whl file or exploded wheel directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for split wheels",
    )
    parser.add_argument(
        "--device-package-prefix",
        type=str,
        required=True,
        help="Package name prefix for device wheels (e.g., 'amd-torch-device')",
    )
    parser.add_argument(
        "--overlay-root",
        type=str,
        required=True,
        help="Directory within wheel where .kpack/ is placed (e.g., 'torch/')",
    )
    parser.add_argument(
        "--device-requires-dist",
        type=str,
        nargs="*",
        default=[],
        help="Additional Requires-Dist entries for device wheels. "
        "Use @GFXARCH@ as placeholder for architecture "
        "(e.g., 'rocm-sdk-device-@GFXARCH@ == 7.1')",
    )
    parser.add_argument(
        "--output-format",
        choices=["wheel", "directory"],
        default="wheel",
        help="Output format: 'wheel' for .whl files, 'directory' for exploded dirs (default: wheel)",
    )
    preset_names = ", ".join(sorted(WHEEL_TYPE_PRESETS.keys()))
    handler_names = ", ".join(list_available_handlers())
    parser.add_argument(
        "--wheel-type",
        choices=sorted(WHEEL_TYPE_PRESETS.keys()),
        default=None,
        help=f"Preset wheel type that selects database handlers. Available: {preset_names}",
    )
    parser.add_argument(
        "--databases",
        nargs="*",
        default=None,
        help=f"Database handlers for arch-specific files. Overrides --wheel-type. "
        f"Available: {handler_names}",
    )
    parser.add_argument(
        "--compression",
        choices=["none", "zstd"],
        default="zstd",
        help="Compression scheme for kpack archives (default: zstd)",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=3,
        help="Zstd compression level, 1-22 (default: 3)",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Temporary directory for intermediate files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers for extraction and transformation (default: 1)",
    )
    parser.add_argument(
        "--generate-variant-wheel",
        action="store_true",
        help="Also generate a PEP 817 variant wheel with variant_properties markers "
        "for wheelnext-aware installers (e.g., uv with variant support)",
    )
    parser.add_argument(
        "--variant-label",
        type=str,
        default="variant",
        help="Label appended to variant wheel filename per PEP 817 "
        "(e.g., 'amd' produces ...-amd.whl). Default: variant",
    )

    Toolchain.configure_argparse(parser)

    args = parser.parse_args()

    try:
        # Validate input
        if not args.input.exists():
            raise WheelSplitError(f"Input does not exist: {args.input}")

        # Set temporary directory
        if args.tmp_dir:
            os.environ["TMPDIR"] = str(args.tmp_dir)
            if args.verbose:
                print(f"Using temporary directory: {args.tmp_dir}")

        # Create toolchain
        toolchain = Toolchain.from_args(args)

        # Resolve database handlers: --databases overrides --wheel-type
        db_handler_names = None
        if args.databases is not None:
            db_handler_names = args.databases
        elif args.wheel_type is not None:
            db_handler_names = WHEEL_TYPE_PRESETS[args.wheel_type]

        database_handlers = None
        if db_handler_names:
            database_handlers = get_database_handlers(db_handler_names)

        # Create splitter and run
        splitter = WheelSplitter(
            device_package_prefix=args.device_package_prefix,
            overlay_root=args.overlay_root,
            toolchain=toolchain,
            device_requires_dist=args.device_requires_dist,
            database_handlers=database_handlers,
            compression=args.compression,
            compression_level=args.compression_level,
            verbose=args.verbose,
            jobs=args.jobs,
            generate_variant_wheel=args.generate_variant_wheel,
            variant_label=args.variant_label,
        )

        print(f"Input: {args.input}")
        print(f"Output: {args.output_dir}")
        print(f"Device package prefix: {args.device_package_prefix}")
        print(f"Overlay root: {args.overlay_root}")
        print(f"Output format: {args.output_format}")
        if args.generate_variant_wheel:
            print(f"Variant wheel: enabled")

        result = splitter.split(args.input, args.output_dir, args.output_format)

        # Print summary
        print(f"\nSplit complete:")
        print(f"  Fat binaries processed: {result.fat_binaries_processed}")
        print(f"  Architectures: {', '.join(result.architectures_found)}")
        print(f"  Host wheel: {result.host_wheel_path}")
        if result.variant_wheel_path:
            print(f"  Variant wheel: {result.variant_wheel_path}")
        for arch, path in sorted(result.device_wheel_paths.items()):
            print(f"  Device wheel ({arch}): {path}")

        return 0

    except WheelSplitError as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
