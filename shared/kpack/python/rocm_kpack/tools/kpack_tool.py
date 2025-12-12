"""Tool for inspecting and extracting from .kpack archives."""

import argparse
import sys
from pathlib import Path

from rocm_kpack.kpack import PackedKernelArchive


def cmd_list(args):
    """List contents of a kpack archive."""
    kpack_path = Path(args.kpack_file)
    if not kpack_path.exists():
        print(f"Error: {kpack_path} does not exist", file=sys.stderr)
        return 1

    archive = PackedKernelArchive.read(kpack_path)

    print(f"Kpack: {kpack_path.name}")
    print(f"  Group:       {archive.group_name}")
    print(f"  Arch family: {archive.gfx_arch_family}")
    print(f"  Binaries:    {len(archive.toc)}")
    print()

    if args.summary:
        # Just show summary stats
        total_kernels = sum(len(arches) for arches in archive.toc.values())
        total_size = sum(
            len(archive.get_kernel(binary, arch))
            for binary, arches in archive.toc.items()
            for arch in arches
        )
        print(f"Total kernels: {total_kernels}")
        print(f"Total size:    {total_size:,} bytes ({total_size / (1024**2):.2f} MB)")
        return 0

    # Detailed listing
    print(f"{'Binary Path':<60} {'Arch':<12} {'Size':>12}")
    print("-" * 85)

    for binary_path in sorted(archive.toc.keys()):
        architectures = sorted(archive.toc[binary_path])
        for i, arch in enumerate(architectures):
            kernel_data = archive.get_kernel(binary_path, arch)
            size_bytes = len(kernel_data)
            size_kb = size_bytes / 1024

            if i == 0:
                # First arch for this binary - show full path
                print(f"{binary_path:<60} {arch:<12} {size_kb:>10.1f} KB")
            else:
                # Subsequent archs - indent
                print(f"{'':60} {arch:<12} {size_kb:>10.1f} KB")

    return 0


def cmd_extract(args):
    """Extract a specific kernel from the archive."""
    kpack_path = Path(args.kpack_file)
    if not kpack_path.exists():
        print(f"Error: {kpack_path} does not exist", file=sys.stderr)
        return 1

    archive = PackedKernelArchive.read(kpack_path)

    # Normalize binary path (remove leading ./ if present)
    binary_path = args.binary_path
    if binary_path.startswith("./"):
        binary_path = binary_path[2:]

    # Check if binary exists
    if binary_path not in archive.toc:
        print(f"Error: Binary '{binary_path}' not found in archive", file=sys.stderr)
        print(f"\nAvailable binaries:", file=sys.stderr)
        for bp in sorted(archive.toc.keys())[:10]:
            print(f"  {bp}", file=sys.stderr)
        if len(archive.toc) > 10:
            print(f"  ... and {len(archive.toc) - 10} more", file=sys.stderr)
        return 1

    # Check if arch exists for this binary
    if args.arch not in archive.toc[binary_path]:
        print(
            f"Error: Architecture '{args.arch}' not found for binary '{binary_path}'",
            file=sys.stderr,
        )
        print(
            f"\nAvailable architectures: {', '.join(sorted(archive.toc[binary_path]))}",
            file=sys.stderr,
        )
        return 1

    # Extract kernel
    kernel_data = archive.get_kernel(binary_path, args.arch)
    output_path = Path(args.output)

    # Create parent directories if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_bytes(kernel_data)
    size_kb = len(kernel_data) / 1024
    print(f"Extracted: {binary_path} ({args.arch}) -> {output_path}")
    print(f"Size: {size_kb:.1f} KB")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and extract from .kpack kernel archives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all contents
  python -m rocm_kpack.tools.kpack_tool list rocm-gfx1100.kpack

  # Show summary only
  python -m rocm_kpack.tools.kpack_tool list --summary rocm-gfx1100.kpack

  # Extract a specific kernel
  python -m rocm_kpack.tools.kpack_tool extract rocm-gfx1100.kpack \\
    lib/librocblas.so.5.1 gfx1100 -o librocblas_gfx1100.hsaco
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List command
    list_parser = subparsers.add_parser("list", help="List contents of kpack archive")
    list_parser.add_argument("kpack_file", help="Path to .kpack file")
    list_parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="Show summary statistics only",
    )

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract", help="Extract a kernel from the archive"
    )
    extract_parser.add_argument("kpack_file", help="Path to .kpack file")
    extract_parser.add_argument(
        "binary_path", help="Binary path in archive (e.g., lib/librocblas.so.5.1)"
    )
    extract_parser.add_argument("arch", help="Architecture to extract (e.g., gfx1100)")
    extract_parser.add_argument(
        "-o", "--output", required=True, help="Output file path"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "list":
        return cmd_list(args)
    elif args.command == "extract":
        return cmd_extract(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
