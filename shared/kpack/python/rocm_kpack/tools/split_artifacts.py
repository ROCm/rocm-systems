#!/usr/bin/env python3

"""
Command-line tool for splitting TheRock artifacts into generic and architecture-specific components.

This tool processes TheRock build artifacts to separate host code from device code,
creating split artifacts suitable for kpack-based distribution.

Supports two modes:
1. Single artifact mode: Process one artifact with explicit component name
2. Batch mode: Process all arch-specific artifacts in a shard directory
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

from rocm_kpack.artifact_splitter import ArtifactSplitter
from rocm_kpack.binutils import Toolchain
from rocm_kpack.database_handlers import get_database_handlers, list_available_handlers


def parse_artifact_name(artifact_dir_name: str) -> Optional[str]:
    """
    Extract artifact prefix (name_component) from artifact directory name.

    Artifact directory names follow pattern: <name>_<component>_<target_family>
    where target_family is the last underscore-separated part. The only semantic
    target_family value is "generic" which indicates device-arch neutral artifacts.

    Examples:
        "blas_lib_gfx110X-dgpu" -> artifact_prefix="blas_lib", target_family="gfx110X-dgpu"
        "blas_dev_generic" -> None (skip generic artifacts)

    Args:
        artifact_dir_name: Directory name like "blas_lib_gfx110X-dgpu"

    Returns:
        Artifact prefix (name_component) like "blas_lib", or None if target_family is "generic"
    """
    parts = artifact_dir_name.split("_")

    # Need at least 2 parts: artifact prefix and target family
    if len(parts) < 2:
        return None

    # Last underscore-separated part is the target family (arch)
    target_family = parts[-1]

    # Skip generic artifacts (the only semantic target_family value)
    if target_family == "generic":
        return None

    # Artifact prefix is everything except the target family
    artifact_prefix = "_".join(parts[:-1])

    return artifact_prefix


def get_database_handlers_for_args(args) -> list:
    """
    Get database handlers based on command-line arguments.

    Args:
        args: Parsed command-line arguments with split_databases and verbose

    Returns:
        List of database handler instances
    """
    if not args.split_databases:
        if args.verbose:
            print("Database splitting disabled (use --split-databases to enable)")
        return []

    database_handlers = get_database_handlers(args.split_databases)
    if args.verbose:
        handler_names = ", ".join(h.name() for h in database_handlers)
        print(f"Database handlers enabled: {handler_names}")
    return database_handlers


def single_split(args, toolchain: Toolchain):
    """
    Process a single artifact in single mode.

    Args:
        args: Parsed command-line arguments
        toolchain: Toolchain instance

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If splitting fails
    """
    database_handlers = get_database_handlers_for_args(args)

    splitter = ArtifactSplitter(
        artifact_prefix=args.artifact_prefix,
        toolchain=toolchain,
        database_handlers=database_handlers,
        verbose=args.verbose,
    )

    print(f"Splitting artifact: {args.input_dir}")
    print(f"Artifact prefix: {args.artifact_prefix}")
    print(f"Output directory: {args.output_dir}")

    splitter.split(args.input_dir, args.output_dir)

    print("Splitting complete!")


def batch_split(args, toolchain: Toolchain):
    """
    Process all arch-specific artifacts in batch mode.

    Args:
        args: Parsed command-line arguments
        toolchain: Toolchain instance

    Raises:
        ValueError: If no artifacts found or configuration invalid
        RuntimeError: If one or more artifacts fail to split
    """
    print("=" * 70)
    print("BATCH ARTIFACT SPLITTING")
    print("=" * 70)
    print(f"Shard directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Find all artifact subdirectories
    artifact_dirs = [d for d in args.input_dir.iterdir() if d.is_dir()]

    if not artifact_dirs:
        raise ValueError(f"No subdirectories found in {args.input_dir}")

    # Get database handlers once for all artifacts
    database_handlers = get_database_handlers_for_args(args)

    total = 0
    success = 0
    skipped = 0
    failures = []

    for artifact_dir in sorted(artifact_dirs):
        # Check if it has artifact_manifest.txt
        manifest_file = artifact_dir / "artifact_manifest.txt"
        if not manifest_file.exists():
            if args.verbose:
                print(f"Skipping {artifact_dir.name}: no artifact_manifest.txt")
            skipped += 1
            continue

        # Parse artifact name to extract artifact prefix
        artifact_prefix = parse_artifact_name(artifact_dir.name)
        if artifact_prefix is None:
            if args.verbose:
                print(f"Skipping {artifact_dir.name}: target_family is 'generic'")
            skipped += 1
            continue

        total += 1
        print(
            f"[{total}] Processing: {artifact_dir.name} (artifact_prefix: {artifact_prefix})"
        )

        # Run split
        try:
            splitter = ArtifactSplitter(
                artifact_prefix=artifact_prefix,
                toolchain=toolchain,
                database_handlers=database_handlers,
                verbose=args.verbose,
            )

            splitter.split(artifact_dir, args.output_dir)
            success += 1
            print(f"    ✓ Success")

        except Exception as e:
            failures.append((artifact_dir.name, str(e)))
            print(f"    ✗ Failed: {e}", file=sys.stderr)
            if args.verbose:
                import traceback

                traceback.print_exc()

    # Print summary
    print()
    print("=" * 70)
    print("BATCH SPLITTING SUMMARY")
    print("=" * 70)
    print(f"Total artifacts found: {len(artifact_dirs)}")
    print(f"Processed: {total}")
    print(f"Successful: {success}")
    print(f"Failed: {len(failures)}")
    print(f"Skipped: {skipped}")
    print()

    if failures:
        print("✗ BATCH SPLITTING COMPLETED WITH ERRORS")
        print("\nFailed artifacts:")
        for name, error in failures:
            print(f"  - {name}: {error}")
        raise RuntimeError(f"{len(failures)} artifact(s) failed to split")
    else:
        print("✓ BATCH SPLITTING COMPLETED SUCCESSFULLY")


def main():
    """Main entry point for the artifact splitter CLI."""
    # Get list of available handlers for help text
    available_handlers = list_available_handlers()

    parser = argparse.ArgumentParser(
        description="Split TheRock artifacts into generic and architecture-specific components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Single Artifact Mode:
  # Split one artifact with explicit artifact prefix
  %(prog)s --artifact-dir artifacts/hip_lib_gfx110X --output-dir output/ --artifact-prefix hip_lib

  # Split with database handlers
  %(prog)s --artifact-dir artifacts/rocblas_lib_gfx110X --output-dir output/ \\
           --artifact-prefix rocblas_lib --split-databases rocblas hipblaslt

Batch Mode:
  # Split all arch-specific artifacts in a shard directory
  %(prog)s --batch-artifact-parent-dir /path/to/shard/gfx110X-build --output-dir output/

  # Batch mode with database handlers (applied to all artifacts)
  %(prog)s --batch-artifact-parent-dir /path/to/shard --output-dir output/ \\
           --split-databases rocblas hipblaslt

  # Batch mode with verbose output
  %(prog)s --batch-artifact-parent-dir /path/to/shard --output-dir output/ --verbose
""",
    )

    # Mutually exclusive group for input mode
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--artifact-dir",
        type=Path,
        help="Single mode: artifact directory containing artifact_manifest.txt",
    )

    input_group.add_argument(
        "--batch-artifact-parent-dir",
        type=Path,
        help="Batch mode: parent directory containing multiple artifact subdirectories",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for split artifacts",
    )

    parser.add_argument(
        "--artifact-prefix",
        type=str,
        default=None,
        help="Artifact prefix name_component (e.g., 'blas_lib' from 'blas_lib_gfx1100'). "
        "Required with --artifact-dir, forbidden with --batch-artifact-parent-dir.",
    )

    parser.add_argument(
        "--split-databases",
        nargs="+",
        choices=available_handlers,
        default=None,
        help=f"Enable database splitting for specified types. Available: {', '.join(available_handlers)}",
    )

    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help=f"Temporary directory for intermediate files (default: {tempfile.gettempdir()})",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--clang-offload-bundler",
        type=Path,
        default=None,
        help="Path to clang-offload-bundler tool (will search PATH if not specified)",
    )

    args = parser.parse_args()

    try:
        # Determine mode and validate arguments
        is_batch_mode = args.batch_artifact_parent_dir is not None

        if is_batch_mode and args.artifact_prefix:
            raise ValueError(
                "--artifact-prefix cannot be used with --batch-artifact-parent-dir"
            )

        if not is_batch_mode and not args.artifact_prefix:
            raise ValueError("--artifact-prefix is required with --artifact-dir")

        # Set input_dir based on mode for backward compatibility with single_split/batch_split
        input_dir = (
            args.batch_artifact_parent_dir if is_batch_mode else args.artifact_dir
        )

        # Validate input directory
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        if not input_dir.is_dir():
            raise ValueError(f"Input path is not a directory: {input_dir}")

        # In single mode, check for artifact_manifest.txt
        if not is_batch_mode:
            manifest_path = input_dir / "artifact_manifest.txt"
            if not manifest_path.exists():
                raise ValueError(
                    f"No artifact_manifest.txt found in {input_dir}\n"
                    f"       This doesn't appear to be a TheRock artifact directory"
                )

        # Set temporary directory environment variable for subprocess tools
        os.environ["TMPDIR"] = str(args.tmp_dir)
        if args.verbose:
            print(f"Using temporary directory: {args.tmp_dir}")

        # Create output directory if it doesn't exist
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Create toolchain once (shared across all splits in batch mode)
        toolchain = Toolchain(clang_offload_bundler=args.clang_offload_bundler)

        # Store input_dir in args for single_split/batch_split
        args.input_dir = input_dir

        # Route to appropriate mode
        if is_batch_mode:
            batch_split(args, toolchain)
        else:
            single_split(args, toolchain)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
