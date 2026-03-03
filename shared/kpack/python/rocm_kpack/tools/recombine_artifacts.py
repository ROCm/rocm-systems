#!/usr/bin/env python3

"""
Tool for recombining split artifacts into package groups.

This tool implements the reduce phase of the map/reduce artifact splitting
workflow. It takes split artifacts from the map phase and recombines them
according to a JSON configuration that defines package groupings.
"""

import argparse
import sys
import traceback
from pathlib import Path

from rocm_kpack.artifact_collector import ArtifactCollector
from rocm_kpack.artifact_combiner import ArtifactCombiner
from rocm_kpack.packaging_config import PackagingConfig


def main():
    """Main entry point for artifact recombination."""
    parser = argparse.ArgumentParser(
        description="Recombine split artifacts into package groups",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool implements the reduce phase of artifact splitting. It combines
generic and architecture-specific artifacts from the map phase according
to a JSON configuration that defines package groupings.

Workflow:
  1. Collect split artifacts from input directory
  2. Load packaging configuration (architecture grouping)
  3. For each package group:
     - Copy generic artifact structure
     - Add architecture-specific kpack files and databases
  4. Create output artifacts ready for downstream packaging

Exit codes:
  0 - Success
  1 - Validation or processing failed
  2 - Invalid arguments or configuration

Examples:
  # Recombine artifacts using configuration file
  %(prog)s \\
      --input-shards-dir /build/shards \\
      --config packaging.json \\
      --output-dir /build/package-groups

  # Verbose mode
  %(prog)s \\
      --input-shards-dir /build/shards \\
      --config packaging.json \\
      --output-dir /build/package-groups \\
      --verbose

Input directory structure:
  /build/shards/
    ├── gfx110X_build/        # Shard from one build job
    │   ├── blas_lib_generic/
    │   ├── blas_lib_gfx1100/
    │   └── blas_lib_gfx1101/
    └── gfx120X_build/        # Shard from another build job
        ├── blas_lib_generic/
        ├── blas_lib_gfx1200/
        └── blas_lib_gfx1201/
""",
    )

    parser.add_argument(
        "--input-shards-dir",
        type=Path,
        required=True,
        help="Directory containing shard subdirectories from map phase",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON configuration file defining package groups",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for combined artifacts",
    )

    parser.add_argument(
        "--component",
        type=str,
        help="Only process specific component (e.g., 'rocblas_lib')",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Validate arguments
    if not args.input_shards_dir.exists():
        print(
            f"Error: Input shards directory does not exist: {args.input_shards_dir}",
            file=sys.stderr,
        )
        return 2

    if not args.input_shards_dir.is_dir():
        print(
            f"Error: Input shards path is not a directory: {args.input_shards_dir}",
            file=sys.stderr,
        )
        return 2

    if not args.config.exists():
        print(
            f"Error: Configuration file does not exist: {args.config}", file=sys.stderr
        )
        return 2

    # Load configuration
    try:
        config = PackagingConfig.from_json(args.config)
    except Exception as e:
        print(f"Error: Failed to load configuration: {e}", file=sys.stderr)
        return 2

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ARTIFACT RECOMBINATION")
    print("=" * 70)
    print(f"Input shards directory: {args.input_shards_dir}")
    print(f"Primary shard: {config.primary_shard}")
    print(f"Configuration: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Collect artifacts
    print("Collecting artifacts...")
    print("-" * 70)

    collector = ArtifactCollector(
        args.input_shards_dir, config.primary_shard, verbose=args.verbose
    )

    try:
        collector.collect()
    except (ValueError, FileNotFoundError, OSError) as e:
        print(f"Error: Failed to collect artifacts: {e}", file=sys.stderr)
        return 1

    components = collector.get_component_names()

    if not components:
        print("Error: No artifacts found", file=sys.stderr)
        return 1

    print(f"Found {len(components)} component(s): {', '.join(sorted(components))}")
    print()

    # Filter to specific component if requested
    if args.component:
        if args.component not in components:
            print(
                f"Error: Component '{args.component}' not found in artifacts",
                file=sys.stderr,
            )
            print(
                f"Available components: {', '.join(sorted(components))}",
                file=sys.stderr,
            )
            return 1
        components = {args.component}

    # Create combiner
    combiner = ArtifactCombiner(collector, verbose=args.verbose)

    # Process each component
    print("Combining artifacts...")
    print("-" * 70)

    success_count = 0
    error_count = 0

    for component_name in sorted(components):
        print(f"\nProcessing component: {component_name}")

        # Process each architecture group
        for group_name, arch_group in config.architecture_groups.items():
            try:
                combiner.combine_component(
                    component_name, group_name, arch_group, args.output_dir
                )
                success_count += 1

            except (ValueError, RuntimeError, OSError) as e:
                print(
                    f"  Error combining {component_name} for group {group_name}: {e}",
                    file=sys.stderr,
                )
                error_count += 1

                if args.verbose:
                    traceback.print_exc()

    # Print summary
    print()
    print("=" * 70)
    print("RECOMBINATION SUMMARY")
    print("=" * 70)
    print(f"Successful combinations: {success_count}")
    print(f"Failed combinations: {error_count}")
    print()

    if error_count > 0:
        print("✗ RECOMBINATION FAILED")
        print(f"  {error_count} combination(s) failed")
        print()
        return 1
    else:
        print("✓ RECOMBINATION COMPLETED SUCCESSFULLY")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
