#!/usr/bin/env python3

"""
Artifact splitter for separating host and device code in TheRock artifacts.

This module provides functionality to split TheRock build artifacts into:
- Generic artifacts: Host-only binaries with kpack manifest references
- Architecture-specific artifacts: Device code (kpack files and kernel databases)
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
from dataclasses import dataclass

from rocm_kpack.artifact_utils import (
    extract_architecture_from_target,
    is_fat_binary,
    read_artifact_manifest,
    scan_directory,
    write_artifact_manifest,
)
from rocm_kpack.binutils import BundledBinary, Toolchain
from rocm_kpack.database_handlers import DatabaseHandler
from rocm_kpack.kpack import PackedKernelArchive
from rocm_kpack.compression import ZstdCompressor
from rocm_kpack.elf import kpack_offload_binary, NotFatBinaryError


@dataclass
class ExtractedKernel:
    """Represents a kernel extracted from a fat binary."""

    target_name: str  # Target identifier from bundler (e.g., "hip-amdgcn-amd-amdhsa-gfx1100")
    kernel_data: bytes  # The actual kernel binary data
    source_binary_relpath: str  # Path to the original fat binary relative to prefix
    source_prefix: str  # The prefix this kernel came from (e.g., "math-libs/BLAS/rocBLAS/stage")
    architecture: str  # Architecture (e.g., "gfx1100")


@dataclass
class KpackInfo:
    """Information about a created kpack file."""

    kpack_path: Path  # Path to the kpack file
    size: int  # Size of the kpack file in bytes
    kernel_count: int  # Number of kernels in the kpack


class FileClassificationVisitor:
    """Visitor that accumulates file classification results during tree scanning."""

    def __init__(
        self,
        toolchain: Toolchain,
        database_handlers: Optional[List[DatabaseHandler]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the classification visitor.

        Args:
            toolchain: Toolchain instance for binary operations
            database_handlers: List of database handler instances
            verbose: Enable verbose output
        """
        self.toolchain = toolchain
        self.database_handlers = database_handlers or []
        self.verbose = verbose

        # Accumulated results
        self.fat_binaries: List[Path] = []
        self.database_files_by_arch: Dict[
            str, List[Tuple[Path, DatabaseHandler]]
        ] = defaultdict(list)
        self.exclude_from_generic: Set[Path] = set()

    def visit_file(self, file_path: Path, prefix_path: Path) -> None:
        """
        Visit a file and classify it.

        Args:
            file_path: Path to the file
            prefix_path: Root of the prefix for relative path computation
        """
        if not file_path.is_file():
            return

        # Check if it's a fat binary
        if is_fat_binary(file_path, self.toolchain):
            self.fat_binaries.append(file_path)
            if self.verbose:
                print(f"  Found fat binary: {file_path.relative_to(prefix_path)}")
            return

        # Check database handlers
        for handler in self.database_handlers:
            arch = handler.detect(file_path, prefix_path)
            if arch:
                self.database_files_by_arch[arch].append((file_path, handler))
                self.exclude_from_generic.add(file_path)
                if self.verbose:
                    print(
                        f"  Found {handler.name()} database file for {arch}: {file_path.relative_to(prefix_path)}"
                    )
                break  # First matching handler wins

    def get_statistics(self) -> str:
        """Get a summary of classification results."""
        total_db_files = sum(
            len(files) for files in self.database_files_by_arch.values()
        )
        return (
            f"  Total: {len(self.fat_binaries)} fat binaries, "
            f"{total_db_files} database files, "
            f"{len(self.exclude_from_generic)} files to exclude from generic"
        )


class GenericCopyVisitor:
    """Visitor that copies files to generic artifact, excluding marked files."""

    def __init__(
        self,
        exclude_files: Set[Path],
        source_prefix: Path,
        dest_prefix: Path,
        verbose: bool = False,
    ):
        """
        Initialize the copy visitor.

        Args:
            exclude_files: Set of files to exclude from copying
            source_prefix: Source prefix directory
            dest_prefix: Destination prefix directory
            verbose: Enable verbose output
        """
        self.exclude_files = exclude_files
        self.source_prefix = source_prefix
        self.dest_prefix = dest_prefix
        self.verbose = verbose
        self.copied_count = 0
        self.excluded_count = 0

    def visit_file(self, file_path: Path) -> None:
        """
        Visit a file and copy it if not excluded.

        Args:
            file_path: Path to the file
        """
        if file_path in self.exclude_files:
            self.excluded_count += 1
            if self.verbose:
                rel_path = file_path.relative_to(self.source_prefix)
                print(f"    Excluding: {rel_path}")
            return

        # Copy the file preserving structure
        rel_path = file_path.relative_to(self.source_prefix)
        dest_path = self.dest_prefix / rel_path

        # Create parent directories
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle symlinks vs regular files
        if file_path.is_symlink():
            # Preserve symlink (don't follow)
            link_target = os.readlink(file_path)
            if dest_path.exists() or dest_path.is_symlink():
                dest_path.unlink()
            os.symlink(link_target, dest_path)
        else:
            # Copy regular file
            shutil.copy2(file_path, dest_path)
        self.copied_count += 1

    def get_statistics(self) -> str:
        """Get a summary of copy results."""
        return (
            f"  Copied {self.copied_count} files, excluded {self.excluded_count} files"
        )


class ArtifactSplitter:
    """Splits TheRock artifacts into generic and architecture-specific components."""

    def __init__(
        self,
        artifact_prefix: str,
        toolchain: Toolchain,
        database_handlers: Optional[List[DatabaseHandler]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the artifact splitter.

        Args:
            artifact_prefix: Artifact prefix name_component (e.g., 'hip_lib', 'blas_lib')
            toolchain: Toolchain instance for binary operations
            database_handlers: Optional list of DatabaseHandler instances for kernel databases
            verbose: Enable verbose output
        """
        self.artifact_prefix = artifact_prefix
        self.toolchain = toolchain
        self.database_handlers = database_handlers or []
        self.verbose = verbose

    def compute_kpack_search_pattern(self, binary_path: Path, prefix_root: Path) -> str:
        """
        Compute the @GFXARCH@ search pattern from a binary to its kpack files.

        The pattern uses @GFXARCH@ as a placeholder that the runtime expands
        with the requested GPU architecture at load time. This eliminates the
        need for .kpm manifest files that enumerate architectures at build time.

        Args:
            binary_path: Path to the binary
            prefix_root: Root of the prefix (where .kpack/ directory will be)

        Returns:
            Relative path pattern like .kpack/{artifact_prefix}_@GFXARCH@.kpack
        """
        # Get the relative path from prefix root to binary
        rel_path = binary_path.relative_to(prefix_root)

        # Count directory levels (excluding the binary file itself)
        depth = len(rel_path.parts) - 1

        # Build the relative path to .kpack directory with @GFXARCH@ pattern
        kpack_pattern = f".kpack/{self.artifact_prefix}_@GFXARCH@.kpack"
        if depth == 0:
            # Binary is at prefix root
            search_pattern = kpack_pattern
        else:
            # Binary is in subdirectories
            up_path = "/".join([".."] * depth)
            search_pattern = f"{up_path}/{kpack_pattern}"

        if self.verbose:
            print(f"  Binary at: {rel_path}")
            print(f"  Search pattern: {search_pattern}")

        return search_pattern

    def scan_prefix(
        self, prefix_path: Path, visitor: FileClassificationVisitor
    ) -> None:
        """
        Scan a prefix directory and classify files using the visitor.

        Args:
            prefix_path: Path to the prefix directory
            visitor: Visitor to accumulate classification results
        """
        if self.verbose:
            print(f"Scanning prefix: {prefix_path}")

        # Walk through all files in the prefix using robust directory traversal
        for file_path, direntry in scan_directory(prefix_path):
            if direntry.is_file(follow_symlinks=False):
                visitor.visit_file(file_path, prefix_path)

        if self.verbose:
            print(visitor.get_statistics())

    def copy_generic_artifact(
        self, prefix_path: Path, dest_prefix: Path, exclude_files: Set[Path]
    ) -> None:
        """
        Copy files to generic artifact, excluding marked files.

        Args:
            prefix_path: Source prefix directory
            dest_prefix: Destination prefix directory
            exclude_files: Set of files to exclude from copying
        """
        if self.verbose:
            print(f"  Creating generic artifact (excluding {len(exclude_files)} files)")

        copy_visitor = GenericCopyVisitor(
            exclude_files, prefix_path, dest_prefix, self.verbose
        )

        # Walk through all files and copy non-excluded ones using robust traversal
        for file_path, direntry in scan_directory(prefix_path):
            # Handle both regular files and symlinks
            if direntry.is_file(follow_symlinks=False) or direntry.is_symlink():
                copy_visitor.visit_file(file_path)

        if self.verbose:
            print(copy_visitor.get_statistics())

    def process_fat_binaries(
        self, fat_binaries: List[Path], prefix: str, prefix_path: Path
    ) -> Dict[str, List[ExtractedKernel]]:
        """
        Process fat binaries to extract device code.

        Args:
            fat_binaries: List of fat binary paths
            prefix: The prefix string (from artifact_manifest.txt)
            prefix_path: The actual prefix directory path

        Returns:
            Dictionary of architecture to list of ExtractedKernel objects
        """
        kernels_by_arch: Dict[str, List[ExtractedKernel]] = defaultdict(list)

        for binary_path in fat_binaries:
            if self.verbose:
                print(f"Processing fat binary: {binary_path.relative_to(prefix_path)}")

            # Create a BundledBinary instance with our toolchain
            binary = BundledBinary(binary_path, toolchain=self.toolchain)

            # Track code object index per (binary, arch) for multi-TU support
            # Each TU in a -fgpu-rdc binary gets a unique index
            code_object_index: Dict[str, int] = defaultdict(int)

            # Extract kernels using context manager
            with binary.unbundle() as unbundled:
                # Process each unbundled target
                for target_name, file_name in unbundled.target_list:
                    # Extract architecture from target name (e.g., "hip-amdgcn-amd-amdhsa-gfx1100")
                    arch = extract_architecture_from_target(target_name)
                    if arch:
                        kernel_path = unbundled.dest_dir / file_name
                        # Read kernel data while the file still exists
                        kernel_data = kernel_path.read_bytes()

                        # Build source_binary_relpath with index suffix
                        # TOC always uses indexed format: "lib/foo.so#0", "lib/foo.so#1"
                        # This matches the co_index in wrapper's reserved1 field
                        base_relpath = str(binary_path.relative_to(prefix_path))
                        index = code_object_index[arch]
                        code_object_index[arch] += 1
                        indexed_relpath = f"{base_relpath}#{index}"

                        # Create ExtractedKernel object
                        extracted_kernel = ExtractedKernel(
                            target_name=target_name,
                            kernel_data=kernel_data,
                            source_binary_relpath=indexed_relpath,
                            source_prefix=prefix,
                            architecture=arch,
                        )

                        kernels_by_arch[arch].append(extracted_kernel)
                        if self.verbose:
                            print(
                                f"    Extracted kernel for {arch}: {file_name} -> {indexed_relpath} ({len(kernel_data)} bytes)"
                            )

        return kernels_by_arch

    def create_kpack_files(
        self, all_kernels_by_arch: Dict[str, List[ExtractedKernel]], output_dir: Path
    ) -> Dict[str, KpackInfo]:
        """
        Create kpack files from all extracted kernels in architecture-specific artifacts.
        Creates a single kpack file per architecture, placed in the first prefix's .kpack directory.

        The kpack file is placed in the original prefix directory structure (not a synthetic
        "kpack/stage" prefix) so that when generic and arch-specific artifacts are overlaid
        during bootstrap, the .kpack directory merges correctly with the .kpm manifest.

        Args:
            all_kernels_by_arch: Dictionary mapping architecture to list of ALL ExtractedKernel objects from all prefixes
            output_dir: Output directory for artifacts

        Returns:
            Dict mapping architecture to KpackInfo
        """
        if not all_kernels_by_arch:
            return {}

        kpack_info_by_arch = {}

        if self.verbose:
            print(
                f"\nCreating kpack files for {len(all_kernels_by_arch)} architectures"
            )

        # Process each architecture
        for arch, kernel_list in all_kernels_by_arch.items():
            if not kernel_list:
                continue

            # Group kernels by prefix
            kernels_by_prefix: Dict[str, List[ExtractedKernel]] = defaultdict(list)
            for kernel in kernel_list:
                kernels_by_prefix[kernel.source_prefix].append(kernel)

            if self.verbose:
                print(
                    f"  Creating kpack for {arch} with {len(kernel_list)} kernels total"
                )
                for prefix, kernels in kernels_by_prefix.items():
                    print(f"    - {len(kernels)} kernels from {prefix}")

            # Create architecture-specific artifact directory
            arch_artifact_name = f"{self.artifact_prefix}_{arch}"
            arch_artifact_dir = output_dir / arch_artifact_name

            # Use the first prefix as the location for the kpack file.
            # This preserves the original directory structure so that when
            # generic and arch-specific artifacts are overlaid, the .kpack/
            # directory containing both .kpm (generic) and .kpack (arch) merge.
            first_prefix = sorted(kernels_by_prefix.keys())[0]
            kpack_prefix_dir = arch_artifact_dir / first_prefix

            # Create .kpack directory (hidden)
            kpack_dir = kpack_prefix_dir / ".kpack"
            kpack_dir.mkdir(parents=True, exist_ok=True)

            # Create kpack archive - single file for all kernels from all prefixes
            kpack_file = kpack_dir / f"{self.artifact_prefix}_{arch}.kpack"

            # Create PackedKernelArchive instance
            # Use the specific architecture directly, no family grouping
            archive = PackedKernelArchive(
                group_name=self.artifact_prefix,
                gfx_arch_family=arch,  # Use specific arch, not family
                gfx_arches=[arch],
            )

            # Add kernels to archive
            for kernel in kernel_list:
                # Build the full relative path including the source prefix
                # This allows the runtime to find the kernel for the right binary
                relative_path = f"{kernel.source_prefix}/{kernel.source_binary_relpath}"

                # Prepare and add kernel
                prepared = archive.prepare_kernel(
                    relative_path=relative_path,
                    gfx_arch=kernel.architecture,
                    hsaco_data=kernel.kernel_data,
                    metadata={
                        "target": kernel.target_name,
                        "source_prefix": kernel.source_prefix,
                    },
                )
                archive.add_kernel(prepared)

            # Finalize and write archive
            archive.finalize_archive()
            archive.write(kpack_file)

            # Validate kpack file was created successfully
            if not kpack_file.exists():
                raise RuntimeError(f"Failed to create kpack file: {kpack_file}")

            kpack_size = kpack_file.stat().st_size
            if kpack_size == 0:
                raise RuntimeError(f"Kpack file is empty: {kpack_file}")

            kernel_count = len(kernel_list)
            kpack_info_by_arch[arch] = KpackInfo(
                kpack_path=kpack_file, size=kpack_size, kernel_count=kernel_count
            )

            if self.verbose:
                print(f"    Written: {kpack_file.relative_to(kpack_prefix_dir)}")
                print(f"    Size: {kpack_size} bytes")

            # Update or create artifact manifest for this architecture artifact
            # Include all prefixes that contributed kernels
            manifest_path = arch_artifact_dir / "artifact_manifest.txt"
            existing_prefixes = []
            if manifest_path.exists():
                with open(manifest_path, "r") as f:
                    existing_prefixes = [line.strip() for line in f if line.strip()]

            # Add all prefixes that had kernels
            for prefix in kernels_by_prefix.keys():
                if prefix not in existing_prefixes:
                    existing_prefixes.append(prefix)

            write_artifact_manifest(arch_artifact_dir, sorted(existing_prefixes))

        return kpack_info_by_arch

    def inject_kpack_references(
        self,
        fat_binaries_by_prefix: Dict[str, List[Path]],
        generic_artifact_dir: Path,
        kpack_info_by_arch: Dict[str, KpackInfo],
    ):
        """
        Inject kpack search patterns into fat binaries and strip device code.

        Embeds @GFXARCH@ patterns in each binary's .rocm_kpack_ref section.
        The runtime expands @GFXARCH@ with the requested GPU architecture at
        load time, eliminating the need for .kpm manifest files.

        Args:
            fat_binaries_by_prefix: Dictionary mapping prefix to list of fat binary paths
            generic_artifact_dir: Path to generic artifact directory
            kpack_info_by_arch: Dict mapping arch to KpackInfo
        """
        if self.verbose:
            print("\nInjecting kpack search patterns and stripping device code")

        for prefix, binary_paths in fat_binaries_by_prefix.items():
            prefix_dir = generic_artifact_dir / prefix

            # Inject kpack search pattern in each binary
            for binary_path in binary_paths:
                # Compute @GFXARCH@ search pattern from binary to .kpack directory
                search_pattern = self.compute_kpack_search_pattern(
                    binary_path, prefix_dir
                )

                if self.verbose:
                    print(
                        f"  Processing {binary_path.relative_to(generic_artifact_dir)}"
                    )
                    print(f"    Search pattern: {search_pattern}")

                # Create temporary output file
                temp_output = binary_path.with_suffix(binary_path.suffix + ".kpacked")

                # Record original size for validation
                original_size = binary_path.stat().st_size

                try:
                    # Compute kernel_name to match TOC key format (without index):
                    # "{prefix}/{binary_relpath}" e.g. "rocrand/lib/librocrand.so.1.1"
                    # Runtime appends "#N" based on wrapper's reserved1 (co_index)
                    binary_relpath = binary_path.relative_to(prefix_dir)
                    kernel_name = f"{prefix}/{binary_relpath}"

                    # Add kpack search pattern and transform binary in one pass
                    # (adds .rocm_kpack_ref, maps to PT_LOAD, rewrites magic, zero-pages .hip_fatbin)
                    kpack_offload_binary(
                        input_path=binary_path,
                        output_path=temp_output,
                        kpack_search_paths=[search_pattern],
                        kernel_name=kernel_name,
                        verbose=self.verbose,
                    )

                    # Remove original first to break hard links, then move temp
                    # (mode already preserved by kpack_offload_binary)
                    binary_path.unlink()
                    shutil.move(temp_output, binary_path)

                    # Validate output was created
                    if not binary_path.exists():
                        raise RuntimeError(
                            f"Binary disappeared after stripping: {binary_path}"
                        )

                    new_size = binary_path.stat().st_size
                    # Note: Size may grow for small .hip_fatbin sections where
                    # structural overhead (padding, PHDR relocation) exceeds
                    # zero-page savings. This is acceptable for small binaries.
                    if self.verbose:
                        size_delta = new_size - original_size
                        if size_delta < 0:
                            print(
                                f"    Device code stripped, new size: {new_size} bytes "
                                f"(saved {-size_delta} bytes)"
                            )
                        else:
                            print(
                                f"    Processed, new size: {new_size} bytes "
                                f"(+{size_delta} bytes overhead)"
                            )

                except NotFatBinaryError:
                    # Binary has no device code - skip silently
                    if self.verbose:
                        print(f"    Skipping (no device code)")
                    continue

                finally:
                    # Always clean up temp file
                    if temp_output.exists():
                        temp_output.unlink()

    def process_database_files(
        self,
        database_files_by_arch: Dict[str, List[Tuple[Path, DatabaseHandler]]],
        prefix: str,
        prefix_path: Path,
        output_dir: Path,
    ):
        """
        Move kernel database files to architecture-specific artifacts.

        Args:
            database_files_by_arch: Dictionary of (file_path, handler) tuples by architecture
            prefix: The prefix string (from artifact_manifest.txt)
            prefix_path: The actual prefix directory path
            output_dir: Output directory for artifacts
        """
        for arch, file_handler_pairs in database_files_by_arch.items():
            # Create architecture-specific artifact directory
            arch_artifact_name = f"{self.artifact_prefix}_{arch}"
            arch_artifact_dir = output_dir / arch_artifact_name
            arch_prefix_dir = arch_artifact_dir / prefix

            if self.verbose:
                print(
                    f"  Moving {len(file_handler_pairs)} database files to {arch_artifact_name}"
                )

            for file_path, handler in file_handler_pairs:
                # Compute destination path preserving structure
                rel_path = file_path.relative_to(prefix_path)
                dest_path = arch_prefix_dir / rel_path

                # Create parent directories
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy the file (will move after generic is created)
                if self.verbose:
                    print(f"    Moving: {rel_path}")
                shutil.copy2(file_path, dest_path)

            # Update or create artifact manifest for this architecture artifact
            manifest_path = arch_artifact_dir / "artifact_manifest.txt"
            existing_prefixes = []
            if manifest_path.exists():
                with open(manifest_path, "r") as f:
                    existing_prefixes = [line.strip() for line in f if line.strip()]

            # Add current prefix if not already present
            if prefix not in existing_prefixes:
                existing_prefixes.append(prefix)
                write_artifact_manifest(arch_artifact_dir, existing_prefixes)

    def split(self, input_dir: Path, output_dir: Path):
        """
        Split an artifact directory into generic and architecture-specific components.

        Args:
            input_dir: Input artifact directory
            output_dir: Output directory for split artifacts
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read artifact manifest
        prefixes = read_artifact_manifest(input_dir)
        if self.verbose:
            print(f"Found {len(prefixes)} prefixes in artifact manifest:")
            for prefix in prefixes:
                print(f"  - {prefix}")

        # Accumulate all kernels from all prefixes
        all_kernels_by_arch: Dict[str, List[ExtractedKernel]] = defaultdict(list)

        # Track fat binaries by prefix for later processing
        fat_binaries_by_prefix: Dict[str, List[Path]] = {}

        # Track prefixes that were actually processed (for manifest)
        processed_prefixes: List[str] = []

        # Process each prefix
        for prefix in prefixes:
            prefix_path = input_dir / prefix

            if not prefix_path.exists():
                # Skip empty prefixes (directories that had no files may not be created in artifact)
                if self.verbose:
                    print(f"\nSkipping prefix (directory does not exist): {prefix}")
                continue

            if self.verbose:
                print(f"\nProcessing prefix: {prefix}")

            # Phase 1: Classify files using visitor
            classifier = FileClassificationVisitor(
                self.toolchain, self.database_handlers, self.verbose
            )
            self.scan_prefix(prefix_path, classifier)

            # Phase 2: Process database files (move to arch-specific artifacts)
            if self.database_handlers and classifier.database_files_by_arch:
                self.process_database_files(
                    classifier.database_files_by_arch, prefix, prefix_path, output_dir
                )

            # Phase 3: Create generic artifact (excluding database files)
            generic_artifact_name = f"{self.artifact_prefix}_generic"
            generic_artifact_dir = output_dir / generic_artifact_name
            generic_prefix_dir = generic_artifact_dir / prefix

            if self.verbose:
                print(f"Creating generic artifact: {generic_artifact_name}")

            # Copy files excluding those marked for exclusion
            self.copy_generic_artifact(
                prefix_path, generic_prefix_dir, classifier.exclude_from_generic
            )

            # Track this prefix for the manifest
            processed_prefixes.append(prefix)

            # Phase 4: Process fat binaries and accumulate kernels
            if classifier.fat_binaries:
                kernels_by_arch = self.process_fat_binaries(
                    classifier.fat_binaries, prefix, prefix_path
                )

                # Accumulate kernels from this prefix
                for arch, kernels in kernels_by_arch.items():
                    all_kernels_by_arch[arch].extend(kernels)

                # Track fat binaries in the generic artifact for later processing
                fat_binaries_in_generic = []
                for binary_path in classifier.fat_binaries:
                    generic_binary_path = generic_prefix_dir / binary_path.relative_to(
                        prefix_path
                    )
                    if generic_binary_path.exists():
                        fat_binaries_in_generic.append(generic_binary_path)

                if fat_binaries_in_generic:
                    fat_binaries_by_prefix[prefix] = fat_binaries_in_generic

        # Phase 5: Create kpack files from all accumulated kernels
        kpack_info_by_arch = {}
        if all_kernels_by_arch:
            kpack_info_by_arch = self.create_kpack_files(
                all_kernels_by_arch, output_dir
            )

        # Phase 6: Inject kpack references and strip device code from fat binaries
        if fat_binaries_by_prefix and kpack_info_by_arch:
            generic_artifact_dir = output_dir / f"{self.artifact_prefix}_generic"
            self.inject_kpack_references(
                fat_binaries_by_prefix, generic_artifact_dir, kpack_info_by_arch
            )

        # Phase 7: Write artifact manifest for generic artifact with all prefixes
        # Always write manifest, even if empty (ensures downstream tools find it)
        generic_artifact_dir = output_dir / f"{self.artifact_prefix}_generic"
        if not generic_artifact_dir.exists():
            generic_artifact_dir.mkdir(parents=True, exist_ok=True)
        write_artifact_manifest(generic_artifact_dir, processed_prefixes)

        if self.verbose:
            print(f"\nSplitting complete. Output in: {output_dir}")
