"""Visitor for packing bundled binaries into .kpack archives."""

import shutil
import threading
from concurrent.futures import Executor
from pathlib import Path

from rocm_kpack.artifact_scanner import ArtifactPath, ArtifactVisitor
from rocm_kpack.binutils import BundledBinary, Toolchain
from rocm_kpack.database_handlers import DatabaseHandler
from rocm_kpack.elf import kpack_offload_binary, NotFatBinaryError
from rocm_kpack.kpack import PackedKernelArchive
from rocm_kpack.parallel import KernelInput, parallel_prepare_kernels


class PackingVisitor(ArtifactVisitor):
    """Visitor that extracts kernels to .kpack files and creates host-only binaries.

    Processes a directory tree containing bundled binaries:
    - Extracts GPU kernels to a single .kpack archive per architecture family
    - Creates host-only versions of binaries (without .hip_fatbin section)
    - Injects .rocm_kpack_ref markers linking binaries to kpack files
    - Copies opaque files verbatim

    The resulting output tree has:
    - Host-only binaries with markers in original locations
    - .kpack/ directory at root with packed kernel archives
    - All other files copied as-is

    Thread-safe: All visitor methods can be called concurrently from multiple threads.
    Fine-grained locking protects shared state (TOC updates, statistics tracking).
    """

    def __init__(
        self,
        output_root: Path,
        group_name: str,
        gfx_arch_family: str,
        gfx_arches: list[str],
        toolchain: Toolchain,
        executor: Executor | None = None,
    ):
        """Initialize packing visitor.

        Args:
            output_root: Root directory where output will be written
            group_name: Build slice name (e.g., "blas", "torch")
            gfx_arch_family: Architecture family identifier (e.g., "gfx1100", "gfx100X")
            gfx_arches: List of actual architectures in this family
            toolchain: Toolchain for binary operations
            executor: Optional Executor for parallel kernel preparation.
                     If None, kernels are prepared sequentially.
        """
        self.output_root = output_root
        self.group_name = group_name
        self.gfx_arch_family = gfx_arch_family
        self.gfx_arches = gfx_arches
        self.toolchain = toolchain
        self.executor = executor

        # Create .kpack directory
        self.kpack_dir = output_root / ".kpack"
        self.kpack_dir.mkdir(parents=True, exist_ok=True)

        # Initialize PackedKernelArchive (in-memory mode)
        self.kpack_filename = PackedKernelArchive.compute_pack_filename(
            group_name, gfx_arch_family
        )
        self.kpack_path = self.kpack_dir / self.kpack_filename
        self.archive = PackedKernelArchive(
            group_name=group_name,
            gfx_arch_family=gfx_arch_family,
            gfx_arches=gfx_arches,
        )

        # Track visited artifacts for reporting
        self.visited_opaque_files: list[Path] = []
        self.visited_bundled_binaries: list[Path] = []
        self.visited_databases: list[Path] = []

        # Lock for thread-safe access to shared state
        self._lock = threading.Lock()

    def visit_opaque_file(self, artifact_path: ArtifactPath) -> None:
        """Copy opaque file verbatim to output tree.

        Preserves symlinks rather than following them.
        Thread-safe: Can be called concurrently. Handles race conditions
        where multiple threads might try to create the same file.

        Args:
            artifact_path: Path information for the opaque file
        """
        # Track visitation (requires lock)
        with self._lock:
            self.visited_opaque_files.append(artifact_path.relative_path)

        # File operations can run in parallel (different paths)
        dest = self.output_root / artifact_path.relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Skip if destination already exists (race condition in parallel mode)
        # Another thread may have already processed this path
        if dest.exists() or dest.is_symlink():
            return

        # Preserve symlinks instead of following them
        if artifact_path.absolute_path.is_symlink():
            link_target = artifact_path.absolute_path.readlink()
            try:
                dest.symlink_to(link_target)
            except FileExistsError:
                # Race condition: another thread created it between our check and now
                pass
        else:
            try:
                shutil.copy2(artifact_path.absolute_path, dest)
            except FileExistsError:
                # Race condition: another thread created it between our check and now
                pass

    def visit_bundled_binary(
        self, artifact_path: ArtifactPath, bundled_binary: BundledBinary
    ) -> None:
        """Extract kernels to kpack and create host-only binary with marker.

        Thread-safe: Can be called concurrently. Unbundling and kernel preparation
        run in parallel; TOC updates are serialized with fine-grained locking.

        Args:
            artifact_path: Path information for the bundled binary
            bundled_binary: BundledBinary instance for the file
        """
        # Track visitation (requires lock)
        with self._lock:
            self.visited_bundled_binaries.append(artifact_path.relative_path)

        # Get list of architectures in this binary
        architectures = bundled_binary.list_bundles()

        # Extract kernels and add to kpack archive
        with bundled_binary.unbundle(delete_on_close=True) as contents:
            # Batch all kernels from this binary for parallel preparation
            kernels_to_prepare = []
            kernel_name = artifact_path.relative_path.as_posix()

            for target_name, filename in contents.target_list:
                # Only process GPU architecture bundles (.hsaco files)
                if filename.endswith(".hsaco"):
                    # Extract architecture from target name
                    # e.g., "hipv4-amdgcn-amd-amdhsa--gfx1100" -> "gfx1100"
                    if "--" in target_name:
                        arch = target_name.split("--")[-1]
                    else:
                        # Fallback: try to extract from filename
                        # e.g., "hipv4-amdgcn-amd-amdhsa--gfx1100.hsaco" -> "gfx1100"
                        arch = filename.replace(".hsaco", "").split("--")[-1]

                    # Read kernel data
                    kernel_path = contents.dest_dir / filename
                    hsaco_data = kernel_path.read_bytes()

                    # Collect for parallel preparation
                    kernels_to_prepare.append(
                        KernelInput(kernel_name, arch, hsaco_data, None)
                    )

            # Prepare all kernels sequentially (parallelism comes from concurrent binary processing)
            # Note: We must NOT use self.executor here to avoid deadlock - the scanner
            # already fills the executor with _process_path tasks, so nested executor.submit()
            # calls would block forever waiting for threads that are all busy waiting.
            # There may be secondary parallelism to exploit later, and we retain the
            # executor for that eventuality.
            prepared_kernels = parallel_prepare_kernels(
                self.archive, kernels_to_prepare, executor=None
            )

            # Add to archive sequentially (requires lock for TOC manipulation)
            with self._lock:
                for prepared in prepared_kernels:
                    self.archive.add_kernel(prepared)

        # Compute relative path from binary location to .kpack directory
        binary_depth = len(artifact_path.relative_path.parent.parts)
        if binary_depth == 0:
            # Binary is at root level
            kpack_relative_path = f".kpack/{self.kpack_filename}"
        else:
            # Binary is in subdirectory - need to go up
            kpack_relative_path = "/".join(
                [".."] * binary_depth + [".kpack", self.kpack_filename]
            )

        # Create host-only binary with kpack reference in a single pass
        # (adds .rocm_kpack_ref, maps to PT_LOAD, rewrites magic, zero-pages .hip_fatbin)
        host_only_dest = self.output_root / artifact_path.relative_path
        host_only_dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            kpack_offload_binary(
                input_path=bundled_binary.file_path,
                output_path=host_only_dest,
                kpack_search_paths=[kpack_relative_path],
                kernel_name=artifact_path.relative_path.as_posix(),
            )
        except NotFatBinaryError:
            # Binary was classified as bundled but has no .hip_fatbin
            # Just copy it as-is
            shutil.copy2(bundled_binary.file_path, host_only_dest)

    def visit_kernel_database(
        self, artifact_path: ArtifactPath, database: DatabaseHandler
    ) -> None:
        """Handle kernel database artifacts.

        Thread-safe: Can be called concurrently.
        TODO: Implement DatabaseHandlers for ad-hoc kernel libraries.
        For now, just track that we visited them.

        Args:
            artifact_path: Path information for the database
            database: Database instance
        """
        # Track visitation (requires lock)
        with self._lock:
            self.visited_databases.append(artifact_path.relative_path)
        # TODO: Inject DatabaseHandlers to process ad-hoc kernel libraries
        # For now, just track visitation

    def finalize(self) -> None:
        """Finalize packing by compressing kernels and writing kpack archive."""
        self.archive.finalize_archive()
        self.archive.write(self.kpack_path)

    def get_stats(self) -> dict[str, int]:
        """Get statistics about visited artifacts.

        Returns:
            Dictionary with counts of visited artifacts by type
        """
        return {
            "opaque_files": len(self.visited_opaque_files),
            "bundled_binaries": len(self.visited_bundled_binaries),
            "kernel_databases": len(self.visited_databases),
        }

    def __repr__(self) -> str:
        """String representation of the visitor."""
        stats = self.get_stats()
        return (
            f"PackingVisitor("
            f"group={self.group_name}, "
            f"family={self.gfx_arch_family}, "
            f"kpack={self.kpack_filename}, "
            f"opaque={stats['opaque_files']}, "
            f"bundled={stats['bundled_binaries']}, "
            f"databases={stats['kernel_databases']})"
        )
