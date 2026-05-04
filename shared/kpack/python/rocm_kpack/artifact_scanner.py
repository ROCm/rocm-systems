"""Artifact scanner for walking directory trees and categorizing ROCm artifacts.

This module provides infrastructure for scanning ROCm installation trees and
identifying different categories of artifacts:
- Opaque files (to be copied verbatim)
- Bundled binaries (fat binaries with multi-arch device code)
- Kernel databases (library-specific kernel collections)
"""

import subprocess
from abc import ABC, abstractmethod
from concurrent.futures import Executor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rocm_kpack.binutils import BundledBinary, Toolchain


@dataclass(frozen=True)
class ArtifactPath:
    """Represents a path relative to a scan root.

    This dual representation allows consumers to work with relative paths
    (for materializing altered copies) while still having access to the
    absolute path for reading the original files.

    Attributes:
        root_dir: The root directory being scanned
        relative_path: Path relative to root_dir
    """

    root_dir: Path
    relative_path: Path

    @property
    def absolute_path(self) -> Path:
        """Compute the absolute path by joining root_dir and relative_path."""
        return self.root_dir / self.relative_path


class KernelArtifact:
    """Represents a single kernel artifact within a database.

    Attributes:
        relative_path: Path relative to the database root
        gfx_target: GPU architecture (e.g., 'gfx1100', 'gfx1201')
        artifact_type: Type of artifact ('code_object', 'metadata', 'hsaco', etc.)
    """

    def __init__(self, relative_path: Path, gfx_target: str, artifact_type: str):
        self.relative_path = relative_path
        self.gfx_target = gfx_target
        self.artifact_type = artifact_type

    def __repr__(self):
        return (
            f"KernelArtifact(relative_path={self.relative_path}, "
            f"gfx_target={self.gfx_target}, artifact_type={self.artifact_type})"
        )


class KernelDatabase(ABC):
    """Base class for different kernel database formats.

    A kernel database represents a collection of GPU kernels organized by
    architecture. Different ROCm libraries use different formats (paired files,
    SQLite databases, etc.).
    """

    def __init__(self, artifact_path: ArtifactPath):
        self.artifact_path = artifact_path

    @property
    def root_dir(self) -> Path:
        """Scan root directory."""
        return self.artifact_path.root_dir

    @property
    def relative_path(self) -> Path:
        """Path relative to scan root."""
        return self.artifact_path.relative_path

    @property
    def absolute_path(self) -> Path:
        """Absolute path to database root."""
        return self.artifact_path.absolute_path

    @abstractmethod
    def get_architectures(self) -> list[str]:
        """Get list of gfx targets (e.g., ['gfx1100', 'gfx1101'])."""

    @abstractmethod
    def get_kernel_artifacts(self) -> Iterator[KernelArtifact]:
        """Iterate over kernel artifacts (paths relative to database root)."""


class ArtifactVisitor(ABC):
    """Visitor interface for processing discovered artifacts.

    Implement this interface to handle different categories of artifacts
    discovered during tree scanning.
    """

    def visit_opaque_file(self, artifact_path: ArtifactPath) -> None:
        """Called for files that should be copied verbatim.

        Args:
            artifact_path: Contains root_dir and relative_path
        """
        pass

    def visit_bundled_binary(
        self, artifact_path: ArtifactPath, bundled_binary: BundledBinary
    ) -> None:
        """Called for fat binaries with bundled single-arch hsaco files.

        Args:
            artifact_path: Location of the bundled binary
            bundled_binary: Parsed binary object
        """
        pass

    def visit_kernel_database(
        self, artifact_path: ArtifactPath, database: KernelDatabase
    ) -> None:
        """Called for recognized kernel database structures.

        Args:
            artifact_path: Root path of the database (e.g., lib/hipblaslt/library)
            database: Parsed database object
        """
        pass


class DatabaseRecognizer(ABC):
    """Plugin interface for recognizing kernel database formats.

    Implement this to add support for new kernel database formats.
    """

    @abstractmethod
    def can_recognize(self, artifact_path: ArtifactPath) -> bool:
        """Quick check if this path might be a database of this type.

        This should be a fast heuristic check (e.g., checking file extensions
        or directory names) to avoid expensive parsing operations.

        Args:
            artifact_path: Path to check

        Returns:
            True if this recognizer might handle this path
        """

    @abstractmethod
    def recognize(self, artifact_path: ArtifactPath) -> KernelDatabase | None:
        """Parse and return the database, or None if not recognized.

        This is called after can_recognize returns True and should perform
        more thorough validation and parsing.

        Args:
            artifact_path: Path to parse

        Returns:
            KernelDatabase instance if recognized, None otherwise
        """


class RecognizerRegistry:
    """Registry of database recognizers.

    Maintains a list of recognizers and tries them in order until one succeeds.
    """

    def __init__(self):
        self.recognizers: list[DatabaseRecognizer] = []

    def register(self, recognizer: DatabaseRecognizer) -> None:
        """Register a new recognizer plugin.

        Recognizers are tried in registration order.

        Args:
            recognizer: The recognizer to register
        """
        self.recognizers.append(recognizer)

    def try_recognize(self, artifact_path: ArtifactPath) -> KernelDatabase | None:
        """Try all recognizers until one succeeds.

        Args:
            artifact_path: Path to attempt recognition on

        Returns:
            KernelDatabase instance if recognized, None otherwise
        """
        for recognizer in self.recognizers:
            if recognizer.can_recognize(artifact_path):
                result = recognizer.recognize(artifact_path)
                if result:
                    return result
        return None


class ArtifactScanner:
    """Scans a directory tree and categorizes artifacts.

    Uses a registry of recognizers to identify kernel databases, and falls back
    to treating files as opaque or bundled binaries.

    Supports parallel scanning when provided with an executor.
    """

    def __init__(
        self,
        recognizer_registry: RecognizerRegistry,
        toolchain: Toolchain | None = None,
        executor: Executor | None = None,
    ):
        """Initialize the scanner.

        Args:
            recognizer_registry: Registry of database recognizers
            toolchain: Toolchain for bundled binary operations (optional)
            executor: Executor for parallel scanning (optional, default: sequential)
        """
        self.registry = recognizer_registry
        self.toolchain = toolchain
        self.executor = executor
        # Track relative paths of visited databases to avoid double-visiting
        self._visited_database_paths: set[Path] = set()

    def scan_tree(self, root_dir: Path, visitor: ArtifactVisitor) -> None:
        """Walk the tree and invoke visitor callbacks.

        If an executor was provided, processes paths in parallel.
        Otherwise, processes sequentially.

        Paths are streamed incrementally - processing begins immediately
        as paths are discovered, rather than collecting all paths upfront.

        Args:
            root_dir: Root directory to scan (stored as root for all ArtifactPaths)
            visitor: Visitor to receive callbacks (must be thread-safe if using executor)
        """
        if self.executor is None:
            # Sequential processing: process as we walk
            for abs_path in self._walk_tree(root_dir):
                relative_path = abs_path.relative_to(root_dir)
                artifact_path = ArtifactPath(root_dir, relative_path)
                self._process_path(artifact_path, visitor)
        else:
            # Parallel processing: submit to executor as we walk
            futures = []
            for abs_path in self._walk_tree(root_dir):
                relative_path = abs_path.relative_to(root_dir)
                artifact_path = ArtifactPath(root_dir, relative_path)
                future = self.executor.submit(
                    self._process_path, artifact_path, visitor
                )
                futures.append(future)

            # Wait for all to complete (will raise exceptions if any occurred)
            for future in as_completed(futures):
                future.result()  # Propagate exceptions

    def _walk_tree(self, root_dir: Path) -> Iterator[Path]:
        """Walk the directory tree, yielding all paths.

        Args:
            root_dir: Root directory to walk

        Yields:
            Absolute paths to all files and directories
        """
        # Use sorted for deterministic ordering in tests
        for path in sorted(root_dir.rglob("*")):
            yield path

    def _process_path(
        self, artifact_path: ArtifactPath, visitor: ArtifactVisitor
    ) -> None:
        """Process a single path and invoke appropriate visitor callback.

        Args:
            artifact_path: Path to process
            visitor: Visitor to invoke
        """
        # Skip if already visited as part of a database
        if any(
            artifact_path.relative_path.is_relative_to(db_rel_path)
            for db_rel_path in self._visited_database_paths
        ):
            return

        # Try database recognition (directories only)
        if artifact_path.absolute_path.is_dir():
            database = self.registry.try_recognize(artifact_path)
            if database:
                visitor.visit_kernel_database(artifact_path, database)
                self._visited_database_paths.add(artifact_path.relative_path)
                return

        # Try bundled binary detection
        if (
            artifact_path.absolute_path.is_file()
            and self.toolchain
            and self._is_bundled_binary(artifact_path)
        ):
            bb = BundledBinary(artifact_path.absolute_path, toolchain=self.toolchain)
            visitor.visit_bundled_binary(artifact_path, bb)
            return

        # Default: opaque file
        if artifact_path.absolute_path.is_file():
            visitor.visit_opaque_file(artifact_path)

    def _is_bundled_binary(self, artifact_path: ArtifactPath) -> bool:
        """Check if a file is a bundled binary with device code.

        Detects ELF binaries (executables and shared libraries) that contain
        bundled GPU device code in .hip_fatbin sections.

        TODO: Add support for Windows PE/COFF binaries. This currently only
        handles ELF binaries using readelf. For COFF binaries, we'll need
        to use a different approach (e.g., llvm-objdump or parse PE headers).
        The repackaging tooling will run on Linux but needs to process both
        Linux ELF and Windows COFF binaries.

        Args:
            artifact_path: Path to check

        Returns:
            True if this is a bundled binary with .hip_fatbin section
        """
        file_path = artifact_path.absolute_path

        # Skip symlinks - they just point to the actual versioned .so files
        # We want to process the actual file, not the symlink
        if file_path.is_symlink():
            return False

        # Use file content introspection instead of extension checking.
        # Check ELF magic bytes to determine if this is an ELF binary.
        # This is more robust than extension checking and handles edge cases
        # like renamed files or non-standard extensions.
        #
        # ELF magic: First 4 bytes are \x7fELF
        try:
            with open(file_path, "rb") as f:
                magic = f.read(4)
                if magic != b"\x7fELF":
                    return False
        except (OSError, IOError):
            # Can't read file, skip
            return False

        # Check for .hip_fatbin ELF section using readelf
        # This only works for ELF binaries (Linux)
        try:
            result = subprocess.run(
                ["readelf", "-S", str(file_path)],
                capture_output=True,
                text=True,
                check=False,  # Don't raise on non-zero exit
            )
            # readelf succeeds for ELF files, fails for non-ELF
            if result.returncode == 0:
                return ".hip_fatbin" in result.stdout
        except FileNotFoundError:
            # readelf not available - fall through to return False
            pass

        # For non-ELF files (Windows COFF) or if readelf fails, return False for now
        # TODO: Implement COFF detection when we have Windows test assets
        return False
