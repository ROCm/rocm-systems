"""
Common utilities for artifact manipulation.

This module provides reusable utilities for working with TheRock artifacts,
including manifest handling, directory traversal, and file classification.
"""

import os
import subprocess
from pathlib import Path
from typing import Iterator, Tuple, Callable, Optional

from rocm_kpack.binutils import Toolchain


def read_artifact_manifest(artifact_dir: Path) -> list[str]:
    """
    Read the artifact manifest file from a TheRock artifact directory.

    The artifact_manifest.txt file format is a simple text file with one prefix
    per line. Each prefix represents a directory path relative to the artifact
    root directory. Empty lines are ignored.

    Example artifact_manifest.txt:
        math-libs/BLAS/rocBLAS/stage
        math-libs/BLAS/hipBLASLt/stage
        kpack/stage

    Args:
        artifact_dir: Path to artifact directory containing artifact_manifest.txt

    Returns:
        List of prefixes (directory paths) from the manifest

    Raises:
        FileNotFoundError: If artifact_manifest.txt does not exist
    """
    manifest_path = artifact_dir / "artifact_manifest.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"artifact_manifest.txt not found in {artifact_dir}")

    with open(manifest_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def write_artifact_manifest(artifact_dir: Path, prefixes: list[str]) -> None:
    """
    Write an artifact manifest file to a TheRock artifact directory.

    Args:
        artifact_dir: Path to artifact directory
        prefixes: List of prefix paths to write
    """
    manifest_path = artifact_dir / "artifact_manifest.txt"
    with open(manifest_path, "w") as f:
        for prefix in prefixes:
            f.write(f"{prefix}\n")


def scan_directory(
    root_dir: Path, predicate: Optional[Callable[[Path, os.DirEntry], bool]] = None
) -> Iterator[Tuple[Path, os.DirEntry]]:
    """
    Robustly scan a directory tree without following symlinks.

    This uses os.scandir for performance and correctly handles symlinks by not
    following them, preventing infinite loops and other issues.

    Args:
        root_dir: Root directory to scan
        predicate: Optional filter function that takes (path, direntry) and returns True to include

    Yields:
        Tuples of (absolute_path, direntry) for each file/directory found
    """

    def scan_recursive(current_dir: Path):
        with os.scandir(current_dir) as it:
            for entry in it:
                full_path = current_dir / entry.name

                # Apply predicate if provided
                if predicate and not predicate(full_path, entry):
                    continue

                yield full_path, entry

                # Recursively scan subdirectories (not following symlinks)
                if entry.is_dir(follow_symlinks=False):
                    yield from scan_recursive(full_path)

    yield from scan_recursive(root_dir)


def is_fat_binary(file_path: Path, toolchain: Toolchain) -> bool:
    """
    Check if a file is a fat binary (contains GPU device code).

    For ELF binaries, this checks for the presence of a .hip_fatbin section.
    Performs a fast ELF magic byte check before running readelf.

    Args:
        file_path: Path to the file to check
        toolchain: Toolchain instance with readelf path

    Returns:
        True if the file contains device code, False if it's not a fat binary

    Raises:
        RuntimeError: If readelf fails (corrupted file, readelf crash, etc.)
        FileNotFoundError: If file doesn't exist
    """
    # Fast check: Is file empty?
    try:
        if file_path.stat().st_size == 0:
            return False  # Empty file, definitely not a fat binary
    except FileNotFoundError:
        raise
    except OSError as e:
        raise RuntimeError(f"Cannot stat file {file_path}: {e}") from e

    # Fast check: Is this even an ELF file?
    try:
        with open(file_path, "rb") as f:
            magic = f.read(4)
            if magic != b"\x7fELF":
                return False  # Not an ELF file, definitely not a fat binary
    except FileNotFoundError:
        raise
    except OSError as e:
        raise RuntimeError(f"Cannot read file {file_path}: {e}") from e

    # It's an ELF file, now check for .hip_fatbin section
    try:
        output = subprocess.check_output(
            [str(toolchain.readelf), "-S", str(file_path)],
            stderr=subprocess.STDOUT,
            text=True,
        )
        return ".hip_fatbin" in output
    except subprocess.CalledProcessError as e:
        # readelf returns 1 for valid ELF files without sections we're looking for
        # Returns 2+ for actual errors
        if e.returncode == 1:
            return False  # Valid ELF, just no .hip_fatbin section
        raise RuntimeError(
            f"readelf failed on {file_path} with code {e.returncode}: {e.output}"
        ) from e
    except FileNotFoundError as e:
        raise RuntimeError(f"readelf not found: {toolchain.readelf}") from e


def extract_architecture_from_target(target: str) -> Optional[str]:
    """
    Extract GPU architecture from a clang target string.

    Handles both simple and "cooked" architectures (e.g., gfx942:xnack+).
    Looks for the last "--" in the target string and takes everything after it.

    Args:
        target: Target string like "hipv4-amdgcn-amd-amdhsa--gfx906"
                or "hipv4-amdgcn-amd-amdhsa--gfx942:xnack+"

    Returns:
        The architecture string (e.g., "gfx906", "gfx942:xnack+") or None if not found
    """
    if not target:
        return None

    # Find the last occurrence of "--" and take everything after it
    parts = target.rsplit("--", 1)
    if len(parts) == 2:
        return parts[1]

    return None
