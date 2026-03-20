"""
Common utilities for artifact manipulation.

This module provides reusable utilities for working with TheRock artifacts,
including manifest handling, directory traversal, and file classification.
"""

import os
from pathlib import Path
from typing import Iterator, Tuple, Callable, Optional

from rocm_kpack.binutils import Toolchain
from rocm_kpack.coff.surgery import CoffSurgery
from rocm_kpack.elf.surgery import ElfSurgery
from rocm_kpack.format_detect import detect_binary_format, UnsupportedBinaryFormat


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

    For ELF binaries, checks for .hip_fatbin section.
    For PE/COFF binaries, checks for .hip_fat section.

    Args:
        file_path: Path to the file to check
        toolchain: Toolchain instance with readelf path

    Returns:
        True if the file contains device code, False if it's not a fat binary

    Raises:
        RuntimeError: If binary analysis fails
        FileNotFoundError: If file doesn't exist
    """
    # Fast check: Is file empty?
    try:
        if file_path.stat().st_size == 0:
            return False
    except FileNotFoundError:
        raise
    except OSError as e:
        raise RuntimeError(f"Cannot stat file {file_path}: {e}") from e

    try:
        fmt = detect_binary_format(file_path)
    except UnsupportedBinaryFormat:
        return False

    if fmt == "elf":
        surgery = ElfSurgery.load(file_path)
        return surgery.find_section(".hip_fatbin") is not None
    else:
        surgery = CoffSurgery.load(file_path)
        return surgery.find_section(".hip_fat") is not None


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
