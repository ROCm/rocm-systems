#!/usr/bin/env python3

"""
Database handlers for kernel database splitting.

This module provides a plugin architecture for handling different types of
kernel databases (rocBLAS, hipBLASLt, aotriton, etc.) during artifact splitting.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List


# Compile regex patterns once at module level
# TODO: Verify this pattern handles xnack (asan) kernels correctly
# These have special syntax like gfx942:xnack+ or gfx90a:sramecc+:xnack-
_GFX_ARCH_PATTERN = re.compile(r"gfx(\d+[a-z]*)")


class DatabaseHandler(ABC):
    """Base class for kernel database handlers."""

    @abstractmethod
    def name(self) -> str:
        """
        Return the handler identifier.

        Returns:
            Handler name for CLI and logging
        """
        pass

    @abstractmethod
    def detect(self, path: Path, prefix_root: Path) -> Optional[str]:
        """
        Detect if a file belongs to this database type and extract architecture.

        Args:
            path: Path to the file
            prefix_root: Root of the prefix for relative path computation

        Returns:
            Architecture string (e.g., 'gfx1100') if file matches, None otherwise
        """
        pass

    def should_move(self, path: Path) -> bool:
        """
        Determine if this file should be moved to architecture-specific artifact.

        Args:
            path: Path to the file

        Returns:
            True if file should be moved, False otherwise
        """
        # By default, if we can detect it, we should move it
        return True


class RocBLASHandler(DatabaseHandler):
    """Handler for rocBLAS Tensile library files."""

    def name(self) -> str:
        return "rocblas"

    def detect(self, path: Path, prefix_root: Path) -> Optional[str]:
        """
        Detect rocBLAS kernel database files.

        Pattern: lib/rocblas/library/*_gfx*.{co,hsaco,dat}
        """
        try:
            rel_path = path.relative_to(prefix_root)
            # Use as_posix() for consistent forward-slash paths on all platforms
            path_str = rel_path.as_posix()

            # Check if it's in rocblas/library directory
            if "rocblas/library" not in path_str:
                return None

            # Check file extension
            if path.suffix not in [".co", ".hsaco", ".dat"]:
                return None

            # Extract architecture from filename
            # Look for patterns like _gfx1100, _gfx1101, gfx1102, etc.
            match = _GFX_ARCH_PATTERN.search(path.name)
            if match:
                return f"gfx{match.group(1)}"

            # Some .dat files don't have architecture suffix but are generic
            # We don't move those
            return None

        except (ValueError, AttributeError):
            return None


class HipBLASLtHandler(DatabaseHandler):
    """Handler for hipBLASLt kernel files."""

    def name(self) -> str:
        return "hipblaslt"

    def detect(self, path: Path, prefix_root: Path) -> Optional[str]:
        """
        Detect hipBLASLt kernel database files.

        Pattern: lib/hipblaslt/library/*_gfx*.{co,hsaco,dat}
        """
        try:
            rel_path = path.relative_to(prefix_root)
            # Use as_posix() for consistent forward-slash paths on all platforms
            path_str = rel_path.as_posix()

            # Check if it's in hipblaslt/library directory
            if "hipblaslt/library" not in path_str:
                return None

            # Check file extension
            if path.suffix not in [".co", ".hsaco", ".dat"]:
                return None

            # Extract architecture from filename
            match = _GFX_ARCH_PATTERN.search(path.name)
            if match:
                return f"gfx{match.group(1)}"

            return None

        except (ValueError, AttributeError):
            return None


class AotritonHandler(DatabaseHandler):
    """Handler for AOTriton kernel directories."""

    def name(self) -> str:
        return "aotriton"

    def detect(self, path: Path, prefix_root: Path) -> Optional[str]:
        """
        Detect AOTriton kernel files.

        Pattern: */aotriton/kernels/gfx*/
        """
        try:
            rel_path = path.relative_to(prefix_root)
            path_parts = rel_path.parts

            # Look for aotriton/kernels in the path
            for i, part in enumerate(path_parts[:-1]):
                if (
                    part == "aotriton"
                    and i + 1 < len(path_parts)
                    and path_parts[i + 1] == "kernels"
                ):
                    # Check if next part is an architecture
                    if i + 2 < len(path_parts):
                        arch_part = path_parts[i + 2]
                        if arch_part.startswith("gfx"):
                            return arch_part
                    break

            return None

        except (ValueError, AttributeError, IndexError):
            return None


class MIOpenHandler(DatabaseHandler):
    """Handler for MIOpen performance database/model files.

    MIOpen installs tuning model files to share/miopen/db/ with filenames
    prefixed by the target architecture (e.g., gfx942.tn.model,
    gfx908_ConvAsm1x1U_decoder.ktn.model).
    """

    def name(self) -> str:
        return "miopen"

    def detect(self, path: Path, prefix_root: Path) -> Optional[str]:
        """
        Detect MIOpen tuning database files.

        Pattern: share/miopen/db/gfx*.{tn.model,ktn.model}
        """
        try:
            rel_path = path.relative_to(prefix_root)
            path_str = rel_path.as_posix()

            if "miopen/db" not in path_str:
                return None

            if not path.name.endswith(".model"):
                return None

            match = _GFX_ARCH_PATTERN.search(path.name)
            if match:
                return f"gfx{match.group(1)}"

            return None

        except (ValueError, AttributeError):
            return None


# Registry of available handlers
AVAILABLE_HANDLERS = {
    "rocblas": RocBLASHandler,
    "hipblaslt": HipBLASLtHandler,
    "aotriton": AotritonHandler,
    "miopen": MIOpenHandler,
}


def get_database_handlers(names: List[str]) -> List[DatabaseHandler]:
    """
    Get database handler instances by name.

    Args:
        names: List of handler names to instantiate

    Returns:
        List of DatabaseHandler instances

    Raises:
        ValueError: If an unknown handler name is provided
    """
    handlers = []
    for name in names:
        if name not in AVAILABLE_HANDLERS:
            available = ", ".join(sorted(AVAILABLE_HANDLERS.keys()))
            raise ValueError(
                f"Unknown database handler: {name}. Available: {available}"
            )
        handlers.append(AVAILABLE_HANDLERS[name]())
    return handlers


def list_available_handlers() -> List[str]:
    """
    Get list of available handler names.

    Returns:
        List of registered handler names
    """
    return sorted(AVAILABLE_HANDLERS.keys())
