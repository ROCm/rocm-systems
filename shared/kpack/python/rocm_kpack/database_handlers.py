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
# Matches architecture IDs like gfx908, gfx90a, gfx942-xnack+, gfx90a-xnack-
# Note: Tensile filenames use hyphens (gfx90a-xnack+), not colons (gfx90a:xnack+)
_GFX_ARCH_PATTERN = re.compile(r"gfx\d+[a-z]*(?:-xnack[+-])?")

# MIOpen-specific arch pattern. MIOpen filenames concatenate arch ID + CU count
# without a separator (e.g., gfx90878 = gfx908 + 78 CUs, gfx942130 = gfx942 + 130).
# Standard _GFX_ARCH_PATTERN would greedily consume too many digits, so we
# enumerate known arch IDs explicitly. This list must be updated when new
# architectures are added to ROCm. Ordered longest-first so alternation
# matches correctly (e.g., gfx1151 before gfx1150 doesn't matter but
# gfx90a before gfx900 ensures the letter variant is tried first).
_MIOPEN_ARCH_PATTERN = re.compile(
    r"gfx(?:"
    r"90a|900|906|908|940|941|942|950"
    r"|1010|1030|1100|1101|1102|1150|1151|1200|1201"
    r")"
)


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
        Detect if a file belongs to this database type and extract bundle key.

        Args:
            path: Path to the file (must be under prefix_root)
            prefix_root: Root of the prefix for relative path computation

        Returns:
            Bundle key (e.g., 'gfx1100', 'gfx11', 'gfx12_0') if file matches,
            None otherwise. Bundle keys correspond to entries in the
            rocm-bootstrap hierarchy.

        Raises:
            ValueError: If path is not under prefix_root (caller bug).
        """
        pass

    def _relative_path(self, path: Path, prefix_root: Path) -> str:
        """Compute forward-slash relative path, raising on bad input."""
        try:
            return path.relative_to(prefix_root).as_posix()
        except ValueError:
            raise ValueError(
                f"{self.name()} handler: path {path} is not under "
                f"prefix_root {prefix_root}"
            ) from None

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
        path_str = self._relative_path(path, prefix_root)

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
            return match.group(0)

        # Some .dat files don't have architecture suffix but are generic
        # We don't move those
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
        path_str = self._relative_path(path, prefix_root)

        # Check if it's in hipblaslt/library directory
        if "hipblaslt/library" not in path_str:
            return None

        # Check file extension
        if path.suffix not in [".co", ".hsaco", ".dat"]:
            return None

        # Extract architecture from filename
        match = _GFX_ARCH_PATTERN.search(path.name)
        if match:
            return match.group(0)

        return None


class HipSparseLtHandler(DatabaseHandler):
    """Handler for hipSPARSELt Tensile kernel files."""

    def name(self) -> str:
        return "hipsparselt"

    def detect(self, path: Path, prefix_root: Path) -> Optional[str]:
        """
        Detect hipSPARSELt kernel database files.

        Pattern: lib/hipsparselt/library/*_gfx*.{co,hsaco,dat}
        """
        path_str = self._relative_path(path, prefix_root)

        if "hipsparselt/library" not in path_str:
            return None

        if path.suffix not in [".co", ".hsaco", ".dat"]:
            return None

        match = _GFX_ARCH_PATTERN.search(path.name)
        if match:
            return match.group(0)

        return None


class AotritonHandler(DatabaseHandler):
    """Handler for AOTriton kernel image directories.

    AOTriton ships precompiled kernel images in per-architecture directories:
        lib/aotriton.images/amd-gfx942/flash/attn_fwd/kernel.aks2
        lib/aotriton.images/amd-gfx11xx/flash/bwd_kernel_dk_dv/kernel.aks2

    Architecture directories use family names (gfx11xx, gfx120x) for ISA
    families, and specific chip names (gfx942, gfx90a, gfx950) for others.

    Returns bundle keys from the rocm-bootstrap hierarchy:
        gfx11xx → gfx11 (family), gfx120x → gfx12_0 (sub-family),
        gfx942 → gfx942 (target), etc.
    """

    # Mapping from aotriton directory suffixes to rocm-bootstrap bundle keys.
    # Entries are only needed for family/sub-family patterns that differ from
    # the raw directory name. Target-level names (gfx942, gfx90a, etc.) pass
    # through unchanged since they are already valid bundle keys.
    _BUNDLE_MAP = {
        "gfx11xx": "gfx11",
        "gfx120x": "gfx12_0",
    }

    def name(self) -> str:
        return "aotriton"

    def detect(self, path: Path, prefix_root: Path) -> Optional[str]:
        """
        Detect AOTriton kernel image files.

        Pattern: */aotriton.images/amd-gfx*/...

        Returns:
            Bundle key (e.g., 'gfx11', 'gfx12_0', 'gfx942') or None.
        """
        path_str = self._relative_path(path, prefix_root)
        path_parts = Path(path_str).parts

        for i, part in enumerate(path_parts[:-1]):
            if part == "aotriton.images" and i + 1 < len(path_parts):
                arch_dir = path_parts[i + 1]
                if arch_dir.startswith("amd-gfx"):
                    raw_name = arch_dir[4:]  # strip "amd-" prefix
                    return self._BUNDLE_MAP.get(raw_name, raw_name)
                break

        return None


class MIOpenHandler(DatabaseHandler):
    """Handler for MIOpen performance database and model files.

    MIOpen installs files to share/miopen/db/ with filenames prefixed by
    the target architecture:
        gfx942130.db.txt          (gfx942 + 130 CUs)
        gfx90878.HIP.fdb.txt      (gfx908 + 78 CUs)
        gfx1030_36.db.txt         (gfx1030 + 36 CUs, underscore separator)
        gfx908_ConvAsm1x1U_decoder.ktn.model
        gfx908.tn.model

    The arch + CU-count concatenation (no separator) requires a specific
    regex that knows ROCm arch ID lengths — see _MIOPEN_ARCH_PATTERN.
    """

    def name(self) -> str:
        return "miopen"

    def detect(self, path: Path, prefix_root: Path) -> Optional[str]:
        """
        Detect MIOpen tuning database files.

        Pattern: share/miopen/db/gfx*.{db.txt,fdb.txt,model}
        """
        path_str = self._relative_path(path, prefix_root)

        if "miopen/db" not in path_str:
            return None

        if not path.is_file():
            return None

        # Match .model, .db.txt, .fdb.txt, .OpenCL.fdb.txt, .HIP.fdb.txt
        name = path.name
        if not (
            name.endswith(".model")
            or name.endswith(".db.txt")
            or name.endswith(".fdb.txt")
        ):
            return None

        match = _MIOPEN_ARCH_PATTERN.search(name)
        if match:
            return match.group(0)

        return None


# Registry of available handlers
AVAILABLE_HANDLERS = {
    "rocblas": RocBLASHandler,
    "hipblaslt": HipBLASLtHandler,
    "hipsparselt": HipSparseLtHandler,
    "aotriton": AotritonHandler,
    "miopen": MIOpenHandler,
}


# Wheel-type presets: each maps to the list of database handlers relevant
# for that kind of wheel.
WHEEL_TYPE_PRESETS = {
    "torch-fat": [
        "rocblas",
        "hipblaslt",
        "hipsparselt",
        "aotriton",
        "miopen",
    ],
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
