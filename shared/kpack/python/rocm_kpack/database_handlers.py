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

# MIOpen CK per-arch shared library pattern.
# Matches libMIOpenCK<name>_<arch>.so (Linux) and MIOpenCK<name>_<arch>.dll (Windows).
_MIOPEN_CK_SO_PATTERN = re.compile(
    r"^(?:lib)?MIOpenCK\w+_(" + _GFX_ARCH_PATTERN.pattern + r")\.(?:so|dll)"
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


# Regex matching a library/<arch>/ path component (for per-arch subdirectory layout).
_LIBRARY_ARCH_DIR_PATTERN = re.compile(
    r"/library/(" + _GFX_ARCH_PATTERN.pattern + r")/"
)


class _TensileHandler(DatabaseHandler):
    """Base class for Tensile-based library handlers (rocBLAS, hipBLASLt, hipSPARSELt).

    Supports two directory layouts:

    Flat (current):
        lib/<lib>/library/TensileLibrary_gfx942.co
        lib/<lib>/library/TensileLibrary_..._fallback.dat  (no arch → generic)

    Per-arch subdirectory (rocm-libraries#5976):
        lib/<lib>/library/gfx942/TensileLibrary_....co
        lib/<lib>/library/gfx942/TensileLibrary_..._fallback.dat
        lib/<lib>/library/gfx942/TensileManifest.txt

    In the flat layout, architecture is extracted from the filename and only
    known Tensile extensions are accepted. In the per-arch layout, the
    architecture comes from the directory name and any file underneath is
    accepted (manifests, fallback .dat files, etc.).
    """

    # Subclasses set this to the directory marker (e.g. "rocblas/library")
    _library_dir: str

    def detect(self, path: Path, prefix_root: Path) -> Optional[str]:
        path_str = self._relative_path(path, prefix_root)

        if self._library_dir not in path_str:
            return None

        # Try filename-based detection (flat layout, known extensions)
        if path.suffix in (".co", ".hsaco", ".dat"):
            match = _GFX_ARCH_PATTERN.search(path.name)
            if match:
                return match.group(0)

        # Try directory-based detection (per-arch subdirectory layout):
        # .../library/<arch>/anything
        dir_match = _LIBRARY_ARCH_DIR_PATTERN.search(path_str)
        if dir_match:
            return dir_match.group(1)

        return None


class RocBLASHandler(_TensileHandler):
    """Handler for rocBLAS Tensile library files."""

    _library_dir = "rocblas/library"

    def name(self) -> str:
        return "rocblas"


class HipBLASLtHandler(_TensileHandler):
    """Handler for hipBLASLt kernel files."""

    _library_dir = "hipblaslt/library"

    def name(self) -> str:
        return "hipblaslt"


class HipSparseLtHandler(_TensileHandler):
    """Handler for hipSPARSELt Tensile kernel files."""

    _library_dir = "hipsparselt/library"

    def name(self) -> str:
        return "hipsparselt"


class AotritonHandler(DatabaseHandler):
    """Handler for AOTriton kernel image directories.

    AOTriton ships precompiled kernel images in per-architecture directories:
        lib/aotriton.images/amd-gfx942/flash/attn_fwd/kernel.aks2
        lib/aotriton.images/amd-gfx11xx/flash/bwd_kernel_dk_dv/kernel.aks2
        lib/aotriton.images/amd-gfx110x/flash/bwd_kernel_dk_dv/kernel.aks2

    Architecture directories use family/sub-family names (gfx11xx, gfx110x,
    gfx115x, gfx120x) for shared ISA assets, and specific chip names
    (gfx942, gfx90a, gfx950) for others.

    Returns bundle keys from the rocm-bootstrap hierarchy:
        gfx11xx → gfx11 (family), gfx110x → gfx110x (sub-family),
        gfx120x → gfx12_0 (sub-family), gfx942 → gfx942 (target), etc.
    """

    # Mapping from aotriton directory suffixes to rocm-bootstrap bundle keys.
    # Entries are only needed for patterns that differ from the raw directory
    # name. Already-valid keys (gfx110x, gfx942, gfx90a, etc.) pass through
    # unchanged.
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
            Bundle key (e.g., 'gfx11', 'gfx110x', 'gfx12_0', 'gfx942') or None.
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
        Detect MIOpen tuning database files and CK per-arch shared libraries.

        Patterns:
        - share/miopen/db/gfx*.{db.txt,fdb.txt,model,kdb}
        - CK per-arch shared libraries matching _MIOPEN_CK_SO_PATTERN
        """
        path_str = self._relative_path(path, prefix_root)
        filename = Path(path_str).name

        # MIOpen CK per-arch shared libraries (dlopen'd at runtime).
        ck_match = _MIOPEN_CK_SO_PATTERN.match(filename)
        if ck_match:
            return ck_match.group(1)

        if "miopen/db" not in path_str:
            return None

        if not path.is_file():
            return None

        # Match .kdb, .model, .db.txt, .fdb.txt, .OpenCL.fdb.txt, .HIP.fdb.txt
        name = path.name
        if not (
            name.endswith(".kdb")
            or name.endswith(".model")
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
