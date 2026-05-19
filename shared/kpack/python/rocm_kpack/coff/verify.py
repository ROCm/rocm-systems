"""
PE/COFF verification utilities.

The CoffVerifier class provides structural validation for PE binaries,
catching common issues that would cause problems at runtime or with tools.

This includes both internal structural checks and external tool invocation.
"""

import subprocess
import sys
from pathlib import Path
from typing import Callable

from ..verify import VerificationResult, BinaryVerifier
from .types import (
    IMAGE_DIRECTORY_ENTRY_BASERELOC,
    IMAGE_DIRECTORY_ENTRY_IMPORT,
    IMAGE_DIRECTORY_ENTRY_EXPORT,
)
from .surgery import CoffSurgery


class CoffVerifier(BinaryVerifier):
    """PE/COFF binary verification.

    Checks for common structural issues that cause problems at runtime
    or with tools like llvm-objdump, dumpbin, etc.

    Usage:
        result = CoffVerifier.verify(Path("foo.dll"))
        if not result.passed:
            print(result)
    """

    def __init__(self, surgery: CoffSurgery):
        """Initialize with a CoffSurgery instance.

        Args:
            surgery: Parsed PE binary
        """
        self._surgery = surgery

    @classmethod
    def verify(cls, path: Path) -> VerificationResult:
        """Verify a PE file on disk.

        Args:
            path: Path to PE binary

        Returns:
            VerificationResult with any errors/warnings
        """
        surgery = CoffSurgery.load(path)
        verifier = cls(surgery)
        return verifier.run_all_checks()

    @classmethod
    def verify_data(cls, data: bytes | bytearray) -> VerificationResult:
        """Verify PE data in memory.

        Args:
            data: PE binary data

        Returns:
            VerificationResult with any errors/warnings
        """
        surgery = CoffSurgery(bytearray(data))
        verifier = cls(surgery)
        return verifier.run_all_checks()

    def run_all_checks(self) -> VerificationResult:
        """Run all internal structural checks."""
        result = VerificationResult()

        # Run each check and merge results
        checks: list[Callable[[], VerificationResult]] = [
            self.check_no_overlapping_sections,
            self.check_section_alignment,
            self.check_section_offsets_in_bounds,
            self.check_data_directory_bounds,
            self.check_size_of_image,
            self.check_file_alignment,
        ]

        for check in checks:
            result.merge(check())

        return result

    # =========================================================================
    # Structural Checks
    # =========================================================================

    def check_no_overlapping_sections(self) -> VerificationResult:
        """Check that no two sections have overlapping file regions.

        This catches corruption where section raw data overlaps.
        """
        result = VerificationResult()

        sections = list(self._surgery.iter_sections())

        for i, sect1 in enumerate(sections):
            if sect1.raw_size == 0:
                continue

            start1 = sect1.file_offset
            end1 = start1 + sect1.raw_size

            for sect2 in sections[i + 1 :]:
                if sect2.raw_size == 0:
                    continue

                start2 = sect2.file_offset
                end2 = start2 + sect2.raw_size

                # Check for overlap
                if start1 < end2 and start2 < end1:
                    result.add_error(
                        f"Sections {sect1.name} and {sect2.name} have overlapping "
                        f"file ranges: [{start1:#x}, {end1:#x}) and [{start2:#x}, {end2:#x})"
                    )

        return result

    def check_section_alignment(self) -> VerificationResult:
        """Check section alignment constraints.

        - PointerToRawData must be aligned to FileAlignment
        - VirtualAddress must be aligned to SectionAlignment
        """
        result = VerificationResult()

        file_align = self._surgery.file_alignment
        sect_align = self._surgery.section_alignment

        for section in self._surgery.iter_sections():
            # Check file offset alignment (if section has raw data)
            if section.raw_size > 0:
                if section.file_offset % file_align != 0:
                    result.add_error(
                        f"Section {section.name} PointerToRawData 0x{section.file_offset:x} "
                        f"not aligned to FileAlignment 0x{file_align:x}"
                    )

            # Check RVA alignment
            if section.rva % sect_align != 0:
                result.add_warning(
                    f"Section {section.name} VirtualAddress 0x{section.rva:x} "
                    f"not aligned to SectionAlignment 0x{sect_align:x}"
                )

        return result

    def check_section_offsets_in_bounds(self) -> VerificationResult:
        """Check that section raw data is within file bounds."""
        result = VerificationResult()

        file_size = len(self._surgery.data)

        for section in self._surgery.iter_sections():
            if section.raw_size == 0:
                continue

            end_offset = section.file_offset + section.raw_size
            if end_offset > file_size:
                result.add_error(
                    f"Section {section.name} raw data extends beyond file: "
                    f"ends at 0x{end_offset:x}, file size is 0x{file_size:x}"
                )

        return result

    def check_data_directory_bounds(self) -> VerificationResult:
        """Check that data directories point to valid locations."""
        result = VerificationResult()

        # Check key data directories
        for idx, name in [
            (IMAGE_DIRECTORY_ENTRY_EXPORT, "Export"),
            (IMAGE_DIRECTORY_ENTRY_IMPORT, "Import"),
            (IMAGE_DIRECTORY_ENTRY_BASERELOC, "BaseReloc"),
        ]:
            dd = self._surgery.get_data_directory(idx)
            if dd is None or not dd.is_present:
                continue

            # Check RVA is valid (can be converted to file offset)
            offset = self._surgery.rva_to_file_offset(dd.VirtualAddress)
            if offset is None:
                result.add_error(
                    f"{name} directory RVA 0x{dd.VirtualAddress:x} "
                    "not in any section"
                )
                continue

            # Check size is within section
            end_rva = dd.VirtualAddress + dd.Size
            end_offset = self._surgery.rva_to_file_offset(end_rva - 1)
            if end_offset is None and dd.Size > 0:
                result.add_warning(
                    f"{name} directory end RVA 0x{end_rva:x} " "extends beyond section"
                )

        return result

    def check_size_of_image(self) -> VerificationResult:
        """Check SizeOfImage covers all sections."""
        result = VerificationResult()

        size_of_image = self._surgery.optional_header.SizeOfImage
        sect_align = self._surgery.section_alignment

        # Find max section end RVA
        max_rva = 0
        for section in self._surgery.iter_sections():
            end_rva = section.rva + section.virtual_size
            max_rva = max(max_rva, end_rva)

        # SizeOfImage should be >= max_rva, aligned to SectionAlignment
        expected_min = max_rva
        if expected_min % sect_align != 0:
            expected_min = (expected_min // sect_align + 1) * sect_align

        if size_of_image < expected_min:
            result.add_error(
                f"SizeOfImage 0x{size_of_image:x} is smaller than "
                f"required 0x{expected_min:x} to cover all sections"
            )

        return result

    def check_file_alignment(self) -> VerificationResult:
        """Check FileAlignment is a power of 2 between 512 and 64K."""
        result = VerificationResult()

        file_align = self._surgery.file_alignment

        # Must be power of 2
        if file_align == 0 or (file_align & (file_align - 1)) != 0:
            result.add_error(f"FileAlignment 0x{file_align:x} is not a power of 2")
        elif file_align < 512:
            result.add_warning(
                f"FileAlignment 0x{file_align:x} is smaller than standard 512"
            )
        elif file_align > 65536:
            result.add_warning(
                f"FileAlignment 0x{file_align:x} is larger than standard 64K"
            )

        return result


# =============================================================================
# External Tool Verification
# =============================================================================


def verify_with_llvm_objdump(
    path: Path,
    llvm_objdump: Path | str = "llvm-objdump",
) -> VerificationResult:
    """Run llvm-objdump and check for warnings/errors.

    Args:
        path: Path to PE binary
        llvm_objdump: Path to llvm-objdump executable

    Returns:
        VerificationResult
    """
    result = VerificationResult()

    try:
        proc = subprocess.run(
            [str(llvm_objdump), "-h", str(path)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        result.add_warning("llvm-objdump not found")
        return result

    # Check for warning patterns
    output = proc.stdout + proc.stderr
    warning_patterns = [
        "corrupt",
        "invalid",
        "warning:",
        "Warning:",
        "error:",
        "Error:",
        "truncated",
    ]

    for line in output.splitlines():
        for pattern in warning_patterns:
            if pattern.lower() in line.lower():
                result.add_error(f"llvm-objdump: {line.strip()}")
                break

    if proc.returncode != 0:
        result.add_error(f"llvm-objdump exited with code {proc.returncode}")

    return result


def verify_with_dumpbin(path: Path) -> VerificationResult:
    """Run dumpbin (Windows SDK) and check for warnings/errors.

    Args:
        path: Path to PE binary

    Returns:
        VerificationResult
    """
    result = VerificationResult()

    try:
        proc = subprocess.run(
            ["dumpbin", "/headers", str(path)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        result.add_warning("dumpbin not found (Windows SDK tool)")
        return result

    if proc.returncode != 0:
        result.add_error(f"dumpbin exited with code {proc.returncode}")

    # Check for error patterns
    output = proc.stdout + proc.stderr
    if "fatal error" in output.lower():
        result.add_error("dumpbin reported fatal error")

    return result


def verify_all(
    path: Path,
    llvm_objdump: Path | str = "llvm-objdump",
) -> VerificationResult:
    """Run all verification checks on a PE binary.

    Args:
        path: Path to PE binary
        llvm_objdump: Path to llvm-objdump executable

    Returns:
        Combined VerificationResult
    """
    result = VerificationResult()

    # Internal structural checks
    internal = CoffVerifier.verify(path)
    result.merge(internal)

    # External tool checks
    result.merge(verify_with_llvm_objdump(path, llvm_objdump))

    # dumpbin is Windows-only (part of MSVC/Windows SDK)
    if sys.platform == "win32":
        result.merge(verify_with_dumpbin(path))

    return result
