"""
ELF verification utilities.

The ElfVerifier class provides structural validation for ELF binaries,
catching common issues that would cause problems at runtime or with tools.

This includes both internal structural checks and external tool invocation.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from ..verify import VerificationResult, BinaryVerifier
from .types import (
    PT_LOAD,
    PT_PHDR,
    SHT_NOBITS,
    PAGE_SIZE,
)
from .surgery import ElfSurgery, ProgramHeader


class ElfVerifier(BinaryVerifier):
    """ELF binary verification.

    Checks for common structural issues that cause problems at runtime
    or with tools like strip, gdb, objcopy.

    Usage:
        result = ElfVerifier.verify(Path("libfoo.so"))
        if not result.passed:
            print(result)
    """

    def __init__(self, surgery: ElfSurgery):
        """Initialize with an ElfSurgery instance.

        Args:
            surgery: Parsed ELF binary
        """
        self._surgery = surgery

    @classmethod
    def verify(cls, path: Path) -> VerificationResult:
        """Verify an ELF file on disk.

        Args:
            path: Path to ELF binary

        Returns:
            VerificationResult with any errors/warnings
        """
        surgery = ElfSurgery.load(path)
        verifier = cls(surgery)
        return verifier.run_all_checks()

    @classmethod
    def verify_data(cls, data: bytes | bytearray) -> VerificationResult:
        """Verify ELF data in memory.

        Args:
            data: ELF binary data

        Returns:
            VerificationResult with any errors/warnings
        """
        surgery = ElfSurgery(bytearray(data))
        verifier = cls(surgery)
        return verifier.run_all_checks()

    def run_all_checks(self) -> VerificationResult:
        """Run all internal structural checks."""
        result = VerificationResult()

        # Run each check and merge results
        checks: list[Callable[[], VerificationResult]] = [
            self.check_no_overlapping_load_segments,
            self.check_phdr_alignment,
            self.check_section_alignment,
            self.check_string_table_bounds,
            self.check_section_header_offsets,
            self.check_phdr_file_offset_consistency,
        ]

        for check in checks:
            result.merge(check())

        return result

    # =========================================================================
    # Structural Checks
    # =========================================================================

    def check_no_overlapping_load_segments(self) -> VerificationResult:
        """Check that no two PT_LOAD segments share file offsets.

        This catches the NOBITS segment offset collision bug where two
        segments had the same p_offset value.
        """
        result = VerificationResult()

        # Collect all PT_LOAD segments with non-zero filesz
        load_segments: list[tuple[int, ProgramHeader]] = []
        for idx, phdr in self._surgery.iter_load_segments():
            load_segments.append((idx, phdr))

        # Check for overlaps
        for i, (idx1, phdr1) in enumerate(load_segments):
            for idx2, phdr2 in load_segments[i + 1 :]:
                # Check file offset overlap (only for segments with file content)
                if phdr1.p_filesz > 0 and phdr2.p_filesz > 0:
                    # Two segments overlap if their file ranges intersect
                    if (
                        phdr1.p_offset < phdr2.p_offset + phdr2.p_filesz
                        and phdr2.p_offset < phdr1.p_offset + phdr1.p_filesz
                    ):
                        result.add_error(
                            f"PT_LOAD segments {idx1} and {idx2} have overlapping "
                            f"file ranges: [{phdr1.p_offset:#x}, {phdr1.end_offset:#x}) "
                            f"and [{phdr2.p_offset:#x}, {phdr2.end_offset:#x})"
                        )

                # Special case: two segments with same offset and at least one has filesz=0
                # This is the NOBITS bug - dynamic linker gets confused
                if phdr1.p_offset == phdr2.p_offset:
                    if phdr1.p_filesz == 0 or phdr2.p_filesz == 0:
                        result.add_error(
                            f"PT_LOAD segments {idx1} and {idx2} share file offset "
                            f"{phdr1.p_offset:#x} (NOBITS collision)"
                        )

        return result

    def check_phdr_alignment(self) -> VerificationResult:
        """Check PT_LOAD segments satisfy mmap alignment requirements.

        For mmap to work: (p_offset % PAGE_SIZE) == (p_vaddr % PAGE_SIZE)
        """
        result = VerificationResult()

        for idx, phdr in self._surgery.iter_load_segments():
            offset_mod = phdr.p_offset % PAGE_SIZE
            vaddr_mod = phdr.p_vaddr % PAGE_SIZE

            if offset_mod != vaddr_mod:
                result.add_error(
                    f"PT_LOAD segment {idx} has misaligned offset/vaddr: "
                    f"offset=0x{phdr.p_offset:x} ({offset_mod:#x} mod PAGE_SIZE), "
                    f"vaddr=0x{phdr.p_vaddr:x} ({vaddr_mod:#x} mod PAGE_SIZE)"
                )

        return result

    def check_section_alignment(self) -> VerificationResult:
        """Check section alignment constraints."""
        result = VerificationResult()

        for section in self._surgery.iter_sections():
            shdr = section.header
            if shdr.sh_addralign > 1:
                # Check offset alignment
                if shdr.sh_offset % shdr.sh_addralign != 0:
                    result.add_warning(
                        f"Section {section.name} offset 0x{shdr.sh_offset:x} "
                        f"not aligned to {shdr.sh_addralign}"
                    )

                # Check address alignment for ALLOC sections
                if shdr.is_alloc and shdr.sh_addr % shdr.sh_addralign != 0:
                    result.add_warning(
                        f"Section {section.name} address 0x{shdr.sh_addr:x} "
                        f"not aligned to {shdr.sh_addralign}"
                    )

        return result

    def check_string_table_bounds(self) -> VerificationResult:
        """Check that string table references are within bounds.

        This catches the .dynstr corruption issue where string offsets
        pointed outside the string table.
        """
        result = VerificationResult()

        # Check section header string table
        ehdr = self._surgery.ehdr
        if ehdr.e_shstrndx < len(self._surgery._shdrs):
            shstrtab = self._surgery._shdrs[ehdr.e_shstrndx]

            for section in self._surgery.iter_sections():
                name_offset = section.header.sh_name
                if name_offset >= shstrtab.sh_size:
                    result.add_error(
                        f"Section header {section.index} name offset "
                        f"{name_offset} exceeds .shstrtab size {shstrtab.sh_size}"
                    )

        # Find .dynstr and check it exists for dynamic binaries
        dynstr = self._surgery.find_section(".dynstr")
        if dynstr is not None:
            # Verify .dynstr content ends with null
            try:
                content = self._surgery.get_section_content(dynstr)
                if content and content[-1] != 0:
                    result.add_warning(".dynstr does not end with null terminator")
            except ValueError:
                pass  # NOBITS section, can't check

        return result

    def check_section_header_offsets(self) -> VerificationResult:
        """Check that section header table is at a valid offset.

        The section header table must not overlap with content.
        """
        result = VerificationResult()

        ehdr = self._surgery.ehdr
        shdr_start = ehdr.e_shoff
        shdr_end = ehdr.e_shoff + ehdr.e_shnum * ehdr.e_shentsize

        # Check against PHDR table
        phdr_start = ehdr.e_phoff
        phdr_end = ehdr.e_phoff + ehdr.e_phnum * ehdr.e_phentsize

        if shdr_start < phdr_end and phdr_start < shdr_end:
            result.add_error(
                f"Section header table [{shdr_start:#x}, {shdr_end:#x}) "
                f"overlaps with program header table [{phdr_start:#x}, {phdr_end:#x})"
            )

        # Check against sections
        for section in self._surgery.iter_sections():
            if section.header.sh_type == SHT_NOBITS:
                continue
            sect_start = section.header.sh_offset
            sect_end = section.header.end_offset

            if shdr_start < sect_end and sect_start < shdr_end:
                if section.name != ".shstrtab":  # shstrtab often abuts section headers
                    result.add_warning(
                        f"Section header table [{shdr_start:#x}, {shdr_end:#x}) "
                        f"may overlap with section {section.name} "
                        f"[{sect_start:#x}, {sect_end:#x})"
                    )

        return result

    def check_phdr_file_offset_consistency(self) -> VerificationResult:
        """Check that PHDR table location matches PT_PHDR segment.

        If a PT_PHDR segment exists, its offset/vaddr should match
        the actual program header table location.
        """
        result = VerificationResult()

        ehdr = self._surgery.ehdr
        for idx, phdr in self._surgery.iter_program_headers():
            if phdr.p_type == PT_PHDR:
                if phdr.p_offset != ehdr.e_phoff:
                    result.add_error(
                        f"PT_PHDR offset {phdr.p_offset:#x} doesn't match "
                        f"e_phoff {ehdr.e_phoff:#x}"
                    )
                expected_filesz = ehdr.e_phnum * ehdr.e_phentsize
                if phdr.p_filesz != expected_filesz:
                    result.add_warning(
                        f"PT_PHDR filesz {phdr.p_filesz} doesn't match "
                        f"expected {expected_filesz} (e_phnum * e_phentsize)"
                    )
                break

        return result


# =============================================================================
# External Tool Verification
# =============================================================================


def verify_with_readelf(path: Path) -> VerificationResult:
    """Run readelf -a and check for warnings/errors.

    Args:
        path: Path to ELF binary

    Returns:
        VerificationResult
    """
    result = VerificationResult()

    try:
        proc = subprocess.run(
            ["readelf", "-a", str(path)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        result.add_error("readelf not found")
        return result

    # Check for common warning patterns
    output = proc.stdout + proc.stderr
    warning_patterns = [
        "corrupt",
        "invalid",
        "warning:",
        "Warning:",
        "ERROR:",
        "truncated",
        "out of range",
        "overlaps",
    ]

    for line in output.splitlines():
        for pattern in warning_patterns:
            if pattern.lower() in line.lower():
                result.add_error(f"readelf: {line.strip()}")
                break

    if proc.returncode != 0:
        result.add_warning(f"readelf exited with code {proc.returncode}")

    return result


def verify_with_strip(path: Path, tmp_dir: Path | None = None) -> VerificationResult:
    """Test that strip doesn't corrupt the binary.

    Args:
        path: Path to ELF binary
        tmp_dir: Directory for temporary output (uses system temp if None)

    Returns:
        VerificationResult
    """
    result = VerificationResult()

    # Create temp file for stripped output
    if tmp_dir is None:
        tmp_dir = Path(tempfile.gettempdir())

    stripped_path = tmp_dir / f"{path.name}.stripped"

    try:
        proc = subprocess.run(
            ["strip", "-o", str(stripped_path), str(path)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        result.add_error("strip not found")
        return result
    finally:
        # Clean up temp file
        if stripped_path.exists():
            stripped_path.unlink()

    if proc.returncode != 0:
        result.add_error(f"strip failed: {proc.stderr.strip()}")

    if proc.stderr:
        for line in proc.stderr.splitlines():
            if "warning" in line.lower() or "error" in line.lower():
                result.add_warning(f"strip: {line.strip()}")

    return result


def verify_with_gdb(path: Path) -> VerificationResult:
    """Check for GDB warnings about the binary.

    Args:
        path: Path to ELF binary

    Returns:
        VerificationResult
    """
    result = VerificationResult()

    try:
        proc = subprocess.run(
            ["gdb", "-batch", "-ex", "info files", str(path)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        result.add_error("gdb not found")
        return result

    # Check for warning patterns
    output = proc.stdout + proc.stderr
    warning_patterns = [
        "corrupt",
        "invalid",
        "warning:",
        "Warning:",
    ]

    for line in output.splitlines():
        for pattern in warning_patterns:
            if pattern in line:
                result.add_error(f"gdb: {line.strip()}")
                break

    return result


def verify_with_ldd(path: Path) -> VerificationResult:
    """Check that ldd can process the binary.

    Args:
        path: Path to ELF binary (should be shared library or executable)

    Returns:
        VerificationResult
    """
    result = VerificationResult()

    try:
        proc = subprocess.run(
            ["ldd", str(path)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        result.add_error("ldd not found")
        return result

    # ldd failures are not necessarily errors (static binaries, etc)
    if "not a dynamic executable" in proc.stdout + proc.stderr:
        result.add_warning("ldd: not a dynamic executable")
    elif proc.returncode != 0:
        result.add_warning(f"ldd exited with code {proc.returncode}")

    return result


def verify_all(path: Path, tmp_dir: Path | None = None) -> VerificationResult:
    """Run all verification checks on an ELF binary.

    Args:
        path: Path to ELF binary
        tmp_dir: Directory for temporary files

    Returns:
        Combined VerificationResult
    """
    result = VerificationResult()

    # Internal structural checks
    internal = ElfVerifier.verify(path)
    result.merge(internal)

    # External tool checks
    result.merge(verify_with_readelf(path))
    result.merge(verify_with_strip(path, tmp_dir))

    # gdb and ldd are Linux-specific and won't work correctly on Windows
    # (even if available via MSYS2, they expect Windows binaries)
    if sys.platform != "win32":
        result.merge(verify_with_gdb(path))
        result.merge(verify_with_ldd(path))

    return result
