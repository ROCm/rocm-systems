import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
from enum import Enum
from typing import Any

import msgpack

from rocm_kpack.ccob_parser import extract_code_objects_from_fatbin


class BinaryType(Enum):
    """Type of bundled binary file."""

    STANDALONE = "standalone"  # .co files - directly in bundler format
    BUNDLED = "bundled"  # Executables/libraries with .hip_fatbin ELF section


class Toolchain:
    """Manages configuration of various toolchain locations.

    Tools are lazily found and cached on first access, so construction never fails.
    Only when a specific tool is accessed will it be searched for and validated.
    """

    def __init__(
        self,
        *,
        clang_offload_bundler: Path | None = None,
        objcopy: Path | None = None,
        objdump: Path | None = None,
        readelf: Path | None = None,
    ):
        # Store explicit paths (may be None)
        self._clang_offload_bundler_path = clang_offload_bundler
        self._objcopy_path = objcopy
        self._objdump_path = objdump
        self._readelf_path = readelf

        # Cached resolved paths
        self._clang_offload_bundler_cached: Path | None = None
        self._objcopy_cached: Path | None = None
        self._objdump_cached: Path | None = None
        self._readelf_cached: Path | None = None

    @staticmethod
    def configure_argparse(p: argparse.ArgumentParser):
        p.add_argument(
            "--clang-offload-bundler", type=Path, help="Path to clang-offload-bundler"
        )

    @staticmethod
    def from_args(args: argparse.Namespace) -> "Toolchain":
        clang_offload_bundler: Path | None = args.clang_offload_bundler
        return Toolchain(clang_offload_bundler=clang_offload_bundler)

    def _validate_or_find(
        self, tool_file_name: str, explicit_path: Path | None
    ) -> Path:
        if explicit_path is None:
            found_path = shutil.which(tool_file_name)
            if found_path is None:
                raise OSError(
                    f"Could not find tool '{tool_file_name}' on system path. "
                    f"Set KPACK_LLVM_BIN to your LLVM bin directory."
                )
            explicit_path = Path(found_path)
        if not explicit_path.exists():
            raise OSError(
                f"Tool '{tool_file_name}' at path {explicit_path} does not exist"
            )
        return explicit_path

    def _validate_or_find_with_fallback(
        self,
        tool_file_name: str,
        fallback_name: str,
        explicit_path: Path | None,
    ) -> Path:
        """Find a tool, trying fallback name if primary not found.

        Args:
            tool_file_name: Primary tool name (e.g., "objdump")
            fallback_name: Fallback tool name (e.g., "llvm-objdump")
            explicit_path: Explicit path if provided by user

        Returns:
            Path to found tool

        Raises:
            OSError: If neither tool can be found
        """
        if explicit_path is not None:
            if not explicit_path.exists():
                raise OSError(
                    f"Tool '{tool_file_name}' at path {explicit_path} does not exist"
                )
            return explicit_path

        # Try primary name first
        found_path = shutil.which(tool_file_name)
        if found_path is not None:
            return Path(found_path)

        # Try fallback name
        found_path = shutil.which(fallback_name)
        if found_path is not None:
            return Path(found_path)

        raise OSError(
            f"Could not find tool '{tool_file_name}' or '{fallback_name}' on system path. "
            f"Set KPACK_LLVM_BIN to your LLVM bin directory."
        )

    @property
    def clang_offload_bundler(self) -> Path:
        """Get clang-offload-bundler path (lazy, cached)."""
        if self._clang_offload_bundler_cached is None:
            self._clang_offload_bundler_cached = self._validate_or_find(
                "clang-offload-bundler", self._clang_offload_bundler_path
            )
        return self._clang_offload_bundler_cached

    @property
    def objcopy(self) -> Path:
        """Get objcopy path (lazy, cached)."""
        if self._objcopy_cached is None:
            self._objcopy_cached = self._validate_or_find("objcopy", self._objcopy_path)
        return self._objcopy_cached

    @property
    def readelf(self) -> Path:
        """Get readelf path (lazy, cached)."""
        if self._readelf_cached is None:
            self._readelf_cached = self._validate_or_find("readelf", self._readelf_path)
        return self._readelf_cached

    @property
    def objdump(self) -> Path:
        """Get objdump path (lazy, cached).

        Tries 'objdump' first, falls back to 'llvm-objdump' if not found.
        This allows the same code to work on Linux (GNU objdump) and
        Windows (LLVM objdump from ROCm).
        """
        if self._objdump_cached is None:
            self._objdump_cached = self._validate_or_find_with_fallback(
                "objdump", "llvm-objdump", self._objdump_path
            )
        return self._objdump_cached

    def exec_capture_text(self, args: list[str | Path]):
        return subprocess.check_output(
            [str(a) for a in args], stderr=subprocess.STDOUT
        ).decode()

    def exec(self, args: list[str | Path]):
        # Use check_output to capture stderr in exceptions (discarding the output)
        subprocess.check_output([str(a) for a in args], stderr=subprocess.STDOUT)


class UnbundledContents:
    """Represents a directory of unbundled contents. This is a context manager that
    will optionally delete the contents on close.
    """

    def __init__(
        self,
        source_binary: "BundledBinary",
        dest_dir: Path,
        delete_on_close: bool,
        target_list: list[tuple[str, str]],
    ):
        self.source_binary = source_binary
        self.dest_dir = dest_dir
        self.delete_on_close = delete_on_close
        self.target_list = target_list

    def __enter__(self) -> "UnbundledContents":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def file_names(self) -> list[str]:
        return [kv[1] for kv in self.target_list]

    def close(self):
        if self.delete_on_close and self.dest_dir.exists():
            shutil.rmtree(self.dest_dir)

    def __repr__(self):
        return f"UnbundledContents(dest_dir={self.dest_dir}, target_list={self.target_list})"


class BundledBinary:
    """Represents a bundled binary at some path.

    Supports two types of bundled binaries:
    1. STANDALONE - Files directly in clang-offload-bundler format (e.g., .co files)
    2. BUNDLED - ELF binaries containing .hip_fatbin section (executables, shared libraries)

    For BUNDLED binaries, the .hip_fatbin section is extracted and treated as
    a bundler-format input.
    """

    def __init__(self, file_path: Path, *, toolchain: Toolchain | None = None):
        # Initialize _temp_dir first to ensure cleanup works even if init fails
        self._temp_dir: Path | None = None  # For extracted .hip_fatbin sections

        self.toolchain = toolchain or Toolchain()
        self.file_path = file_path
        self.binary_type = self._detect_binary_type()

    def unbundle(
        self, *, dest_dir: Path | None = None, delete_on_close: bool = True
    ) -> UnbundledContents:
        """Unbundles the binary, returning a context manager which can be used
        to hold the unbundled files open for as long as needed.
        """
        if dest_dir is None:
            dest_dir = Path(tempfile.TemporaryDirectory(delete=False).name)
        target_list = self._list_bundled_targets(self.file_path)
        contents = UnbundledContents(
            self, dest_dir, delete_on_close=delete_on_close, target_list=target_list
        )
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            self._unbundle(
                targets=[kv[0] for kv in target_list],
                outputs=[dest_dir / kv[1] for kv in target_list],
            )
        except:
            contents.close()
            raise
        return contents

    def _detect_binary_type(self) -> BinaryType:
        """Detect if this is a standalone bundler file or bundled ELF binary.

        Uses readelf to check for .hip_fatbin section to determine type.
        Files with .hip_fatbin section are BUNDLED (executables, libraries).
        Files without (or non-ELF files) are STANDALONE (.co files in bundler format).

        Returns:
            BinaryType indicating the file type

        Raises:
            RuntimeError: For unexpected errors during detection
        """
        try:
            result = subprocess.run(
                [str(self.toolchain.readelf), "-S", str(self.file_path.resolve())],
                capture_output=True,
                text=True,
                check=True,  # Raise CalledProcessError on non-zero exit
            )
            # readelf succeeded - this is an ELF file
            # Check for .hip_fatbin section
            if ".hip_fatbin" in result.stdout:
                return BinaryType.BUNDLED
            else:
                # ELF file without .hip_fatbin section
                return BinaryType.STANDALONE

        except subprocess.CalledProcessError:
            # readelf failed - likely not an ELF file
            # Assume STANDALONE (bundler format file like .co)
            return BinaryType.STANDALONE
        except Exception as e:
            # Unexpected error - fail fast
            raise RuntimeError(
                f"Unexpected error detecting binary type for {self.file_path}: {e}"
            )

    def _get_bundler_input(self) -> Path:
        """Get the file path to use as input to clang-offload-bundler.

        For STANDALONE files, returns the file path directly.
        For BUNDLED binaries, extracts the .hip_fatbin section to a temp file.

        Returns:
            Path to file in bundler format
        """
        if self.binary_type == BinaryType.STANDALONE:
            return self.file_path

        # Extract .hip_fatbin section from bundled binary
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp())

        fatbin_path = self._temp_dir / "fatbin.o"
        # Resolve to absolute paths for objcopy
        abs_file_path = self.file_path.resolve()
        abs_fatbin_path = fatbin_path.resolve()

        try:
            self.toolchain.exec(
                [
                    self.toolchain.objcopy,
                    "--dump-section",
                    f".hip_fatbin={abs_fatbin_path}",
                    abs_file_path,
                ]
            )
        except subprocess.CalledProcessError as e:
            # Include the actual stderr/stdout from objcopy
            error_output = e.output.decode() if e.output else "(no output)"
            raise RuntimeError(
                f"Failed to extract .hip_fatbin section from {self.file_path}. "
                f"objcopy exit code: {e.returncode}. Output: {error_output}"
            ) from e

        return fatbin_path

    def _list_bundled_targets(self, file_path: Path) -> list[tuple[str, str]]:
        """Returns a list of (target_name, file_name) for all bundles.

        Uses our own bundle parser for all formats: CCOB-compressed, single
        uncompressed, and concatenated uncompressed (RDC case).

        For concatenated bundles (RDC), filenames are indexed to distinguish
        multiple code objects with the same target triple.
        """
        bundler_input = self._get_bundler_input()
        data = bundler_input.read_bytes()
        code_objects = extract_code_objects_from_fatbin(data)

        # Check if we need indexed filenames (duplicate targets from RDC)
        targets = [obj.target for obj in code_objects]
        needs_indexing = len(targets) != len(set(targets))

        result = []
        for i, obj in enumerate(code_objects):
            base_ext = ".hsaco" if obj.target.startswith("hip") else ".elf"
            if needs_indexing:
                filename = f"{obj.target}_{i}{base_ext}"
            else:
                filename = f"{obj.target}{base_ext}"
            result.append((obj.target, filename))

        return result

    def _unbundle(self, *, targets: list[str], outputs: list[Path]):
        """Unbundle targets from the binary.

        Args:
            targets: List of target names to unbundle
            outputs: List of output paths (must match length of targets)
        """
        if not targets:
            return

        bundler_input = self._get_bundler_input()
        data = bundler_input.read_bytes()
        code_objects = extract_code_objects_from_fatbin(data)

        if len(code_objects) != len(outputs):
            raise ValueError(
                f"Output count mismatch: {len(code_objects)} code objects "
                f"but {len(outputs)} output paths"
            )

        for obj, output_path in zip(code_objects, outputs):
            output_path.write_bytes(obj.data)

    def list_bundles(self) -> list[str]:
        """List all architecture bundles in the binary.

        Returns:
            List of architecture strings (e.g., ['gfx1100', 'gfx1101'])
            Only returns GPU architectures, not host bundles.
        """
        target_list = self._list_bundled_targets(self.file_path)
        architectures = []
        for target_name, _ in target_list:
            # Extract architecture from target names like:
            # "hipv4-amdgcn-amd-amdhsa--gfx1100" -> "gfx1100"
            if target_name.startswith("hip"):
                parts = target_name.split("--")
                if len(parts) >= 2:
                    architectures.append(parts[-1])
        return architectures

    def remove_section_simple(self, output_path: Path, section_name: str) -> None:
        """Remove a section using objcopy (simple header removal, no space reclaimed).

        This is a simple section removal that only updates ELF headers without
        reclaiming disk space. Useful for benchmarking or comparison purposes.

        For production use with .hip_fatbin removal, use kpack_offload_binary()
        from rocm_kpack.elf instead, which properly reclaims disk space.

        Args:
            output_path: Path where modified binary will be written
            section_name: Name of section to remove (e.g., ".hip_fatbin")

        Raises:
            RuntimeError: If objcopy fails
        """
        try:
            self.toolchain.exec(
                [
                    self.toolchain.objcopy,
                    "--remove-section",
                    section_name,
                    self.file_path,
                    output_path,
                ]
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to remove section {section_name} from {self.file_path}: {e}"
            )

    def cleanup(self) -> None:
        """Clean up temporary files created during operations."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()


def get_section_vaddr(
    toolchain: Toolchain, binary_path: Path, section_name: str
) -> int | None:
    """
    Get the virtual address of a section in an ELF binary.

    Args:
        toolchain: Toolchain instance providing readelf
        binary_path: Path to ELF binary
        section_name: Name of section (e.g., ".custom_data", ".rocm_kpack_ref")

    Returns:
        Virtual address (sh_addr) of the section if it exists and has ALLOC flag,
        None otherwise.

    Note:
        Only returns addresses for sections with the ALLOC flag (A), which indicates
        they are mapped to memory at load time (part of a PT_LOAD segment).
    """
    try:
        # Run readelf to get section headers
        result = subprocess.run(
            [str(toolchain.readelf), "-S", str(binary_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None

    # Parse section headers
    # Format (two-line entries):
    # Line 1: [Nr] Name              Type             Address           Offset
    # Line 2:      Size              EntSize          Flags  Link  Info  Align
    lines = result.stdout.split("\n")
    for i, line in enumerate(lines):
        if section_name in line:
            parts = line.split()
            # Check if this is a section header line (starts with [Nr])
            if len(parts) >= 5 and parts[0].startswith("["):
                try:
                    # Address column is at index 3
                    vaddr = int(parts[3], 16)

                    # Check flags on the next line
                    if i + 1 < len(lines):
                        next_parts = lines[i + 1].split()
                        if len(next_parts) >= 3:
                            flags = next_parts[2]
                            # Only return address if section has ALLOC flag (A)
                            if "A" in flags:
                                return vaddr

                except (ValueError, IndexError):
                    continue

    return None


def get_section_size(
    toolchain: Toolchain, binary_path: Path, section_name: str
) -> int | None:
    """
    Get the size of a section in an ELF binary.

    Args:
        toolchain: Toolchain instance providing readelf
        binary_path: Path to ELF binary
        section_name: Name of section (e.g., ".hipFatBinSegment")

    Returns:
        Size (sh_size) of the section if found, None otherwise.
    """
    try:
        result = subprocess.run(
            [str(toolchain.readelf), "-S", str(binary_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None

    # Parse section headers
    # Format (two-line entries):
    # Line 1: [Nr] Name              Type             Address           Offset
    # Line 2:      Size              EntSize          Flags  Link  Info  Align
    lines = result.stdout.split("\n")
    for i, line in enumerate(lines):
        if section_name in line:
            parts = line.split()
            # Check if this is a section header line (starts with [Nr])
            if len(parts) >= 5 and parts[0].startswith("["):
                try:
                    # Size is on the next line, first column
                    if i + 1 < len(lines):
                        next_parts = lines[i + 1].split()
                        if len(next_parts) >= 1:
                            size = int(next_parts[0], 16)
                            return size
                except (ValueError, IndexError):
                    continue

    return None


def has_section(
    binary_path: Path,
    section_name: str,
    *,
    toolchain: Toolchain | None = None,
) -> bool:
    """Check if a binary has a specific section.

    Args:
        binary_path: Path to binary
        section_name: Name of section to check for (e.g., ".hip_fatbin", ".rocm_kpack_ref")
        toolchain: Toolchain instance (created if not provided)

    Returns:
        True if section exists, False otherwise

    Note:
        This function abstracts binary format tooling (readelf for ELF, etc.)
        to support cross-platform binary analysis.
    """
    if toolchain is None:
        toolchain = Toolchain()

    try:
        output = toolchain.exec_capture_text(
            [toolchain.readelf, "-S", str(binary_path)]
        )
        return section_name in output
    except Exception:
        return False


def get_section_type(
    binary_path: Path,
    section_name: str,
    *,
    toolchain: Toolchain | None = None,
) -> str | None:
    """Get the type of a section in a binary (e.g., PROGBITS, NOBITS).

    Args:
        binary_path: Path to binary
        section_name: Name of section (e.g., ".hip_fatbin")
        toolchain: Toolchain instance (created if not provided)

    Returns:
        Section type string (e.g., "PROGBITS", "NOBITS"), or None if section doesn't exist

    Note:
        This function abstracts binary format tooling (readelf for ELF, etc.)
        to support cross-platform binary analysis.
    """
    if toolchain is None:
        toolchain = Toolchain()

    try:
        output = toolchain.exec_capture_text(
            [toolchain.readelf, "-S", str(binary_path)]
        )

        # Parse section headers to find the type
        # Format: [Nr] Name              Type             Address           Offset
        for line in output.splitlines():
            if section_name in line:
                parts = line.split()
                # Check if this is a section header line (starts with [Nr])
                if len(parts) >= 3 and parts[0].startswith("["):
                    # Type is at index 2
                    return parts[2]

        return None

    except Exception:
        return None
