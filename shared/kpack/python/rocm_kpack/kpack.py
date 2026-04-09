"""Packed kernel archive format for efficient kernel storage and distribution.

This module provides the PackedKernelArchive class for creating and reading
.kpack files - binary archives with aligned blobs and MessagePack TOC.
"""

import struct
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import msgpack

from .compression import (
    Compressor,
    CompressionInput,
    NoOpCompressor,
    create_compressor_from_toc,
)


@dataclass
class PreparedKernel:
    """Opaque result from prepare_kernel() - holds compressed/prepared kernel data.

    This object can be created concurrently (via prepare_kernel) and then
    added to the archive sequentially (via add_kernel with locking).
    """

    relative_path: str
    gfx_arch: str
    compression_input: CompressionInput
    kernel_id: str  # For debugging
    original_size: int
    metadata: dict[str, Any] | None = None


class PackedKernelArchive:
    """A packed kernel archive containing kernels for a specific architecture family.

    File format optimized for fast reads and memory-mapping:

    [Fixed Binary Header - 16 bytes]
      Magic:      "KPAK" (4 bytes)
      Version:    uint32 (4 bytes)
      TOC Offset: uint64 (8 bytes)

    [Padding to 64-byte boundary]

    [Blob data - format depends on compression scheme]

    [MessagePack TOC at TOC Offset]
    {
      "format_version": 1,
      "group_name": "blas",
      "gfx_arch_family": "gfx100X",
      "gfx_arches": [...],
      "compression_scheme": "none" | "zstd-per-kernel",

      # Compression scheme-specific fields:
      # For "none":
      "blobs": [
        {"offset": 64, "size": 7472},      # ordinal 0
        {"offset": 7552, "size": 4928},    # ordinal 1
        ...
      ],

      # For "zstd-per-kernel":
      "zstd_offset": 64,
      "zstd_size": 12345,

      "toc": {
        "bin/hipcc": {
          "gfx1030": {
            "type": "hsaco",
            "ordinal": 0,           # index into compression blob/array
            "original_size": 7472   # uncompressed size (optional)
          }
        }
      }
    }

    Compression Design:
    - compression_scheme at TOC level identifies compressor ("none", "zstd-per-kernel")
    - Compressor-specific metadata stored at TOC level (blobs array, zstd_offset/size, etc.)
    - Per-kernel TOC entries reference kernels by ordinal (0..num_kernels-1)
    - Runtime initializes compressor from TOC once, then uses ordinals for O(1) lookups
    - This allows efficient random access decompression with proper lookup tables
    """

    MAGIC = b"KPAK"
    FORMAT_VERSION = 1
    HEADER_SIZE = 16  # 4 (magic) + 4 (version) + 8 (toc_offset)
    BLOB_ALIGNMENT = 64

    def __init__(
        self,
        group_name: str,
        gfx_arch_family: str,
        gfx_arches: list[str],
        output_path: Path | None = None,
        compressor: Compressor | None = None,
    ):
        """Initialize a new packed kernel archive.

        Args:
            group_name: Name of the build slice (e.g., "blas", "torch")
            gfx_arch_family: Architecture family identifier (e.g., "gfx1100", "gfx100X")
            gfx_arches: List of actual architectures in this family
            output_path: If provided, enables streaming write mode where blobs are
                        written to disk immediately. Call finalize() when done.
                        If None, accumulates in memory until write() is called.
                        Note: Streaming mode not yet supported.
            compressor: Compressor for kernel data. If None, defaults to NoOpCompressor.
                       All kernels are processed using map/reduce pattern.
                       Must call finalize_archive() before write().
        """
        self.group_name = group_name
        self.gfx_arch_family = gfx_arch_family
        self.gfx_arches = gfx_arches
        # TOC: binary_path -> gfx_arch -> entry (with ordinal)
        self.toc: dict[str, dict[str, dict[str, Any]]] = {}

        # File path for read-mode archives
        self._file_path: Path | None = None

        # Compression support - always use a compressor
        if compressor is None:
            compressor = NoOpCompressor()

        self._compressor: Compressor = compressor
        self._compression_inputs: list[tuple[str, CompressionInput]] = []
        self._compressed_blob: bytes | None = None
        self._compression_metadata: dict[str, Any] | None = None
        self._archive_finalized: bool = False
        self._kernel_ordinal_counter: int = 0  # Next ordinal to assign

        # Streaming write mode not yet supported
        if output_path is not None:
            raise ValueError(
                "Streaming write mode not yet supported. "
                "Use in-memory mode (output_path=None) and call write(path) when done."
            )

    def prepare_kernel(
        self,
        relative_path: str,
        gfx_arch: str,
        hsaco_data: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> PreparedKernel:
        """Prepare a kernel for addition to the archive (concurrent-safe).

        This method can be called concurrently from multiple threads to prepare
        kernels in parallel. The actual addition to the archive (TOC manipulation)
        is done later via add_kernel() which requires locking.

        Args:
            relative_path: Path to binary relative to install tree root
            gfx_arch: GPU architecture (e.g., "gfx1100")
            hsaco_data: Raw HSACO kernel data
            metadata: Optional metadata dictionary for extensibility

        Returns:
            PreparedKernel object to pass to add_kernel()
        """
        # Normalize path to use forward slashes (replace backslashes)
        relative_path = relative_path.replace("\\", "/")

        # Generate unique kernel ID for debugging
        kernel_id = f"{relative_path}@{gfx_arch}"

        # Compress/prepare kernel (map phase - concurrent-safe)
        compression_input = self._compressor.prepare_kernel(hsaco_data, kernel_id)

        return PreparedKernel(
            relative_path=relative_path,
            gfx_arch=gfx_arch,
            compression_input=compression_input,
            kernel_id=kernel_id,
            original_size=len(hsaco_data),
            metadata=metadata,
        )

    def add_kernel(self, prepared: PreparedKernel) -> None:
        """Add a kernel to the archive (cheap metadata manipulation - lock this).

        Call prepare_kernel() first to create the PreparedKernel, which separates
        concurrent kernel preparation from sequential TOC manipulation for better
        parallelism.

        Args:
            prepared: PreparedKernel from prepare_kernel()

        Raises:
            ValueError: If kernel already exists for (relative_path, gfx_arch)
        """
        # Extract fields from PreparedKernel
        relative_path = prepared.relative_path
        gfx_arch = prepared.gfx_arch
        compression_input = prepared.compression_input
        kernel_id = prepared.kernel_id
        original_size = prepared.original_size
        metadata = prepared.metadata

        # Check for duplicates
        if relative_path in self.toc:
            if gfx_arch in self.toc[relative_path]:
                raise ValueError(
                    f"Kernel already exists for {relative_path} @ {gfx_arch}"
                )

        # Add to TOC (cheap metadata manipulation)
        if relative_path not in self.toc:
            self.toc[relative_path] = {}

        entry = {
            "type": "hsaco",
            "ordinal": self._kernel_ordinal_counter,
            "original_size": original_size,
        }
        if metadata:
            entry["metadata"] = metadata

        # Store compression input for finalize
        self._compression_inputs.append((kernel_id, compression_input))
        self._kernel_ordinal_counter += 1

        self.toc[relative_path][gfx_arch] = entry

    def finalize_archive(self) -> None:
        """Finalize archive by running reduce phase.

        This must be called after all kernels are added but before write().
        This is the "reduce" phase that performs cross-kernel optimization
        (compression, deduplication, etc.) and produces the final blob(s).

        Raises:
            RuntimeError: If already finalized
        """
        if self._archive_finalized:
            raise RuntimeError("Archive already finalized")

        # Run reduce phase: finalize all compression inputs
        # Returns (blob_data, toc_metadata)
        self._compressed_blob, self._compression_metadata = self._compressor.finalize(
            self._compression_inputs
        )
        self._archive_finalized = True

        # Clear compression inputs to free memory
        self._compression_inputs.clear()

    def write(self, output_path: Path) -> None:
        """Write the packed archive to a file.

        Must call finalize_archive() before calling this method.

        Format:
        1. Write fixed header (magic, version, toc_offset placeholder)
        2. Pad to BLOB_ALIGNMENT boundary
        3. Write compressed blob
        4. Write MessagePack TOC at end
        5. Seek back and update header with TOC offset

        Args:
            output_path: Path where .kpack file will be written

        Raises:
            RuntimeError: If archive not finalized
        """
        if not self._archive_finalized:
            raise RuntimeError("Archive not finalized. Call finalize_archive() first.")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("wb") as f:
            # Write header with placeholder TOC offset
            header = struct.pack(
                "<4sIQ",  # little-endian: 4-byte string, uint32, uint64
                self.MAGIC,
                self.FORMAT_VERSION,
                0,  # TOC offset placeholder
            )
            f.write(header)

            # Pad to first blob alignment boundary
            current_pos = f.tell()
            padding = (
                self.BLOB_ALIGNMENT - (current_pos % self.BLOB_ALIGNMENT)
            ) % self.BLOB_ALIGNMENT
            f.write(b"\x00" * padding)

            # Write compressed blob
            blob_start_offset = f.tell()
            f.write(self._compressed_blob)

            # Build TOC with compression metadata
            compression_scheme = self._compressor.SCHEME_NAME
            toc_metadata = {"compression_scheme": compression_scheme}

            # Add compression-specific metadata (blobs array or zstd_offset/size)
            toc_metadata.update(self._compression_metadata)

            # For schemes that use offsets, fix up the placeholder offsets
            if compression_scheme == "zstd-per-kernel":
                toc_metadata["zstd_offset"] = blob_start_offset
            elif compression_scheme == "none":
                # Fix up blob offsets to be absolute file offsets
                for blob in toc_metadata["blobs"]:
                    blob["offset"] += blob_start_offset

            # Write MessagePack TOC
            toc_offset = f.tell()
            toc_data = {
                "format_version": self.FORMAT_VERSION,
                "group_name": self.group_name,
                "gfx_arch_family": self.gfx_arch_family,
                "gfx_arches": self.gfx_arches,
                "toc": self.toc,
                **toc_metadata,
            }
            msgpack.pack(toc_data, f, use_bin_type=True)

            # Backpatch header with TOC offset
            f.seek(8)  # Skip magic (4 bytes) and version (4 bytes)
            f.write(struct.pack("<Q", toc_offset))

    @staticmethod
    def read(input_path: Path) -> "PackedKernelArchive":
        """Read a packed archive from a file.

        Reads the binary header, seeks to the TOC, and loads metadata.
        Automatically creates the appropriate compressor from the TOC.
        Kernel data remains in the file and is accessed on-demand.

        Args:
            input_path: Path to .kpack file

        Returns:
            PackedKernelArchive instance loaded from file

        Raises:
            ValueError: If magic number, format version is invalid, or
                       compression scheme is unknown
        """
        with input_path.open("rb") as f:
            # Read and validate header
            header_bytes = f.read(PackedKernelArchive.HEADER_SIZE)
            magic, version, toc_offset = struct.unpack("<4sIQ", header_bytes)

            if magic != PackedKernelArchive.MAGIC:
                raise ValueError(
                    f"Invalid magic number: {magic!r} (expected {PackedKernelArchive.MAGIC!r})"
                )

            if version != PackedKernelArchive.FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported format version: {version} (expected {PackedKernelArchive.FORMAT_VERSION})"
                )

            # Seek to TOC and load it
            f.seek(toc_offset)
            toc_data = msgpack.unpack(f, raw=False)

        # Create archive instance
        archive = PackedKernelArchive(
            group_name=toc_data["group_name"],
            gfx_arch_family=toc_data["gfx_arch_family"],
            gfx_arches=toc_data["gfx_arches"],
        )
        archive.toc = toc_data["toc"]
        archive._file_path = input_path

        # Initialize compressor from TOC
        archive._compressor = create_compressor_from_toc(toc_data, input_path)
        archive._archive_finalized = True

        return archive

    def get_kernel(self, relative_path: str, gfx_arch: str) -> bytes | None:
        """Retrieve kernel data for a specific binary and architecture.

        Automatically decompresses using the configured compressor.

        Args:
            relative_path: Path to binary relative to install tree root
            gfx_arch: GPU architecture

        Returns:
            Kernel data bytes (decompressed), or None if not found
        """
        # Normalize path to use forward slashes (replace backslashes)
        relative_path = relative_path.replace("\\", "/")

        if relative_path not in self.toc:
            return None
        if gfx_arch not in self.toc[relative_path]:
            return None

        entry = self.toc[relative_path][gfx_arch]
        ordinal = entry["ordinal"]

        # Read mode: decompress using ordinal
        if self._archive_finalized:
            return self._compressor.decompress_kernel(ordinal)

        # Building mode: this shouldn't happen - archive must be finalized before reading
        raise RuntimeError(
            "Cannot get_kernel() before archive is finalized. Call finalize_archive() first."
        )

    @staticmethod
    def compute_pack_filename(group_name: str, gfx_arch_family: str) -> str:
        """Compute the standard filename for a pack file.

        Args:
            group_name: Build slice name
            gfx_arch_family: Architecture family

        Returns:
            Filename like "blas-gfx100X.kpack"
        """
        return f"{group_name}-{gfx_arch_family}.kpack"

    def __repr__(self) -> str:
        """String representation of the archive."""
        num_binaries = len(self.toc)
        num_kernels = sum(len(arches) for arches in self.toc.values())
        return (
            f"PackedKernelArchive("
            f"group={self.group_name}, "
            f"family={self.gfx_arch_family}, "
            f"binaries={num_binaries}, "
            f"kernels={num_kernels})"
        )
