"""Compression support for packed kernel archives.

This module provides a pluggable compression interface with map/reduce pattern,
supporting both simple per-kernel compression and complex cross-kernel optimization.

Compression Scheme Design:
- Each compressor has a SCHEME_NAME used in the TOC
- Compressor-specific metadata is stored at the TOC level
- Per-kernel TOC entries reference kernels by ordinal (0..num_kernels-1)
- Runtime initializes compressor from TOC once for efficient random access
"""

from abc import ABC, abstractmethod
from pathlib import Path
import struct
import zstandard as zstd


class CompressionInput:
    """Opaque result from map phase - base class for compressor-specific data.

    Subclasses can hold:
    - Compressed bytes (simple schemes like ZstdCompressor)
    - Raw data + statistics (dictionary training)
    - Fingerprints (deduplication)
    - Block structure analysis (block sorting)
    """

    pass


class Compressor(ABC):
    """Abstract base class for compression strategies with map/reduce pattern.

    Design:
    - SCHEME_NAME: Identifier stored in TOC compression_scheme field
    - prepare_kernel(): Map phase (parallel-safe preprocessing)
    - finalize(): Reduce phase (returns blob + TOC metadata)
    - from_toc(): Initialize compressor from TOC for reading
    - decompress_kernel(): Extract kernel by ordinal
    """

    SCHEME_NAME: str = NotImplemented

    @abstractmethod
    def prepare_kernel(self, kernel_data: bytes, kernel_id: str) -> CompressionInput:
        """Map phase: analyze/preprocess kernel (parallel-safe).

        This method is called in parallel for each kernel. Depending on the
        compression strategy, it may:
        - Compress immediately (simple schemes)
        - Extract samples and statistics (dictionary training)
        - Compute hashes (deduplication)
        - Analyze structure (block sorting)

        Args:
            kernel_data: Raw kernel bytes
            kernel_id: Unique identifier for this kernel (for debugging)

        Returns:
            CompressionInput subclass instance with prepared data
        """
        pass

    @abstractmethod
    def finalize(
        self, inputs: list[tuple[str, CompressionInput]]
    ) -> tuple[bytes, dict[str, object]]:
        """Reduce phase: compress all kernels with cross-kernel optimization (exclusive).

        This method is called once with all prepared inputs. It performs
        cross-kernel optimization and writes the final compressed archive blob.

        For simple schemes, this just concatenates pre-compressed frames.
        For complex schemes, this may build shared dictionaries, deduplicate,
        or perform other global optimizations.

        Args:
            inputs: List of (kernel_id, CompressionInput) tuples from prepare_kernel

        Returns:
            Tuple of (blob_data, toc_metadata) where:
            - blob_data: Opaque compressed blob to write to archive
            - toc_metadata: Compression-specific fields to add to TOC
        """
        pass

    @staticmethod
    @abstractmethod
    def from_toc(toc_data: dict[str, object], file_path: Path) -> "Compressor":
        """Initialize compressor from TOC metadata for reading.

        The runtime calls this once when opening a kpack file, then uses
        the returned compressor instance for all decompress_kernel() calls.

        Args:
            toc_data: The full TOC dictionary from the kpack file
            file_path: Path object for reading blob data

        Returns:
            Compressor instance initialized for reading
        """
        pass

    @abstractmethod
    def decompress_kernel(self, ordinal: int) -> bytes:
        """Runtime: extract and decompress kernel by ordinal.

        The compressor was initialized via from_toc() with all necessary
        metadata to perform efficient random access.

        Args:
            ordinal: Kernel index (0..num_kernels-1)

        Returns:
            Decompressed kernel bytes
        """
        pass


class NoOpCompressionInput(CompressionInput):
    """Compression input for uncompressed data."""

    def __init__(self, data: bytes):
        self.data = data


class NoOpCompressor(Compressor):
    """Passthrough compressor that stores data uncompressed.

    TOC structure:
    - compression_scheme: "none"
    - blobs: [{offset: uint64, size: uint32}, ...] indexed by ordinal

    Useful for testing and as a baseline for comparison.
    """

    SCHEME_NAME = "none"

    def __init__(self):
        """Initialize compressor for writing."""
        # For reading mode
        self._file_path = None
        self._blobs = None

    def prepare_kernel(self, kernel_data: bytes, kernel_id: str) -> CompressionInput:
        """Store kernel data without compression."""
        return NoOpCompressionInput(data=kernel_data)

    def finalize(
        self, inputs: list[tuple[str, CompressionInput]]
    ) -> tuple[bytes, dict[str, object]]:
        """Concatenate all kernel data and build blob metadata.

        Returns:
            (concatenated_blobs, {"blobs": [{"offset": ..., "size": ...}, ...]})
        """
        result = bytearray()
        blobs = []
        current_offset = 0

        for kernel_id, comp_input in inputs:
            assert isinstance(comp_input, NoOpCompressionInput)
            data = comp_input.data

            # Append data
            result.extend(data)

            # Record blob metadata (offset is relative to start of blob section)
            blobs.append({"offset": current_offset, "size": len(data)})
            current_offset += len(data)

        toc_metadata = {"blobs": blobs}
        return bytes(result), toc_metadata

    @staticmethod
    def from_toc(toc_data: dict[str, object], file_path: Path) -> "NoOpCompressor":
        """Initialize from TOC for reading."""
        compressor = NoOpCompressor()
        compressor._file_path = file_path
        compressor._blobs = toc_data["blobs"]
        return compressor

    def decompress_kernel(self, ordinal: int) -> bytes:
        """Extract uncompressed kernel by ordinal."""
        if self._blobs is None or self._file_path is None:
            raise RuntimeError("Compressor not initialized from TOC")

        if ordinal < 0 or ordinal >= len(self._blobs):
            raise ValueError(
                f"Ordinal {ordinal} out of range (0..{len(self._blobs)-1})"
            )

        blob = self._blobs[ordinal]
        offset = blob["offset"]  # Absolute file offset (fixed up by PackArchive.write)
        size = blob["size"]

        # Read from file at absolute offset
        with self._file_path.open("rb") as f:
            f.seek(offset)
            return f.read(size)


class ZstdCompressionInput(CompressionInput):
    """Compression input containing pre-compressed zstd frame."""

    def __init__(self, kernel_id: str, compressed_frame: bytes, original_size: int):
        self.kernel_id = kernel_id
        self.compressed_frame = compressed_frame
        self.original_size = original_size


class ZstdCompressor(Compressor):
    """Per-kernel zstd compression (matches current ROCm practice).

    TOC structure:
    - compression_scheme: "zstd-per-kernel"
    - zstd_offset: uint64 (offset to compressed blob)
    - zstd_size: uint64 (size of compressed blob)

    The compressed blob contains:
    - Header: num_kernels (uint32)
    - Frame entries: [size (uint32), compressed_frame (variable bytes)] * num_kernels
    - Frames are indexed by ordinal in order

    This allows O(1) lookup by building an index on first access.
    """

    SCHEME_NAME = "zstd-per-kernel"

    def __init__(self, compression_level: int = 3):
        """Initialize zstd compressor.

        Args:
            compression_level: Zstd compression level (1-22, default 3)
                              3 is the zstd default, good balance of speed/ratio
        """
        self.compression_level = compression_level

        # For reading mode
        self._file_path = None
        self._zstd_offset = None
        self._zstd_size = None
        self._frame_index = (
            None  # Built on first access: [(offset, size, original_size), ...]
        )
        self._decompressor = None  # Created lazily for reading

    def prepare_kernel(self, kernel_data: bytes, kernel_id: str) -> CompressionInput:
        """Compress kernel immediately (work done in parallel).

        Creates a fresh compressor instance for thread-safety.
        ZstdCompressor objects are not thread-safe, so we create one per call.
        """
        compressor = zstd.ZstdCompressor(level=self.compression_level)
        compressed = compressor.compress(kernel_data)
        return ZstdCompressionInput(
            kernel_id=kernel_id,
            compressed_frame=compressed,
            original_size=len(kernel_data),
        )

    def finalize(
        self, inputs: list[tuple[str, CompressionInput]]
    ) -> tuple[bytes, dict[str, object]]:
        """Concatenate pre-compressed frames sequentially.

        Format:
        [Header]
          num_kernels: uint32

        [Frame entries]
          compressed_size: uint32
          compressed_frame: variable bytes
          (repeated for each kernel in order)

        Returns:
            (compressed_blob, {"zstd_offset": ..., "zstd_size": ...})
        """
        result = bytearray()

        # Write header
        num_kernels = len(inputs)
        result.extend(struct.pack("<I", num_kernels))

        # Write frames sequentially
        for kernel_id, comp_input in inputs:
            assert isinstance(comp_input, ZstdCompressionInput)
            frame = comp_input.compressed_frame

            # Write frame size then frame data
            result.extend(struct.pack("<I", len(frame)))
            result.extend(frame)

        # TOC metadata will be filled in by PackArchive with actual offset/size
        toc_metadata = {
            "zstd_offset": 0,  # Placeholder, filled by PackArchive
            "zstd_size": len(result),
        }
        return bytes(result), toc_metadata

    @staticmethod
    def from_toc(toc_data: dict[str, object], file_path: Path) -> "ZstdCompressor":
        """Initialize from TOC for reading."""
        compressor = ZstdCompressor()
        compressor._file_path = file_path
        compressor._zstd_offset = toc_data["zstd_offset"]
        compressor._zstd_size = toc_data["zstd_size"]
        return compressor

    def _build_frame_index(self) -> None:
        """Build frame index for random access (called once on first access)."""
        if self._frame_index is not None:
            return

        # Read compressed blob
        with self._file_path.open("rb") as f:
            f.seek(self._zstd_offset)
            blob_data = f.read(self._zstd_size)

        # Parse header
        offset = 0
        num_kernels = struct.unpack("<I", blob_data[offset : offset + 4])[0]
        offset += 4

        # Build index of frames
        self._frame_index = []
        for _ in range(num_kernels):
            frame_size = struct.unpack("<I", blob_data[offset : offset + 4])[0]
            offset += 4
            frame_offset = offset
            offset += frame_size

            # Store (offset_in_blob, size) for each frame
            self._frame_index.append((frame_offset, frame_size))

        # Cache blob data for faster access (trade memory for speed)
        self._blob_data = blob_data

    def decompress_kernel(self, ordinal: int) -> bytes:
        """Extract and decompress kernel by ordinal."""
        if self._zstd_offset is None or self._file_path is None:
            raise RuntimeError("Compressor not initialized from TOC")

        # Build index on first access
        self._build_frame_index()

        if ordinal < 0 or ordinal >= len(self._frame_index):
            raise ValueError(
                f"Ordinal {ordinal} out of range (0..{len(self._frame_index)-1})"
            )

        # Get frame from index
        frame_offset, frame_size = self._frame_index[ordinal]
        compressed_frame = self._blob_data[frame_offset : frame_offset + frame_size]

        # Decompress (create decompressor lazily)
        if self._decompressor is None:
            self._decompressor = zstd.ZstdDecompressor()
        return self._decompressor.decompress(compressed_frame)


# Registry of compression schemes
COMPRESSION_SCHEMES = {
    NoOpCompressor.SCHEME_NAME: NoOpCompressor,
    ZstdCompressor.SCHEME_NAME: ZstdCompressor,
}


def create_compressor_from_toc(
    toc_data: dict[str, object], file_path: Path
) -> Compressor:
    """Factory function to create compressor from TOC metadata.

    Args:
        toc_data: Full TOC dictionary
        file_path: Path to kpack file

    Returns:
        Initialized compressor instance

    Raises:
        ValueError: If compression scheme is unknown
    """
    scheme = toc_data.get("compression_scheme", "none")

    if scheme not in COMPRESSION_SCHEMES:
        raise ValueError(f"Unknown compression scheme: {scheme}")

    compressor_class = COMPRESSION_SCHEMES[scheme]
    return compressor_class.from_toc(toc_data, file_path)
