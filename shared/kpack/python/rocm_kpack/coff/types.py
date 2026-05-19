"""
Unified PE/COFF type definitions for 64-bit Windows binaries.

This module consolidates all PE/COFF struct definitions needed for
binary surgery operations on Windows executables and DLLs.

We use dataclasses instead of NamedTuples for mutability - PE surgery
often needs to modify header fields.

References:
- Microsoft PE/COFF Specification
- https://learn.microsoft.com/en-us/windows/win32/debug/pe-format
"""

import struct
from dataclasses import dataclass
from typing import ClassVar

# =============================================================================
# Constants
# =============================================================================

# Alignment constants (typical values)
PAGE_SIZE = 0x1000  # 4KB pages (same as ELF)
FILE_ALIGNMENT_DEFAULT = 0x200  # 512 bytes (typical for PE)
SECTION_ALIGNMENT_DEFAULT = 0x1000  # 4KB (typical for PE)

# DOS Header
DOS_MAGIC = 0x5A4D  # "MZ" in little-endian

# PE Signature
PE_SIGNATURE = b"PE\x00\x00"
PE_SIGNATURE_OFFSET_LOCATION = 0x3C  # Offset in DOS header where e_lfanew lives

# Machine types
IMAGE_FILE_MACHINE_UNKNOWN = 0x0
IMAGE_FILE_MACHINE_I386 = 0x14C
IMAGE_FILE_MACHINE_AMD64 = 0x8664
IMAGE_FILE_MACHINE_ARM64 = 0xAA64

# Optional header magic
IMAGE_NT_OPTIONAL_HDR32_MAGIC = 0x10B
IMAGE_NT_OPTIONAL_HDR64_MAGIC = 0x20B  # PE32+

# Section characteristics
IMAGE_SCN_TYPE_NO_PAD = 0x00000008
IMAGE_SCN_CNT_CODE = 0x00000020
IMAGE_SCN_CNT_INITIALIZED_DATA = 0x00000040
IMAGE_SCN_CNT_UNINITIALIZED_DATA = 0x00000080
IMAGE_SCN_LNK_INFO = 0x00000200
IMAGE_SCN_LNK_REMOVE = 0x00000800
IMAGE_SCN_LNK_COMDAT = 0x00001000
IMAGE_SCN_GPREL = 0x00008000
IMAGE_SCN_ALIGN_1BYTES = 0x00100000
IMAGE_SCN_ALIGN_2BYTES = 0x00200000
IMAGE_SCN_ALIGN_4BYTES = 0x00300000
IMAGE_SCN_ALIGN_8BYTES = 0x00400000
IMAGE_SCN_ALIGN_16BYTES = 0x00500000
IMAGE_SCN_ALIGN_32BYTES = 0x00600000
IMAGE_SCN_ALIGN_64BYTES = 0x00700000
IMAGE_SCN_ALIGN_128BYTES = 0x00800000
IMAGE_SCN_ALIGN_256BYTES = 0x00900000
IMAGE_SCN_ALIGN_512BYTES = 0x00A00000
IMAGE_SCN_ALIGN_1024BYTES = 0x00B00000
IMAGE_SCN_ALIGN_2048BYTES = 0x00C00000
IMAGE_SCN_ALIGN_4096BYTES = 0x00D00000
IMAGE_SCN_ALIGN_8192BYTES = 0x00E00000
IMAGE_SCN_LNK_NRELOC_OVFL = 0x01000000
IMAGE_SCN_MEM_DISCARDABLE = 0x02000000
IMAGE_SCN_MEM_NOT_CACHED = 0x04000000
IMAGE_SCN_MEM_NOT_PAGED = 0x08000000
IMAGE_SCN_MEM_SHARED = 0x10000000
IMAGE_SCN_MEM_EXECUTE = 0x20000000
IMAGE_SCN_MEM_READ = 0x40000000
IMAGE_SCN_MEM_WRITE = 0x80000000

# File characteristics
IMAGE_FILE_RELOCS_STRIPPED = 0x0001
IMAGE_FILE_EXECUTABLE_IMAGE = 0x0002
IMAGE_FILE_LINE_NUMS_STRIPPED = 0x0004
IMAGE_FILE_LOCAL_SYMS_STRIPPED = 0x0008
IMAGE_FILE_LARGE_ADDRESS_AWARE = 0x0020
IMAGE_FILE_32BIT_MACHINE = 0x0100
IMAGE_FILE_DEBUG_STRIPPED = 0x0200
IMAGE_FILE_REMOVABLE_RUN_FROM_SWAP = 0x0400
IMAGE_FILE_NET_RUN_FROM_SWAP = 0x0800
IMAGE_FILE_SYSTEM = 0x1000
IMAGE_FILE_DLL = 0x2000
IMAGE_FILE_UP_SYSTEM_ONLY = 0x4000

# DLL characteristics
IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA = 0x0020
IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE = 0x0040  # ASLR
IMAGE_DLLCHARACTERISTICS_FORCE_INTEGRITY = 0x0080
IMAGE_DLLCHARACTERISTICS_NX_COMPAT = 0x0100
IMAGE_DLLCHARACTERISTICS_NO_ISOLATION = 0x0200
IMAGE_DLLCHARACTERISTICS_NO_SEH = 0x0400
IMAGE_DLLCHARACTERISTICS_NO_BIND = 0x0800
IMAGE_DLLCHARACTERISTICS_APPCONTAINER = 0x1000
IMAGE_DLLCHARACTERISTICS_WDM_DRIVER = 0x2000
IMAGE_DLLCHARACTERISTICS_GUARD_CF = 0x4000
IMAGE_DLLCHARACTERISTICS_TERMINAL_SERVER_AWARE = 0x8000

# Data directory indices
IMAGE_DIRECTORY_ENTRY_EXPORT = 0
IMAGE_DIRECTORY_ENTRY_IMPORT = 1
IMAGE_DIRECTORY_ENTRY_RESOURCE = 2
IMAGE_DIRECTORY_ENTRY_EXCEPTION = 3
IMAGE_DIRECTORY_ENTRY_SECURITY = 4
IMAGE_DIRECTORY_ENTRY_BASERELOC = 5
IMAGE_DIRECTORY_ENTRY_DEBUG = 6
IMAGE_DIRECTORY_ENTRY_ARCHITECTURE = 7
IMAGE_DIRECTORY_ENTRY_GLOBALPTR = 8
IMAGE_DIRECTORY_ENTRY_TLS = 9
IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG = 10
IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT = 11
IMAGE_DIRECTORY_ENTRY_IAT = 12
IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT = 13
IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR = 14
IMAGE_NUMBEROF_DIRECTORY_ENTRIES = 16

# Base relocation types
IMAGE_REL_BASED_ABSOLUTE = 0  # Padding, skip
IMAGE_REL_BASED_HIGH = 1
IMAGE_REL_BASED_LOW = 2
IMAGE_REL_BASED_HIGHLOW = 3  # 32-bit pointer
IMAGE_REL_BASED_HIGHADJ = 4
IMAGE_REL_BASED_DIR64 = 10  # 64-bit pointer

# Structure sizes
DOS_HEADER_SIZE = 64
COFF_HEADER_SIZE = 20
OPTIONAL_HEADER64_SIZE = 240  # Including data directories
DATA_DIRECTORY_SIZE = 8
SECTION_HEADER_SIZE = 40


# =============================================================================
# PE/COFF Structures
# =============================================================================


@dataclass
class DosHeader:
    """DOS MZ header (IMAGE_DOS_HEADER).

    The DOS header is 64 bytes and exists for backwards compatibility.
    The only field we really care about is e_lfanew which points to the PE signature.
    """

    e_magic: int  # "MZ" = 0x5A4D
    e_cblp: int
    e_cp: int
    e_crlc: int
    e_cparhdr: int
    e_minalloc: int
    e_maxalloc: int
    e_ss: int
    e_sp: int
    e_csum: int
    e_ip: int
    e_cs: int
    e_lfarlc: int
    e_ovno: int
    e_res: bytes  # 8 bytes reserved
    e_oemid: int
    e_oeminfo: int
    e_res2: bytes  # 20 bytes reserved
    e_lfanew: int  # Offset to PE signature

    STRUCT_FMT: ClassVar[str] = "<HHHHHHHHHHHHHH8sHH20sI"
    SIZE: ClassVar[int] = 64

    @classmethod
    def from_bytes(cls, data: bytes | bytearray, offset: int = 0) -> "DosHeader":
        """Parse DOS header from binary data."""
        if len(data) < offset + cls.SIZE:
            raise ValueError(
                f"Data too short for DOS header: {len(data)} < {offset + cls.SIZE}"
            )

        fields = struct.unpack_from(cls.STRUCT_FMT, data, offset)

        if fields[0] != DOS_MAGIC:
            raise ValueError(f"Not a DOS/PE file (bad magic: 0x{fields[0]:04X})")

        return cls(*fields)

    def to_bytes(self) -> bytes:
        """Serialize DOS header to binary data."""
        return struct.pack(
            self.STRUCT_FMT,
            self.e_magic,
            self.e_cblp,
            self.e_cp,
            self.e_crlc,
            self.e_cparhdr,
            self.e_minalloc,
            self.e_maxalloc,
            self.e_ss,
            self.e_sp,
            self.e_csum,
            self.e_ip,
            self.e_cs,
            self.e_lfarlc,
            self.e_ovno,
            self.e_res,
            self.e_oemid,
            self.e_oeminfo,
            self.e_res2,
            self.e_lfanew,
        )

    def write_to(self, data: bytearray, offset: int = 0) -> None:
        """Write DOS header to mutable buffer at offset."""
        struct.pack_into(
            self.STRUCT_FMT,
            data,
            offset,
            self.e_magic,
            self.e_cblp,
            self.e_cp,
            self.e_crlc,
            self.e_cparhdr,
            self.e_minalloc,
            self.e_maxalloc,
            self.e_ss,
            self.e_sp,
            self.e_csum,
            self.e_ip,
            self.e_cs,
            self.e_lfarlc,
            self.e_ovno,
            self.e_res,
            self.e_oemid,
            self.e_oeminfo,
            self.e_res2,
            self.e_lfanew,
        )


@dataclass
class CoffHeader:
    """COFF file header (IMAGE_FILE_HEADER).

    This 20-byte header comes right after the PE signature.
    """

    Machine: int  # Target machine type (e.g., AMD64)
    NumberOfSections: int
    TimeDateStamp: int
    PointerToSymbolTable: int  # Usually 0 for executables
    NumberOfSymbols: int  # Usually 0 for executables
    SizeOfOptionalHeader: int
    Characteristics: int  # File characteristics flags

    STRUCT_FMT: ClassVar[str] = "<HHIIIHH"
    SIZE: ClassVar[int] = 20

    @classmethod
    def from_bytes(cls, data: bytes | bytearray, offset: int = 0) -> "CoffHeader":
        """Parse COFF header from binary data."""
        if len(data) < offset + cls.SIZE:
            raise ValueError(
                f"Data too short for COFF header: {len(data)} < {offset + cls.SIZE}"
            )

        fields = struct.unpack_from(cls.STRUCT_FMT, data, offset)
        return cls(*fields)

    def to_bytes(self) -> bytes:
        """Serialize COFF header to binary data."""
        return struct.pack(
            self.STRUCT_FMT,
            self.Machine,
            self.NumberOfSections,
            self.TimeDateStamp,
            self.PointerToSymbolTable,
            self.NumberOfSymbols,
            self.SizeOfOptionalHeader,
            self.Characteristics,
        )

    def write_to(self, data: bytearray, offset: int) -> None:
        """Write COFF header to mutable buffer at offset."""
        struct.pack_into(
            self.STRUCT_FMT,
            data,
            offset,
            self.Machine,
            self.NumberOfSections,
            self.TimeDateStamp,
            self.PointerToSymbolTable,
            self.NumberOfSymbols,
            self.SizeOfOptionalHeader,
            self.Characteristics,
        )

    @property
    def is_dll(self) -> bool:
        """Check if this is a DLL."""
        return bool(self.Characteristics & IMAGE_FILE_DLL)

    @property
    def is_executable(self) -> bool:
        """Check if this is an executable image."""
        return bool(self.Characteristics & IMAGE_FILE_EXECUTABLE_IMAGE)


@dataclass
class DataDirectory:
    """Data directory entry (IMAGE_DATA_DIRECTORY).

    Each entry points to a data structure in the image.
    """

    VirtualAddress: int  # RVA of the data
    Size: int  # Size of the data

    STRUCT_FMT: ClassVar[str] = "<II"
    SIZE: ClassVar[int] = 8

    @classmethod
    def from_bytes(cls, data: bytes | bytearray, offset: int = 0) -> "DataDirectory":
        """Parse data directory from binary data."""
        if len(data) < offset + cls.SIZE:
            raise ValueError("Data too short for data directory")
        fields = struct.unpack_from(cls.STRUCT_FMT, data, offset)
        return cls(*fields)

    def to_bytes(self) -> bytes:
        """Serialize data directory to binary data."""
        return struct.pack(self.STRUCT_FMT, self.VirtualAddress, self.Size)

    def write_to(self, data: bytearray, offset: int) -> None:
        """Write data directory to mutable buffer at offset."""
        struct.pack_into(self.STRUCT_FMT, data, offset, self.VirtualAddress, self.Size)

    @property
    def is_present(self) -> bool:
        """Check if this data directory is present."""
        return self.VirtualAddress != 0 or self.Size != 0


@dataclass
class OptionalHeader64:
    """PE32+ optional header (IMAGE_OPTIONAL_HEADER64).

    This header is required for executable images despite its name.
    The "optional" refers to object files which don't have it.

    Note: Data directories are stored separately as a list.
    """

    Magic: int  # 0x20B for PE32+
    MajorLinkerVersion: int
    MinorLinkerVersion: int
    SizeOfCode: int
    SizeOfInitializedData: int
    SizeOfUninitializedData: int
    AddressOfEntryPoint: int
    BaseOfCode: int
    ImageBase: int  # 8 bytes for PE32+
    SectionAlignment: int
    FileAlignment: int
    MajorOperatingSystemVersion: int
    MinorOperatingSystemVersion: int
    MajorImageVersion: int
    MinorImageVersion: int
    MajorSubsystemVersion: int
    MinorSubsystemVersion: int
    Win32VersionValue: int
    SizeOfImage: int
    SizeOfHeaders: int
    CheckSum: int
    Subsystem: int
    DllCharacteristics: int
    SizeOfStackReserve: int  # 8 bytes for PE32+
    SizeOfStackCommit: int  # 8 bytes for PE32+
    SizeOfHeapReserve: int  # 8 bytes for PE32+
    SizeOfHeapCommit: int  # 8 bytes for PE32+
    LoaderFlags: int
    NumberOfRvaAndSizes: int

    # Struct format for the fixed part (before data directories)
    # 2 + 1 + 1 + 4*6 + 8 + 4*9 + 2*2 + 8*4 + 4*2 = 112 bytes
    STRUCT_FMT: ClassVar[str] = "<HBBIIIIIQIIHHHHHH" "IIIIHHQQQQII"
    SIZE: ClassVar[int] = 112  # Fixed part, before data directories

    @classmethod
    def from_bytes(cls, data: bytes | bytearray, offset: int = 0) -> "OptionalHeader64":
        """Parse optional header from binary data."""
        if len(data) < offset + cls.SIZE:
            raise ValueError(
                f"Data too short for optional header: {len(data)} < {offset + cls.SIZE}"
            )

        fields = struct.unpack_from(cls.STRUCT_FMT, data, offset)

        if fields[0] != IMAGE_NT_OPTIONAL_HDR64_MAGIC:
            raise ValueError(
                f"Not a PE32+ file (magic: 0x{fields[0]:04X}, "
                f"expected 0x{IMAGE_NT_OPTIONAL_HDR64_MAGIC:04X})"
            )

        return cls(*fields)

    def to_bytes(self) -> bytes:
        """Serialize optional header to binary data."""
        return struct.pack(
            self.STRUCT_FMT,
            self.Magic,
            self.MajorLinkerVersion,
            self.MinorLinkerVersion,
            self.SizeOfCode,
            self.SizeOfInitializedData,
            self.SizeOfUninitializedData,
            self.AddressOfEntryPoint,
            self.BaseOfCode,
            self.ImageBase,
            self.SectionAlignment,
            self.FileAlignment,
            self.MajorOperatingSystemVersion,
            self.MinorOperatingSystemVersion,
            self.MajorImageVersion,
            self.MinorImageVersion,
            self.MajorSubsystemVersion,
            self.MinorSubsystemVersion,
            self.Win32VersionValue,
            self.SizeOfImage,
            self.SizeOfHeaders,
            self.CheckSum,
            self.Subsystem,
            self.DllCharacteristics,
            self.SizeOfStackReserve,
            self.SizeOfStackCommit,
            self.SizeOfHeapReserve,
            self.SizeOfHeapCommit,
            self.LoaderFlags,
            self.NumberOfRvaAndSizes,
        )

    def write_to(self, data: bytearray, offset: int) -> None:
        """Write optional header to mutable buffer at offset."""
        struct.pack_into(
            self.STRUCT_FMT,
            data,
            offset,
            self.Magic,
            self.MajorLinkerVersion,
            self.MinorLinkerVersion,
            self.SizeOfCode,
            self.SizeOfInitializedData,
            self.SizeOfUninitializedData,
            self.AddressOfEntryPoint,
            self.BaseOfCode,
            self.ImageBase,
            self.SectionAlignment,
            self.FileAlignment,
            self.MajorOperatingSystemVersion,
            self.MinorOperatingSystemVersion,
            self.MajorImageVersion,
            self.MinorImageVersion,
            self.MajorSubsystemVersion,
            self.MinorSubsystemVersion,
            self.Win32VersionValue,
            self.SizeOfImage,
            self.SizeOfHeaders,
            self.CheckSum,
            self.Subsystem,
            self.DllCharacteristics,
            self.SizeOfStackReserve,
            self.SizeOfStackCommit,
            self.SizeOfHeapReserve,
            self.SizeOfHeapCommit,
            self.LoaderFlags,
            self.NumberOfRvaAndSizes,
        )

    @property
    def has_aslr(self) -> bool:
        """Check if ASLR (dynamic base) is enabled."""
        return bool(self.DllCharacteristics & IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE)


@dataclass
class SectionHeader:
    """PE/COFF section header (IMAGE_SECTION_HEADER).

    Each section header is 40 bytes.
    """

    Name: bytes  # 8 bytes, null-padded (NOT null-terminated if 8 chars)
    VirtualSize: int  # Size in memory (can be > SizeOfRawData)
    VirtualAddress: int  # RVA of section
    SizeOfRawData: int  # Size in file (rounded to FileAlignment)
    PointerToRawData: int  # File offset
    PointerToRelocations: int  # Usually 0 for executables
    PointerToLinenumbers: int  # Deprecated, usually 0
    NumberOfRelocations: int
    NumberOfLinenumbers: int
    Characteristics: int  # Section flags

    STRUCT_FMT: ClassVar[str] = "<8sIIIIIIHHI"
    SIZE: ClassVar[int] = 40

    @classmethod
    def from_bytes(cls, data: bytes | bytearray, offset: int = 0) -> "SectionHeader":
        """Parse section header from binary data."""
        if len(data) < offset + cls.SIZE:
            raise ValueError(
                f"Data too short for section header: {len(data)} < {offset + cls.SIZE}"
            )

        fields = struct.unpack_from(cls.STRUCT_FMT, data, offset)
        return cls(*fields)

    def to_bytes(self) -> bytes:
        """Serialize section header to binary data."""
        return struct.pack(
            self.STRUCT_FMT,
            self.Name,
            self.VirtualSize,
            self.VirtualAddress,
            self.SizeOfRawData,
            self.PointerToRawData,
            self.PointerToRelocations,
            self.PointerToLinenumbers,
            self.NumberOfRelocations,
            self.NumberOfLinenumbers,
            self.Characteristics,
        )

    def write_to(self, data: bytearray, offset: int) -> None:
        """Write section header to mutable buffer at offset."""
        struct.pack_into(
            self.STRUCT_FMT,
            data,
            offset,
            self.Name,
            self.VirtualSize,
            self.VirtualAddress,
            self.SizeOfRawData,
            self.PointerToRawData,
            self.PointerToRelocations,
            self.PointerToLinenumbers,
            self.NumberOfRelocations,
            self.NumberOfLinenumbers,
            self.Characteristics,
        )

    @property
    def name_str(self) -> str:
        """Get section name as string (strips null padding)."""
        # Name is 8 bytes, may or may not be null-terminated
        null_pos = self.Name.find(b"\x00")
        if null_pos >= 0:
            return self.Name[:null_pos].decode("ascii", errors="replace")
        return self.Name.decode("ascii", errors="replace")

    @property
    def end_rva(self) -> int:
        """RVA of end of section in memory."""
        return self.VirtualAddress + self.VirtualSize

    @property
    def end_file_offset(self) -> int:
        """File offset of end of section data."""
        return self.PointerToRawData + self.SizeOfRawData

    @property
    def is_code(self) -> bool:
        """Check if this section contains code."""
        return bool(self.Characteristics & IMAGE_SCN_CNT_CODE)

    @property
    def is_initialized_data(self) -> bool:
        """Check if this section contains initialized data."""
        return bool(self.Characteristics & IMAGE_SCN_CNT_INITIALIZED_DATA)

    @property
    def is_uninitialized_data(self) -> bool:
        """Check if this section contains uninitialized data (BSS)."""
        return bool(self.Characteristics & IMAGE_SCN_CNT_UNINITIALIZED_DATA)

    @property
    def is_readable(self) -> bool:
        """Check if this section is readable."""
        return bool(self.Characteristics & IMAGE_SCN_MEM_READ)

    @property
    def is_writable(self) -> bool:
        """Check if this section is writable."""
        return bool(self.Characteristics & IMAGE_SCN_MEM_WRITE)

    @property
    def is_executable(self) -> bool:
        """Check if this section is executable."""
        return bool(self.Characteristics & IMAGE_SCN_MEM_EXECUTE)

    def contains_rva(self, rva: int) -> bool:
        """Check if an RVA falls within this section."""
        return self.VirtualAddress <= rva < self.end_rva

    def contains_file_offset(self, offset: int) -> bool:
        """Check if a file offset falls within this section's raw data."""
        if self.SizeOfRawData == 0:
            return False
        return self.PointerToRawData <= offset < self.end_file_offset


@dataclass
class BaseRelocationBlock:
    """Base relocation block header.

    The base relocation table consists of blocks, each covering a 4KB page.
    Each block has this header followed by TypeOffset entries.
    """

    PageRVA: int  # Base RVA for this block (page-aligned)
    BlockSize: int  # Size including header and all entries

    STRUCT_FMT: ClassVar[str] = "<II"
    SIZE: ClassVar[int] = 8

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray, offset: int = 0
    ) -> "BaseRelocationBlock":
        """Parse relocation block header from binary data."""
        if len(data) < offset + cls.SIZE:
            raise ValueError("Data too short for relocation block")
        fields = struct.unpack_from(cls.STRUCT_FMT, data, offset)
        return cls(*fields)

    def to_bytes(self) -> bytes:
        """Serialize relocation block to binary data."""
        return struct.pack(self.STRUCT_FMT, self.PageRVA, self.BlockSize)

    @property
    def num_entries(self) -> int:
        """Number of TypeOffset entries in this block."""
        return (self.BlockSize - self.SIZE) // 2


@dataclass
class BaseRelocationEntry:
    """Single base relocation entry (2 bytes).

    Each entry is a 16-bit value:
    - High 4 bits: relocation type
    - Low 12 bits: offset within the page
    """

    raw: int  # 16-bit packed value

    STRUCT_FMT: ClassVar[str] = "<H"
    SIZE: ClassVar[int] = 2

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray, offset: int = 0
    ) -> "BaseRelocationEntry":
        """Parse relocation entry from binary data."""
        if len(data) < offset + cls.SIZE:
            raise ValueError("Data too short for relocation entry")
        (raw,) = struct.unpack_from(cls.STRUCT_FMT, data, offset)
        return cls(raw)

    def to_bytes(self) -> bytes:
        """Serialize relocation entry to binary data."""
        return struct.pack(self.STRUCT_FMT, self.raw)

    @property
    def reloc_type(self) -> int:
        """Relocation type (high 4 bits)."""
        return self.raw >> 12

    @property
    def offset(self) -> int:
        """Offset within page (low 12 bits)."""
        return self.raw & 0xFFF

    @property
    def is_absolute(self) -> bool:
        """Check if this is a padding/skip entry."""
        return self.reloc_type == IMAGE_REL_BASED_ABSOLUTE

    @property
    def is_dir64(self) -> bool:
        """Check if this is a 64-bit pointer relocation."""
        return self.reloc_type == IMAGE_REL_BASED_DIR64

    @property
    def is_highlow(self) -> bool:
        """Check if this is a 32-bit pointer relocation."""
        return self.reloc_type == IMAGE_REL_BASED_HIGHLOW


# =============================================================================
# Helper Functions
# =============================================================================


def round_up_to_alignment(value: int, alignment: int) -> int:
    """Round value up to next alignment boundary."""
    if alignment == 0:
        return value
    return (value + alignment - 1) & ~(alignment - 1)


def round_down_to_alignment(value: int, alignment: int) -> int:
    """Round value down to previous alignment boundary."""
    if alignment == 0:
        return value
    return value & ~(alignment - 1)


def round_up_to_page(addr: int) -> int:
    """Round address up to next page boundary."""
    return round_up_to_alignment(addr, PAGE_SIZE)


def round_down_to_page(addr: int) -> int:
    """Round address down to previous page boundary."""
    return round_down_to_alignment(addr, PAGE_SIZE)


def section_name_to_bytes(name: str) -> bytes:
    """Convert section name string to 8-byte padded bytes.

    Section names are limited to 8 characters in PE/COFF.
    """
    if len(name) > 8:
        raise ValueError(f"Section name too long (max 8 chars): {name}")
    return name.encode("ascii").ljust(8, b"\x00")
