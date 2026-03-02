"""
Unified ELF type definitions for 64-bit little-endian ELF.

This module consolidates all ELF struct definitions that were previously
duplicated across elf_modify_load.py and elf_offload_kpacker.py.

We use dataclasses instead of NamedTuples for mutability - ELF surgery
often needs to modify header fields.
"""

import struct
from dataclasses import dataclass
from typing import ClassVar

# =============================================================================
# Constants
# =============================================================================

PAGE_SIZE = 0x1000  # 4KB pages
ELF_MAGIC = b"\x7fELF"

# ELF header offsets (for direct byte access)
E_IDENT_OFFSET = 0
E_IDENT_SIZE = 16
E_TYPE_OFFSET = 16
E_MACHINE_OFFSET = 18
E_VERSION_OFFSET = 20
E_ENTRY_OFFSET = 24
E_PHOFF_OFFSET = 32
E_SHOFF_OFFSET = 40
E_FLAGS_OFFSET = 48
E_EHSIZE_OFFSET = 52
E_PHENTSIZE_OFFSET = 54
E_PHNUM_OFFSET = 56
E_SHENTSIZE_OFFSET = 58
E_SHNUM_OFFSET = 60
E_SHSTRNDX_OFFSET = 62

ELF64_EHDR_SIZE = 64
ELF64_PHDR_SIZE = 56
ELF64_SHDR_SIZE = 64

# ELF type (e_type)
ET_NONE = 0
ET_REL = 1
ET_EXEC = 2
ET_DYN = 3  # Shared object (or PIE executable)
ET_CORE = 4

# Program header types (p_type)
PT_NULL = 0
PT_LOAD = 1
PT_DYNAMIC = 2
PT_INTERP = 3
PT_NOTE = 4
PT_SHLIB = 5
PT_PHDR = 6
PT_TLS = 7
PT_GNU_EH_FRAME = 0x6474E550
PT_GNU_STACK = 0x6474E551
PT_GNU_RELRO = 0x6474E552

# Program header flags (p_flags)
PF_X = 0x1  # Execute
PF_W = 0x2  # Write
PF_R = 0x4  # Read

# Section header types (sh_type)
SHT_NULL = 0
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4
SHT_HASH = 5
SHT_DYNAMIC = 6
SHT_NOTE = 7
SHT_NOBITS = 8
SHT_REL = 9
SHT_SHLIB = 10
SHT_DYNSYM = 11

# Section flags (sh_flags)
SHF_WRITE = 0x1
SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4
SHF_MERGE = 0x10
SHF_STRINGS = 0x20
SHF_INFO_LINK = 0x40
SHF_LINK_ORDER = 0x80
SHF_GROUP = 0x200

# Relocation types (x86_64)
R_X86_64_NONE = 0
R_X86_64_64 = 1
R_X86_64_PC32 = 2
R_X86_64_GOT32 = 3
R_X86_64_PLT32 = 4
R_X86_64_COPY = 5
R_X86_64_GLOB_DAT = 6
R_X86_64_JUMP_SLOT = 7
R_X86_64_RELATIVE = 8  # Used for PIE/shared library pointer adjustments

# Dynamic section tags (d_tag)
DT_NULL = 0
DT_NEEDED = 1
DT_PLTRELSZ = 2
DT_PLTGOT = 3
DT_HASH = 4
DT_STRTAB = 5
DT_SYMTAB = 6
DT_RELA = 7
DT_RELASZ = 8
DT_RELAENT = 9
DT_STRSZ = 10
DT_SYMENT = 11
DT_INIT = 12
DT_FINI = 13
DT_SONAME = 14
DT_RPATH = 15
DT_SYMBOLIC = 16
DT_REL = 17
DT_RELSZ = 18
DT_RELENT = 19
DT_PLTREL = 20
DT_DEBUG = 21
DT_TEXTREL = 22
DT_JMPREL = 23
DT_BIND_NOW = 24
DT_INIT_ARRAY = 25
DT_FINI_ARRAY = 26
DT_INIT_ARRAYSZ = 27
DT_FINI_ARRAYSZ = 28
DT_RUNPATH = 29
DT_FLAGS = 30
DT_PREINIT_ARRAY = 32
DT_PREINIT_ARRAYSZ = 33
DT_SYMTAB_SHNDX = 34
DT_VERSYM = 0x6FFFFFF0
DT_VERDEF = 0x6FFFFFFC
DT_VERNEED = 0x6FFFFFFE

# Dynamic section tags that contain virtual addresses (not sizes or counts)
DT_ADDR_TAGS: frozenset[int] = frozenset(
    {
        DT_PLTGOT,
        DT_HASH,
        DT_STRTAB,
        DT_SYMTAB,
        DT_RELA,
        DT_INIT,
        DT_FINI,
        DT_REL,
        DT_JMPREL,
        DT_INIT_ARRAY,
        DT_FINI_ARRAY,
        DT_PREINIT_ARRAY,
        DT_SYMTAB_SHNDX,
        DT_VERSYM,
        DT_VERDEF,
        DT_VERNEED,
    }
)


# =============================================================================
# ELF Structures
# =============================================================================


@dataclass
class ElfHeader:
    """ELF64 file header (Elf64_Ehdr).

    All fields match the standard ELF64 header layout.
    """

    e_ident: bytes  # 16 bytes: magic, class, endianness, version, OS/ABI, padding
    e_type: int  # Object file type (ET_*)
    e_machine: int  # Architecture (EM_*)
    e_version: int  # ELF version
    e_entry: int  # Entry point virtual address
    e_phoff: int  # Program header table file offset
    e_shoff: int  # Section header table file offset
    e_flags: int  # Processor-specific flags
    e_ehsize: int  # ELF header size
    e_phentsize: int  # Program header entry size
    e_phnum: int  # Number of program headers
    e_shentsize: int  # Section header entry size
    e_shnum: int  # Number of section headers
    e_shstrndx: int  # Section name string table index

    # Struct format for parsing/writing (little-endian)
    STRUCT_FMT: ClassVar[str] = "<16sHHIQQQIHHHHHH"

    @classmethod
    def from_bytes(cls, data: bytes | bytearray) -> "ElfHeader":
        """Parse ELF header from binary data.

        Args:
            data: At least 64 bytes of ELF header data

        Returns:
            Parsed ElfHeader

        Raises:
            ValueError: If not a valid 64-bit little-endian ELF
        """
        if len(data) < ELF64_EHDR_SIZE:
            raise ValueError(
                f"Data too short for ELF header: {len(data)} < {ELF64_EHDR_SIZE}"
            )

        if data[:4] != ELF_MAGIC:
            raise ValueError("Not an ELF file (bad magic)")

        if data[4] != 2:
            raise ValueError("Only 64-bit ELF supported (ELFCLASS64)")

        if data[5] != 1:
            raise ValueError("Only little-endian ELF supported (ELFDATA2LSB)")

        fields = struct.unpack_from(cls.STRUCT_FMT, data, 0)
        return cls(*fields)

    def to_bytes(self) -> bytes:
        """Serialize ELF header to binary data."""
        return struct.pack(
            self.STRUCT_FMT,
            self.e_ident,
            self.e_type,
            self.e_machine,
            self.e_version,
            self.e_entry,
            self.e_phoff,
            self.e_shoff,
            self.e_flags,
            self.e_ehsize,
            self.e_phentsize,
            self.e_phnum,
            self.e_shentsize,
            self.e_shnum,
            self.e_shstrndx,
        )

    def write_to(self, data: bytearray, offset: int = 0) -> None:
        """Write ELF header to mutable buffer at offset."""
        struct.pack_into(
            self.STRUCT_FMT,
            data,
            offset,
            self.e_ident,
            self.e_type,
            self.e_machine,
            self.e_version,
            self.e_entry,
            self.e_phoff,
            self.e_shoff,
            self.e_flags,
            self.e_ehsize,
            self.e_phentsize,
            self.e_phnum,
            self.e_shentsize,
            self.e_shnum,
            self.e_shstrndx,
        )

    @property
    def is_pie_or_shared(self) -> bool:
        """Check if this is a PIE executable or shared library (ET_DYN)."""
        return self.e_type == ET_DYN


@dataclass
class ProgramHeader:
    """ELF64 program header (Elf64_Phdr).

    Program headers define segments - how the file is loaded into memory.
    """

    p_type: int  # Segment type (PT_*)
    p_flags: int  # Segment flags (PF_*)
    p_offset: int  # File offset
    p_vaddr: int  # Virtual address
    p_paddr: int  # Physical address (usually same as vaddr)
    p_filesz: int  # Size in file
    p_memsz: int  # Size in memory (may be > filesz for BSS)
    p_align: int  # Alignment (power of 2)

    # Struct format for parsing/writing (little-endian)
    STRUCT_FMT: ClassVar[str] = "<IIQQQQQQ"

    @classmethod
    def from_bytes(cls, data: bytes | bytearray, offset: int = 0) -> "ProgramHeader":
        """Parse program header from binary data at offset."""
        if len(data) < offset + ELF64_PHDR_SIZE:
            raise ValueError("Data too short for program header")
        fields = struct.unpack_from(cls.STRUCT_FMT, data, offset)
        return cls(*fields)

    def to_bytes(self) -> bytes:
        """Serialize program header to binary data."""
        return struct.pack(
            self.STRUCT_FMT,
            self.p_type,
            self.p_flags,
            self.p_offset,
            self.p_vaddr,
            self.p_paddr,
            self.p_filesz,
            self.p_memsz,
            self.p_align,
        )

    def write_to(self, data: bytearray, offset: int) -> None:
        """Write program header to mutable buffer at offset."""
        struct.pack_into(
            self.STRUCT_FMT,
            data,
            offset,
            self.p_type,
            self.p_flags,
            self.p_offset,
            self.p_vaddr,
            self.p_paddr,
            self.p_filesz,
            self.p_memsz,
            self.p_align,
        )

    @property
    def end_offset(self) -> int:
        """File offset of end of segment content."""
        return self.p_offset + self.p_filesz

    @property
    def end_vaddr(self) -> int:
        """Virtual address of end of segment (file-backed portion)."""
        return self.p_vaddr + self.p_filesz

    @property
    def end_memsz_vaddr(self) -> int:
        """Virtual address of end of segment (including BSS)."""
        return self.p_vaddr + self.p_memsz

    def contains_vaddr(self, vaddr: int) -> bool:
        """Check if a virtual address falls within this segment."""
        return self.p_vaddr <= vaddr < self.end_memsz_vaddr

    def contains_offset(self, offset: int) -> bool:
        """Check if a file offset falls within this segment's file content."""
        return self.p_offset <= offset < self.end_offset


@dataclass
class SectionHeader:
    """ELF64 section header (Elf64_Shdr).

    Section headers describe the layout of the file for linking/debugging.
    """

    sh_name: int  # Offset into section name string table
    sh_type: int  # Section type (SHT_*)
    sh_flags: int  # Section flags (SHF_*)
    sh_addr: int  # Virtual address (if SHF_ALLOC set)
    sh_offset: int  # File offset
    sh_size: int  # Section size
    sh_link: int  # Link to another section (section-type dependent)
    sh_info: int  # Additional info (section-type dependent)
    sh_addralign: int  # Alignment (power of 2, 0 or 1 means none)
    sh_entsize: int  # Entry size if section holds table

    # Struct format for parsing/writing (little-endian)
    STRUCT_FMT: ClassVar[str] = "<IIQQQQIIQQ"

    @classmethod
    def from_bytes(cls, data: bytes | bytearray, offset: int = 0) -> "SectionHeader":
        """Parse section header from binary data at offset."""
        if len(data) < offset + ELF64_SHDR_SIZE:
            raise ValueError("Data too short for section header")
        fields = struct.unpack_from(cls.STRUCT_FMT, data, offset)
        return cls(*fields)

    def to_bytes(self) -> bytes:
        """Serialize section header to binary data."""
        return struct.pack(
            self.STRUCT_FMT,
            self.sh_name,
            self.sh_type,
            self.sh_flags,
            self.sh_addr,
            self.sh_offset,
            self.sh_size,
            self.sh_link,
            self.sh_info,
            self.sh_addralign,
            self.sh_entsize,
        )

    def write_to(self, data: bytearray, offset: int) -> None:
        """Write section header to mutable buffer at offset."""
        struct.pack_into(
            self.STRUCT_FMT,
            data,
            offset,
            self.sh_name,
            self.sh_type,
            self.sh_flags,
            self.sh_addr,
            self.sh_offset,
            self.sh_size,
            self.sh_link,
            self.sh_info,
            self.sh_addralign,
            self.sh_entsize,
        )

    @property
    def end_offset(self) -> int:
        """File offset of end of section content."""
        return self.sh_offset + self.sh_size

    @property
    def end_addr(self) -> int:
        """Virtual address of end of section."""
        return self.sh_addr + self.sh_size

    @property
    def is_alloc(self) -> bool:
        """Check if this section occupies memory at runtime."""
        return bool(self.sh_flags & SHF_ALLOC)

    @property
    def is_nobits(self) -> bool:
        """Check if this section has no file content (like BSS)."""
        return self.sh_type == SHT_NOBITS


@dataclass
class RelaEntry:
    """ELF64 relocation entry with addend (Elf64_Rela).

    Used in SHT_RELA sections for relocations that need an explicit addend.
    """

    r_offset: int  # Address to apply relocation
    r_info: int  # Symbol index and relocation type
    r_addend: int  # Addend (signed)

    # Struct format for parsing/writing (little-endian)
    STRUCT_FMT: ClassVar[str] = "<QQq"  # Note: r_addend is signed (q not Q)
    SIZE: ClassVar[int] = 24

    @classmethod
    def from_bytes(cls, data: bytes | bytearray, offset: int = 0) -> "RelaEntry":
        """Parse RELA entry from binary data at offset."""
        if len(data) < offset + cls.SIZE:
            raise ValueError("Data too short for RELA entry")
        fields = struct.unpack_from(cls.STRUCT_FMT, data, offset)
        return cls(*fields)

    def to_bytes(self) -> bytes:
        """Serialize RELA entry to binary data."""
        return struct.pack(self.STRUCT_FMT, self.r_offset, self.r_info, self.r_addend)

    def write_to(self, data: bytearray, offset: int) -> None:
        """Write RELA entry to mutable buffer at offset."""
        struct.pack_into(
            self.STRUCT_FMT, data, offset, self.r_offset, self.r_info, self.r_addend
        )

    @property
    def r_type(self) -> int:
        """Extract relocation type from r_info."""
        return self.r_info & 0xFFFFFFFF

    @property
    def r_sym(self) -> int:
        """Extract symbol table index from r_info."""
        return self.r_info >> 32

    @staticmethod
    def make_info(sym: int, type_: int) -> int:
        """Create r_info value from symbol index and type."""
        return (sym << 32) | (type_ & 0xFFFFFFFF)

    def get_target_address(self) -> int | None:
        """Get the target address this relocation points to.

        Returns:
            Target address for known relocation types, None for unknown types.

        Currently supports:
            - R_X86_64_RELATIVE: target is r_addend
        """
        if self.r_type == R_X86_64_RELATIVE:
            return self.r_addend
        return None

    def targets_range(self, range_start: int, range_size: int) -> bool | None:
        """Check if this relocation targets an address within a range.

        Args:
            range_start: Start address of the range
            range_size: Size of the range in bytes

        Returns:
            True: relocation target is within range
            False: relocation target is known to be outside range
            None: relocation type not understood (caller decides how to handle)
        """
        target = self.get_target_address()
        if target is None:
            return None
        range_end = range_start + range_size
        return range_start <= target < range_end


@dataclass
class RelEntry:
    """ELF64 relocation entry without addend (Elf64_Rel).

    Used in SHT_REL sections. Less common than RELA on x86_64.
    """

    r_offset: int  # Address to apply relocation
    r_info: int  # Symbol index and relocation type

    # Struct format for parsing/writing (little-endian)
    STRUCT_FMT: ClassVar[str] = "<QQ"
    SIZE: ClassVar[int] = 16

    @classmethod
    def from_bytes(cls, data: bytes | bytearray, offset: int = 0) -> "RelEntry":
        """Parse REL entry from binary data at offset."""
        if len(data) < offset + cls.SIZE:
            raise ValueError("Data too short for REL entry")
        fields = struct.unpack_from(cls.STRUCT_FMT, data, offset)
        return cls(*fields)

    def to_bytes(self) -> bytes:
        """Serialize REL entry to binary data."""
        return struct.pack(self.STRUCT_FMT, self.r_offset, self.r_info)

    def write_to(self, data: bytearray, offset: int) -> None:
        """Write REL entry to mutable buffer at offset."""
        struct.pack_into(self.STRUCT_FMT, data, offset, self.r_offset, self.r_info)

    @property
    def r_type(self) -> int:
        """Extract relocation type from r_info."""
        return self.r_info & 0xFFFFFFFF

    @property
    def r_sym(self) -> int:
        """Extract symbol table index from r_info."""
        return self.r_info >> 32


# =============================================================================
# Helper Functions
# =============================================================================


def round_up_to_page(addr: int) -> int:
    """Round address up to next page boundary."""
    return (addr + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)


def round_down_to_page(addr: int) -> int:
    """Round address down to previous page boundary."""
    return addr & ~(PAGE_SIZE - 1)


def page_align_offset(offset: int, vaddr: int) -> int:
    """Calculate file offset that maintains mmap alignment with vaddr.

    For mmap to work correctly:
        (p_offset % PAGE_SIZE) == (p_vaddr % PAGE_SIZE)

    This function finds the smallest offset >= the given offset that
    satisfies this constraint for the given vaddr.
    """
    vaddr_page_offset = vaddr & (PAGE_SIZE - 1)
    offset_page_offset = offset & (PAGE_SIZE - 1)

    if offset_page_offset == vaddr_page_offset:
        return offset
    elif offset_page_offset < vaddr_page_offset:
        return (offset & ~(PAGE_SIZE - 1)) + vaddr_page_offset
    else:
        return round_up_to_page(offset) + vaddr_page_offset


def get_section_name(
    data: bytes | bytearray, shstrtab_offset: int, name_idx: int
) -> str:
    """Get section name from the section header string table.

    Args:
        data: Full ELF binary data
        shstrtab_offset: File offset of .shstrtab section
        name_idx: sh_name field from section header

    Returns:
        Section name as string
    """
    name_offset = shstrtab_offset + name_idx
    if name_offset >= len(data):
        return ""
    end = data.find(b"\x00", name_offset)
    if end == -1:
        return ""
    return data[name_offset:end].decode("ascii", errors="ignore")
