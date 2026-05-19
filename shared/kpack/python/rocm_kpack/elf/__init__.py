"""
ELF binary manipulation package for rocm-kpack.

This package provides clean abstractions for ELF surgery operations:
- types: Unified ELF struct definitions
- surgery: High-level ElfSurgery class for modifications
- verify: ElfVerifier for post-surgery validation
- phdr_manager: ProgramHeaderManager for PHDR table operations
- zero_page: Page-level optimization utilities

The design prioritizes composition and correctness over cleverness.
"""

from .surgery import (
    ElfSurgery,
    SectionInfo,
    Modification,
    RelocationInfo,
    AddSectionResult,
)
from ..verify import VerificationResult
from .verify import (
    ElfVerifier,
    verify_with_readelf,
    verify_with_strip,
    verify_with_gdb,
    verify_with_ldd,
    verify_all,
)
from .phdr_manager import (
    ProgramHeaderManager,
    PhdrResizeResult,
    create_load_segment,
)
from .operations import (
    map_section_to_load,
    set_pointer,
    update_relocation_addend,
    find_relocation_for_pointer,
    MapSectionResult,
    SetPointerResult,
    UpdateRelocationResult,
)
from .zero_page import (
    conservative_zero_page,
    zero_page_section,
    calculate_aligned_range,
    ZeroPageResult,
)
from .kpack_transform import (
    kpack_offload_binary,
    add_kpack_ref_section,
    rewrite_hipfatbin_magic,
    verify_no_fatbin_relocations,
    read_kpack_ref_marker,
    NotFatBinaryError,
    ProblematicRelocation,
    HIPF_MAGIC,
    HIPK_MAGIC,
    WRAPPER_SIZE,
)
from .types import (
    ElfHeader,
    ProgramHeader,
    SectionHeader,
    RelaEntry,
    RelEntry,
    # Constants
    PAGE_SIZE,
    ELF_MAGIC,
    # Program header types
    PT_NULL,
    PT_LOAD,
    PT_DYNAMIC,
    PT_INTERP,
    PT_NOTE,
    PT_PHDR,
    PT_TLS,
    PT_GNU_EH_FRAME,
    PT_GNU_STACK,
    PT_GNU_RELRO,
    # Section header types
    SHT_NULL,
    SHT_PROGBITS,
    SHT_SYMTAB,
    SHT_STRTAB,
    SHT_RELA,
    SHT_HASH,
    SHT_DYNAMIC,
    SHT_NOTE,
    SHT_NOBITS,
    SHT_REL,
    SHT_DYNSYM,
    # Program header flags
    PF_X,
    PF_W,
    PF_R,
    # Section flags
    SHF_WRITE,
    SHF_ALLOC,
    SHF_EXECINSTR,
    # Relocation types (x86_64)
    R_X86_64_NONE,
    R_X86_64_64,
    R_X86_64_RELATIVE,
    # ELF types
    ET_EXEC,
    ET_DYN,
    # Dynamic section tags
    DT_NULL,
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
    DT_ADDR_TAGS,
)

__all__ = [
    # High-level classes
    "ElfSurgery",
    "SectionInfo",
    "Modification",
    "RelocationInfo",
    "ElfVerifier",
    "VerificationResult",
    "ProgramHeaderManager",
    "PhdrResizeResult",
    # Operations
    "map_section_to_load",
    "set_pointer",
    "update_relocation_addend",
    "find_relocation_for_pointer",
    "MapSectionResult",
    "SetPointerResult",
    "UpdateRelocationResult",
    "create_load_segment",
    # Zero-page operations
    "conservative_zero_page",
    "zero_page_section",
    "calculate_aligned_range",
    "ZeroPageResult",
    # Kpack operations
    "kpack_offload_binary",
    "add_kpack_ref_section",
    "rewrite_hipfatbin_magic",
    "verify_no_fatbin_relocations",
    "read_kpack_ref_marker",
    "NotFatBinaryError",
    "ProblematicRelocation",
    "HIPF_MAGIC",
    "HIPK_MAGIC",
    "WRAPPER_SIZE",
    # Section addition
    "AddSectionResult",
    # Verification functions
    "verify_with_readelf",
    "verify_with_strip",
    "verify_with_gdb",
    "verify_with_ldd",
    "verify_all",
    # Structs
    "ElfHeader",
    "ProgramHeader",
    "SectionHeader",
    "RelaEntry",
    "RelEntry",
    # Constants
    "PAGE_SIZE",
    "ELF_MAGIC",
    # Program header types
    "PT_NULL",
    "PT_LOAD",
    "PT_DYNAMIC",
    "PT_INTERP",
    "PT_NOTE",
    "PT_PHDR",
    "PT_TLS",
    "PT_GNU_EH_FRAME",
    "PT_GNU_STACK",
    "PT_GNU_RELRO",
    # Section header types
    "SHT_NULL",
    "SHT_PROGBITS",
    "SHT_SYMTAB",
    "SHT_STRTAB",
    "SHT_RELA",
    "SHT_HASH",
    "SHT_DYNAMIC",
    "SHT_NOTE",
    "SHT_NOBITS",
    "SHT_REL",
    "SHT_DYNSYM",
    # Program header flags
    "PF_X",
    "PF_W",
    "PF_R",
    # Section flags
    "SHF_WRITE",
    "SHF_ALLOC",
    "SHF_EXECINSTR",
    # Relocation types
    "R_X86_64_NONE",
    "R_X86_64_64",
    "R_X86_64_RELATIVE",
    # ELF types
    "ET_EXEC",
    "ET_DYN",
    # Dynamic section tags
    "DT_NULL",
    "DT_PLTGOT",
    "DT_HASH",
    "DT_STRTAB",
    "DT_SYMTAB",
    "DT_RELA",
    "DT_INIT",
    "DT_FINI",
    "DT_REL",
    "DT_JMPREL",
    "DT_INIT_ARRAY",
    "DT_FINI_ARRAY",
    "DT_PREINIT_ARRAY",
    "DT_SYMTAB_SHNDX",
    "DT_VERSYM",
    "DT_VERDEF",
    "DT_VERNEED",
    "DT_ADDR_TAGS",
]
