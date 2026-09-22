/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 **********************************************************************************************************************/
/**
 ***********************************************************************************************************************
 * @file  palElf.h
 * @brief Standard ELF Structures, Enums, and Constants.
 * Based off of http://man7.org/linux/man-pages/man5/elf.5.html
 ***********************************************************************************************************************
 */

#pragma once
#include "palUtil.h"
#include "palSpan.h"

// $OpenBSD: elf.5,v 1.12 2003/10/27 20:23:58 jmc Exp $
// Copyright (c) 1999 Jeroen Ruigrok van der Werven
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.

// THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

namespace Util
{
namespace Elf
{

/// Used to specify if the the ELF is 32 or 64bit.
enum IdentClass : uint8
{
    ElfClass32 = 1,
    ElfClass64 = 2
};

/// Used to specify if the the ELF is little or big endian.
enum IdentEndianness : uint8
{
    ElfLittleEndian = 1,
    ElfBigEndian    = 2
};

constexpr uint32 IdentSize        = 16;            ///< Identification size.
constexpr uint32 IdentPaddingSize = IdentSize - 9; ///< Identification padding size.

constexpr uint32 ElfMagic         = 0x464C457F;    ///< '\x7f','E','L','F' in little endian.
constexpr uint8  ElfVersion       = 1;             ///< Identification version

#pragma pack (push, 1)

/// The File header describes the ELF file. It is located at the beginning of the ELF file and is
/// used to locate other parts of the ELF. This struct is known as Elf64_Ehdr in the spec.
struct FileHeader
{
    union
    {
        uint8  e_ident[IdentSize]; ///< ELF identification information.
        struct
        {
            uint32 ei_magic;      ///< Contains a 'magic number,' identifying the file as an ELF
                                  ///  object file. Contains the characters '\x7f','E','L','F'.
            uint8  ei_class;      ///< Identifies the class of the object file, or its capacity.
            uint8  ei_data;       ///< Specifies the data encoding of the object file data
                                  ///  structures.
            uint8  ei_version;    ///< Identifies the version of the object file format.
            uint8  ei_osabi;      ///< Identifies the operating system and ABI for which the object
                                  ///  is prepared.
            uint8  ei_abiversion; ///< Identifies the version of the ABI for which the object is
                                  ///  prepared.
            uint8  ei_pad[IdentPaddingSize]; ///< Padding bytes.
        };
    };
    uint16 e_type;      ///< Identifies the object file type.
    uint16 e_machine;   ///< Identifies the target architecture.
    uint32 e_version;   ///< Identifies the version of the object file format.
    uint64 e_entry;     ///< The virtual address of the program entry point. If there is no entry
                        ///  point, this field contains zero.
    uint64 e_phoff;     ///< The file offset, in bytes, of the program header table.
    uint64 e_shoff;     ///< The file offset, in bytes, of the section header table.
    uint32 e_flags;     ///< Processor-specific flags.
    uint16 e_ehsize;    ///< Size, in bytes, of the ELF header.
    uint16 e_phentsize; ///< Size, in bytes, of a program header table entry.
    uint16 e_phnum;     ///< Number of entries in the program header table.
    uint16 e_shentsize; ///< Size, in bytes, of a section header table entry.
    uint16 e_shnum;     ///< Number of entries in the section header table.
    uint16 e_shstrndx;  ///< Section header table index of the section containing the section name
                        ///  string table. If there is no section name string table, this field has
                        ///  the value ShnUndef.
};

/// The Section header describes a section.  This struct is known as Elf64_Shdr in the spec.
struct SectionHeader
{
    uint32 sh_name;      ///< Offset, in bytes, to the section name, relative to the start of the
                         ///  section name string table.
    uint32 sh_type;      ///< Identifies the section type.  See SectionHeaderType.
    uint64 sh_flags;     ///< Identifies the attributes of the section. See SectionHeaderFlags.
    uint64 sh_addr;      ///< Virtual address of the beginning of the section in memory. If the
                         ///  section is not allocated to the memory image of the program, this
                         ///  field should be zero
    uint64 sh_offset;    ///< Offset, in bytes, of the beginning of the section contents in the file.
    uint64 sh_size;      ///< Size, in bytes, of the section. Except for ShtNoBits sections, this
                         ///  is the amount of space occupied in the file.
    uint32 sh_link;      ///< Contains the section index of an associated section.
    uint32 sh_info;      ///< Contains extra information about the section.
    uint64 sh_addralign; ///< Alignment required. This field must be a power of two.
    uint64 sh_entsize;   ///< For sections that contain fixed-size entries, this field contains the
                         ///  size, in bytes, of each entry. Otherwise, this field contains zero.
};

// String tables
// String table sections contain strings used for section names and symbol names. A string table is
// just an array of bytes containing null-terminated strings. Section header table entries, and
// symbol table entries refer to strings in a string table with an index relative to the beginning
// of the string table. The first byte in a string table is defined to be null, so that the index 0
// always refers to a null or nonexistent name.

/// The section data of a symbol section contains a symbol table. This is an entry in that table.
/// This struct is known as Elf64_Sym in the spec.
struct SymbolTableEntry
{
    uint32 st_name;  ///< Offset, in bytes, to the symbol name, relative to the start of the symbol
                     ///  string table. If this field contains zero, the symbol has no name.
    union
    {
        struct
        {
            uint8 type     : 4; ///< See SymbolTableType.
            uint8 binding  : 4; ///< Binding attributes. See SymbolTableBinding.
        };
        uint8 all;
    } st_info;       ///< This field contains the symbol type and its binding attributes (that is,
                     ///  its scope).
    uint8  st_other; ///< Reserved for future use; must be zero.
    uint16 st_shndx; ///< Section index of the section in which the symbol is 'defined'. For
                     ///  undefined symbols, this field contains ShnUndef; for absolute symbols,
                     ///  it contains ShnAbs; and for common symbols, it contains ShnCommon.
    uint64 st_value; ///< Contains the value of the symbol. This may be an absolute value or a
                     ///  relocatable address. In relocatable files, this field contains the
                     ///  alignment constraint for common symbols, and a section-relative offset for
                     ///  defined relocatable symbols. In executable and shared object files, this
                     ///  field contains a virtual address for defined relocatable symbols.
    uint64 st_size;  ///< Size associated with the symbol. If a symbol does not have an associated
                     ///  size, or the size is unknown, this field contains zero.
};

/// Sections of type Rel contain a relocation table.  This is an entry in that table. The addend
/// part of the relocation is obtained from the original value of the word being relocated.
/// This struct is known as Elf64_Rel in the spec.
struct RelTableEntry
{
    uint64 r_offset; ///< Indicates the location at which the relocation should be applied. For a
                     ///  relocatable file, this is the offset, in bytes, from the beginning of the
                     ///  section to the beginning of the storage unit being relocated. For an
                     ///  executable or shared object, this is the virtual address of the storage
                     ///  unit being relocated.
    union
    {
        struct
        {
            uint32 type; ///< Relocation types are processor specific.
            uint32 sym;  ///< The symbol table index identifies the symbol whose value should be
                         ///  used in the relocation.
        };
        uint64 all;   ///< Contains both a symbol table index and a relocation type.
    } r_info;
};

/// Sections of type Rela contain a relocation table. This is an entry in that table. The Rela type
/// provides an explicit field for a full-width addend.
/// This struct is known as Elf64_Rela in the spec.
struct RelaTableEntry
{
    uint64 r_offset; ///< \copydoc Rel.r_offset
    union
    {
        struct
        {
            uint32 type; ///< \copydoc Rel.r_info.type
            uint32 sym;  ///< \copydoc Rel.r_info.sym
        };
        uint64 all;      ///< \copydoc Rel.r_info.all
    } r_info;
    uint64 r_addend; ///< Specifies a constant addend used to compute the value to be stored in the
                     ///  relocated field.
};

/// In executable and shared object files, sections are grouped into segments for loading. The
/// program header describes one of these segments. This struct is known as Elf64_Phdr in the spec.
struct ProgramHeader
{
    uint32 p_type;   ///< Identifies the type of segment. See SegmentType.
    uint32 p_flags;  ///< Segment attributes. See SegmentFlags.
    uint64 p_offset; ///< Offset, in bytes, of the segment from the beginning of the file.
    uint64 p_vaddr;  ///< Virtual address at which the first byte of the segment resides in memory.
    uint64 p_paddr;  ///< Reserved for systems with physical addressing.
    uint64 p_filesz; ///< Size, in bytes, of the file image of the segment.
    uint64 p_memsz;  ///< Size, in bytes, of the memory image of the segment.
    uint64 p_align;  ///< Alignment constraint for the segment. Must be a power of two. The values
                     ///  of p_offset and p_vaddr must be congruent modulo the alignment.
};

/// Sections of type SectionHeaderType::Note and SegmentType::Note can be used.
/// Additional variable fields are the name field which identifies the entry's owner or originator.
/// The name field contains a null terminated string, with padding as necessary to ensure 8-byte
/// alignment for the descriptor field.
///
/// The desc field contains the contents of the note, followed by padding as necessary to ensure
/// 8-byte alignment for the next note entry. The format and interpretation of the note contents are
/// determined solely by the name and type fields, and are unspecified by the ELF standard.
struct NoteTableEntryHeader
{
    uint32 n_namesz; ///< Identifies the length, in bytes, of the name field.
    uint32 n_descsz; ///< Identifies the length of the note descriptor field.
    uint32 n_type;   ///< Determines, along with the originator's name, the interpretation of the
                     ///  note contents. Each originator controls its own types.
};

/// Sections of type Dyn contain a dynamic table. This is an entry in that
/// table. Refer to Section 11 of the spec for efficient dynamic table access
/// using a hash table. This struct is known as Elf64_Dyn in the spec.
struct DynamicTableEntry
{
    uint64 d_tag;     ///< Identifies the type of dynamic table entry. The type determines the
                      ///  interpretation of the d_un union.
    union {
        uint64 d_val; ///< This union member is used to represent integer values.
        uint64 d_ptr; ///< This union member is used to represent program virtual addresses. These
                      ///  addresses are link-time virtual addresses, and must be relocated to
                      ///  match the object file's actual load address. This relocation must be
                      ///  done implicitly; there are no dynamic relocations for these entries.
    } d_un;
};

// Hash Table for accessing dynamic table efficiently.
#pragma pack (pop)

/// ELF Object File Type: e_type
enum class ObjectFileType : uint32
{
    None = 0, ///< No file type.
    Rel  = 1, ///< Relocatable object file.
    Exec = 2, ///< Executable file.
    Dyn  = 3, ///< Shared object file.
    Core = 4, ///< Core file.
};

/// ELF Machine Type: e_machine
enum class MachineType : uint16
{
    AmdGpu = 0xe0, ///< EM_AMDGPU.  AMDGPU machine architecture magic number.
};

/// ELF Section Header Index
enum class SectionHeaderIndex : uint16
{
    Undef  = 0,      ///< Used to mark an undefined or meaningless section reference.
    Abs    = 0xfff1, ///< Indicates that the corresponding reference is an absolute value.
    Common = 0xfff2, ///< Indicates a symbol that has been declared as a common block.
};

/// ELF Section Header Type: sh_type
enum class SectionHeaderType : uint32
{
    Null     = 0,  ///< Marks an unused section header.
    ProgBits = 1,  ///< Contains information defined by the program.
    SymTab   = 2,  ///< Contains a linker symbol table.
    StrTab   = 3,  ///< Contains a string table.
    Rela     = 4,  ///< Contains 'Rela' type relocation entries.
    Hash     = 5,  ///< Contains a symbol hash table.
    Dynamic  = 6,  ///< Contains dynamic linking tables.
    Note     = 7,  ///< Contains note information.
    NoBits   = 8,  ///< Contains uninitialized space; does not occupy any space in the file.
    Rel      = 9,  ///< Contains 'Rel' type relocation entries.
    ShLib    = 10, ///< Reserved.
    DynSym   = 11, ///< Contains a dynamic loader symbol table.
};

/// ELF Section Header Flags: sh_flags
enum SectionHeaderFlags : uint32
{
    ShfWrite     = 0x1, ///< Section contains writable data.
    ShfAlloc     = 0x2, ///< Section is allocated in memory image of program.
    ShfExecInstr = 0x4, ///< Section contains executable instructions.
};

/// ELF Symbol Table Binding: st_info.binding
enum class SymbolTableEntryBinding : uint32
{
    Local  = 0, ///< Not visible outside the object file.
    Global = 1, ///< Global symbol, visible to all object files.
    Weak   = 2, ///< Global scope, but with lower precedence than global symbols.
};

/// ELF Symbol Table Type: st_info.type
enum class SymbolTableEntryType : uint32
{
    None  = 0,   ///< No type specified (e.g., an absolute symbol).
    Object  = 1, ///< Data object.
    Func    = 2, ///< Function entry point.
    Section = 3, ///< Symbol is associated with a section.
    File    = 4, ///< Source file associated with the object file.
};

/// ELF Segment Type: p_type
enum class SegmentType : uint32
{
    Null    = 0, ///< Unused entry.
    Load    = 1, ///< Loadable segment.
    Dynamic = 2, ///< Dynamic linking tables.
    Interp  = 3, ///< Program interpreter path name.
    Note    = 4, ///< Note sections.
    ShLib   = 5, ///< Reserved.
    PhDr    = 6, ///< Program header table.
    Count
};

/// ELF Segment Flags: p_flags
enum SegmentFlags : uint32
{
    PfExecute = 0x1,  ///< Execute permission.
    PfWrite   = 0x2,  ///< Write permission.
    PfRead    = 0x4,  ///< Read permission.
};

/// ELF Dynamic Table Type: d_tag
enum class DynamicTableEntryType : uint32
{
    Null        = 0,  ///< d_un: ignored. Marks the end of the dynamic array.
    Needed      = 1,  ///< d_un: d_val. The string table offset of the name of a needed library.
    PltRelSz    = 2,  ///< d_un: d_val. The total size, in bytes, of the relocation entries
                      ///  associated with the procedure linkage table.
    PltGot      = 3,  ///< d_un: d_ptr. Contains an address associated with the linkage table. The
                      ///  specific meaning of this field is processor dependent.
    Hash        = 4,  ///< d_un: d_ptr. Address of the symbol hash table.
    StrTab      = 5,  ///< d_un: d_ptr. Address of the dynamic string table.
    SymTab      = 6,  ///< d_un: d_ptr. Address of the dynamic symbol table.
    Rela        = 7,  ///< d_un: d_ptr. Address of a relocation table with Rela entries.
    RelaSz      = 8,  ///< d_un: d_val. Total size, in bytes, of the Rela relocation table
    RelaEnt     = 9,  ///< d_un: d_val. Size, in bytes, of each Rela relocation entry
    StrSz       = 10, ///< d_un: d_val. Total size, in bytes, of the string table.
    SymEnt      = 11, ///< d_un: d_val. Size, in bytes, of each symbol table entry.
    Init        = 12, ///< d_un: d_ptr. Address of the initialization function.
    Fini        = 13, ///< d_un: d_ptr. Address of the termination function.
    SoName      = 14, ///< d_un: d_val. The string table offset of the name of this shared object.
    RPath       = 15, ///< d_un: d_val. The string table offset of a shared library search path
                      ///  string.
    Symbolic    = 16, ///< d_un: ignored. The presence of this dynamic table entry modifies the
                      ///  symbol resolution algorithm for references within the library. Symbols
                      ///  defined within the library are used to resolve references before the
                      ///  dynamic linker searches the usual search path.
    Rel         = 17, ///< d_un: d_ptr. Address of a relocation table with Rela entries.
    RelSz       = 18, ///< d_un: d_val. Total size, in bytes, of the Rel relocation table.
    RelEnt      = 19, ///< d_un: d_val. Size, in bytes, of each Rel relocation entry.
    PltRel      = 20, ///< d_un: d_val. Type of relocation entry used for the procedure linkage
                      ///  table. The d_val member contains either Rel or Rela.
    Debug       = 21, ///< d_un: d_ptr. Reserved for debugger use.
    TextRel     = 22, ///< d_un: ignored. The presence of this dynamic table entry signals that the
                      ///  relocation table contains relocations for a non-writable segment.
    JmpRel      = 23, ///< d_un: d_ptr. Address of the relocations associated with the procedure
                      ///  linkage table.
    BindNow     = 24, ///< d_un: ignored. The presence of this dynamic table entry signals that the
                      ///  dynamic loader should process all relocations for this object before
                      ///  transferring control to the program.
    InitArray   = 25, ///< d_un: d_ptr. Pointer to an array of pointers to initialization functions.
    FiniArray   = 26, ///< d_un: d_ptr. Pointer to an array of pointers to termination functions.
    InitArraySz = 27, ///< d_un: d_val. Size, in bytes, of the array of initialization functions.
    FiniArraySz = 28, ///< d_un: d_val. Size, in bytes, of the array of termination functions.
};

/// The SectionType is used to describe standard sections.
enum class SectionType : uint32
{
    Null = 0,   ///< Null section.
    Bss,        ///< Uninitialized data.
    Data,       ///< Initialized data.
    Interp,     ///< Program interpreter path name.
    RoData,     ///< Read-only data (constants and literals).
    Text,       ///< Executable code.

    Comment,    ///< Version control information.
    Dynamic,    ///< Dynamic linking tables.
    DynStr,     ///< String table for .dynamic section.
    DynSym,     ///< Symbol table for dynamic linking.
    Got,        ///< Global offset table.
    Hash,       ///< Symbol hash table.
    Note,       ///< Note section.
    Plt,        ///< Procedure linkage table.
    Rel,        ///< Relocations.
    Rela,       ///< Relocations.
    ShStrTab,   ///< Section name string table.
    StrTab,     ///< String table.
    SymTab,     ///< Linker symbol table.
    Count
};

/// Used to access values in SectionHeaderInfo
struct SectionHeaderInfo
{
    SectionHeaderType type;
    uint32            flags;
};

/// A mapping from SectionType to the corresponding name of that section.
constexpr const char* SectionNameStringTable[] =
{
    "",
    ".bss",
    ".data",
    ".interp",
    ".rodata",
    ".text",
    ".comment",
    ".dynamic",
    ".dynstr",
    ".dynsym",
    ".got",
    ".hash",
    ".note",
    ".plt",
    ".rel",
    ".rela",
    ".shstrtab",
    ".strtab",
    ".symtab",
};

/// A mapping from SectionType to the corresponding SectionHeaderType and flags.
constexpr SectionHeaderInfo SectionHeaderInfoTable[] =
{
    // SectionType::Null
    {
        SectionHeaderType::Null,
        0
    },
    // SectionType::Bss
    {
        SectionHeaderType::NoBits,
        ShfAlloc | ShfWrite
    },
    // SectionType::Data
    {
        SectionHeaderType::ProgBits,
        ShfAlloc | ShfWrite
    },
    // SectionType::Interp
    {
        SectionHeaderType::ProgBits,
        ShfAlloc // [A]?
    },
    // SectionType::RoData
    {
        SectionHeaderType::ProgBits,
        ShfAlloc
    },
    // SectionType::Text
    {
        SectionHeaderType::ProgBits,
        ShfAlloc | ShfExecInstr
    },
    // SectionType::Comment
    {
        SectionHeaderType::ProgBits,
        0
    },
    // SectionType::Dynamic
    {
        SectionHeaderType::Dynamic,
        ShfAlloc | ShfWrite // [W]?
    },
    // SectionType::DynStr
    {
        SectionHeaderType::StrTab,
        ShfAlloc | ShfWrite
    },
    // SectionType::DynSym
    {
        SectionHeaderType::DynSym,
        ShfAlloc | ShfWrite
    },
    // SectionType::Got
    {
        SectionHeaderType::ProgBits,
        0 // Machine dependent.
    },
    // SectionType::Hash
    {
        SectionHeaderType::Hash,
        ShfAlloc
    },
    // SectionType::Note
    {
        SectionHeaderType::Note,
        0
    },
    // SectionType::Plt
    {
        SectionHeaderType::ProgBits,
        0 // Machine dependent.
    },
    // SectionType::Rel
    {
        // Append <name> to have .rel<name> where <name> is the section.
        SectionHeaderType::Rel,
        0
    },
    // SectionType::Rela
    {
        // Append <name> to have .rela<name> where <name> is the section.
        SectionHeaderType::Rela,
        0
    },
    // SectionType::ShStrTab
    {
        SectionHeaderType::StrTab,
        0
    },
    // SectionType::StrTab
    {
        SectionHeaderType::StrTab,
        0
    },
    // SectionType::SymTab
    {
        SectionHeaderType::SymTab,
        0
    },
};

///@{
/// @returns true if the given binary blob identifies as an ELF file.
/// @param [in] pData/binary  The binary blob to check.
inline bool IsElf(const void*      pData)  { return static_cast<const FileHeader*>(pData)->ei_magic == ElfMagic; }
inline bool IsElf(Span<const void> binary)
    { return (binary.SizeInBytes() >= sizeof(Util::Elf::FileHeader)) && IsElf(binary.Data()); }
///@}

} // Elf
} // Util
