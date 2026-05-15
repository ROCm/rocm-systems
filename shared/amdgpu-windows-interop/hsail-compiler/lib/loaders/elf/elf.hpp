/*
************************************************************************************************************************
*
*  Copyright (C) 2020 Advanced Micro Devices, Inc.  All rights reserved.
*
***********************************************************************************************************************/

//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef ELF_HPP_
#define ELF_HPP_

#include <map>

#include "top.hpp"
#include "elf_utils.hpp"
#if !defined(WITH_LIGHTNING_COMPILER)
#include "caltarget.h" // using CALtargetEnum
#endif // !defined(WITH_LIGHTNING_COMPILER)

#include "libelf.h"
#include "gelf.h"

// Not sure where to put these in the libelf
#define AMD_BIF2 2 // AMD BIF Version 2.0
#define AMD_BIF3 3 // AMD BIF Version 3.0

// These two definitions need to stay in sync with
// the definitions elfdefinitions.h until they get
// properly upstreamed to gcc/libelf.
#ifndef EM_HSAIL
#define EM_HSAIL 0xAF5A
#endif
#ifndef EM_HSAIL_64
#define EM_HSAIL_64 0xAF5B
#endif
#ifndef EM_AMDIL
#define EM_AMDIL 0x4154
#endif
#ifndef EM_ATI_CALIMAGE_BINARY
#define EM_ATI_CALIMAGE_BINARY 125
#endif
#ifndef EM_AMDGPU
#define EM_AMDGPU 224
#endif
#ifndef ELFOSABI_AMD_OPENCL
#define ELFOSABI_AMD_OPENCL 201
#endif
#ifndef ELFOSABI_HSAIL
#define ELFOSABI_HSAIL 202
#endif
#ifndef ELFOSABI_AMDIL
#define ELFOSABI_AMDIL 203
#endif
#ifndef ELFOSABI_CALIMAGE
#define ELFOSABI_CALIMAGE 100
#endif
namespace amd {

// Test: is it ELF file (with a given bitness) ?
bool isElfHeader(const char* p, signed char ec);
bool isElfMagic(const char* p);

// Test: is it ELF for CAL ?
bool isCALTarget(const char* p, signed char ec);

// Symbol handle
typedef struct symbol_handle *Sym_Handle;

class OclElf
{
public:
    enum {
           CAL_BASE = 1001,         // A number that is not dependent on libelf.h
           CPU_BASE = 2001,
           CPU_FEATURES_FIRST = 0,  // Never generated, but keep it for simplicity.
           CPU_FEATURES_LAST  = 0xF // This should be consistent with cpudevice.hpp
        } oclElfBase;

    typedef enum {
        // NOTE!!! Never remove an entry or change the order.

#if !defined(WITH_LIGHTNING_COMPILER)
        //  All CAL targets are within [CAL_FIRST, CAL_LAST].
        CAL_FIRST      = CAL_TARGET_600  + CAL_BASE,
        CAL_LAST       = CAL_TARGET_LAST + CAL_BASE,
#endif // !defined(WITH_LIGHTNING_COMPILER)

        // All CPU targets are within [CPU_FIRST, CPU_LAST]
        CPU_FIRST      = CPU_FEATURES_FIRST + CPU_BASE,
        CPU_LAST       = CPU_FEATURES_LAST  + CPU_BASE,

        OCL_TARGETS_LAST,
    } oclElfTargets;

    typedef enum {
        CAL_PLATFORM = 0,
        CPU_PLATFORM = 1,
        COMPLIB_PLATFORM = 2,
        LC_PLATFORM = 3,
        LAST_PLATFORM = 4
    } oclElfPlatform;

    typedef enum {
        LLVMIR = 0,
        SOURCE,
        ILTEXT,
        ASTEXT,
        CAL,
        DLL,
        STRTAB,
        SYMTAB,
        RODATA,
        SHSTRTAB,
        NOTES,
        COMMENT,
        ILDEBUG,
        DEBUG_INFO,
        DEBUG_ABBREV,
        DEBUG_LINE,
        DEBUG_PUBNAMES,
        DEBUG_PUBTYPES,
        DEBUG_LOC,
        DEBUG_ARANGES,
        DEBUG_RANGES,
        DEBUG_MACINFO,
        DEBUG_STR,
        DEBUG_FRAME,
        JITBINARY,
        CODEGEN,
        TEXT,
        INTERNAL,
        SPIR,
        SPIRV,
        RUNTIME_METADATA,
        OCL_ELF_SECTIONS_LAST
    } oclElfSections;

    typedef struct {
        char*     sec_name;    //!   section name
        char*     sec_addr;    //!   section address
        uint64_t  sec_size;    //!   section size
        char*     sym_name;    //!   symbol name
        char*     address;     //!   address of corresponding to symbol data
        uint64_t  size;        //!   size of data corresponding to symbol
    } SymbolInfo;

private:

    // file descriptor
    int        _fd;

    // file name
    const char* _fname;

    // pointer to libelf structure
    ::Elf*     _e;

    // Error Object
    mutable OclElfErr  _err;

    // Bitness of the Elf object.
    unsigned char _eclass;

    // Raw ELF bytes in memory from which Elf object is initialized
    // The memory is owned by the client, not this OclElf object !
    const char* _rawElfBytes;
    uint64_t    _rawElfSize;

    // Read, write, or read and write for this Elf object
    const Elf_Cmd  _elfCmd;

    // Memory management
    typedef std::map<void*, size_t> EMemory;
    EMemory  _elfMemory;

    // Indexes of .shstrtab and .strtab (for convenience)
    Elf64_Word    _shstrtab_ndx;
    Elf64_Word    _strtab_ndx;

public:

    /*
       OclElf object can be created for reading or writing (it could be created for
       both reading and writing, which is not supported yet at this time). Currently,
       it has two forms:

        1)  OclElf(eclass, rawElfBytes, rawElfSize, 0, ELF_C_READ)

            To load ELF from raw bytes in memory and generate OclElf object. And this
            object is for reading only.

        2)  OclElf(eclass,  NULL, 0, elfFileName|NULL, ELF_C_WRITE)

            To create an ELF for writing and save it into a file 'elfFileName' (if it
            is NULL, the OclElf will create a temporary file and set it to 'elfFileName'.

            Since we need to read the ELF into memory, this file 'elfFileName' is created
            with both read and write, so that the runtime can use dumpImage() to get ELF
            raw bytes by reading this file.

        'eclass' is ELF's bitness and it must be the same as the eclass of ELF to
        be loaded (for example, rawElfBytes).

        Return values of all public APIs with bool return type
           true  : on success;
           false : on error.
     */
    OclElf (
        unsigned char eclass,       // eclass for this ELF
        const char*   rawElfBytes,  // raw ELF bytes to be loaded
        uint64_t      rawElfSize,   // size of the ELF raw bytes
        const char*   elfFileName,  // File to save this ELF.
        Elf_Cmd       elfcmd        // ELF_C_READ/ELF_C_WRITE
        );

    ~OclElf ();

    /*
       dumpImage() will finalize the ELF and write it into the file. It then reads
       it into the memory; and returns it via <buff, len>.

       The memory pointed by buff is owned by OclElf object.
     */
    bool dumpImage(char** buff, size_t* len);

    /*
       addSection() is used to create a single ELF section with data <d_buf, d_size>. If
       do_copy is true, the OclElf object will make a copy of d_buf and uses that copy to
       create an ELF section.

       When setting do_copy = false,  the caller should make sure that <d_buf, d_size> will
       be unchanged and available during the lifetime of this OclElf object; ie before
       calling dumpImage().
     */
    bool addSection (
        oclElfSections id,
        const void*    d_buf,
        size_t         d_size,
        bool           do_copy = true
        );

    /*
       getSection() will return the whole section in <dst, sz>.

       The memory pointed by <dst, sz> is owned by the OclElf object.
     */
    bool getSection(oclElfSections id, char** dst, size_t* sz) const;

    /*
        addSymbol() adds a symbol with name 'symbolName' and data <buffer, size>
        into the ELF.  'id' indicates which section  <buffer, size> will go
        into.  The meaning of 'do_copy' is the same as addSection().
     */
    bool addSymbol(
        oclElfSections id,             // Section in which symbol is added
        const char*    symbolName,     // Name of symbol
        const void*    buffer,         // Symbol's data
        size_t         size,           // Symbol's size
        bool           do_copy = true  // If true, add a copy of buffer into the section
        );

    /*
     * getSymbol() will return the data associated with
     * the symbol from the Elf.
     *
     * The memory pointed by <buffer, size> is owned by the OclElf object
     */
    bool getSymbol(
        oclElfSections id,        // Section in which symbol is in
        const char* symbolName,   // Name of the symbol to retrieve
        char** buffer,            // Symbol's data
        size_t* size              // Symbol's size
        ) const;

    /*
       nextSymbol() and getSymbolInfo() use the symbol handle to access symbols

       For example:
          for( Sym_Handle s = nextSymbol(NULL); s ; s = nextSymbol(s)) {
              SymbolInfo si;
              if (!getSymbolInfo(s, &si)) {
                  Error;
              }
              use si
          }

          where nextSymbol(NULL) will return the first symbol.

       Note that memory space pointed to by si is owned by OclElf.
     */
    bool getSymbolInfo(Sym_Handle sym, SymbolInfo* symInfo) const;
    Sym_Handle nextSymbol(Sym_Handle symhandle) const;

    /*
        Adds a note with name 'noteName' and description "noteDesc"
        into the .note section of ELF. Length of note name is 'nameSize'.
        Length of note description is "descSize'.
    */
    bool addNote(const char* noteName, const char* noteDesc,
                 size_t nameSize, size_t descSize);

    /*
        Returns the description of a note whose name is 'noteName'
        in 'noteDesc'.
        Returns the length of the description in 'descSize'.
    */
    bool getNote(const char* noteName, char** noteDesc, size_t *descSize);

    /*
       Get/set machine and platform (target) for which elf is built.
     */
    bool getTarget(uint16_t& machine, oclElfPlatform& platform);
    bool setTarget(uint16_t machine, oclElfPlatform platform);

    /*
       Get/set elf type field from header
     */
    bool getType(uint16_t &type);
    bool setType(uint16_t  type);

    /*
       Get/set elf flag field from header.
     */
    bool getFlags(uint32_t &flag);
    bool setFlags(uint32_t  flag);

    /*
       Clear() will return the status of OclElf to just after ctor() is invoked.
       However, it will not regenerate a temporary file name like ctor() does.

       It is useful when the ELF content needs to be discarded for some reason.
     */
    bool Clear();

    bool              hasError() { return (_err.getOclElfError())[0] != 0; }
    const char*      getErrMsg() { return _err.getOclElfError(); }
    unsigned char  getELFClass() { return _eclass; }

    bool isHsaCo() const { return (_eclass == ELFCLASS32) ?
      (elf32_getehdr(_e)->e_machine == EM_AMDGPU) : (elf64_getehdr(_e)->e_machine == EM_AMDGPU); }

private:

    /* Initialization */
    bool Init();

    /*
       Initialize ELF object by creating ELF header and key sections such as
      .shstrtab, .strtab, and .symtab.
     */
    bool InitElf ();

    // Wraper for creating a section header and Elf_Data
    bool createShdr (
        oclElfSections id,
        Elf_Scn*&      scn,
        Elf64_Word     shname,
        Elf64_Word     shlink = 0
        );

    Elf_Data* createElfData(
        Elf_Scn*&      scn,
        oclElfSections id,
        void*          d_buf,
        uint64_t       d_size,
        bool           do_copy
        );

    /*
       Create a new section (id) with data <d_buf, d_size>. If do_copy is true,
       make a copy of d_buf and create a new section with that copy.

       Return the valid Elf_Scn on success; return NULL on error.

       Note that newSection() uses Section Header's size, so make sure elf_update()
       is invoked properly before invoking newSection().
     */
    Elf_Scn* newSection (
        oclElfSections id,
        const void*    d_buf,
        size_t         d_size,
        bool           do_copy = true  // if true, add a copy of d_buf
        );

    /*
        Add a new data into a section by creating a new data descriptor.
        And the new data's offset is returned in 'outOffset'.
     */
    bool addSectionData(
        Elf64_Xword&   outOffset,
        oclElfSections id,
        const void*    buffer,
        size_t         size,
        bool           do_copy=true  // if true, add a copy of buffer
        );

    // Return Elf_Data for this section 'id'
    bool getSectionData(Elf_Data*& data, oclElfSections id) const;

    // Return Elf_Scn for this section 'id'
    bool getSectionDesc(Elf_Scn*& scn, oclElfSections id) const;

    //
    bool getShstrtabNdx(Elf64_Word& outNdx, const char*);

    void* oclelf_allocAndCopy(void* p, size_t sz);
    void* oclelf_calloc(size_t sz);

    void elfMemoryRelease();
};

} // namespace amd

#endif
