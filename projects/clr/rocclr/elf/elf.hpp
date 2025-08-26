/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef ELF_HPP_
#define ELF_HPP_

#include <map>

#include "top.hpp"
#include "elfio/elfio.hpp"
#include <sstream>
using amd::ELFIO::Elf64_Ehdr;
using amd::ELFIO::Elf64_Shdr;

// Not sure where to put these in the libelf
#define AMD_BIF2 2  // AMD BIF Version 2.0
#define AMD_BIF3 3  // AMD BIF Version 3.0

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
#ifndef EM_AMDIL_64
#define EM_AMDIL_64 0x4155
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
using namespace amd::ELFIO;

class Elf {
 public:
  enum {
    CAL_BASE = 1001,  // A number that is not dependent on libelf.h
    CPU_BASE = 2001,
    CPU_FEATURES_FIRST = 0,  // Never generated, but keep it for simplicity.
    CPU_FEATURES_LAST = 0xF  // This should be consistent with cpudevice.hpp
  } ElfBase;

  typedef enum {
    // NOTE!!! Never remove an entry or change the order.

    // All CPU targets are within [CPU_FIRST, CPU_LAST]
    CPU_FIRST = CPU_FEATURES_FIRST + CPU_BASE,
    CPU_LAST = CPU_FEATURES_LAST + CPU_BASE,

    OCL_TARGETS_LAST,
  } ElfTargets;

  typedef enum {
    CAL_PLATFORM = 0,
    CPU_PLATFORM = 1,
    COMPLIB_PLATFORM = 2,
    LC_PLATFORM = 3,
    LAST_PLATFORM = 4
  } ElfPlatform;

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
    ELF_SECTIONS_LAST
  } ElfSections;

  typedef enum {
    ELF_C_NULL = 0,
    ELF_C_CLR,
    ELF_C_FDDONE,
    ELF_C_FDREAD,
    ELF_C_RDWR,
    ELF_C_READ,
    ELF_C_SET,
    ELF_C_WRITE,
    ELF_C_NUM
  } ElfCmd;

  struct SymbolInfo {
    std::string sec_name;  //!   section name
    const char* sec_addr;  //!   section address
    uint64_t sec_size;     //!   section size
    std::string sym_name;  //!   symbol name
    const char* address;   //!   address of corresponding to symbol data
    uint64_t size;         //!   size of data corresponding to symbol
    SymbolInfo()
        : sec_name(), sec_addr(nullptr), sec_size(0), sym_name(), address(nullptr), size(0) {}

    SymbolInfo(const char* sename, const char* seaddr, uint64_t sesize, const char* syname,
               const char* syaddr, uint64_t sysize)
        : sec_name(sename),
          sec_addr(seaddr),
          sec_size(sesize),
          sym_name(syname),
          address(syaddr),
          size(sysize) {}
  };

  /*
   * Note descriptors.
   * Follow https://docs.oracle.com/cd/E19683-01/816-1386/6m7qcoblj/index.html#chapter6-18048
   */
  struct ElfNote {
    uint32_t n_namesz; /* Length of note's name. */
    uint32_t n_descsz; /* Length of note's value. */
    uint32_t n_type;   /* Type of note. */
  };

 private:
  // elfio object for reading and writting
  elfio _elfio;

  // file name
  std::string _fname;

  // Bitness of the Elf object.
  unsigned char _eclass;

  // Raw ELF bytes in memory from which Elf object is initialized
  // The memory is owned by the client, not this Elf object !
  const char* _rawElfBytes;
  uint64_t _rawElfSize;

  // Read, write, or read and write for this Elf object
  const ElfCmd _elfCmd;

  // Memory management
  typedef std::map<void*, size_t> EMemory;
  EMemory _elfMemory;

  Elf64_Word _shstrtab_ndx;  // Indexes of .shstrtab. Must be valid.
  Elf64_Word _strtab_ndx;    // Indexes of .strtab. Must be valid.
  Elf64_Word _symtab_ndx;    // Indexes of .symtab. May be SHN_UNDEF.

  bool _successful;

 public:
  /*
     Elf object can be created for reading or writing (it could be created for
     both reading and writing, which is not supported yet at this time). Currently,
     it has two forms:

      1)  Elf(eclass, rawElfBytes, rawElfSize, 0, ELF_C_READ)

          To load ELF from raw bytes in memory and generate Elf object. And this
          object is for reading only.

      2)  Elf(eclass,  nullptr, 0, elfFileName|nullptr, ELF_C_WRITE)

          To create an ELF for writing and save it into a file 'elfFileName' (if it
          is nullptr, the Elf will create a stream in memory.

          Since we need to read the ELF into memory, the runtime can use dumpImage() to get ELF
          raw bytes by reading this file/stream.

      'eclass' is ELF's bitness and it must be the same as the eclass of ELF to
      be loaded (for example, rawElfBytes).


      Return values of all public APIs with bool return type
         true  : on success;
         false : on error.
   */
  Elf(unsigned char eclass,     // eclass for this ELF
      const char* rawElfBytes,  // raw ELF bytes to be loaded
      uint64_t rawElfSize,      // size of the ELF raw bytes
      const char* elfFileName,  // File to save this ELF.
      ElfCmd elfcmd             // ELF_C_READ/ELF_C_WRITE
  );

  ~Elf();

  /*
   * dumpImage() will finalize the ELF and write it into the file/stream. It then reads
   * it into the memory; and returns it via <buff, len>.
   * The memory pointed by buff is new'ed in Elf and should be deleted by caller
   * if dumpImage() succeeds.
   */
  bool dumpImage(char** buff, size_t* len);
  bool dumpImage(std::istream& is, char** buff, size_t* len) const;

  /*
   * If the session doesn't exist, create a new ELF section with data <d_buf, d_size>;
   * otherwise, append the data.
   */
  bool addSection(ElfSections id, const void* d_buf, size_t d_size);

  /*
   * Return the whole section in <dst, sz>.
   * The memory pointed by <dst, sz> is owned by the Elf object.
   */
  bool getSection(ElfSections id, char** dst, size_t* sz) const;

  /*
   * Add a symbol with name 'symbolName' and data <buffer, size>
   * into the ELF.  'id' indicates which section  <buffer, size> will go
   * into.
   */
  bool addSymbol(ElfSections id,          // Section in which symbol is added
                 const char* symbolName,  // Name of symbol
                 const void* buffer,      // Symbol's data
                 size_t size              // Symbol's size
  );

  /*
   * Return the data associated with the symbol from the Elf.
   * The memory pointed by <buffer, size> is owned by the Elf object
   */
  bool getSymbol(ElfSections id,          // Section in which symbol is in
                 const char* symbolName,  // Name of the symbol to retrieve
                 char** buffer,           // Symbol's data
                 size_t* size             // Symbol's size
  ) const;

  /* Return number of symbols in SYMTAB section */
  unsigned int getSymbolNum() const;

  /* Return SymbolInfo of the index-th symbol in SYMTAB section */
  bool getSymbolInfo(unsigned int index, SymbolInfo* symInfo) const;

  /*
   * Adds a note with name 'noteName' and description "noteDesc"
   * into the .note section of ELF. Length of note description is "descSize'.
   */
  bool addNote(const char* noteName, const char* noteDesc, size_t descSize);

  /*
   * Return the description of a note whose name is 'noteName'
   * in 'noteDesc'.
   * Return the length of the description in 'descSize'.
   * The memory pointed by <noteDesc, descSize> is owned by the Elf object.
   */
  bool getNote(const char* noteName, char** noteDesc, size_t* descSize);


  /* Get/set machine and platform (target) for which elf is built */
  bool getTarget(uint16_t& machine, ElfPlatform& platform) const;
  bool setTarget(uint16_t machine, ElfPlatform platform);

  /* Get/set elf type field from header */
  bool getType(uint16_t& type) const;
  bool setType(uint16_t type);

  /* Get/set elf flag field from header */
  bool getFlags(uint32_t& flag) const;
  bool setFlags(uint32_t flag);

  /*
   * Clear() will return the status of Elf to just after ctor() is invoked.
   * It is useful when the ELF content needs to be discarded for some reason.
   */
  bool Clear();

  unsigned char getELFClass() const { return _eclass; }

  bool isSuccessful() const { return _successful; }

  bool isHsaCo() const { return _elfio.get_machine() == EM_AMDGPU; }

  /* Return number of segments */
  unsigned int getSegmentNum() const;

  /* Return segment at index */
  bool getSegment(const unsigned int index, segment*& seg) const;

  /* Return size of elf file */
  static uint64_t getElfSize(const void* emi);

  /* is it ELF */
  static bool isElfMagic(const char* p);

  // is it ELF for CAL ?
  static bool isCALTarget(const char* p, signed char ec);

 private:
  /* Initialization */
  bool Init();

  /*
   * Initialize ELF object by creating ELF header and key sections such as
   * .shstrtab, .strtab, and .symtab.
   */
  bool InitElf();

  /* Setup a section header */
  bool setupShdr(ElfSections id, section* section, Elf64_Word shlink = 0) const;

  /*
   * Create a new data into an existing section.
   * And the section is returned in 'sec'.
   */
  bool createElfData(section*& sec, ElfSections id, const char* d_buf, size_t d_size);

  /*
   * Assumes that .shstrtab and .strtab have been created already.
   * Create a new section (id) with data <d_buf, d_size>.
   * Return the valid section* on success; nullptr on error.
   */
  section* newSection(ElfSections id, const char* d_buf, size_t d_size);

  /*
   * Add a new data into the existing section.
   * And the new data's offset is returned in 'outOffset'.
   */
  bool addSectionData(Elf_Xword& outOffset, ElfSections id, const void* buffer, size_t size);

  /*
   * Return an index to the .shstrtab in 'outNdx' for "name" if it
   * is in .shstrtab (outNdx == 0 means it is not in .shstrtab).
   */
  bool getShstrtabNdx(Elf64_Word& outNdx, const char*);

  /*
   * Generate UUID string
   */
  static std::string generateUUIDV4();

  /*
   * Return newly-allocated memory or nullptr
   * The allocated memory is guaranteed to be initialized to zero.
   */
  void* xmalloc(const size_t len);

  void* allocAndCopy(void* p, size_t sz);
  void* calloc(size_t sz);

  void elfMemoryRelease();
};

}  // namespace amd

#endif
