/*
************************************************************************************************************************
*
*  Copyright (C) 2020 Advanced Micro Devices, Inc.  All rights reserved.
*
***********************************************************************************************************************/

//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//
#include "elf.hpp"

#include <cstring>
#include <cassert>
#include <string>

#if defined(__linux__)
#include <unistd.h>
#endif

#include "os/os.hpp"
#include "_libelf.h"
namespace amd {

using namespace oclelfutils;

#if !defined(ELFMAG)
#define ELFMAG  "\177ELF"
#define SELFMAG 4
#endif

/*
   Opague data type definition.
*/
struct symbol_handle {
  union {
    Elf64_Sym sym64;
    Elf32_Sym sym32;
  } u;
};

typedef struct {
  OclElf::oclElfSections id;
  const char  *name;
  Elf_Type    d_type;
  uint64_t    d_align;  // section alignment in bytes
  Elf32_Word  sh_type;  // section type
  Elf32_Word  sh_flags; // section flags
  const char  *desc;
} OclElfSectionsDesc;

namespace {
  // Objects that are visible only within this module

  const OclElfSectionsDesc oclElfSecDesc[] =
  {
    { OclElf::LLVMIR,         ".llvmir",         ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "ASIC-independent LLVM IR" },
    { OclElf::SOURCE,         ".source",         ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "OpenCL source" },
    { OclElf::ILTEXT,         ".amdil",          ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "AMD IL text" },
    { OclElf::ASTEXT,         ".astext",         ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "X86 assembly text" },
    { OclElf::CAL,            ".text",           ELF_T_BYTE, 1, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR,
      "AMD CalImage" },
    { OclElf::DLL,            ".text",           ELF_T_BYTE, 1, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR,
      "x86 dll" },
    { OclElf::STRTAB,         ".strtab",         ELF_T_BYTE, 1, SHT_STRTAB,   SHF_STRINGS,
      "String table" },
    { OclElf::SYMTAB,         ".symtab",         ELF_T_SYM,  sizeof(Elf64_Xword), SHT_SYMTAB,   0,
      "Symbol table" },
    { OclElf::RODATA,         ".rodata",         ELF_T_BYTE, 1, SHT_PROGBITS, SHF_ALLOC,
      "Read-only data" },
    { OclElf::SHSTRTAB,       ".shstrtab",       ELF_T_BYTE, 1, SHT_STRTAB,   SHF_STRINGS,
      "Section names" },
    { OclElf::NOTES,          ".note",          ELF_T_NOTE, 1, SHT_NOTE,     0,
      "used by loader for notes" },
    { OclElf::COMMENT,        ".comment",        ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "Version string" },
    { OclElf::ILDEBUG,        ".debugil",        ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "AMD Debug IL" },
    { OclElf::DEBUG_INFO,     ".debug_info",     ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "Dwarf debug info" },
    { OclElf::DEBUG_ABBREV,   ".debug_abbrev",   ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "Dwarf debug abbrev" },
    { OclElf::DEBUG_LINE,     ".debug_line",     ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "Dwarf debug line" },
    { OclElf::DEBUG_PUBNAMES, ".debug_pubnames", ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "Dwarf debug pubnames" },
    { OclElf::DEBUG_PUBTYPES, ".debug_pubtypes", ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "Dwarf debug pubtypes" },
    { OclElf::DEBUG_LOC,      ".debug_loc",      ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "Dwarf debug loc" },
    { OclElf::DEBUG_ARANGES,  ".debug_aranges",  ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "Dwarf debug aranges" },
    { OclElf::DEBUG_RANGES,   ".debug_ranges",   ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "Dwarf debug ranges" },
    { OclElf::DEBUG_MACINFO,  ".debug_macinfo",  ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "Dwarf debug macinfo" },
    { OclElf::DEBUG_STR,      ".debug_str",      ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "Dwarf debug str" },
    { OclElf::DEBUG_FRAME,    ".debug_frame",    ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "Dwarf debug frame" },
    { OclElf::JITBINARY,      ".text",           ELF_T_BYTE, 1, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR,
      "x86 JIT Binary" },
    { OclElf::CODEGEN,         ".cg",            ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "Target dependent IL" },
    { OclElf::TEXT,            ".text",          ELF_T_BYTE, 1, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR,
      "Device specific ISA" },
    { OclElf::INTERNAL,        ".internal",      ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "Internal usage" },
    { OclElf::SPIR,            ".spir",          ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "Vendor/Device-independent LLVM IR" },
    { OclElf::SPIRV,           ".spirv",         ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "SPIR-V Binary" },
    { OclElf::RUNTIME_METADATA,".AMDGPU.runtime_metadata",  ELF_T_BYTE, 1, SHT_PROGBITS, 0,
      "AMDGPU runtime metadata" },
  };

  // index 0 is reserved and must be there (NULL section)
  const char shstrtab[] = {
    /* index  0 */ '\0',
    /* index  1 */ '.', 's', 'h', 's', 't', 'r', 't', 'a', 'b', '\0',
    /* index 11 */ '.', 's', 't', 'r', 't', 'a', 'b', '\0'
  };

#define SHSTRTAB_NAME_NDX  1
#define STRTAB_NAME_NDX   11

  // index 0 is reserved and must be there (NULL name)
  const char strtab[] = {
    /* index 0 */ '\0'
  };

}

  bool
isElfMagic(const char* p)
{
  if (p==NULL || strncmp(p, ELFMAG, SELFMAG) != 0) {
    return false;
  }
  return true;
}

//
  bool
isElfHeader(const char* p, signed char ec)
{
  if (!isElfMagic(p)) {
    return false;
  }
  signed char libVersion  = elf_version(EV_CURRENT);
  signed char fileVersion = p[EI_VERSION];
  signed char elfClass    = p[EI_CLASS];
  if( fileVersion > libVersion) {
    return false;
  }

  // class check:
  if ( elfClass != ec) {
    return false;
  }

  return true;
}

  bool
isCALTarget(const char* p, signed char ec)
{
  if (!isElfMagic(p)) {
    return false;
  }

  Elf64_Half machine;
  if (ec == ELFCLASS32) {
    machine = ((Elf32_Ehdr*)p)->e_machine;

  }
  else {
    machine = ((Elf64_Ehdr*)p)->e_machine;
  }

#if !defined(WITH_LIGHTNING_COMPILER)
  if ( (machine >= OclElf::CAL_FIRST) && (machine <= OclElf::CAL_LAST) ) {
    return true;
  }
#endif // !defined(WITH_LIGHTNING_COMPILER)

  return false;
}

///////////////////////////////////////////////////////////////
////////////////////// elf initializers ///////////////////////
///////////////////////////////////////////////////////////////

OclElf::OclElf (
    unsigned char eclass,
    const char*   rawElfBytes,
    uint64_t      rawElfSize,
    const char*   elfFileName,
    Elf_Cmd       elfcmd
    )
: _fd (-1),
  _fname (elfFileName),
  _e (0),
  _err (),
  _eclass (eclass),
  _rawElfBytes (rawElfBytes),
  _rawElfSize (rawElfSize),
  _elfCmd (elfcmd),
  _elfMemory(),
  _shstrtab_ndx (0),
  _strtab_ndx (0)
{
  if (rawElfBytes != NULL) {
    /*
       In general, 'eclass' should be the same as rawElfBytes's. 'eclass' is what the runtime
       will use for generating an ELF, and therefore it expects the input ELF to have this 'eclass'.
       However, GPU needs to accept both 32-bit and 64-bit ELF for compatibility (we used to
       generate 64-bit ELF, which is the bad design in the first place). Here we just uses eclass
       from rawElfBytes, and overrides the input 'eclass'.
       */
    _eclass = (unsigned char)rawElfBytes[EI_CLASS];
  }
  (void)Init();
}

OclElf::~OclElf()
{
#if 0
  Elf_Cmd c = (_errCmd == ELF_C_READ) ? ELF_C_NULL : _errCmd;
  if (elf_update(_e, c < 0) {
      _err.xfail("OclElf::Fini() : elf_update() failed: %s", elf_errmsg(-1);
        return;
        }
#endif
        _err.Fini();

        elf_end(_e);
        _e = 0;

        if (_fd != -1) {
        xclose(_err, _fname, _fd);
        char* tname= const_cast<char*>(_fname);
        if (tname) {
        unlink(tname);
        free(tname);
        }
        _fd = -1;
        _fname = NULL;

        }

        elfMemoryRelease();
}

  bool
OclElf::Clear()
{
  if (_e) {
    elf_end(_e);
    _e = NULL;
  }

  if (_fd != -1) {
    if (xclose(_err, _fname, _fd) < 0) {
      return false;
    }
    _fd = -1;
  }

  elfMemoryRelease();

  _err.Fini();

  // Re-initialize the object
  Init();

  return !hasError();
}

/*
   Initialize OclElf object
   */
  bool
OclElf::Init()
{
  _err.Init();

  // Create a temporary file if it is needed
  if (_elfCmd != ELF_C_READ) {
    if (_fname != NULL) {
      size_t  sz = strlen(_fname) + 1;

      char* tname = (char*)xmalloc(_err, sz);
      if (tname == 0) {
        _err.xfail("OclElf::Init() failed to malloc()");
        return false;
      }
      strcpy(tname, _fname);
      _fname = static_cast<const char*>(tname);
    }
  }

  if (elf_version(EV_CURRENT) == EV_NONE) {
    _err.xfail("OclElf::Init(): Application expects CURRENT elf version");
    return false;
  }
  int oflag, pmode;
#if defined(_MSC_VER)
  if (_elfCmd == ELF_C_READ) {
    oflag = _O_RDONLY | _O_BINARY;
  }
  else {
    oflag = _O_CREAT | _O_RDWR | _O_TRUNC | _O_BINARY;
  }
  pmode = _S_IREAD  | _S_IWRITE;
#else
  if (_elfCmd == ELF_C_READ) {
    oflag = O_RDONLY;
  }
  else {
    oflag = O_CREAT | O_RDWR | O_TRUNC;
  }
  pmode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH; // 0644
#endif
  if ((_fd == -1) && (_rawElfBytes == NULL)) {
    // case 1: elf object is in file '_fname'

    _fd = xopen(_err, _fname, oflag, pmode);
    if (_fd == -1) {
      _err.xfail("OclElf::Init(): Cannot Open File %s!", _fname);
      return false;
    }

    _e = elf_begin(_fd, _elfCmd, NULL, NULL);
    if (_e == NULL) {
      _err.xfail ("OclElf::Init(): elf_begin failed");
      return false;
    }
  }
  else if (_fd == -1) {
    // case 2: elf object is in memory
    if (_elfCmd == ELF_C_READ) {
      assert ((_fname == NULL) && "ELF file name should not be provided for a read only elf.");
    } else {
      _fd = xopen(_err, _fname, oflag, pmode);
      if (_fd == -1) {
        _err.xfail("OclElf::Init(): Cannot Open File %s!", _fname);
        return false;
      }
    }

    // const_cast is safe
    _e = elf_memory(const_cast<char*>(_rawElfBytes), _rawElfSize, NULL);
    if ( _e == NULL) {
      _err.xfail("OclElf::Init(): elf_memory failed: %s",
          elf_errmsg(-1));
      return false;
    }
    // If _fd != -1, then we are a read/write and not just a read, so change accordingly.
    if (_fd != -1) {
        _e->e_fd = _fd;
        _e->e_cmd = _elfCmd;
    }
  }
  else { // _fd != -1
    // case 3: elf object is in a file with file descriptor '_fd'

    _e = elf_begin(_fd, _elfCmd, NULL, NULL);
    if (_e == NULL) {
      _err.xfail ("OclElf::Init(): elf_begin failed: %s",
          elf_errmsg(-1));
      return false;
    }
  }

  if (!InitElf()) {
    return false;
  }

  // Success
  return true;
}

/*
   Return  true:  if InitElf() is successful
   Return false:  if InitElf() failed.
   */
  bool
OclElf::InitElf ()
{
  assert (_e && "libelf object should have been created already");

  if (_elfCmd != ELF_C_WRITE) {
    // Set up _shstrtab_ndx and _strtab_ndx
    GElf_Ehdr gehdr;
    if (gelf_getehdr(_e, &gehdr) == NULL) {
      _err.xfail("OclElf::InitElf() failed in gelf_getehdr()- %s",
          elf_errmsg(-1));
      return false;
    }

    _shstrtab_ndx = gehdr.e_shstrndx;

    Elf_Scn* scn;
    if (!getSectionDesc(scn, STRTAB)) {
      _err.xfail("OclElf::InitElf() failed in getSectionDesc(STRTAB)");
      return false;
    }

    // Sanity check.  Each ELF binary should have STRTAB !
    if (scn != NULL) {
      _strtab_ndx = elf_ndxscn(scn);
    }

    return true;
  }

  /*********************************/
  /******** ELF_C_WRITE ************/
  /*********************************/

  //
  // 1. Create ELF header
  //
  if (_eclass == ELFCLASS32) {
    Elf32_Ehdr* ehdr32 = elf32_newehdr(_e);
    if (ehdr32 == NULL) {
      _err.xfail("OclElf::InitElf() failed in elf32_newehdr: %s.",
          elf_errmsg(-1));
      return false;
    }
  }
  else {
    Elf64_Ehdr* ehdr64 = elf64_newehdr(_e);
    if (ehdr64 == NULL) {
      _err.xfail("OclElf::InitElf() failed in elf32_newehdr : %s.",
          elf_errmsg(-1));
      return false;
    }
  }

#if 0
  if (elf_update(_e,  ELF_C_NULL) < 0) {
    _err.xfail("elf_update() failed");
    return -1;
  }
#endif

  //
  // 2. Create ELF shstrtab
  //
  Elf_Scn* scn_shstrtab = elf_newscn(_e);
  if (scn_shstrtab == NULL) {
    _err.xfail("Elf::InitElf() failed in elf_newscn : %s", elf_errmsg(-1));
    return false;
  }

  /* addng ELF_Data descriptor associated with section scn */
  Elf_Data* data_shstrtab = createElfData(scn_shstrtab, SHSTRTAB,
      const_cast<char*>(shstrtab), (uint64_t)sizeof(shstrtab), false);
  if (data_shstrtab == NULL) {
    return false;
  }

  if (!createShdr(SHSTRTAB, scn_shstrtab, SHSTRTAB_NAME_NDX)) {
    return false;
  }

  // Save shstrtab section index
  _shstrtab_ndx = elf_ndxscn(scn_shstrtab);
#if defined(BSD_LIBELF)
  elf_setshstrndx(_e, _shstrtab_ndx);
#else
  elfx_update_shstrndx(_e, _shstrtab_ndx);
#endif

#if 0
  if (elf_update(_e,  ELF_C_NULL) < 0) {
    _err.xfail("elf_update() failed");
    return -1;
  }
#endif

  //
  // 3. Create .strtab section
  //
  Elf_Scn* scn_strtab = elf_newscn(_e);
  if (scn_strtab == NULL) {
    _err.xfail("Elf::InitElf() failed in elf_newscn : %s", elf_errmsg(-1));
    return false;
  }

  /* addng ELF_Data descriptor associated with section scn */
  Elf_Data* data_strtab = createElfData(scn_strtab, STRTAB,
      const_cast<char*>(strtab), (uint64_t)sizeof(strtab), false);
  if (data_strtab == NULL) {
    return false;
  }

  if (!createShdr(STRTAB, scn_strtab, STRTAB_NAME_NDX)) {
    return false;
  }

  // Save strtab section index
  _strtab_ndx = elf_ndxscn(scn_strtab);

  // Need to update section header
  if (elf_update(_e,  ELF_C_NULL) < 0) {
    _err.xfail("elf_update() failed");
    return false;
  }

  //
  // 4. Create the symbol table
  //

  // Create the first reserved symbol (undefined symbol)
  size_t sym_sz = (_eclass == ELFCLASS32) ? sizeof(Elf32_Sym) : sizeof(Elf64_Sym);
  void*  sym = oclelf_calloc(sym_sz);
  if (sym == NULL) {
    _err.xfail("OclElf::InitElf() failed to alloc memory");
    return false;
  }

  Elf_Scn* scn_symtab = newSection(SYMTAB, sym, sym_sz, false);
  if (scn_symtab == NULL) {
    // Use newSection()'s error message.
    return false;
  }

  return true;
}

Elf_Data*
OclElf::createElfData(
    Elf_Scn*&      scn,
    oclElfSections id,
    void*          d_buf,
    uint64_t       d_size,
    bool           do_copy
    )
{
  /* addng Elf_Data descriptor associated with section scn */
  Elf_Data*   data = elf_newdata(scn);
  if (data == NULL) {
    _err.xfail("OclElf::createElfData() failed in elf_newdata() - %s",
        elf_errmsg(-1));
    return NULL;
  }

  void* newbuf;
  if (do_copy) {
    newbuf = oclelf_allocAndCopy((void*)d_buf, d_size);
  }
  else {
    newbuf = d_buf;
  }

  data->d_align   = oclElfSecDesc[id].d_align;
  data->d_off     = 0LL;
  data->d_buf     = newbuf;
  data->d_type    = oclElfSecDesc[id].d_type;
  data->d_size    = d_size;
  data->d_version = EV_CURRENT ;

  if (elf_update(_e,  ELF_C_NULL) < 0) {
    _err.xfail("elf_update() failed");
    return NULL;
  }
  return data;
}

bool
OclElf::createShdr (
    oclElfSections id,
    Elf_Scn*&      scn,
    Elf64_Word     shname,
    Elf64_Word     shlink
    )
{
  if (_eclass == ELFCLASS32) {
    Elf32_Shdr* shdr32 = elf32_getshdr(scn);
    if (shdr32 == NULL) {
      _err.xfail("Elf::createShdr() failed in elf32_getshdr(): %s.", elf_errmsg(-1));
      return false;
    }

    shdr32->sh_name  = (Elf32_Word)shname;
    shdr32->sh_type  = (Elf32_Word)oclElfSecDesc[id].sh_type;
    shdr32->sh_flags = (Elf32_Word)oclElfSecDesc[id].sh_flags;

    shdr32->sh_link  = (Elf32_Word)shlink;
  }
  else {
    Elf64_Shdr* shdr64 = elf64_getshdr(scn);
    if (shdr64 == NULL) {
      _err.xfail("Elf::InitElf() failed in elf64_getshdr(): %s.", elf_errmsg(-1));
      return false;
    }

    shdr64->sh_name  = (Elf64_Word)shname;
    shdr64->sh_type  = (Elf64_Word)oclElfSecDesc[id].sh_type;
    shdr64->sh_flags = (Elf64_Xword)oclElfSecDesc[id].sh_flags;

    shdr64->sh_link  = (Elf64_Word)shlink;
  }
  return true;
}

  bool
OclElf::getTarget(uint16_t& machine, oclElfPlatform& platform)
{
  assert(_e != 0);

  GElf_Ehdr ehdrO;
  GElf_Ehdr *ehdr = gelf_getehdr(_e, &ehdrO);
  if (ehdr == NULL) {
    return false;
  }

  Elf64_Half mach = ehdr->e_machine;
  if ((mach >= CPU_FIRST) && (mach <= CPU_LAST)) {
    platform = CPU_PLATFORM;
    machine = mach - CPU_BASE;
  }
#if !defined(WITH_LIGHTNING_COMPILER)
  else if ( (mach >= CAL_FIRST) && (mach <= CAL_LAST)) {
    platform = CAL_PLATFORM;
    machine = mach - CAL_BASE;
  }
#endif // !defined(WITH_LIGHTNING_COMPILER)
  else if (mach == EM_386
      || mach == EM_HSAIL
      || mach == EM_HSAIL_64
      || mach == EM_AMDIL
      || mach == EM_AMDIL_64
      || mach == EM_X86_64) {
    platform = COMPLIB_PLATFORM;
    machine = mach;
  } else {
    // Invalid machine
    return false;
  }

  return true;
}

  bool
OclElf::setTarget(uint16_t machine, oclElfPlatform platform)
{
  assert(_e != 0);

  Elf64_Half mach;
  if (platform == CPU_PLATFORM)
    mach = machine + CPU_BASE;
  else if (platform == CAL_PLATFORM)
    mach = machine + CAL_BASE;
  else
    mach = machine;

  if (_eclass == ELFCLASS32) {
    Elf32_Ehdr* ehdr32 = elf32_getehdr(_e);

    if (ehdr32 == NULL) {
      _err.xfail("setTarget() : failed in elf32_getehdr()- %s.", elf_errmsg(-1));
      return false;
    }

    ehdr32->e_ident[EI_DATA] = ELFDATA2LSB;
    ehdr32->e_type = ET_NONE;
    ehdr32->e_machine = (Elf32_Half)mach;
  }
  else {
    Elf64_Ehdr* ehdr64 = elf64_getehdr(_e);

    if (ehdr64 == NULL) {
      _err.xfail("setTarget() : failed in elf64_getehdr()- %s.", elf_errmsg(-1));
      return false;
    }

    ehdr64->e_ident[EI_DATA] = ELFDATA2LSB;
    ehdr64->e_type = ET_NONE;
    ehdr64->e_machine = mach;
  }

  return true;
}

bool
OclElf::getType(uint16_t &type) {
  assert(_e != 0);

  if (_eclass == ELFCLASS32) {
    Elf32_Ehdr* ehdr32 = elf32_getehdr(_e);

    if (ehdr32 == NULL) {
      _err.xfail("setTarget() : failed in elf32_getehdr()- %s.", elf_errmsg(-1));
      return false;
    }

    type = ehdr32->e_type;
  }
  else {
    Elf64_Ehdr* ehdr64 = elf64_getehdr(_e);

    if (ehdr64 == NULL) {
      _err.xfail("setTarget() : failed in elf64_getehdr()- %s.", elf_errmsg(-1));
      return false;
    }

    type = ehdr64->e_type;
  }

  return true;
}

bool
OclElf::setType(uint16_t type) {
  assert(_e != 0);

  if (_eclass == ELFCLASS32) {
    Elf32_Ehdr* ehdr32 = elf32_getehdr(_e);

    if (ehdr32 == NULL) {
      _err.xfail("setTarget() : failed in elf32_getehdr()- %s.", elf_errmsg(-1));
      return false;
    }

    ehdr32->e_type = type;
  }
  else {
    Elf64_Ehdr* ehdr64 = elf64_getehdr(_e);

    if (ehdr64 == NULL) {
      _err.xfail("setTarget() : failed in elf64_getehdr()- %s.", elf_errmsg(-1));
      return false;
    }

    ehdr64->e_type = type;
  }

  return true;
}

bool
OclElf::getFlags(uint32_t &flag) {
  assert(_e != 0);

  if (_eclass == ELFCLASS32) {
    Elf32_Ehdr* ehdr32 = elf32_getehdr(_e);

    if (ehdr32 == NULL) {
      _err.xfail("setTarget() : failed in elf32_getehdr()- %s.", elf_errmsg(-1));
      return false;
    }

    flag = ehdr32->e_flags;
  }
  else {
    Elf64_Ehdr* ehdr64 = elf64_getehdr(_e);

    if (ehdr64 == NULL) {
      _err.xfail("setTarget() : failed in elf64_getehdr()- %s.", elf_errmsg(-1));
      return false;
    }

    flag = ehdr64->e_flags;
  }

  return true;
}

bool
OclElf::setFlags(uint32_t flag) {
  assert(_e != 0);

  if (_eclass == ELFCLASS32) {
    Elf32_Ehdr* ehdr32 = elf32_getehdr(_e);

    if (ehdr32 == NULL) {
      _err.xfail("setTarget() : failed in elf32_getehdr()- %s.", elf_errmsg(-1));
      return false;
    }

    ehdr32->e_flags = flag;
  }
  else {
    Elf64_Ehdr* ehdr64 = elf64_getehdr(_e);

    if (ehdr64 == NULL) {
      _err.xfail("setTarget() : failed in elf64_getehdr()- %s.", elf_errmsg(-1));
      return false;
    }

    ehdr64->e_flags = flag;
  }

  return true;
}

/*
   returns true if success; return false if fail.
   scn will return scn for 'id'.
   */
  bool
OclElf::getSectionDesc(Elf_Scn*& scn, OclElf::oclElfSections id) const
{
  if ( ((id == SHSTRTAB) && (_shstrtab_ndx != 0)) ||
      ((id == STRTAB)   && (_strtab_ndx != 0)) ) {
    // Special (efficient) processing of SHSTRTAB/STRTAB
    size_t idx = (id == SHSTRTAB) ? _shstrtab_ndx : _strtab_ndx;
    if ((scn = elf_getscn(_e, idx)) == NULL) {
      _err.xfail("OclElf::addSectionDesc(): elf_getscn() failed - %s",
          elf_errmsg(-1));
      return false;
    }
  }
  else {
    /* Search sections */
    const char* sname = oclElfSecDesc[id].name;
    for (scn = elf_nextscn(_e, 0);
        scn != NULL;
        scn = elf_nextscn(_e, scn))
    {
      size_t idx = elf_ndxscn(scn);
      if ( ((idx == _shstrtab_ndx) && (_shstrtab_ndx != 0)) ||
          ((idx == _strtab_ndx)   && (_strtab_ndx   != 0)) ) {
        continue;
      }

      GElf_Shdr shdr;
      if (gelf_getshdr(scn, &shdr) != &shdr) {
        _err.xfail("OclElf::getSectionDesc() : failed in gelf_getshdr()- %s.",
            elf_errmsg(-1));
        return false;
      }

      /* Convert an index (to the shdr string table) to a char pointer */
      char *nm = elf_strptr(_e, _shstrtab_ndx, shdr.sh_name);
      if (strcmp(sname, nm ? nm : "") == 0) {
        // Found !
        break;
      }
    }
  }
  return true;
}

/*
   Return true if success; return false if fail.
   data will return Elf_Data.
   */
  bool
OclElf::getSectionData(Elf_Data*& data, OclElf::oclElfSections id) const
{
  assert(_e != 0);

  data = NULL;
  Elf_Scn* scn;
  if (!getSectionDesc(scn, id)) {
    return false;
  }
  if (scn != NULL) {
    // There is only one data descriptor (we are reading!)
    data = elf_getdata(scn, 0);
  }
  return true;
}

/*
   Get the whole section, assuming that there is only one data descriptor
   */
  bool
OclElf::getSection(OclElf::oclElfSections id, char** dst, size_t* sz) const
{
  assert((oclElfSecDesc[id].id == id) &&
      "oclElfSecDesc[] should be in the same order as enum oclElfSections");

  Elf_Data* data;
  if (!getSectionData(data, id)) {
    _err.xfail("OclElf::getSection() failed in getSectionData()");
    return false;
  }
  if (data == NULL) {
    *dst = NULL;
    *sz  = 0;
  }
  else {
    *sz = (size_t)data->d_size;
    *dst = (char*)data->d_buf;
  }

  return true;
}

/*
   API routines for manipulating symbols
   */
  Sym_Handle
OclElf::nextSymbol(Sym_Handle symHandle) const
{
  size_t sz;
  char*  beg, *end;

  if (!getSection(SYMTAB, &beg, &sz)) {
    _err.xfail("OclElf::nextSymbol() failed in getSection()");
    return NULL;
  }

  if ( (beg == 0) || (sz == 0) ) {
    return NULL;
  }

  end = beg + sz;
  if (_eclass == ELFCLASS64) {
    // Skip the first dummy symbol (STT_NOTYPE)
    beg += sizeof(Elf64_Sym);

    if (beg == end) { // No valid symbols in the table
      return NULL;
    }

    if (symHandle == NULL) {
      // Return the first symbol
      return reinterpret_cast<Sym_Handle>(beg);
    }

    // Return the next symbol
    Elf64_Sym* sym64 = reinterpret_cast<Elf64_Sym*>(symHandle);
    sym64++;
    if (reinterpret_cast<char*>(sym64) == end) {
      return NULL;
    }
    return reinterpret_cast<Sym_Handle>(sym64);
  }
  else {
    // Skip the first dummy symbol (STT_NOTYPE)
    beg += sizeof(Elf32_Sym);

    if (beg == end) { // No valid symbols in the table
      return NULL;
    }

    if (symHandle == NULL) {
      // Return the first symbol
      return reinterpret_cast<Sym_Handle>(beg);
    }

    Elf32_Sym* sym32 = reinterpret_cast<Elf32_Sym*>(symHandle);
    sym32++;
    if (reinterpret_cast<char*>(sym32) == end) {
      return NULL;
    }
    return reinterpret_cast<Sym_Handle>(sym32);
  }

  // UNREACHABLE
  return NULL;
}

/*
   Given a symbol handle, return info for this symbol

   Fails with symbols which have special section indexes (like absolute symbols).
   It is impossible to return valid SymbolInfo for such symbols because
   correct section names are unknown (unspecified in ELF).
   */
  bool
OclElf::getSymbolInfo(Sym_Handle  symHandle, SymbolInfo* symInfo) const
{
  assert(_e != 0);

  Elf_Scn *scn;
  char*       sym_name;
  Elf64_Addr  st_value;    /* visibility */
  Elf64_Xword st_size;     /* index of related section */

  if (_eclass == ELFCLASS64) {
    Elf64_Sym* sym64 = reinterpret_cast<Elf64_Sym*>(symHandle);
    if (sym64->st_shndx >= SHN_LORESERVE && sym64->st_shndx <= SHN_HIRESERVE) {
      return false;
    }

    sym_name = elf_strptr(_e, _strtab_ndx, sym64->st_name);
    st_value = (Elf64_Addr)(sym64->st_value);
    st_size  = (Elf64_Xword)(sym64->st_size);

    // get section
    scn = elf_getscn(_e, sym64->st_shndx);
  }
  else {
    Elf32_Sym* sym32 = reinterpret_cast<Elf32_Sym*>(symHandle);
    if (sym32->st_shndx >= SHN_LORESERVE && sym32->st_shndx <= SHN_HIRESERVE) {
      return false;
    }

    sym_name = elf_strptr(_e, _strtab_ndx, sym32->st_name);
    st_value = (Elf64_Addr)(sym32->st_value);
    st_size  = (Elf64_Xword)(sym32->st_size);

    // get section
    scn = elf_getscn(_e, sym32->st_shndx);
  }

  GElf_Shdr gshdr;
  if (gelf_getshdr(scn, &gshdr) == NULL) {
    _err.xfail("OclElf::getSymbolInfo() failed in gelf_getshdr() - %s.",
        elf_errmsg(-1));
    return false;
  }
  char* sec_name = elf_strptr(_e, _shstrtab_ndx, gshdr.sh_name);

  // Assume there is only one Elf_Data. For reading, it's always true
  Elf_Data* data = elf_getdata(scn, 0);
  if (data == NULL) {
      symInfo->sec_addr = (char*)NULL;
      symInfo->sec_size = 0;
      symInfo->address  = (char*)NULL;
      symInfo->size     = (uint64_t)0;
  }
  else {
      symInfo->sec_addr = (char*)data->d_buf;
      symInfo->sec_size = data->d_size;
      symInfo->address  = symInfo->sec_addr + (size_t)st_value;
      symInfo->size     = (uint64_t)st_size;
  }
  symInfo->sec_name = sec_name;
  symInfo->sym_name = sym_name;

  return true;
}

/*
   AddSectionData() will add data into a section. Return the offset
   of the data in this section if success; return -1 if fail.
   */
bool
OclElf::addSectionData (
    Elf64_Xword&   outOffset,
    oclElfSections id,
    const void*    buffer,
    size_t         size,
    bool           do_copy    // true if buffer needs to be copied
    )
{
  outOffset = 0;
  const char* secName = oclElfSecDesc[id].name;
  GElf_Shdr shdr;
  Elf_Scn* scn;
  if (!getSectionDesc(scn, id)) {
    return false;
  }
  assert (scn && "Elf_Scn should have been created already");

  if (gelf_getshdr(scn, &shdr) != &shdr) {
    _err.xfail("OclElf::addSectionData(): gelf_getshdr() failed - %s",
        elf_errmsg(-1));
    return false;
  }
  outOffset = (Elf64_Xword)shdr.sh_size;

  /* addng Elf_Data descriptor associated with section scn */
  Elf_Data* data = createElfData(scn, id, const_cast<void*>(buffer),
      (uint64_t)size, do_copy);
  if (data == NULL) {
    return false;
  }

  return true;
}

/*
   getShdrNdx() returns an index to the .shstrtab in 'outNdx' for "name" if it
   is in .shstrtab (outNdx == 0 means it is not in .shstrtab). It return true if
   it is successful; return false if en error occured.
   */
  bool
OclElf::getShstrtabNdx(Elf64_Word& outNdx, const char* name)
{
  outNdx = 0;

  // .shstrtab must be created already
  Elf_Scn* scn = elf_getscn(_e, _shstrtab_ndx);
  if (scn == NULL) {
    _err.xfail("OclElf::getShdrNdx() failed in elf_getscn for section .shstrtab - %s",
        elf_errmsg(-1));
    return false;
  }

  Elf_Data* data = elf_getdata(scn, NULL);
  if (data == NULL) {
    _err.xfail("Elf::getShdrNdx() failed in elf_getdata for section .shstrtab - %s",
        elf_errmsg(-1));
    return false;
  }

  size_t name_sz = strlen(name);
  uint64_t data_offset = 0;
  do {
    if (data->d_size > name_sz) {
      char* base = (char*)data->d_buf;
      char* end = base + (size_t)data->d_size;
      char* b = base;
      char* e;

      while ( b != end) {
        e = b;

        // find the next 0 char
        while ( (e != end) && (*e != 0) ) {
          e++;
        }

        if ((e != end) && ((size_t)(e - b) == name_sz) &&
            (strcmp(b, name) == 0)) {
          outNdx = (Elf64_Word)((b - base) + data_offset);
          return true;
        }
        b = e+1;
      }
    }
    data_offset += data->d_size;
  } while ((data = elf_getdata(scn, data)) != NULL);

  return true;
}

/*
   newSection() assumes that .shstrtab and .strtab have been created already.
   Return the pointer to the new section if success;  return 0 if fail.
   */
Elf_Scn*
OclElf::newSection (
    OclElf::oclElfSections id,
    const void*            d_buf,
    size_t                 d_size,
    bool                   do_copy
    )
{
  Elf64_Word sh_name;
  if (!getShstrtabNdx(sh_name, oclElfSecDesc[id].name)) {
    _err.xfail("OclElf::newSection() failed in getShstrtabNdx() for section %s",
        oclElfSecDesc[id].name);
    return NULL;
  }

  if (sh_name == 0) { // Need to create a new entry for this section name
    Elf64_Xword offset;
    if (!addSectionData(offset, SHSTRTAB, oclElfSecDesc[id].name,
          strlen(oclElfSecDesc[id].name) + 1, false)) {
      _err.xfail("OclElf::newSection() failed in getSectionData() for section %s",
          oclElfSecDesc[id].name);
      return NULL;
    }
    sh_name = (Elf64_Word)offset;
  }

  // Create a new section
  Elf_Scn* scn = elf_newscn(_e);
  if (scn == NULL) {
    _err.xfail("OclElf::newSection() failed in elf_newscn() - %s.",
        elf_errmsg(-1));
    return NULL;
  }

  // If there is no data, skip creating Elf_Data
  if ((d_buf != NULL) && (d_size != 0)) {
    Elf_Data* data = createElfData(scn, id,
        const_cast<void*>(d_buf), (uint64_t)d_size, do_copy);
    if (data == NULL) {
      return NULL;
    }
  }

  if (!createShdr(id, scn, sh_name, (id == SYMTAB) ? _strtab_ndx : 0)) {
    return NULL;
  }

  if (elf_update(_e,  ELF_C_NULL) < 0) {
    _err.xfail("OclElf::newSection(): elf_update() failed");
    return NULL;
  }

  return scn;
}

/*
   Return  true:  success
false:  fail
*/
bool
OclElf::addSection (
    oclElfSections id,
    const void*    d_buf,
    size_t         d_size,
    bool           do_copy
    )
{
  assert(oclElfSecDesc[id].id == id &&
      "struct oclElfSecDesc should be ordered by id same as enum Elf::oclElfSections");

  /* If section is already in elf object, simply return its address */
  Elf_Scn* scn;
  if (!getSectionDesc(scn, id)) {
    // Failed
    return false;
  }

  if (scn != NULL) {
    Elf64_Xword sec_offset;
    if (!addSectionData(sec_offset, id, d_buf, d_size, do_copy)) {
      _err.xfail("OclElf::addSection() failed in addSectionData() for section name %s.",
          oclElfSecDesc[id].name);
      return false;
    }
  }
  else {
    scn = newSection(id, d_buf, d_size, do_copy);
    if (scn == NULL) {
      _err.xfail("OclElf::addSection() failed in newSection() for section name %s.",
          oclElfSecDesc[id].name);
      return false;
    }
  }
  return true;
}

bool
OclElf::addSymbol(
    oclElfSections id,
    const char* symbolName,
    const void* buffer,
    size_t size,
    bool do_copy
    )
{
  assert(oclElfSecDesc[id].id == id &&
      "The order of oclElfSecDesc[] and Elf::oclElfSections mismatches.");

  const char* sectionName = oclElfSecDesc[id].name;

  bool isFunction = ((id == OclElf::CAL) || (id == OclElf::DLL) || (id == OclElf::JITBINARY)) ? true : false;

  // Get section index
  Elf_Scn* scn;
  if (!getSectionDesc(scn, id)) {
    _err.xfail("OclElf::addSymbol() failed in getSectionDesc");
    return false;
  }
  if (scn == NULL) {
    // Create a new section.
    if ((scn = newSection(id, NULL, 0, false)) == NULL) {
      _err.xfail("OclElf::addSymbol() failed in newSection");
      return false;
    }
  }
  size_t sec_ndx = elf_ndxscn(scn);
  if (sec_ndx == SHN_UNDEF) {
    _err.xfail("OclElf::addSymbol() failed in elf_ndxscn() - %s.",
        elf_errmsg(-1));
    return false;
  }

  // Put symbolName into .strtab section
  Elf64_Xword strtab_offset;
  if (!addSectionData(strtab_offset, STRTAB, (void*)symbolName,
        strlen(symbolName)+1, true)) {
    _err.xfail("OclElf::addSymbol() failed in addSectionData(.strtab)");
    return false;
  }

  // Put buffer into section
  Elf64_Xword sec_offset = 0;
  if ( (buffer != NULL) && (size != 0) ) {
    if (!addSectionData(sec_offset, id, buffer, size, do_copy)) {
      _err.xfail("OclElf::addSymbol() failed in addSectionData(%s)", sectionName);
      return false;
    }
  }

  bool retvalue;
  Elf64_Xword symtab_offset;
  if (_eclass == ELFCLASS64) {
    Elf64_Sym* sym64 = (Elf64_Sym*)oclelf_calloc(sizeof(Elf64_Sym));

    sym64->st_name  = (Elf64_Word)strtab_offset;
    sym64->st_value = (Elf64_Addr)sec_offset;
    sym64->st_size  = (Elf64_Xword)size;
    sym64->st_info  = (isFunction)? STT_FUNC : STT_OBJECT;
    sym64->st_shndx = (Elf64_Section)sec_ndx;

    retvalue = addSectionData(symtab_offset, SYMTAB, sym64, sizeof(Elf64_Sym), false);
  }
  else {  // _eclass == ELFCLASS32
    Elf32_Sym* sym32 = (Elf32_Sym*)oclelf_calloc(sizeof(Elf32_Sym));

    sym32->st_name  = (Elf32_Word)strtab_offset;
    sym32->st_value = (Elf32_Addr)sec_offset;
    sym32->st_size  = (Elf32_Word)size;
    sym32->st_info  = (isFunction)? STT_FUNC : STT_OBJECT;
    sym32->st_shndx = (Elf32_Section)sec_ndx;

    retvalue = addSectionData(symtab_offset, SYMTAB, sym32, sizeof(Elf32_Sym), false);
  }

  if (!retvalue) {
    _err.xfail("OclElf::addSymbol() failed in addSectionData(.symtab)");
    return false;
  }

  if (elf_update(_e,  ELF_C_NULL) < 0) {
    _err.xfail("OclElf::addSymbol() : elf_update() failed");
    return false;
  }

  return true;
}

bool
OclElf::getSymbol(
    oclElfSections id,
    const char* symbolName,
    char** buffer,
    size_t* size
    ) const
{
  assert(oclElfSecDesc[id].id == id &&
      "The order of oclElfSecDesc[] and Elf::oclElfSections mismatches.");
  if (!size || !buffer || !symbolName) {
    return false;
  }
  // Initialize the size and buffer to invalid data points.
  (*size) = 0;
  (*buffer) = NULL;
  for (amd::Sym_Handle s = nextSymbol(NULL); s; s = nextSymbol(s)) {
    amd::OclElf::SymbolInfo si;
    // Continue if symbol information is not retrieved.
    if (!getSymbolInfo(s, &si)) {
      continue;
    }
    // Continue if the symbol is in the wrong section.
    if (strcmp(oclElfSecDesc[id].name, si.sec_name)) {
      continue;
    }
    // Continue if the symbol name doesn't match.
    if (strcmp(symbolName, si.sym_name)) {
      continue;
    }
    // Set the size and the address and return true.
    (*size) = si.size;
    (*buffer) = si.address;
    return true;
  }
  return false;
}

bool
OclElf::addNote(
    const char* noteName,
    const char* noteDesc,
    size_t nameSize,
    size_t descSize
    )
{
  if ((nameSize == 0 && descSize == 0)
      || (nameSize != 0 && noteName == NULL)
      || (descSize != 0 && noteDesc == NULL)) {
    _err.xfail("OclElf::addNote() empty note");
    return false;
  }

  const oclElfSections sid = NOTES;
  assert(oclElfSecDesc[sid].id == sid &&
      "The order of oclElfSecDesc[] and Elf::oclElfSections mismatches.");

  // Get section
  Elf_Scn* scn;
  if (!getSectionDesc(scn, sid)) {
    _err.xfail("OclElf::addNote() failed in getSectionDesc");
    return false;
  }
  if (scn == NULL) {
    // Create a new section.
    if ((scn = newSection(sid, NULL, 0, false)) == NULL) {
      _err.xfail("OclElf::addNote() failed in newSection");
      return false;
    }
  }

  // Put note into section
  Elf64_Xword sec_offset = 0;
  size_t bufsize = sizeof(Elf_Note) + nameSize + descSize;
  char* buffer = (char*)oclelf_calloc(bufsize);
  if (buffer == NULL) return false;

  Elf_Note* en = reinterpret_cast<Elf_Note*>(buffer);
  en->n_namesz = nameSize;
  en->n_descsz = descSize;
  en->n_type = 0;
  memcpy(buffer+sizeof(Elf_Note), noteName, nameSize);
  memcpy(buffer+sizeof(Elf_Note)+nameSize, noteDesc, descSize);
  if (!addSectionData(sec_offset, sid, buffer, bufsize, false/*not copy*/)) {
    const char* sectionName = oclElfSecDesc[sid].name;
    _err.xfail("OclElf::addNote() failed in addSectionData(%s)", sectionName);
    return false;
  }

  if (elf_update(_e,  ELF_C_NULL) < 0) {
    _err.xfail("OclElf::addNote() : elf_update() failed");
    return false;
  }

  return true;
}

bool
OclElf::getNote(
    const char* noteName,
    char** noteDesc,
    size_t *descSize
    )
{
  if (!descSize || !noteDesc || !noteName) {
    return false;
  }

  const oclElfSections sid = NOTES;
  assert(oclElfSecDesc[sid].id == sid &&
      "The order of oclElfSecDesc[] and Elf::oclElfSections mismatches.");

  // Get section
  Elf_Scn* scn;
  if (!getSectionDesc(scn, sid)) {
    _err.xfail("OclElf::getNote() failed in getSectionDesc");
    return false;
  }
  if (scn == NULL) {
    _err.xfail("OclElf::getNote() failed: .note section not found");
    return false;
  }

  // read the whole .note section
  Elf_Data* data = elf_getdata(scn, 0);

  // Initialize the size and buffer to invalid data points.
  *descSize = 0;
  *noteDesc = NULL;

  // look for the specified note
  char* ptr = (char*)data->d_buf;
  while (ptr < (char*)data->d_buf + data->d_size) {
    Elf_Note* note = reinterpret_cast<Elf_Note*>(ptr);

    // Continue if the note name doesn't match.
    if (strlen(noteName) != note->n_namesz
        || strncmp(noteName, ptr+sizeof(Elf_Note), note->n_namesz) != 0) {
      ptr += sizeof(Elf_Note) + note->n_namesz + note->n_descsz;
      continue;
    }
    // Set the size and the address and return true.
    *descSize = note->n_descsz;
    *noteDesc = ptr + sizeof(Elf_Note) + note->n_namesz;
    return true;
  }
  return false;
}

  bool
OclElf::dumpImage(char** buff, size_t* len)
{
  if (buff == NULL || len == NULL ) {
    return false;
  }

  assert ((_fd != -1) && "_fd in Elf::dumpImage should be defined");

  // Now, write the ELF into the file
  if (elf_update(_e, ELF_C_WRITE) < 0) {
    _err.xfail("OclElf::dumpImage() : elf_update() failed - %s",
        elf_errmsg(-1));
    return false;
  }

  int buff_sz = xlseek(_err, _fname, _fd, 0, SEEK_END);
  if (buff_sz == -1) {
    return false;
  }

  /*
     The memory is owned by caller, and caller assumes that the memory is new'ed.
     So, use new instead of malloc
     */
  *buff = new char[buff_sz];
  if (*buff == NULL) {
    _err.xfail("OclElf::dumpImage() : new char[sz] failed");
    return false;
  }

  if (xlseek(_err, _fname, _fd, 0, SEEK_SET) == -1) {
    _err.xfail("OclElf::dumpImage() failed in xlseek()");
    delete [] *buff;
    return false;
  }

  if (xread(_err, _fname, _fd, *buff, buff_sz) != buff_sz) {
    _err.xfail("OclElf::dumpImage() failed in xread()");
    delete [] *buff;
    *buff = 0;
    return false;
  }

  *len = buff_sz;
  return true;
}

  void*
OclElf::oclelf_allocAndCopy(void* p, size_t sz)
{
  if (p == 0 || sz == 0) return p;

  void* buf = xmalloc(_err, sz);
  if (buf == 0) {
    _err.xfail("OclElf::oclelf_allocAndCopy() failed");
    return 0;
  }

  memcpy(buf, p, sz);
  _elfMemory.insert( std::make_pair(buf, sz));
  return buf;
}

  void*
OclElf::oclelf_calloc(size_t sz)
{
  void* buf = xmalloc(_err, sz);
  if (buf == 0) {
    _err.xfail("OclElf::oclelf_calloc() failed");
    return 0;
  }
  _elfMemory.insert( std::make_pair(buf, sz));
  return buf;
}

  void
OclElf::elfMemoryRelease()
{
  for(EMemory::iterator it = _elfMemory.begin(); it != _elfMemory.end(); it++) {
    free(it->first);
  }
  _elfMemory.clear();
}

} // namespace amd
