/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

#include "elf.hpp"

#include <cstring>
#include <cassert>
#include <string>

#if defined(__linux__)
#include <unistd.h>
#endif

#include "os/os.hpp"
#include <thread>
#include <utils/flags.hpp>
#include <utils/debug.hpp>
#include <random>
#include <sstream>


// #define DEBUG_DETAIL // For detailed debug log

#ifdef DEBUG_DETAIL

#define ElfTrace(level)                                                                            \
  ClPrint(level, amd::LOG_CODE, "%-5d: [%zx] %p %s: ", getpid(), std::this_thread::get_id(), this, \
          __func__)

#define logElfInfo(msg)                                                                            \
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "%-5d: [%zx] %p %s: " msg, getpid(),                       \
          std::this_thread::get_id(), this, __func__)

#define logElfWarning(msg)                                                                         \
  ClPrint(amd::LOG_WARNING, amd::LOG_CODE, "%-5d: [%zx] %p %s: " msg, getpid(),                    \
          std::this_thread::get_id(), this, __func__)

#define LogElfDebug(format, ...)                                                                   \
  ClPrint(amd::LOG_DEBUG, amd::LOG_CODE, "%-5d: [%zx] %p %s: " format, getpid(),                   \
          std::this_thread::get_id(), this, __func__, __VA_ARGS__)

#define LogElfWarning(format, ...)                                                                 \
  ClPrint(amd::LOG_WARNING, amd::LOG_CODE, "%-5d: [%zx] %p %s: " format, getpid(),                 \
          std::this_thread::get_id(), this, __func__, __VA_ARGS__)

#define LogElfInfo(format, ...)                                                                    \
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "%-5d: [%zx] %p %s: " format, getpid(),                    \
          std::this_thread::get_id(), this, __func__, __VA_ARGS__)
#else
#define ElfTrace(level)
#define logElfInfo(msg)
#define logElfWarning(msg)
#define LogElfDebug(format, ...)
#define LogElfWarning(format, ...)
#define LogElfInfo(format, ...)
#endif

#define logElfError(msg)                                                                           \
  ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "%-5d: [%zx] %p %s: " msg, getpid(),                      \
          std::this_thread::get_id(), this, __func__)

#define LogElfError(format, ...)                                                                   \
  ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "%-5d: [%zx] %p %s: " format, getpid(),                   \
          std::this_thread::get_id(), this, __func__, __VA_ARGS__)

namespace amd {
using namespace amd::ELFIO;

#if !defined(ELFMAG)
#define ELFMAG "\177ELF"
#define SELFMAG 4
#endif

typedef struct {
  Elf::ElfSections id;
  const char* name;
  uint64_t d_align;     // section alignment in bytes
  Elf32_Word sh_type;   // section type
  Elf32_Word sh_flags;  // section flags
  const char* desc;
} ElfSectionsDesc;

namespace {
// Objects that are visible only within this module
constexpr ElfSectionsDesc ElfSecDesc[] = {
    {Elf::LLVMIR, ".llvmir", 1, SHT_PROGBITS, 0, "ASIC-independent LLVM IR"},
    {Elf::SOURCE, ".source", 1, SHT_PROGBITS, 0, "OpenCL source"},
    {Elf::ILTEXT, ".amdil", 1, SHT_PROGBITS, 0, "AMD IL text"},
    {Elf::ASTEXT, ".astext", 1, SHT_PROGBITS, 0, "X86 assembly text"},
    {Elf::CAL, ".text", 1, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, "AMD CalImage"},
    {Elf::DLL, ".text", 1, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, "x86 dll"},
    {Elf::STRTAB, ".strtab", 1, SHT_STRTAB, SHF_STRINGS, "String table"},
    {Elf::SYMTAB, ".symtab", sizeof(Elf64_Xword), SHT_SYMTAB, 0, "Symbol table"},
    {Elf::RODATA, ".rodata", 1, SHT_PROGBITS, SHF_ALLOC, "Read-only data"},
    {Elf::SHSTRTAB, ".shstrtab", 1, SHT_STRTAB, SHF_STRINGS, "Section names"},
    {Elf::NOTES, ".note", 1, SHT_NOTE, 0, "used by loader for notes"},
    {Elf::COMMENT, ".comment", 1, SHT_PROGBITS, 0, "Version string"},
    {Elf::ILDEBUG, ".debugil", 1, SHT_PROGBITS, 0, "AMD Debug IL"},
    {Elf::DEBUG_INFO, ".debug_info", 1, SHT_PROGBITS, 0, "Dwarf debug info"},
    {Elf::DEBUG_ABBREV, ".debug_abbrev", 1, SHT_PROGBITS, 0, "Dwarf debug abbrev"},
    {Elf::DEBUG_LINE, ".debug_line", 1, SHT_PROGBITS, 0, "Dwarf debug line"},
    {Elf::DEBUG_PUBNAMES, ".debug_pubnames", 1, SHT_PROGBITS, 0, "Dwarf debug pubnames"},
    {Elf::DEBUG_PUBTYPES, ".debug_pubtypes", 1, SHT_PROGBITS, 0, "Dwarf debug pubtypes"},
    {Elf::DEBUG_LOC, ".debug_loc", 1, SHT_PROGBITS, 0, "Dwarf debug loc"},
    {Elf::DEBUG_ARANGES, ".debug_aranges", 1, SHT_PROGBITS, 0, "Dwarf debug aranges"},
    {Elf::DEBUG_RANGES, ".debug_ranges", 1, SHT_PROGBITS, 0, "Dwarf debug ranges"},
    {Elf::DEBUG_MACINFO, ".debug_macinfo", 1, SHT_PROGBITS, 0, "Dwarf debug macinfo"},
    {Elf::DEBUG_STR, ".debug_str", 1, SHT_PROGBITS, 0, "Dwarf debug str"},
    {Elf::DEBUG_FRAME, ".debug_frame", 1, SHT_PROGBITS, 0, "Dwarf debug frame"},
    {Elf::JITBINARY, ".text", 1, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, "x86 JIT Binary"},
    {Elf::CODEGEN, ".cg", 1, SHT_PROGBITS, 0, "Target dependent IL"},
    {Elf::TEXT, ".text", 1, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, "Device specific ISA"},
    {Elf::INTERNAL, ".internal", 1, SHT_PROGBITS, 0, "Internal usage"},
    {Elf::SPIR, ".spir", 1, SHT_PROGBITS, 0, "Vendor/Device-independent LLVM IR"},
    {Elf::SPIRV, ".spirv", 1, SHT_PROGBITS, 0, "SPIR-V Binary"},
    {Elf::RUNTIME_METADATA, ".AMDGPU.runtime_metadata", 1, SHT_PROGBITS, 0,
     "AMDGPU runtime metadata"},
};
}  // namespace

///////////////////////////////////////////////////////////////
////////////////////// elf initializers ///////////////////////
///////////////////////////////////////////////////////////////

Elf::Elf(unsigned char eclass, const char* rawElfBytes, uint64_t rawElfSize,
         const char* elfFileName, ElfCmd elfcmd)
    : _fname(elfFileName ? elfFileName : ""),
      _eclass(eclass),
      _rawElfBytes(rawElfBytes),
      _rawElfSize(rawElfSize),
      _elfCmd(elfcmd),
      _elfMemory(),
      _shstrtab_ndx(SHN_UNDEF),
      _strtab_ndx(SHN_UNDEF),
      _symtab_ndx(SHN_UNDEF),
      _successful(false) {
  LogElfInfo("fname=%s, rawElfSize=%lu, elfcmd=%d, %s", _fname.c_str(), _rawElfSize, _elfCmd,
             _elfCmd == ELF_C_WRITE ? "writer" : "reader");

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

Elf::~Elf() {
  LogElfInfo("fname=%s, rawElfSize=%lu, elfcmd=%d", _fname.c_str(), _rawElfSize, _elfCmd);
  elfMemoryRelease();
}

bool Elf::Clear() {
  ElfTrace(amd::LOG_INFO);

  _elfio.clean();
  elfMemoryRelease();

  // Re-initialize the object
  Init();

  return isSuccessful();
}

/*
 Initialize Elf object
 */
bool Elf::Init() {
  _successful = false;

  switch (_elfCmd) {
    case ELF_C_WRITE:
      _elfio.create(_eclass, ELFDATA2LSB);
      break;

    case ELF_C_READ:
      if (_rawElfBytes == nullptr || _rawElfSize == 0) {
        logElfError("failed: _rawElfBytes = nullptr or _rawElfSize = 0");
        return false;
      }
      {
        std::istringstream is{std::string(_rawElfBytes, _rawElfSize)};
        if (!_elfio.load(is)) {
          LogElfError("failed in _elfio.load(%p, %lu)", _rawElfBytes, _rawElfSize);
          return false;
        }
      }
      break;

    default:
      LogElfError("failed: unexpected cmd %d", _elfCmd);
      return false;  // Don't support other mode
  }

  if (!InitElf()) {
    return false;
  }

  // Success
  _successful = true;

  return true;
}

bool Elf::InitElf() {
  if (_elfCmd == ELF_C_READ) {
    assert(_elfio.sections.size() > 0 && "elfio object should have been created already");

    // Set up _shstrtab_ndx
    _shstrtab_ndx = _elfio.get_section_name_str_index();
    if (_shstrtab_ndx == SHN_UNDEF) {
      logElfError("failed: _shstrtab_ndx = SHN_UNDEF");
      return false;
    }

    // Set up _strtab_ndx
    section* strtab_sec = _elfio.sections[ElfSecDesc[STRTAB].name];
    if (strtab_sec == nullptr) {
      logElfError("failed: null sections(STRTAB)");
      return false;
    }

    _strtab_ndx = strtab_sec->get_index();

    section* symtab_sec = _elfio.sections[ElfSecDesc[SYMTAB].name];

    if (symtab_sec != nullptr) {
      _symtab_ndx = symtab_sec->get_index();
    }
    // It's ok for empty SYMTAB
  } else if (_elfCmd == ELF_C_WRITE) {
    /*********************************/
    /******** ELF_C_WRITE ************/
    /*********************************/

    //
    // 1. Create ELF header
    //
    _elfio.create(_eclass, ELFDATA2LSB);

    //
    // 2. Check created ELF shstrtab
    //
    section* shstrtab_sec = _elfio.sections[ElfSecDesc[SHSTRTAB].name];
    if (shstrtab_sec == nullptr) {
      logElfError("failed: shstrtab_sec = nullptr");
      return false;
    }

    if (!setupShdr(SHSTRTAB, shstrtab_sec)) {
      return false;
    }

    // Save shstrtab section index
    _shstrtab_ndx = shstrtab_sec->get_index();

    //
    // 3. Create .strtab section
    //
    auto* strtab_sec = _elfio.sections.add(ElfSecDesc[STRTAB].name);
    if (strtab_sec == nullptr) {
      logElfError("failed to add section STRTAB");
      return false;
    }

    // adding null string data associated with section
    // index 0 is reserved and must be there (NULL name)
    constexpr char strtab[] = {/* index 0 */ '\0'};
    strtab_sec->set_data(const_cast<char*>(strtab), sizeof(strtab));

    if (!setupShdr(STRTAB, strtab_sec)) {
      return false;
    }

    // Save strtab section index
    _strtab_ndx = strtab_sec->get_index();

    //
    // 4. Create the symbol table
    //

    // Create the first reserved dummy symbol (undefined symbol)
    size_t sym_sz = (_eclass == ELFCLASS32) ? sizeof(Elf32_Sym) : sizeof(Elf64_Sym);
    char* sym = static_cast<char*>(::calloc(1, sym_sz));
    if (sym == nullptr) {
      logElfError("failed to calloc memory for SYMTAB section");
      return false;
    }

    auto* symtab_sec = newSection(SYMTAB, sym, sym_sz);
    free(sym);

    if (symtab_sec == nullptr) {
      logElfError("failed to create SYMTAB");
      return false;
    }

    _symtab_ndx = symtab_sec->get_index();
  } else {
    LogElfError("failed: wrong cmd %d", _elfCmd);
    return false;
  }

  LogElfInfo("succeeded: secs=%d, segs=%d, _shstrtab_ndx=%u, _strtab_ndx=%u, _symtab_ndx=%u",
             _elfio.sections.size(), _elfio.segments.size(), _shstrtab_ndx, _strtab_ndx,
             _symtab_ndx);
  return true;
}

bool Elf::createElfData(section*& sec, ElfSections id, const char* d_buf, size_t d_size) {
  assert((ElfSecDesc[id].id == id) &&
         "ElfSecDesc[] should be in the same order as enum ElfSections");

  sec = _elfio.sections[ElfSecDesc[id].name];
  if (sec == nullptr) {
    LogElfError("failed: null sections(%s)", ElfSecDesc[id].name);
    return false;
  }

  sec->set_data(d_buf, d_size);
  return true;
}

bool Elf::setupShdr(ElfSections id, section* section, Elf64_Word shlink) const {
  section->set_addr_align(ElfSecDesc[id].d_align);
  section->set_type(ElfSecDesc[id].sh_type);
  section->set_flags(ElfSecDesc[id].sh_flags);
  section->set_link(shlink);

  auto class_num = _elfio.get_class();
  size_t entry_size = 0;
  switch (id) {
    case SYMTAB:
      if (class_num == ELFCLASS32) {
        entry_size = sizeof(Elf32_Sym);
      } else {
        entry_size = sizeof(Elf64_Sym);
      }
      break;
    default:
      // .dynsym and .relaNAME also have table entries
      break;
  }
  if (entry_size > 0) {
    section->set_entry_size(entry_size);
  }
  return true;
}

bool Elf::getTarget(uint16_t& machine, ElfPlatform& platform) const {
  Elf64_Half mach = _elfio.get_machine();
  if ((mach >= CPU_FIRST) && (mach <= CPU_LAST)) {
    platform = CPU_PLATFORM;
    machine = mach - CPU_BASE;
  } else if (mach == EM_386 || mach == EM_HSAIL || mach == EM_HSAIL_64 || mach == EM_AMDIL ||
             mach == EM_AMDIL_64 || mach == EM_X86_64) {
    platform = COMPLIB_PLATFORM;
    machine = mach;
  } else {
    // Invalid machine
    LogElfError("failed: Invalid machine=0x%04x(%d)", mach, mach);
    return false;
  }
  LogElfInfo("succeeded: machine=0x%04x, platform=%d", machine, platform);
  return true;
}

bool Elf::setTarget(uint16_t machine, ElfPlatform platform) {
  Elf64_Half mach;
  if (platform == CPU_PLATFORM)
    mach = machine + CPU_BASE;
  else if (platform == CAL_PLATFORM)
    mach = machine + CAL_BASE;
  else
    mach = machine;

  _elfio.set_machine(mach);
  LogElfInfo("succeeded: machine=0x%04x(%d), platform=%d", machine, machine, platform);

  return true;
}

bool Elf::getType(uint16_t& type) const {
  type = _elfio.get_type();
  return true;
}

bool Elf::setType(uint16_t type) {
  _elfio.set_type(type);
  return true;
}

bool Elf::getFlags(uint32_t& flag) const {
  flag = _elfio.get_flags();
  return true;
}

bool Elf::setFlags(uint32_t flag) {
  _elfio.set_flags(flag);
  return true;
}

bool Elf::getSection(Elf::ElfSections id, char** dst, size_t* sz) const {
  assert((ElfSecDesc[id].id == id) &&
         "ElfSecDesc[] should be in the same order as enum ElfSections");

  section* sec = _elfio.sections[ElfSecDesc[id].name];
  if (sec == nullptr) {
    LogElfError("failed: null sections(%s)", ElfSecDesc[id].name);
    return false;
  }

  // There is only one data descriptor (we are reading!)
  *dst = const_cast<char*>(sec->get_data());
  *sz = sec->get_size();

  LogElfInfo("succeeded: *dst=%p, *sz=%zu", *dst, *sz);
  return true;
}

unsigned int Elf::getSymbolNum() const {
  if (_symtab_ndx == SHN_UNDEF) {
    logElfError(" failed: _symtab_ndx = SHN_UNDEF");
    return 0;  // No SYMTAB
  }
  symbol_section_accessor symbol_reader(_elfio, _elfio.sections[_symtab_ndx]);
  auto num = symbol_reader.get_symbols_num() - 1;  // Exclude the first dummy symbol
  LogElfInfo(": num=%lu", num);
  return num;
}

unsigned int Elf::getSegmentNum() const { return _elfio.segments.size(); }

bool Elf::getSegment(const unsigned int index, segment*& seg) const {
  bool ret = false;
  if (index < _elfio.segments.size()) {
    seg = _elfio.segments[index];
    ret = true;
  }
  return ret;
}

bool Elf::getSymbolInfo(unsigned int index, SymbolInfo* symInfo) const {
  if (_symtab_ndx == SHN_UNDEF) {
    logElfError(" failed: _symtab_ndx = SHN_UNDEF");
    return false;  // No SYMTAB
  }
  symbol_section_accessor symbol_reader(_elfio, _elfio.sections[_symtab_ndx]);

  auto num = getSymbolNum();

  if (index >= num) {
    LogElfError(" failed: wrong index %u >= symbols num %lu", index, num);
    return false;
  }

  std::string sym_name;
  Elf64_Addr value = 0;
  Elf_Xword size = 0;
  unsigned char bind = 0;
  unsigned char type = 0;
  Elf_Half sec_index = 0;
  unsigned char other = 0;

  // index++ for real index on top of the first dummy symbol
  bool ret = symbol_reader.get_symbol(++index, sym_name, value, size, bind, type, sec_index, other);
  if (!ret) {
    LogElfError("failed to get_symbol(%u)", index);
    return false;
  }
  section* sec = _elfio.sections[sec_index];
  if (sec == nullptr) {
    LogElfError("failed: null section at %u", sec_index);
    return false;
  }

  symInfo->sec_addr = sec->get_data();
  symInfo->sec_size = sec->get_size();
  symInfo->address = symInfo->sec_addr + (size_t)value;
  symInfo->size = (uint64_t)size;

  symInfo->sec_name = sec->get_name();
  symInfo->sym_name = sym_name;
#if 0
  // For debug purpose
  LogElfDebug("succeeded at index=%u: sec_addr=%p, sec_size=%lu, address=%p,"
              " size=%lu, sec_name=%s, sym_name=%s",
              index, symInfo->sec_addr, symInfo->sec_size, symInfo->address, symInfo->size,
              symInfo->sec_name.c_str(), symInfo->sym_name.c_str());
#endif
  return true;
}

bool Elf::addSectionData(Elf_Xword& outOffset, ElfSections id, const void* buffer, size_t size) {
  assert(ElfSecDesc[id].id == id &&
         "struct ElfSecDesc should be ordered by id same as enum Elf::ElfSections");

  outOffset = 0;
  section* sec = _elfio.sections[ElfSecDesc[id].name];
  if (sec == nullptr) {
    LogElfError("failed: null sections(%s)", ElfSecDesc[id].name);
    return false;
  }

  outOffset = sec->get_size();

  sec->append_data(static_cast<const char*>(buffer), size);
  LogElfInfo("succeeded: buffer=%p, size=%zu", buffer, size);

  return true;
}

bool Elf::getShstrtabNdx(Elf64_Word& outNdx, const char* name) {
  outNdx = 0;
  auto* section = _elfio.sections[name];
  if (section == nullptr) {
    LogElfError("failed: sections[%s] = nullptr", name);
    return false;
  }

  // .shstrtab must be created already
  auto idx = section->get_name_string_offset();

  if (idx <= 0) {
    LogElfError("failed: idx=%d", idx);
    return false;
  }
  outNdx = idx;
  LogElfDebug("Succeeded: name=%s, idx=%d", name, idx);
  return true;
}

section* Elf::newSection(Elf::ElfSections id, const char* d_buf, size_t d_size) {
  assert(ElfSecDesc[id].id == id &&
         "struct ElfSecDesc should be ordered by id same as enum Elf::ElfSections");

  section* sec = _elfio.sections[ElfSecDesc[id].name];
  if (sec == nullptr) {
    sec = _elfio.sections.add(ElfSecDesc[id].name);
  }
  if (sec == nullptr) {
    LogElfError("failed: sections.add(%s) = nullptr", ElfSecDesc[id].name);
    return sec;
  }

  if (d_buf != nullptr && d_size > 0) {
    sec->set_data(d_buf, d_size);
  }

  if (!setupShdr(id, sec, (id == SYMTAB) ? _strtab_ndx : 0)) {
    return nullptr;
  }

  LogElfDebug("succeeded: name=%s, d_buf=%p, d_size=%zu", ElfSecDesc[id].name, d_buf, d_size);
  return sec;
}

bool Elf::addSection(ElfSections id, const void* d_buf, size_t d_size) {
  assert(ElfSecDesc[id].id == id &&
         "struct ElfSecDesc should be ordered by id same as enum Elf::ElfSections");

  section* sec = _elfio.sections[ElfSecDesc[id].name];

  if (sec != nullptr) {
    Elf_Xword sec_offset = 0;
    if (!addSectionData(sec_offset, id, d_buf, d_size)) {
      LogElfError("failed in addSectionData(name=%s, d_buf=%p, d_size=%zu)", ElfSecDesc[id].name,
                  d_buf, d_size);
      return false;
    }
  } else {
    sec = newSection(id, static_cast<const char*>(d_buf), d_size);
    if (sec == nullptr) {
      LogElfError("failed in newSection(name=%s, d_buf=%p, d_size=%zu)", ElfSecDesc[id].name, d_buf,
                  d_size);
      return false;
    }
  }

  LogElfDebug("succeeded: name=%s, d_buf=%p, d_size=%zu", ElfSecDesc[id].name, d_buf, d_size);
  return true;
}

bool Elf::addSymbol(ElfSections id, const char* symbolName, const void* buffer, size_t size) {
  assert(ElfSecDesc[id].id == id && "The order of ElfSecDesc[] and Elf::ElfSections mismatches.");

  if (_symtab_ndx == SHN_UNDEF) {
    logElfError("failed: _symtab_ndx = SHN_UNDEF");
    return false;  // No SYMTAB
  }

  const char* sectionName = ElfSecDesc[id].name;

  bool isFunction = ((id == Elf::CAL) || (id == Elf::DLL) || (id == Elf::JITBINARY)) ? true : false;

  // Get section index
  section* sec = _elfio.sections[sectionName];
  if (sec == nullptr) {
    // Create a new section.
    if ((sec = newSection(id, nullptr, 0)) == NULL) {
      LogElfError("failed in newSection(name=%s)", sectionName);
      return false;
    }
  }
  size_t sec_ndx = sec->get_index();
  if (sec_ndx == SHN_UNDEF) {
    logElfError("failed: sec->get_index() = SHN_UNDEF");
    return false;
  }

  // Put symbolName into .strtab section
  Elf_Xword strtab_offset = 0;
  if (!addSectionData(strtab_offset, STRTAB, symbolName, strlen(symbolName) + 1)) {
    LogElfError("failed in addSectionData(name=%s, symbolName=%s, length=%zu)",
                ElfSecDesc[STRTAB].name, symbolName, strlen(symbolName) + 1);
    return false;
  }

  // Put buffer into section
  Elf_Xword sec_offset = 0;
  if ((buffer != nullptr) && (size != 0)) {
    if (!addSectionData(sec_offset, id, buffer, size)) {
      LogElfError("failed in addSectionData(name=%s, buffer=%p, size=%zu)", sectionName, buffer,
                  size);
      return false;
    }
  }

  symbol_section_accessor symbol_writter(_elfio, _elfio.sections[_symtab_ndx]);

  auto ret = symbol_writter.add_symbol(strtab_offset, sec_offset, size, 0,
                                       (isFunction) ? STT_FUNC : STT_OBJECT, 0, sec_ndx);

  LogElfDebug(
      "%s: sectionName=%s symbolName=%s strtab_offset=%lu, sec_offset=%lu, "
      "size=%zu, sec_ndx=%zu, ret=%d",
      ret >= 1 ? "succeeded" : "failed", sectionName, symbolName, strtab_offset, sec_offset, size,
      sec_ndx, ret);
  return ret >= 1;
}

bool Elf::getSymbol(ElfSections id, const char* symbolName, char** buffer, size_t* size) const {
  assert(ElfSecDesc[id].id == id && "The order of ElfSecDesc[] and Elf::ElfSections mismatches.");

  if (!size || !buffer || !symbolName) {
    logElfError("failed: invalid parameters");
    return false;
  }
  if (_symtab_ndx == SHN_UNDEF) {
    logElfError("failed: _symtab_ndx = SHN_UNDEF");
    return false;  // No SYMTAB
  }

  *size = 0;
  *buffer = nullptr;
  symbol_section_accessor symbol_reader(_elfio, _elfio.sections[_symtab_ndx]);

  Elf64_Addr value = 0;
  Elf_Xword size0 = 0;
  unsigned char bind = 0;
  unsigned char type = 0;
  unsigned char other = 0;
  Elf_Half sec_ndx = SHN_UNDEF;

  // Search by symbolName, sectionName
  bool ret = symbol_reader.get_symbol(symbolName, ElfSecDesc[id].name, value, size0, bind, type,
                                      sec_ndx, other);

  if (ret) {
    *buffer = const_cast<char*>(_elfio.sections[sec_ndx]->get_data() + value);
    *size = static_cast<size_t>(size0);
  }
#if 0
  // For debug purpose
  LogElfDebug("%s: sectionName=%s symbolName=%s value=%lu, buffer=%p, size=%zu, sec_ndx=%u",
              ret ? "succeeded" : "failed",
              ElfSecDesc[id].name, symbolName, value, *buffer, *size, sec_ndx);
#endif
  return ret;
}

bool Elf::addNote(const char* noteName, const char* noteDesc, size_t descSize) {
  if (descSize == 0 || noteName == nullptr || (descSize != 0 && noteDesc == nullptr)) {
    logElfError("failed: empty note");
    return false;
  }

  // Get section
  section* sec = _elfio.sections[ElfSecDesc[NOTES].name];
  if (sec == nullptr) {
    // Create a new section.
    if ((sec = newSection(NOTES, nullptr, 0)) == nullptr) {
      logElfError("failed in newSection(NOTES)");
      return false;
    }
  }

  note_section_accessor note_writer(_elfio, sec);
  // noteName is null terminated
  note_writer.add_note(0, noteName, noteDesc, descSize);

  LogElfDebug("Succeed: add_note(%s, %s)", noteName, std::string(noteDesc, descSize).c_str());

  return true;
}

bool Elf::getNote(const char* noteName, char** noteDesc, size_t* descSize) {
  if (!descSize || !noteDesc || !noteName) {
    logElfError("failed: empty note");
    return false;
  }

  // Get section
  section* sec = _elfio.sections[ElfSecDesc[NOTES].name];
  if (sec == nullptr) {
    logElfError("failed: null sections(NOTES)");
    return false;
  }

  // Initialize the size and buffer to invalid data points.
  *descSize = 0;
  *noteDesc = nullptr;

  note_section_accessor note_reader(_elfio, sec);

  auto num = note_reader.get_notes_num();
  Elf_Word type = 0;
  void* desc = nullptr;
  Elf_Word descSize1 = 0;

  for (unsigned int i = 0; i < num; i++) {
    std::string name;
    if (note_reader.get_note(i, type, name, desc, descSize1)) {
      if (name == noteName) {
        *noteDesc = static_cast<char*>(desc);
        *descSize = descSize1;
        LogElfDebug("Succeed: get_note(%s, %s)", name.c_str(),
                    std::string(*noteDesc, *descSize).c_str());
        return true;
      }
    }
  }

  return false;
}

std::string Elf::generateUUIDV4() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(0, 15);
  static std::uniform_int_distribution<> dis2(8, 11);
  std::stringstream ss;
  int i;
  ss << std::hex;
  for (i = 0; i < 8; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 4; i++) {
    ss << dis(gen);
  }
  ss << "-4";
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  ss << dis2(gen);
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 12; i++) {
    ss << dis(gen);
  };
  return ss.str();
}

bool Elf::dumpImage(char** buff, size_t* len) {
  bool ret = false;
  std::string dumpFile = _fname;
  if (_fname.empty()) {
    dumpFile = generateUUIDV4();
    dumpFile += ".bin";
    LogElfInfo("Generated temporary dump file: %s", dumpFile.c_str());
  }

  if (!_elfio.save(dumpFile)) {
    LogElfError("failed in _elfio.save(%s)", dumpFile.c_str());
    return false;
  }

  if (buff != nullptr && len != nullptr) {
    std::ifstream is;
    is.open(dumpFile, std::ifstream::in | std::ifstream::binary);  // open input file
    if (!is.good()) {
      LogElfError("failed in is.open(%s)", dumpFile.c_str());
      return false;
    }
    ret = dumpImage(is, buff, len);
    is.close();  // close file
  }

  if (_fname.empty()) {
    std::remove(dumpFile.c_str());
  }
  LogElfInfo("%s: buff=%p, len=%zu\n", ret ? "Succeed" : "failed", *buff, *len);
  return ret;
}

bool Elf::dumpImage(std::istream& is, char** buff, size_t* len) const {
  if (buff == nullptr || len == nullptr) {
    return false;
  }
  is.seekg(0, std::ios::end);  // go to the end
  *len = is.tellg();           // report location (this is the length)
  is.seekg(0, std::ios::beg);  // go back to the beginning
  *buff = new char[*len];      // allocate memory which should be deleted by caller
  is.read(*buff, *len);        // read the whole file into the buffer
  return true;
}

uint64_t Elf::getElfSize(const void* emi) {
  const unsigned char eclass = static_cast<const unsigned char*>(emi)[EI_CLASS];
  uint64_t total_size = 0;
  if (eclass == ELFCLASS32) {
    auto ehdr = static_cast<const Elf32_Ehdr*>(emi);
    auto shdr = reinterpret_cast<const Elf32_Shdr*>(static_cast<const char*>(emi) + ehdr->e_shoff);

    auto max_offset = ehdr->e_shoff;
    total_size = max_offset + ehdr->e_shentsize * ehdr->e_shnum;

    for (decltype(ehdr->e_shnum) i = 0; i < ehdr->e_shnum; ++i) {
      auto cur_offset = shdr[i].sh_offset;
      if (max_offset < cur_offset) {
        max_offset = cur_offset;
        total_size = max_offset;
        if (SHT_NOBITS != shdr[i].sh_type) {
          total_size += shdr[i].sh_size;
        }
      }
    }
  } else if (eclass == ELFCLASS64) {
    auto ehdr = static_cast<const Elf64_Ehdr*>(emi);
    auto shdr = reinterpret_cast<const Elf64_Shdr*>(static_cast<const char*>(emi) + ehdr->e_shoff);

    auto max_offset = ehdr->e_shoff;
    total_size = max_offset + ehdr->e_shentsize * ehdr->e_shnum;

    for (decltype(ehdr->e_shnum) i = 0; i < ehdr->e_shnum; ++i) {
      auto cur_offset = shdr[i].sh_offset;
      if (max_offset < cur_offset) {
        max_offset = cur_offset;
        total_size = max_offset;
        if (SHT_NOBITS != shdr[i].sh_type) {
          total_size += shdr[i].sh_size;
        }
      }
    }
  }
  return total_size;
}

bool Elf::isElfMagic(const char* p) {
  if (p == nullptr || strncmp(p, ELFMAG, SELFMAG) != 0) {
    return false;
  }
  return true;
}

bool Elf::isCALTarget(const char* p, signed char ec) {
  if (!isElfMagic(p)) {
    return false;
  }

  return false;
}

void* Elf::xmalloc(const size_t len) {
  void* retval = ::calloc(1, len);
  if (retval == nullptr) {
    logElfError("failed: out of memory");
    return nullptr;
  }
  return retval;
}

void* Elf::allocAndCopy(void* p, size_t sz) {
  if (p == 0 || sz == 0) return p;

  void* buf = xmalloc(sz);
  if (buf == nullptr) {
    logElfError("failed: out of memory");
    return 0;
  }

  memcpy(buf, p, sz);
  _elfMemory.insert(std::make_pair(buf, sz));
  return buf;
}

void* Elf::calloc(size_t sz) {
  void* buf = xmalloc(sz);
  if (buf == nullptr) {
    logElfError("failed: out of memory");
    return 0;
  }
  _elfMemory.insert(std::make_pair(buf, sz));
  return buf;
}

void Elf::elfMemoryRelease() {
  for (EMemory::iterator it = _elfMemory.begin(); it != _elfMemory.end(); ++it) {
    free(it->first);
  }
  _elfMemory.clear();
}

}  // namespace amd
