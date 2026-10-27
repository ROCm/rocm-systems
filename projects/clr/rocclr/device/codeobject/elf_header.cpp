/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/codeobject/elf_header.hpp"

#include "elf/elfio/elf_types.hpp"

#include <cstring>
#include <string_view>

namespace amd::codeobject {

using amd::ELFIO::Elf64_Ehdr;
using amd::ELFIO::Elf64_Phdr;
using amd::ELFIO::Elf64_Shdr;
using amd::ELFIO::Elf64_Sym;

namespace {

// AMDGPU ELF header values, from ROCr's amd_hsa_elf.h (also LLVM's BinaryFormat/ELF.h). Kept
// here because the copies of that header in the PAL build don't have the newer processors.
constexpr uint16_t kEmAmdgpu = 224;
constexpr uint32_t kOsAbiAmdgpuHsa = 64;
constexpr uint32_t kAbiVersionV3 = 1;
constexpr uint32_t kAbiVersionV4 = 2;
constexpr uint32_t kAbiVersionV5 = 3;
constexpr uint32_t kAbiVersionV6 = 4;

constexpr uint32_t kMachMask = 0x0ff;
constexpr uint32_t kXnackV3 = 0x100;
constexpr uint32_t kSrameccV3 = 0x200;
constexpr uint32_t kXnackV4Mask = 0x300;
constexpr uint32_t kXnackOffV4 = 0x200;
constexpr uint32_t kXnackOnV4 = 0x300;
constexpr uint32_t kSrameccV4Mask = 0xc00;
constexpr uint32_t kSrameccOffV4 = 0x800;
constexpr uint32_t kSrameccOnV4 = 0xc00;
constexpr uint32_t kGenericVersionMask = 0xff000000;
constexpr uint32_t kGenericVersionOffset = 24;

//! One processor of the loader's mach table
struct MachInfo {
  uint32_t mach;          //!< EF_AMDGPU_MACH_AMDGCN_* value of e_flags
  const char* name;       //!< Processor name
  bool xnackSupported;    //!< The processor has the xnack feature
  bool srameccSupported;  //!< The processor has the sramecc feature
};

// ROCr's GetMachInfo (libamdhsacode/amd_hsa_code.cpp): the processors its loader accepts
constexpr MachInfo kMachTable[] = {
    {0x020, "gfx600", false, false},
    {0x021, "gfx601", false, false},
    {0x03a, "gfx602", false, false},
    {0x022, "gfx700", false, false},
    {0x023, "gfx701", false, false},
    {0x024, "gfx702", false, false},
    {0x025, "gfx703", false, false},
    {0x026, "gfx704", false, false},
    {0x03b, "gfx705", false, false},
    {0x028, "gfx801", true, false},
    {0x029, "gfx802", false, false},
    {0x02a, "gfx803", false, false},
    {0x03c, "gfx805", false, false},
    {0x02b, "gfx810", true, false},
    {0x02c, "gfx900", true, false},
    {0x02d, "gfx902", true, false},
    {0x02e, "gfx904", true, false},
    {0x02f, "gfx906", true, true},
    {0x030, "gfx908", true, true},
    {0x031, "gfx909", true, false},
    {0x03f, "gfx90a", true, true},
    {0x032, "gfx90c", true, false},
    {0x04c, "gfx942", true, true},
    {0x04f, "gfx950", true, true},
    {0x033, "gfx1010", true, false},
    {0x034, "gfx1011", true, false},
    {0x035, "gfx1012", true, false},
    {0x042, "gfx1013", true, false},
    {0x036, "gfx1030", false, false},
    {0x037, "gfx1031", false, false},
    {0x038, "gfx1032", false, false},
    {0x039, "gfx1033", false, false},
    {0x03e, "gfx1034", false, false},
    {0x03d, "gfx1035", false, false},
    {0x045, "gfx1036", false, false},
    {0x041, "gfx1100", false, false},
    {0x046, "gfx1101", false, false},
    {0x047, "gfx1102", false, false},
    {0x044, "gfx1103", false, false},
    {0x043, "gfx1150", false, false},
    {0x04a, "gfx1151", false, false},
    {0x055, "gfx1152", false, false},
    {0x058, "gfx1153", false, false},
    {0x05d, "gfx1170", false, false},
    {0x05e, "gfx1171", false, false},
    {0x048, "gfx1200", false, false},
    {0x04e, "gfx1201", false, false},
    {0x049, "gfx1250", true, true},
    {0x0eb, "gfx1250-strict", true, true},
    {0x051, "gfx9-generic", true, false},
    {0x05f, "gfx9-4-generic", true, true},
    {0x052, "gfx10-1-generic", true, false},
    {0x053, "gfx10-3-generic", false, false},
    {0x054, "gfx11-generic", false, false},
    {0x062, "gfx11-7-generic", false, false},
    {0x059, "gfx12-generic", false, false},
    {0x05b, "gfx12-5-generic", false, false},
};

const MachInfo* FindMach(uint32_t mach) {
  for (const MachInfo& info : kMachTable) {
    if (info.mach == mach) return &info;
  }
  return nullptr;
}

//! ELF note header, the same for 32- and 64-bit ELF
struct NoteHeader {
  uint32_t nameSize;  //!< n_namesz
  uint32_t descSize;  //!< n_descsz
  uint32_t type;      //!< n_type
};

constexpr uint64_t AlignNote(uint64_t size) { return (size + 3) & ~uint64_t(3); }

//! @returns The section headers of @p image once the image is validated against @p size.
const Elf64_Shdr* SectionHeaders(const void* image, size_t size) {
  if (size == kUnknownSize || ElfImageSize(image, size) == 0) {
    return nullptr;
  }
  const auto* ehdr = static_cast<const Elf64_Ehdr*>(image);
  return reinterpret_cast<const Elf64_Shdr*>(static_cast<const char*>(image) + ehdr->e_shoff);
}

}  // namespace

// ================================================================================================
bool IsElf(const void* image) {
  return image != nullptr && std::memcmp(image, "\177ELF", 4) == 0;
}

// ================================================================================================
uint64_t ElfImageSize(const void* image, size_t bound) {
  if (image == nullptr || bound <= EI_CLASS) {
    return 0;
  }
  const bool sizeKnown = (bound != kUnknownSize);
  const unsigned char eclass = static_cast<const unsigned char*>(image)[EI_CLASS];
  // Only 64-bit AMDGPU code objects are supported.
  if (eclass != ELFCLASS64 || bound < sizeof(Elf64_Ehdr)) {
    return 0;
  }

  const uint64_t addrSpaceLimit = ~uint64_t(0) - reinterpret_cast<uintptr_t>(image);
  const uint64_t limit = sizeKnown ? bound : addrSpaceLimit;

  // end = off + sz, returns false on overflow.
  auto checkedEnd = [](uint64_t off, uint64_t sz, uint64_t& end) -> bool {
    if (off > ~uint64_t(0) - sz) return false;
    end = off + sz;
    return true;
  };

  auto ehdr = static_cast<const Elf64_Ehdr*>(image);
  uint64_t totalSize = sizeof(Elf64_Ehdr);

  // Validate and bound the section-header table before any dereference.
  if (ehdr->e_shnum != 0) {
    if (ehdr->e_shentsize != sizeof(Elf64_Shdr) || ehdr->e_shoff >= limit) {
      return 0;
    }
    const uint64_t shtab = static_cast<uint64_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr);
    uint64_t shtabEnd = 0;
    if (!checkedEnd(ehdr->e_shoff, shtab, shtabEnd) || shtabEnd > limit) {
      return 0;
    }
    if (shtabEnd > totalSize) totalSize = shtabEnd;

    auto shdr =
        reinterpret_cast<const Elf64_Shdr*>(static_cast<const char*>(image) + ehdr->e_shoff);
    for (uint64_t i = 0; i < ehdr->e_shnum; ++i) {
      if (SHT_NOBITS == shdr[i].sh_type) continue;
      uint64_t end = 0;
      if (!checkedEnd(shdr[i].sh_offset, shdr[i].sh_size, end)) return 0;
      if (end > limit) return 0;
      if (end > totalSize) totalSize = end;
    }
  }

  // Validate and bound the program-header table before any dereference.
  if (ehdr->e_phnum != 0) {
    if (ehdr->e_phentsize != sizeof(Elf64_Phdr) || ehdr->e_phoff >= limit) {
      return 0;
    }
    const uint64_t phtab = static_cast<uint64_t>(ehdr->e_phnum) * sizeof(Elf64_Phdr);
    uint64_t phtabEnd = 0;
    if (!checkedEnd(ehdr->e_phoff, phtab, phtabEnd) || phtabEnd > limit) {
      return 0;
    }
    if (phtabEnd > totalSize) totalSize = phtabEnd;

    auto phdr =
        reinterpret_cast<const Elf64_Phdr*>(static_cast<const char*>(image) + ehdr->e_phoff);
    for (uint64_t i = 0; i < ehdr->e_phnum; ++i) {
      if (PT_NULL == phdr[i].p_type) continue;
      uint64_t end = 0;
      if (!checkedEnd(phdr[i].p_offset, phdr[i].p_filesz, end)) return 0;
      if (end > limit) return 0;
      if (end > totalSize) totalSize = end;
    }
  }

  return (totalSize > limit) ? 0 : totalSize;
}

// ================================================================================================
bool GetIsaFromHeader(uint32_t elfClass, uint16_t machine, uint32_t osAbi, uint32_t abiVersion,
                      uint32_t eFlags, std::string& isaName, unsigned* genericVersion) {
  isaName.clear();

  // Generic versioning starts at 1, so zero means no generic version.
  if (genericVersion) {
    *genericVersion = 0;
  }

  // AMDHSA always uses ELFCLASS64
  if (elfClass != ELFCLASS64 || machine != kEmAmdgpu) {
    return false;
  }

  uint32_t codeObjectMajorVersion = 0;
  switch (abiVersion) {
    case kAbiVersionV3: codeObjectMajorVersion = 3; break;
    case kAbiVersionV4: codeObjectMajorVersion = 4; break;
    case kAbiVersionV5: codeObjectMajorVersion = 5; break;
    case kAbiVersionV6: codeObjectMajorVersion = 6; break;
    default:
      // Code object v2 keeps its ISA in a note
      return false;
  }

  // Only AMDHSA code objects run on the ROCm runtime
  if (osAbi != kOsAbiAmdgpuHsa) {
    return false;
  }

  const MachInfo* info = FindMach(eFlags & kMachMask);
  if (info == nullptr) {
    return false;
  }
  std::string name = info->name;

  if (codeObjectMajorVersion == 3) {
    if (eFlags & kSrameccV3) {
      name += ":sramecc+";
    } else if (info->srameccSupported) {
      name += ":sramecc-";
    }

    if (eFlags & kXnackV3) {
      name += ":xnack+";
    } else if (info->xnackSupported) {
      name += ":xnack-";
    }
  } else {
    switch (eFlags & kSrameccV4Mask) {
      case kSrameccOffV4: name += ":sramecc-"; break;
      case kSrameccOnV4: name += ":sramecc+"; break;
    }

    switch (eFlags & kXnackV4Mask) {
      case kXnackOffV4: name += ":xnack-"; break;
      case kXnackOnV4: name += ":xnack+"; break;
    }

    // The generic version is not part of the ISA name
    if (genericVersion && codeObjectMajorVersion >= 6) {
      *genericVersion = (eFlags & kGenericVersionMask) >> kGenericVersionOffset;
    }
  }

  isaName = "amdgcn-amd-amdhsa--" + name;
  return true;
}

// ================================================================================================
bool ForEachNote(const void* image, size_t size, const char* name, uint32_t type,
                 const std::function<bool(const char*, uint32_t)>& fn) {
  const Elf64_Shdr* shdr = SectionHeaders(image, size);
  if (shdr == nullptr) {
    return false;
  }
  const std::string_view wanted(name);
  const uint16_t numSections = static_cast<const Elf64_Ehdr*>(image)->e_shnum;
  for (uint16_t s = 0; s < numSections; ++s) {
    if (shdr[s].sh_type != SHT_NOTE) {
      continue;
    }
    // ElfImageSize checked that the section lies inside the image
    const char* data = static_cast<const char*>(image) + shdr[s].sh_offset;
    const uint64_t dataSize = shdr[s].sh_size;
    uint64_t offset = 0;
    while (offset < dataSize) {
      if (dataSize - offset < sizeof(NoteHeader)) {
        return false;
      }
      NoteHeader note;
      std::memcpy(&note, data + offset, sizeof(note));
      const uint64_t nameAligned = AlignNote(note.nameSize);
      const uint64_t descAligned = AlignNote(note.descSize);
      const uint64_t entrySize = sizeof(NoteHeader) + nameAligned + descAligned;
      if (entrySize > dataSize - offset) {
        return false;
      }
      if (note.type == type) {
        // The name may or may not count its terminating null
        const char* noteName = data + offset + sizeof(NoteHeader);
        std::string_view nameView(noteName, note.nameSize);
        if (!nameView.empty() && nameView.back() == '\0') {
          nameView.remove_suffix(1);
        }
        if (nameView == wanted && !fn(noteName + nameAligned, note.descSize)) {
          return true;
        }
      }
      offset += entrySize;
    }
  }
  return true;
}

// ================================================================================================
bool DynamicSymbolNames(const void* image, size_t size, std::vector<std::string>& funcs,
                        std::vector<std::string>& vars) {
  const Elf64_Shdr* shdr = SectionHeaders(image, size);
  if (shdr == nullptr) {
    return false;
  }
  const uint16_t numSections = static_cast<const Elf64_Ehdr*>(image)->e_shnum;
  const Elf64_Shdr* dynsym = nullptr;
  for (uint16_t s = 0; s < numSections && dynsym == nullptr; ++s) {
    if (shdr[s].sh_type == SHT_DYNSYM) {
      dynsym = &shdr[s];
    }
  }
  if (dynsym == nullptr) {
    return true;
  }
  if (dynsym->sh_entsize != sizeof(Elf64_Sym) || dynsym->sh_link >= numSections ||
      shdr[dynsym->sh_link].sh_type != SHT_STRTAB) {
    return false;
  }
  // ElfImageSize checked that both sections lie inside the image
  const char* base = static_cast<const char*>(image);
  const Elf64_Shdr& strtab = shdr[dynsym->sh_link];
  const char* strings = base + strtab.sh_offset;
  const uint64_t count = dynsym->sh_size / sizeof(Elf64_Sym);
  for (uint64_t i = 0; i < count; ++i) {
    Elf64_Sym sym;
    std::memcpy(&sym, base + dynsym->sh_offset + i * sizeof(Elf64_Sym), sizeof(sym));
    const unsigned type = ELF_ST_TYPE(sym.st_info);
    if ((type != STT_FUNC && type != STT_OBJECT) || sym.st_name == 0) {
      continue;
    }
    if (sym.st_name >= strtab.sh_size) {
      return false;
    }
    const char* symName = strings + sym.st_name;
    const auto* end = static_cast<const char*>(
        std::memchr(symName, '\0', static_cast<size_t>(strtab.sh_size - sym.st_name)));
    if (end == nullptr) {
      return false;  // not terminated inside .dynstr
    }
    const size_t length = static_cast<size_t>(end - symName);
    if (length != 0) {
      (type == STT_FUNC ? funcs : vars).emplace_back(symName, length);
    }
  }
  return true;
}

}  // namespace amd::codeobject
