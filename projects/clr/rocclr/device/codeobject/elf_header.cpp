/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/codeobject/elf_header.hpp"

#include "device/device.hpp"
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

//! The amd::Isa row of the processor with EF_AMDGPU_MACH_* value @p mach, or null. Processors
//! CLR has no row for (gfx6, gfx7) can't run on any device, so their code objects stay unnamed.
const amd::Isa* FindMach(uint32_t mach) {
  for (const amd::Isa* isa = amd::Isa::begin(); isa != amd::Isa::end(); ++isa) {
    if (isa->elfMach() == mach) return isa;
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

//! A @p T copied from @p offset in @p image: code objects sit at any offset in a bundle or a
//! file, so their headers can't be read through typed pointers
template <typename T> T ReadAt(const void* image, uint64_t offset) {
  T value;
  std::memcpy(&value, static_cast<const char*>(image) + offset, sizeof(value));
  return value;
}

//! Reads the ELF header of @p image once the image is validated against @p size.
bool ReadElfHeader(const void* image, size_t size, Elf64_Ehdr& ehdr) {
  if (size == kUnknownSize || ElfImageSize(image, size) == 0) {
    return false;
  }
  ehdr = ReadAt<Elf64_Ehdr>(image, 0);
  return true;
}

//! Section header @p index of a validated image
Elf64_Shdr ReadSection(const void* image, const Elf64_Ehdr& ehdr, uint64_t index) {
  return ReadAt<Elf64_Shdr>(image, ehdr.e_shoff + index * sizeof(Elf64_Shdr));
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

  const auto ehdr = ReadAt<Elf64_Ehdr>(image, 0);
  uint64_t totalSize = sizeof(Elf64_Ehdr);

  // Validate and bound the section-header table before any read.
  if (ehdr.e_shnum != 0) {
    if (ehdr.e_shentsize != sizeof(Elf64_Shdr) || ehdr.e_shoff >= limit) {
      return 0;
    }
    const uint64_t shtab = static_cast<uint64_t>(ehdr.e_shnum) * sizeof(Elf64_Shdr);
    uint64_t shtabEnd = 0;
    if (!checkedEnd(ehdr.e_shoff, shtab, shtabEnd) || shtabEnd > limit) {
      return 0;
    }
    if (shtabEnd > totalSize) totalSize = shtabEnd;

    for (uint64_t i = 0; i < ehdr.e_shnum; ++i) {
      const Elf64_Shdr shdr = ReadSection(image, ehdr, i);
      if (SHT_NOBITS == shdr.sh_type) continue;
      uint64_t end = 0;
      if (!checkedEnd(shdr.sh_offset, shdr.sh_size, end)) return 0;
      if (end > limit) return 0;
      if (end > totalSize) totalSize = end;
    }
  }

  // Validate and bound the program-header table before any read.
  if (ehdr.e_phnum != 0) {
    if (ehdr.e_phentsize != sizeof(Elf64_Phdr) || ehdr.e_phoff >= limit) {
      return 0;
    }
    const uint64_t phtab = static_cast<uint64_t>(ehdr.e_phnum) * sizeof(Elf64_Phdr);
    uint64_t phtabEnd = 0;
    if (!checkedEnd(ehdr.e_phoff, phtab, phtabEnd) || phtabEnd > limit) {
      return 0;
    }
    if (phtabEnd > totalSize) totalSize = phtabEnd;

    for (uint64_t i = 0; i < ehdr.e_phnum; ++i) {
      const auto phdr = ReadAt<Elf64_Phdr>(image, ehdr.e_phoff + i * sizeof(Elf64_Phdr));
      if (PT_NULL == phdr.p_type) continue;
      uint64_t end = 0;
      if (!checkedEnd(phdr.p_offset, phdr.p_filesz, end)) return 0;
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

  const amd::Isa* isa = FindMach(eFlags & kMachMask);
  if (isa == nullptr) {
    return false;
  }
  std::string name = isa->targetId();

  if (codeObjectMajorVersion == 3) {
    if (eFlags & kSrameccV3) {
      name += ":sramecc+";
    } else if (isa->sramecc() != amd::Isa::Feature::Unsupported) {
      name += ":sramecc-";
    }

    if (eFlags & kXnackV3) {
      name += ":xnack+";
    } else if (isa->xnack() != amd::Isa::Feature::Unsupported) {
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
  Elf64_Ehdr ehdr;
  if (!ReadElfHeader(image, size, ehdr)) {
    return false;
  }
  const std::string_view wanted(name);
  for (uint16_t s = 0; s < ehdr.e_shnum; ++s) {
    const Elf64_Shdr shdr = ReadSection(image, ehdr, s);
    if (shdr.sh_type != SHT_NOTE) {
      continue;
    }
    // ElfImageSize checked that the section lies inside the image
    const char* data = static_cast<const char*>(image) + shdr.sh_offset;
    const uint64_t dataSize = shdr.sh_size;
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
  Elf64_Ehdr ehdr;
  if (!ReadElfHeader(image, size, ehdr)) {
    return false;
  }
  Elf64_Shdr dynsym{};
  bool found = false;
  for (uint16_t s = 0; s < ehdr.e_shnum && !found; ++s) {
    dynsym = ReadSection(image, ehdr, s);
    found = (dynsym.sh_type == SHT_DYNSYM);
  }
  if (!found) {
    return true;
  }
  if (dynsym.sh_entsize != sizeof(Elf64_Sym) || dynsym.sh_link >= ehdr.e_shnum) {
    return false;
  }
  const Elf64_Shdr strtab = ReadSection(image, ehdr, dynsym.sh_link);
  if (strtab.sh_type != SHT_STRTAB) {
    return false;
  }
  // ElfImageSize checked that both sections lie inside the image
  const char* base = static_cast<const char*>(image);
  const char* strings = base + strtab.sh_offset;
  const uint64_t count = dynsym.sh_size / sizeof(Elf64_Sym);
  for (uint64_t i = 0; i < count; ++i) {
    const auto sym = ReadAt<Elf64_Sym>(image, dynsym.sh_offset + i * sizeof(Elf64_Sym));
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
