// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_CODE_AMDGPU_ELF_H_
#define ROCJITSU_CODE_AMDGPU_ELF_H_

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <span>

namespace rocjitsu {

using Elf_Half = uint16_t;
using Elf_Word = uint32_t;
using Elf64_Addr = uint64_t;
using Elf64_Off = uint64_t;

inline constexpr int EI_CLASS = 4;
inline constexpr int EI_DATA = 5;
inline constexpr int EI_VERSION = 6;
inline constexpr int EI_OSABI = 7;
inline constexpr int EI_ABIVERSION = 8;
inline constexpr int EI_PAD = 9;
inline constexpr int EI_NIDENT = 16;

inline constexpr char EI_MAGIC[] = {0x7f, 'E', 'L', 'F'};
inline constexpr int EI_MAGIC_SIZE = sizeof(EI_MAGIC);

inline constexpr uint8_t ELFCLASSNONE = 0;
inline constexpr uint8_t ELFCLASS32 = 1;
inline constexpr uint8_t ELFCLASS64 = 2;

inline constexpr uint8_t ELFOSABI_NONE = 0;
inline constexpr uint8_t ELFOSABI_AMDGPU_HSA = 64;

inline constexpr int ELFABIVERSION_AMDGPU_HSA_V2 = 0;
inline constexpr int ELFABIVERSION_AMDGPU_HSA_V3 = 1;
inline constexpr int ELFABIVERSION_AMDGPU_HSA_V4 = 2;
inline constexpr int ELFABIVERSION_AMDGPU_HSA_V5 = 3;
inline constexpr int ELFABIVERSION_AMDGPU_HSA_V6 = 4;

inline constexpr Elf_Half EM_X86_64 = 62;
inline constexpr Elf_Half EM_AMDGPU = 224;

inline constexpr Elf_Half ET_REL = 1;
inline constexpr Elf_Half ET_DYN = 3;

inline constexpr Elf_Half SHN_UNDEF = 0;
inline constexpr Elf_Half SHN_ABS = 0xfff1;

inline constexpr uint32_t EF_AMDGPU_MACH = 0x0ff;
inline constexpr uint32_t EF_AMDGPU_MACH_NONE = 0;
inline constexpr uint32_t EF_AMDGPU_MACH_AMDGCN_GFX908 = 0x30;
inline constexpr uint32_t EF_AMDGPU_MACH_AMDGCN_GFX90A = 0x3f;
inline constexpr uint32_t EF_AMDGPU_MACH_AMDGCN_GFX940 = 0x40;
inline constexpr uint32_t EF_AMDGPU_MACH_AMDGCN_GFX941 = 0x4b;
inline constexpr uint32_t EF_AMDGPU_MACH_AMDGCN_GFX942 = 0x4c;
inline constexpr uint32_t EF_AMDGPU_MACH_AMDGCN_GFX950 = 0x4f;
inline constexpr uint32_t EF_AMDGPU_MACH_AMDGCN_GFX1010 = 0x33;
inline constexpr uint32_t EF_AMDGPU_MACH_AMDGCN_GFX1030 = 0x36;
inline constexpr uint32_t EF_AMDGPU_MACH_AMDGCN_GFX1100 = 0x41;
inline constexpr uint32_t EF_AMDGPU_MACH_AMDGCN_GFX1150 = 0x43;
inline constexpr uint32_t EF_AMDGPU_MACH_AMDGCN_GFX1200 = 0x48;
inline constexpr uint32_t EF_AMDGPU_MACH_AMDGCN_GFX1250 = 0x49;
inline constexpr uint32_t EF_AMDGPU_MACH_AMDGCN_GFX1251 = 0x5a;
inline constexpr uint32_t EF_AMDGPU_MACH_AMDGCN_GFX1201 = 0x4e;

/*
 * \NPI new GPU: add its EF_AMDGPU_MACH_AMDGCN_* constant above (value from the \
 * LLVM AMDGPU backend), then handle it in elf_mach_for_arch, arch_for_elf_mach, \
 * and elf_mach_name below.
 */
inline constexpr uint32_t elf_mach_for_arch(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return EF_AMDGPU_MACH_AMDGCN_GFX908;
  case ROCJITSU_CODE_ARCH_CDNA2:
    return EF_AMDGPU_MACH_AMDGCN_GFX90A;
  case ROCJITSU_CODE_ARCH_CDNA3:
    return EF_AMDGPU_MACH_AMDGCN_GFX942;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return EF_AMDGPU_MACH_AMDGCN_GFX950;
  case ROCJITSU_CODE_ARCH_RDNA1:
    return EF_AMDGPU_MACH_AMDGCN_GFX1010;
  case ROCJITSU_CODE_ARCH_RDNA2:
    return EF_AMDGPU_MACH_AMDGCN_GFX1030;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return EF_AMDGPU_MACH_AMDGCN_GFX1100;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return EF_AMDGPU_MACH_AMDGCN_GFX1150;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return EF_AMDGPU_MACH_AMDGCN_GFX1200;
  case ROCJITSU_CODE_ARCH_CDNA5:
    return EF_AMDGPU_MACH_AMDGCN_GFX1250;
  default:
    return EF_AMDGPU_MACH_NONE;
  }
}

/// @brief Return the rocjitsu ISA family used by an AMDGPU ELF MACH value.
///
/// @details Several concrete GPU steppings share one rocjitsu decoder/translator
/// family. Keep this mapping next to the ELF constants so code-object parsing,
/// load-time hooks, and tests do not each grow their own copy.
inline constexpr rj_code_arch_t arch_for_elf_mach(uint32_t mach) {
  switch (mach & EF_AMDGPU_MACH) {
  case EF_AMDGPU_MACH_AMDGCN_GFX908:
    return ROCJITSU_CODE_ARCH_CDNA1;
  case EF_AMDGPU_MACH_AMDGCN_GFX90A:
    return ROCJITSU_CODE_ARCH_CDNA2;
  case EF_AMDGPU_MACH_AMDGCN_GFX940:
  case EF_AMDGPU_MACH_AMDGCN_GFX941:
  case EF_AMDGPU_MACH_AMDGCN_GFX942:
    return ROCJITSU_CODE_ARCH_CDNA3;
  case EF_AMDGPU_MACH_AMDGCN_GFX950:
    return ROCJITSU_CODE_ARCH_CDNA4;
  case EF_AMDGPU_MACH_AMDGCN_GFX1010:
    return ROCJITSU_CODE_ARCH_RDNA1;
  case EF_AMDGPU_MACH_AMDGCN_GFX1030:
    return ROCJITSU_CODE_ARCH_RDNA2;
  case EF_AMDGPU_MACH_AMDGCN_GFX1100:
    return ROCJITSU_CODE_ARCH_RDNA3;
  case EF_AMDGPU_MACH_AMDGCN_GFX1150:
    return ROCJITSU_CODE_ARCH_RDNA3_5;
  case EF_AMDGPU_MACH_AMDGCN_GFX1200:
  case EF_AMDGPU_MACH_AMDGCN_GFX1201:
    return ROCJITSU_CODE_ARCH_RDNA4;
  case EF_AMDGPU_MACH_AMDGCN_GFX1250:
  case EF_AMDGPU_MACH_AMDGCN_GFX1251:
    return ROCJITSU_CODE_ARCH_CDNA5;
  default:
    return ROCJITSU_CODE_ARCH_INVALID;
  }
}

/// @brief Return the canonical gfx name for an AMDGPU ELF MACH value.
inline constexpr const char *elf_mach_name(uint32_t mach) {
  switch (mach & EF_AMDGPU_MACH) {
  case EF_AMDGPU_MACH_AMDGCN_GFX908:
    return "gfx908";
  case EF_AMDGPU_MACH_AMDGCN_GFX90A:
    return "gfx90a";
  case EF_AMDGPU_MACH_AMDGCN_GFX940:
    return "gfx940";
  case EF_AMDGPU_MACH_AMDGCN_GFX941:
    return "gfx941";
  case EF_AMDGPU_MACH_AMDGCN_GFX942:
    return "gfx942";
  case EF_AMDGPU_MACH_AMDGCN_GFX950:
    return "gfx950";
  case EF_AMDGPU_MACH_AMDGCN_GFX1010:
    return "gfx1010";
  case EF_AMDGPU_MACH_AMDGCN_GFX1030:
    return "gfx1030";
  case EF_AMDGPU_MACH_AMDGCN_GFX1100:
    return "gfx1100";
  case EF_AMDGPU_MACH_AMDGCN_GFX1150:
    return "gfx1150";
  case EF_AMDGPU_MACH_AMDGCN_GFX1200:
    return "gfx1200";
  case EF_AMDGPU_MACH_AMDGCN_GFX1201:
    return "gfx1201";
  case EF_AMDGPU_MACH_AMDGCN_GFX1250:
    return "gfx1250";
  case EF_AMDGPU_MACH_AMDGCN_GFX1251:
    return "gfx1251";
  default:
    return "unknown";
  }
}

inline constexpr uint32_t SHT_NULL = 0;
inline constexpr uint32_t SHT_PROGBITS = 1;
inline constexpr uint32_t SHT_SYMTAB = 2;
inline constexpr uint32_t SHT_STRTAB = 3;
inline constexpr uint32_t SHT_RELA = 4;
inline constexpr uint32_t SHT_NOTE = 7;
inline constexpr uint32_t SHT_NOBITS = 8; // ELF spec: section occupies no file space (e.g. .bss)
inline constexpr uint32_t SHT_REL = 9;
inline constexpr uint32_t SHT_DYNSYM = 11;

// AMDGPU ELF relocation types used by linked code objects. Keep the numeric
// values local to the ELF layer so DBT does not depend on LLVM's ELF headers
// merely to inspect loader relocations.
inline constexpr uint32_t R_AMDGPU_NONE = 0;
inline constexpr uint32_t R_AMDGPU_ABS32_LO = 1;
inline constexpr uint32_t R_AMDGPU_ABS32_HI = 2;
inline constexpr uint32_t R_AMDGPU_ABS64 = 3;
inline constexpr uint32_t R_AMDGPU_ABS32 = 6;
inline constexpr uint32_t R_AMDGPU_RELATIVE64 = 13;

inline constexpr uint8_t kElfSymbolBindLocal = 0;
inline constexpr uint8_t kElfSymbolBindGlobal = 1;
inline constexpr uint8_t kElfSymbolBindWeak = 2;
inline constexpr uint8_t kElfSymbolVisibilityHidden = 2;     // STV_HIDDEN
inline constexpr uint8_t kElfSymbolVisibilityInternal = 1;   // STV_INTERNAL
inline constexpr uint8_t kElfSymbolTypeNone = 0;             // STT_NOTYPE
inline constexpr uint8_t kElfSymbolTypeObject = 1;           // STT_OBJECT
inline constexpr uint8_t kElfSymbolTypeFunc = 2;             // STT_FUNC
inline constexpr uint8_t kElfSymbolTypeSection = 3;          // STT_SECTION
inline constexpr uint8_t kElfSymbolTypeAmdGpuHsaKernel = 10; // STT_AMDGPU_HSA_KERNEL

inline constexpr uint8_t elf_symbol_bind(uint8_t info) { return info >> 4; }
inline constexpr uint8_t elf_symbol_type(uint8_t info) { return info & 0xf; }
inline constexpr uint8_t elf_symbol_visibility(uint8_t other) { return other & 0x3; }

/// @brief Whether a host could resolve this symbol by name and obtain its address.
///
/// @details Only external linkage reaches a loader: the HSA executable-symbol API and the dynamic
/// linker expose `STB_GLOBAL`/`STB_WEAK` definitions that are not hidden or internal. A local
/// symbol names a compilation-unit-private body -- an anonymous-namespace device function, say --
/// and no interface hands its address out, so the only way its address can enter the machine is
/// through this object's own code or relocations.
inline constexpr bool elf_symbol_is_externally_resolvable(uint8_t info, uint8_t other) {
  const uint8_t bind = elf_symbol_bind(info);
  if (bind != kElfSymbolBindGlobal && bind != kElfSymbolBindWeak)
    return false;
  const uint8_t visibility = elf_symbol_visibility(other);
  return visibility != kElfSymbolVisibilityHidden && visibility != kElfSymbolVisibilityInternal;
}
inline constexpr uint8_t elf_symbol_info(uint8_t bind, uint8_t type) {
  return static_cast<uint8_t>((bind << 4) | (type & 0xf));
}

// ELF64 relocation info accessors (r_info packs symbol index in the high 32 bits
// and the relocation type in the low 32 bits).
inline constexpr uint32_t elf_reloc_sym(uint64_t info) { return static_cast<uint32_t>(info >> 32); }
inline constexpr uint32_t elf_reloc_type(uint64_t info) {
  return static_cast<uint32_t>(info & 0xffffffffu);
}

/// @brief Whether an AMDGPU ELF relocation record has no effect.
inline constexpr bool elf_relocation_is_inert(uint64_t info) {
  return elf_reloc_type(info) == R_AMDGPU_NONE;
}

// R_AMDGPU_RELATIVE64 uses symbol index 0 and forms its value from the load bias
// plus r_addend, so its addend can name an in-.text virtual address with no
// owning symbol.

inline constexpr uint64_t SHF_WRITE = 1u << 0;
inline constexpr uint64_t SHF_ALLOC = 1u << 1;
inline constexpr uint64_t SHF_EXECINSTR = 1u << 2;

/// @brief AMDGPU vendor specific notes for Code Object V3.
inline constexpr uint32_t NT_AMDGPU_METADATA = 32;

// Program header types.
inline constexpr uint32_t PT_LOAD = 1;
inline constexpr uint32_t PT_DYNAMIC = 2;
inline constexpr uint32_t PT_NOTE = 4;

// Dynamic section tags.
inline constexpr int64_t DT_NULL = 0;
inline constexpr int64_t DT_HASH = 4;
inline constexpr int64_t DT_STRTAB = 5;
inline constexpr int64_t DT_SYMTAB = 6;
inline constexpr int64_t DT_STRSZ = 10;
inline constexpr int64_t DT_SYMENT = 11;

/**
 * @brief ELF dynamic section entry.
 */
struct Elf64_Dyn {
  int64_t d_tag;
  union {
    uint64_t d_val;
    uint64_t d_ptr;
  } d_un;
};

/**
 * @brief ELF header.
 */
struct Elf64_Ehdr {
  uint8_t e_ident[EI_NIDENT];
  uint16_t e_type;
  uint16_t e_machine;
  uint32_t e_version;
  Elf64_Addr entry;
  Elf64_Off e_phoff;
  Elf64_Off e_shoff;
  uint32_t e_flags;
  uint16_t e_ehsize;
  uint16_t e_phentsize;
  uint16_t e_phnum;
  uint16_t e_shentsize;
  uint16_t e_shnum;
  uint16_t e_shstrndx;
};

/**
 * @brief ELF program header.
 */
struct Elf64_Phdr {
  uint32_t p_type;
  uint32_t p_flags;
  Elf64_Off p_offset;
  Elf64_Addr p_vaddr;
  Elf64_Addr p_paddr;
  uint64_t p_filesz;
  uint64_t p_memsz;
  uint64_t p_align;
};

/**
 * @brief ELF section header.
 */
struct Elf64_Shdr {
  uint32_t sh_name;
  uint32_t sh_type;
  uint64_t sh_flags;
  Elf64_Addr sh_addr;
  Elf64_Off sh_offset;
  uint64_t sh_size;
  uint32_t sh_link;
  uint32_t sh_info;
  uint64_t sh_addralign;
  uint64_t sh_entsize;
};

struct Elf64_Nhdr {
  uint32_t n_namesz;
  uint32_t n_descsz;
  uint32_t n_type;
};

/**
 * @brief ELF symbol.
 */
struct Elf64_Sym {
  uint32_t st_name;
  uint8_t st_info;
  uint8_t st_other;
  uint16_t st_shndx;
  Elf64_Addr st_value;
  uint64_t st_size;
};

/**
 * @brief ELF relocation.
 */
struct Elf64_Rel {
  Elf64_Addr r_offset;
  uint64_t r_info;
};

/**
 * @brief ELF relocation with addend.
 */
struct Elf64_Rela {
  Elf64_Addr r_offset;
  uint64_t r_info;
  int64_t r_addend;
};

/// @brief Whether ROCr's dynamic loader forms this relocation from a symbol value.
inline constexpr bool rocr_dynamic_relocation_uses_symbol_value(uint32_t relocation_type) {
  switch (relocation_type) {
  case R_AMDGPU_ABS32_LO:
  case R_AMDGPU_ABS32_HI:
  case R_AMDGPU_ABS64:
  case R_AMDGPU_ABS32:
    return true;
  default:
    return false;
  }
}

/// @brief Whether ROCr derives this dynamic symbol's runtime address from st_value.
inline constexpr bool rocr_dynamic_symbol_address_follows_st_value(uint8_t symbol_type) {
  switch (symbol_type) {
  case kElfSymbolTypeObject:
  case kElfSymbolTypeFunc:
  case kElfSymbolTypeAmdGpuHsaKernel:
    return true;
  default:
    return false;
  }
}

/// @brief Whether this symbol type can introduce a relocation-backed code entry.
inline constexpr bool elf_symbol_is_executable_entry(uint8_t symbol_type) {
  return symbol_type == kElfSymbolTypeFunc;
}

/// @brief How ROCr dispatches one relocation section in a modern code object.
enum class RocrRelocationSectionMode : uint8_t {
  NotApplicable,
  Dynamic,
  ExplicitTarget,
  Malformed,
};

/// @brief Classify the relocation section using ROCr's targetSection() decision.
///
/// @details ROCr materializes SHT_RELA sections, resolves sh_info to a section pointer, and uses
/// dynamic relocation processing when that pointer is null. Only section zero safely represents
/// that null target: ROCr skips index zero during its later section passes, but dereferences every
/// nonzero section entry before relocation dispatch. A valid non-null section selects the
/// explicit/static path; a nonzero SHT_NULL entry or out-of-range index is malformed. Other ELF
/// and relocation-section forms retain their existing non-ROCr handling.
inline constexpr RocrRelocationSectionMode
classify_rocr_relocation_section(const Elf64_Ehdr &ehdr, std::span<const Elf64_Shdr> shdrs,
                                 const Elf64_Shdr &relocs) {
  if (ehdr.e_type != ET_DYN || relocs.sh_type != SHT_RELA)
    return RocrRelocationSectionMode::NotApplicable;
  if (relocs.sh_info == SHN_UNDEF)
    return RocrRelocationSectionMode::Dynamic;
  if (relocs.sh_info >= shdrs.size() || shdrs[relocs.sh_info].sh_type == SHT_NULL)
    return RocrRelocationSectionMode::Malformed;
  return RocrRelocationSectionMode::ExplicitTarget;
}

/// @brief ROCr's handling of an R_AMDGPU_NONE record before place or symbol lookup.
enum class RocrNoneRelocationAction : uint8_t {
  NotNone,
  Ignored,
  Rejected,
};

/// @brief Classify R_AMDGPU_NONE from the shared ROCr relocation-section mode.
inline constexpr RocrNoneRelocationAction
classify_rocr_none_relocation(RocrRelocationSectionMode mode, uint64_t relocation_info) {
  if (elf_reloc_type(relocation_info) != R_AMDGPU_NONE)
    return RocrNoneRelocationAction::NotNone;
  if (mode == RocrRelocationSectionMode::Dynamic || mode == RocrRelocationSectionMode::Malformed) {
    return RocrNoneRelocationAction::Rejected;
  }
  return RocrNoneRelocationAction::Ignored;
}

/// @brief How a relocation reference to an ordinary symbol in rewritten .text must be handled.
enum class TextSymbolRelocationAction : uint8_t {
  Ignored,
  RequiresSymbolMapping,
  RequiresExecutableEntry,
  Unsupported,
};

/// @brief Classify one relocation reference to a symbol defined in rewritten .text.
///
/// @details A required executable entry also requires the symbol's st_value to be remapped.
/// ROCr rejects R_AMDGPU_NONE in target-less dynamic relocation sections, but skips
/// explicit-target/static relocation sections for supported code objects. Other ROCr dynamic
/// relocations follow the loader's supported symbol-valued forms, while explicit-target/static
/// sections retain the broader pre-existing policy that any ordinary zero-addend RELA reference
/// follows the symbol value.
inline constexpr TextSymbolRelocationAction
classify_text_symbol_relocation(RocrRelocationSectionMode mode, uint64_t relocation_info,
                                bool has_explicit_addend, int64_t addend, const Elf64_Sym &symbol) {
  const uint32_t relocation_type = elf_reloc_type(relocation_info);
  if (mode == RocrRelocationSectionMode::Malformed)
    return TextSymbolRelocationAction::Unsupported;
  const bool rocr_dynamic = mode == RocrRelocationSectionMode::Dynamic;
  if (relocation_type == R_AMDGPU_NONE) {
    return rocr_dynamic ? TextSymbolRelocationAction::Unsupported
                        : TextSymbolRelocationAction::Ignored;
  }
  if (!has_explicit_addend || addend != 0 || elf_reloc_sym(relocation_info) == 0)
    return TextSymbolRelocationAction::Unsupported;

  const uint8_t symbol_type = elf_symbol_type(symbol.st_info);
  if (rocr_dynamic) {
    if (relocation_type == R_AMDGPU_RELATIVE64) {
      return symbol_type == kElfSymbolTypeSection
                 ? TextSymbolRelocationAction::Unsupported
                 : TextSymbolRelocationAction::RequiresSymbolMapping;
    }
    if (!rocr_dynamic_relocation_uses_symbol_value(relocation_type) ||
        !rocr_dynamic_symbol_address_follows_st_value(symbol_type)) {
      return TextSymbolRelocationAction::Unsupported;
    }
    return elf_symbol_is_executable_entry(symbol_type)
               ? TextSymbolRelocationAction::RequiresExecutableEntry
               : TextSymbolRelocationAction::RequiresSymbolMapping;
  }

  if (symbol_type == kElfSymbolTypeSection)
    return TextSymbolRelocationAction::Unsupported;
  if (elf_symbol_is_executable_entry(symbol_type) ||
      (symbol_type == kElfSymbolTypeNone && relocation_type == R_AMDGPU_ABS64)) {
    return TextSymbolRelocationAction::RequiresExecutableEntry;
  }
  return TextSymbolRelocationAction::RequiresSymbolMapping;
}

/// @brief Return whether a relocation record applies to storage loaded at runtime.
///
/// Relocation sections with an explicit sh_info target inherit that section's allocation
/// state. Generic ET_DYN relocation sections use SHN_UNDEF and identify their target storage
/// through each record's virtual r_offset instead. Keeping this ABI rule here prevents entry
/// discovery and final ELF patching from disagreeing about debug-only relocation metadata.
[[nodiscard]] inline bool elf_relocation_place_is_allocated(const Elf64_Ehdr &ehdr,
                                                            std::span<const Elf64_Shdr> shdrs,
                                                            const Elf64_Shdr &relocs,
                                                            uint64_t relocation_offset) {
  const RocrRelocationSectionMode mode = classify_rocr_relocation_section(ehdr, shdrs, relocs);
  if (mode == RocrRelocationSectionMode::Malformed)
    return false;
  if (mode == RocrRelocationSectionMode::ExplicitTarget) {
    return (shdrs[relocs.sh_info].sh_flags & SHF_ALLOC) != 0;
  }
  if (mode == RocrRelocationSectionMode::NotApplicable && relocs.sh_info != SHN_UNDEF) {
    return relocs.sh_info < shdrs.size() && (shdrs[relocs.sh_info].sh_flags & SHF_ALLOC) != 0;
  }
  if (mode == RocrRelocationSectionMode::NotApplicable && ehdr.e_type != ET_DYN)
    return false;
  for (const Elf64_Shdr &section : shdrs) {
    if ((section.sh_flags & SHF_ALLOC) != 0 && relocation_offset >= section.sh_addr &&
        relocation_offset - section.sh_addr < section.sh_size) {
      return true;
    }
  }
  return false;
}

} // namespace rocjitsu

#endif // ROCJITSU_CODE_AMDGPU_ELF_H_
