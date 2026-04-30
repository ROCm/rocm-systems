// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_CODE_AMDGPU_ELF_H_
#define ROCJITSU_CODE_AMDGPU_ELF_H_

#include <cstdint>

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
inline constexpr uint32_t EF_AMDGPU_MACH_AMDGCN_GFX1201 = 0x4e;

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
  default:
    return EF_AMDGPU_MACH_NONE;
  }
}

inline constexpr uint32_t SHT_NULL = 0;
inline constexpr uint32_t SHT_PROGBITS = 1;
inline constexpr uint32_t SHT_SYMTAB = 2;
inline constexpr uint32_t SHT_STRTAB = 3;
inline constexpr uint32_t SHT_NOTE = 7;
inline constexpr uint32_t SHT_DYNSYM = 11;

/// @brief AMDGPU vendor specific notes for Code Object V3.
inline constexpr uint32_t NT_AMDGPU_METADATA = 32;

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

} // namespace rocjitsu

#endif // ROCJITSU_CODE_AMDGPU_ELF_H_
