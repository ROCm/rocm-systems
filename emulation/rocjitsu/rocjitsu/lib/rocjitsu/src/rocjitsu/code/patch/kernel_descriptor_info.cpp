// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/kernel_descriptor_info.h"

#include "rocjitsu/code/amdgpu_elf.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <cstring>
#include <optional>

namespace rocjitsu {

namespace {

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

[[nodiscard]] bool is_cdna_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

[[nodiscard]] bool is_rdna_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4;
}

[[nodiscard]] bool kernel_descriptor_symbol(const Elf64_Sym &sym, const char *strtab,
                                            size_t strtab_size) {
  if (sym.st_size != sizeof(KD))
    return false;
  // Named descriptors conventionally end in ".kd"; stripped or unnamed symbol
  // tables are accepted by size so DBT still works on minimized objects.
  if (strtab == nullptr || strtab_size == 0 || sym.st_name == 0)
    return true;
  if (sym.st_name >= strtab_size)
    return false;

  const char *name = strtab + sym.st_name;
  const size_t len = std::strlen(name);
  return len < 3 || std::strcmp(name + len - 3, ".kd") == 0;
}

[[nodiscard]] std::optional<uint64_t> text_vaddr_for_section(uint64_t text_offset,
                                                             uint64_t text_size,
                                                             const Elf64_Ehdr &ehdr,
                                                             const Elf64_Shdr *shdr) {
  // Find the .text section's virtual address for kernel-entry offset calculation.
  for (int i = 0; i < ehdr.e_shnum; ++i) {
    if (shdr[i].sh_offset == text_offset && shdr[i].sh_size == text_size)
      return shdr[i].sh_addr;
  }
  return std::nullopt;
}

[[nodiscard]] KernelWorkGroupIdInfo workgroup_id_info(uint64_t entry_text_offset, const KD &desc) {
  const uint32_t rsrc2 = desc.compute_pgm_rsrc2;
  // USER_SGPR_COUNT is the compiler-authored count of preloaded user SGPRs.
  // CP/SPI places enabled system SGPRs, including workgroup IDs, immediately
  // after that count; the enable bits say which ones exist, not their base.
  uint32_t sgpr = AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);

  KernelWorkGroupIdInfo info{entry_text_offset, -1, -1, -1};
  if (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X))
    info.sgpr_wg_id_x = static_cast<int8_t>(sgpr++);
  if (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y))
    info.sgpr_wg_id_y = static_cast<int8_t>(sgpr++);
  if (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z))
    info.sgpr_wg_id_z = static_cast<int8_t>(sgpr++);
  return info;
}

} // namespace

std::vector<KernelDescriptorInfo> collect_kernel_descriptor_info(std::span<const uint8_t> image,
                                                                 uint64_t text_offset,
                                                                 uint64_t text_size) {
  std::vector<KernelDescriptorInfo> infos;

  if (image.size() < sizeof(Elf64_Ehdr))
    return infos;

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(image.data());
  if (ehdr->e_shoff + static_cast<uint64_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr) > image.size())
    return infos;

  const auto *shdr = reinterpret_cast<const Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto text_vaddr = text_vaddr_for_section(text_offset, text_size, *ehdr, shdr);
  if (!text_vaddr)
    return infos;

  for (int i = 0; i < ehdr->e_shnum; ++i) {
    if (shdr[i].sh_type != SHT_SYMTAB && shdr[i].sh_type != SHT_DYNSYM)
      continue;
    if (shdr[i].sh_offset + shdr[i].sh_size > image.size() || shdr[i].sh_entsize == 0)
      continue;

    const char *strtab = nullptr;
    size_t strtab_size = 0;
    if (shdr[i].sh_link < ehdr->e_shnum) {
      const auto &strtab_shdr = shdr[shdr[i].sh_link];
      if (strtab_shdr.sh_offset + strtab_shdr.sh_size <= image.size()) {
        strtab = reinterpret_cast<const char *>(image.data() + strtab_shdr.sh_offset);
        strtab_size = strtab_shdr.sh_size;
      }
    }

    const auto *symtab = reinterpret_cast<const Elf64_Sym *>(image.data() + shdr[i].sh_offset);
    const size_t nsyms = shdr[i].sh_size / sizeof(Elf64_Sym);
    for (size_t j = 0; j < nsyms; ++j) {
      if (!kernel_descriptor_symbol(symtab[j], strtab, strtab_size))
        continue;

      const uint16_t sec_idx = symtab[j].st_shndx;
      if (sec_idx >= ehdr->e_shnum || symtab[j].st_value < shdr[sec_idx].sh_addr)
        continue;

      const uint64_t file_off =
          shdr[sec_idx].sh_offset + (symtab[j].st_value - shdr[sec_idx].sh_addr);
      if (file_off + sizeof(KD) > image.size())
        continue;

      const auto *desc = reinterpret_cast<const KD *>(image.data() + file_off);
      const uint64_t entry_vaddr = symtab[j].st_value + desc->kernel_code_entry_byte_offset;
      if (entry_vaddr < *text_vaddr || entry_vaddr >= *text_vaddr + text_size)
        continue;

      const uint64_t entry_text_offset = entry_vaddr - *text_vaddr;
      const bool wave32 = AMDHSA_BITS_GET(desc->kernel_code_properties,
                                          kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);

      KernelDescriptorInfo info;
      info.entry_text_offset = entry_text_offset;
      info.wavefront_size = wave32 ? 32 : 64;
      info.workgroup_id = workgroup_id_info(entry_text_offset, *desc);
      infos.push_back(info);
    }
  }

  return infos;
}

uint8_t kernel_wavefront_size(rj_code_arch_t guest_arch, const KernelDescriptorInfo *kernel) {
  if (is_cdna_arch(guest_arch))
    return 64;

  if (is_rdna_arch(guest_arch))
    return kernel && kernel->wavefront_size ? *kernel->wavefront_size : 32;

  return 64;
}

} // namespace rocjitsu
