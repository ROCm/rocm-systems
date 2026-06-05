// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/vgpr_msb.h"
#include "util/bit.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace rocjitsu {

namespace {

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

static_assert(sizeof(KD) == 64, "AMDHSA kernel descriptor size changed");

constexpr uint32_t kMaxVgprGranulatedField = 63;
constexpr uint32_t kMaxSgprGranulatedField = 15;
constexpr uint16_t kScalarOperandTtmpBase = 108;
constexpr uint16_t kTtmpRdna4GridYz = 7;
constexpr uint16_t kTtmpRdna4GridX = 9;
constexpr uint8_t kSoppWaitLoadcnt = 64;
constexpr uint8_t kSoppWaitKmcnt = 71;
constexpr uint8_t kSmemEncoding = 0x3d;
constexpr uint8_t kSmemSoffsetNone = 0x7c;

[[nodiscard]] constexpr uint16_t build_hwreg(uint8_t reg_id, uint8_t offset, uint8_t size) {
  return static_cast<uint16_t>((reg_id & 0x3Fu) | ((offset & 0x1Fu) << 6) |
                               (((size - 1u) & 0x1Fu) << 11));
}

[[nodiscard]] constexpr uint32_t build_sopk(uint8_t op, uint16_t simm16, uint8_t sdst = 0) {
  return 0xB0000000u | (simm16 & 0xFFFFu) | ((sdst & 0x7Fu) << 16) | ((op & 0x1Fu) << 23);
}

// -----------------------------------------------------------------------------
// ISA-family helpers.
// -----------------------------------------------------------------------------

[[nodiscard]] bool is_cdna_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

[[nodiscard]] bool is_rdna_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4;
}

[[nodiscard]] bool is_gfx1250_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_GFX1250;
}

[[nodiscard]] bool arch_supports_wave_size(rj_code_arch_t arch, uint32_t wf) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return supports_wave_size<cdna1::Isa>(wf);
  case ROCJITSU_CODE_ARCH_CDNA2:
    return supports_wave_size<cdna2::Isa>(wf);
  case ROCJITSU_CODE_ARCH_CDNA3:
    return supports_wave_size<cdna3::Isa>(wf);
  case ROCJITSU_CODE_ARCH_CDNA4:
    return supports_wave_size<cdna4::Isa>(wf);
  case ROCJITSU_CODE_ARCH_GFX1250:
    return supports_wave_size<gfx1250::Isa>(wf);
  case ROCJITSU_CODE_ARCH_RDNA1:
    return supports_wave_size<rdna1::Isa>(wf);
  case ROCJITSU_CODE_ARCH_RDNA2:
    return supports_wave_size<rdna2::Isa>(wf);
  case ROCJITSU_CODE_ARCH_RDNA3:
    return supports_wave_size<rdna3::Isa>(wf);
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return supports_wave_size<rdna3_5::Isa>(wf);
  case ROCJITSU_CODE_ARCH_RDNA4:
    return supports_wave_size<rdna4::Isa>(wf);
  default:
    return false;
  }
}

[[nodiscard]] uint32_t arch_default_wave_size(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::Isa::WF_SIZE;
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::Isa::WF_SIZE;
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::Isa::WF_SIZE;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::Isa::WF_SIZE;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::Isa::WF_SIZE;
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::Isa::WF_SIZE;
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::Isa::WF_SIZE;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::Isa::WF_SIZE;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::Isa::WF_SIZE;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::Isa::WF_SIZE;
  default:
    return 64;
  }
}

[[nodiscard]] uint32_t arch_max_sgprs(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::Isa::MAX_SGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::Isa::MAX_SGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::Isa::MAX_SGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::Isa::MAX_SGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::Isa::MAX_SGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::Isa::MAX_SGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::Isa::MAX_SGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::Isa::MAX_SGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::Isa::MAX_SGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::Isa::MAX_SGPRS_PER_WF;
  default:
    return 0;
  }
}

[[nodiscard]] uint32_t arch_max_vgprs(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::Isa::MAX_VGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::Isa::MAX_VGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::Isa::MAX_VGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::Isa::MAX_VGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return gfx1250::Isa::MAX_ADDRESSABLE_VGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::Isa::MAX_VGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::Isa::MAX_VGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::Isa::MAX_VGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::Isa::MAX_VGPRS_PER_WF;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::Isa::MAX_VGPRS_PER_WF;
  default:
    return 0;
  }
}

[[nodiscard]] bool arch_has_accvgpr(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return HasAccVgpr<cdna1::Isa>;
  case ROCJITSU_CODE_ARCH_CDNA2:
    return HasAccVgpr<cdna2::Isa>;
  case ROCJITSU_CODE_ARCH_CDNA3:
    return HasAccVgpr<cdna3::Isa>;
  case ROCJITSU_CODE_ARCH_CDNA4:
    return HasAccVgpr<cdna4::Isa>;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return HasAccVgpr<gfx1250::Isa>;
  case ROCJITSU_CODE_ARCH_RDNA1:
    return HasAccVgpr<rdna1::Isa>;
  case ROCJITSU_CODE_ARCH_RDNA2:
    return HasAccVgpr<rdna2::Isa>;
  case ROCJITSU_CODE_ARCH_RDNA3:
    return HasAccVgpr<rdna3::Isa>;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return HasAccVgpr<rdna3_5::Isa>;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return HasAccVgpr<rdna4::Isa>;
  default:
    return false;
  }
}

[[nodiscard]] uint8_t choose_host_wavefront_size(uint8_t guest_wavefront_size,
                                                 rj_code_arch_t host_arch) {
  if (guest_wavefront_size == 64 && arch_supports_wave_size(host_arch, 64))
    return 64;
  if (guest_wavefront_size == 32 && arch_supports_wave_size(host_arch, 32))
    return 32;
  if (arch_supports_wave_size(host_arch, 64))
    return 64;
  return static_cast<uint8_t>(arch_default_wave_size(host_arch));
}

// -----------------------------------------------------------------------------
// ELF kernel-descriptor discovery.
// -----------------------------------------------------------------------------

[[nodiscard]] bool kernel_descriptor_symbol(const Elf64_Sym &sym, const char *strtab,
                                            size_t strtab_size) {
  if (sym.st_size != sizeof(KD))
    return false;

  // AMDHSA kernel descriptors are global object symbols. Size alone is not a
  // durable signal because unrelated data objects can also be 64 bytes.
  if (elf_symbol_type(sym.st_info) != kElfSymbolTypeObject ||
      elf_symbol_bind(sym.st_info) != kElfSymbolBindGlobal)
    return false;

  // AMDHSA descriptors are named "<kernel>.kd". An unnamed 64-byte global
  // object is ambiguous, so require the ABI suffix instead of treating stripped
  // or minimized symbol records as descriptors.
  if (strtab == nullptr || strtab_size == 0 || sym.st_name == 0)
    return false;
  if (sym.st_name >= strtab_size)
    return false;

  const char *name = strtab + sym.st_name;
  const size_t len = strnlen(name, strtab_size - sym.st_name);
  return len > 3 && std::strcmp(name + len - 3, ".kd") == 0;
}

[[nodiscard]] std::optional<uint64_t> text_vaddr_for_section(uint64_t text_offset,
                                                             uint64_t text_size,
                                                             const Elf64_Ehdr &ehdr,
                                                             const Elf64_Shdr *shdr) {
  for (int i = 0; i < ehdr.e_shnum; ++i) {
    if (shdr[i].sh_offset == text_offset && shdr[i].sh_size == text_size)
      return shdr[i].sh_addr;
  }
  return std::nullopt;
}

[[nodiscard]] uint64_t executable_code_range_size(uint64_t text_vaddr, uint64_t text_size,
                                                  const Elf64_Ehdr &ehdr, const Elf64_Shdr *shdr) {
  constexpr uint64_t max_u64 = std::numeric_limits<uint64_t>::max();
  if (text_vaddr > max_u64 - text_size)
    return 0;

  uint64_t code_end = text_vaddr + text_size;

  // DBT places expansion bodies in .rj_translations and addresses them as a
  // .text-relative continuation. Include executable sections at or after .text
  // so a redirected kernel descriptor entry can still be parsed.
  for (int i = 0; i < ehdr.e_shnum; ++i) {
    constexpr uint64_t executable_load_flags = SHF_ALLOC | SHF_EXECINSTR;
    if ((shdr[i].sh_flags & executable_load_flags) != executable_load_flags)
      continue;
    if (shdr[i].sh_addr < text_vaddr)
      continue;
    if (shdr[i].sh_addr > max_u64 - shdr[i].sh_size)
      continue;
    code_end = std::max(code_end, shdr[i].sh_addr + shdr[i].sh_size);
  }

  return code_end - text_vaddr;
}

using KernelDescriptorVisitor =
    std::function<void(uint64_t descriptor_file_offset, uint64_t entry_text_offset,
                       std::string_view symbol_name, const KD &desc)>;

void visit_kernel_descriptors(std::span<const uint8_t> image, uint64_t text_offset,
                              uint64_t text_size, const KernelDescriptorVisitor &callback) {
  if (image.size() < sizeof(Elf64_Ehdr))
    return;

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(image.data());
  if (ehdr->e_shoff + static_cast<uint64_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr) > image.size())
    return;

  const auto *shdr = reinterpret_cast<const Elf64_Shdr *>(image.data() + ehdr->e_shoff);
  auto text_vaddr = text_vaddr_for_section(text_offset, text_size, *ehdr, shdr);
  if (!text_vaddr)
    return;
  const uint64_t code_range_size = executable_code_range_size(*text_vaddr, text_size, *ehdr, shdr);

  // .symtab and .dynsym may both describe the same descriptor. Translation is
  // keyed by descriptor bytes, so visit each file offset once.
  std::unordered_set<uint64_t> seen_descriptor_offsets;
  for (int i = 0; i < ehdr->e_shnum; ++i) {
    if (shdr[i].sh_type != SHT_SYMTAB && shdr[i].sh_type != SHT_DYNSYM)
      continue;
    if (shdr[i].sh_offset + shdr[i].sh_size > image.size() || shdr[i].sh_entsize == 0)
      continue;
    if (shdr[i].sh_entsize != sizeof(Elf64_Sym))
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
    const size_t nsyms = shdr[i].sh_size / shdr[i].sh_entsize;
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
      if (!seen_descriptor_offsets.insert(file_off).second)
        continue;

      const auto *desc = reinterpret_cast<const KD *>(image.data() + file_off);
      const int64_t entry_vaddr_signed =
          static_cast<int64_t>(symtab[j].st_value) + desc->kernel_code_entry_byte_offset;
      if (entry_vaddr_signed < 0)
        continue;
      const uint64_t entry_vaddr = static_cast<uint64_t>(entry_vaddr_signed);
      if (entry_vaddr < *text_vaddr || entry_vaddr >= *text_vaddr + code_range_size)
        continue;

      const uint64_t entry_text_offset = entry_vaddr - *text_vaddr;
      const char *name = strtab + symtab[j].st_name;
      const size_t name_len = strnlen(name, strtab_size - symtab[j].st_name);
      callback(file_off, entry_text_offset, std::string_view(name, name_len), *desc);
    }
  }
}

// -----------------------------------------------------------------------------
// Kernel descriptor field helpers.
// -----------------------------------------------------------------------------

[[nodiscard]] uint8_t kernel_wavefront_size(rj_code_arch_t guest_arch, const KD &desc) {
  // CDNA kernels are Wave64 in the code objects currently translated here.
  if (is_cdna_arch(guest_arch))
    return 64;

  // gfx1250 is GFX12-like for descriptor format, but unlike RDNA4 targets it
  // is Wave32-only. Treat malformed descriptors that leave the RDNA wave32 bit
  // clear as Wave32 rather than inventing unsupported Wave64 execution.
  if (is_gfx1250_arch(guest_arch))
    return 32;

  // RDNA descriptors opt into Wave32 with ENABLE_WAVEFRONT_SIZE32. If the bit is
  // clear, launch hardware interprets the descriptor as Wave64.
  if (is_rdna_arch(guest_arch)) {
    const bool wave32 = AMDHSA_BITS_GET(desc.kernel_code_properties,
                                        kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
    return wave32 ? 32 : 64;
  }

  return 64;
}

[[nodiscard]] uint32_t user_sgpr_count(const KD &desc) {
  return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);
}

[[nodiscard]] uint32_t kernarg_preload_length(const KD &desc) {
  return AMDHSA_BITS_GET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH);
}

[[nodiscard]] uint32_t kernarg_preload_offset(const KD &desc) {
  return AMDHSA_BITS_GET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_OFFSET);
}

[[nodiscard]] int16_t workgroup_id_sgpr(const KD &desc, uint32_t dimension) {
  const uint32_t rsrc2 = desc.compute_pgm_rsrc2;
  const bool enabled[3] = {
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0,
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0,
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0,
  };

  // Workgroup-id SGPRs are allocated immediately after the user-SGPR block, in
  // X/Y/Z order, but disabled dimensions consume no SGPR. That makes the
  // descriptor-selected register number a direct function of USER_SGPR_COUNT
  // plus the enabled dimensions that precede the requested one.
  uint32_t sgpr = user_sgpr_count(desc);
  for (uint32_t i = 0; i < 3; ++i) {
    if (!enabled[i])
      continue;
    if (i == dimension)
      return static_cast<int16_t>(sgpr);
    ++sgpr;
  }
  return -1;
}

[[nodiscard]] bool uses_gfx90a_accum_offset(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA2 || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
         arch == ROCJITSU_CODE_ARCH_CDNA4;
}

[[nodiscard]] uint32_t descriptor_vgpr_granularity_for_wavefront(rj_code_arch_t arch,
                                                                 uint32_t wavefront_size) {
  // This is the AMDHSA kernel-descriptor encoding granularity for
  // COMPUTE_PGM_RSRC1.GRANULATED_WORKITEM_VGPR_COUNT, not the physical VGPR
  // allocation block from the ISA manuals. For example, RDNA3/RDNA4 manuals
  // describe Wave64 physical allocation in blocks of 8 VGPRs (or 12 on
  // 1536-VGPR/SIMD parts), while the AMDHSA descriptor table encodes
  // GFX10-GFX12 Wave64 as max(0, ceil(vgprs_used / 4) - 1).
  //
  // gfx1250 is wave32-only and IREE gfx1250 HSACOs encode 263/276 VGPR kernel
  // metadata as RSRC1 values 16/17, matching 16-register descriptor granules.
  //
  // If/when occupancy modeling needs the physical allocation block size, add a
  // separate helper for that policy. Reusing this descriptor helper for
  // occupancy would mix two different hardware contracts.
  if (arch == ROCJITSU_CODE_ARCH_CDNA1)
    return 4;
  if (is_cdna_arch(arch))
    return 8;
  if (is_gfx1250_arch(arch))
    return 16;
  if (is_rdna_arch(arch))
    return wavefront_size == 32 ? 8 : 4;
  return 1;
}

[[nodiscard]] uint32_t granulated_count_to_registers(uint32_t granulated, uint32_t granularity) {
  return (granulated + 1) * std::max(granularity, 1u);
}

[[nodiscard]] uint32_t register_count_to_granulated(uint32_t registers, uint32_t granularity) {
  granularity = std::max(granularity, 1u);
  if (registers == 0)
    return 0;
  return (registers + granularity - 1) / granularity - 1;
}

[[nodiscard]] uint32_t max_vgpr_granulated_field(rj_code_arch_t arch, uint32_t wavefront_size) {
  // gfx1201 accepts the GFX12 high-bank MODE bits, but launches hang when the
  // descriptor VGPR field exceeds 31. Keep the semantic VGPR requirement in
  // target_vgpr_count for diagnostics while capping the hardware launch field.
  if (arch == ROCJITSU_CODE_ARCH_RDNA4 && wavefront_size == 32)
    return 31;
  return kMaxVgprGranulatedField;
}

[[nodiscard]] uint32_t align_up_register_count(uint32_t count, uint32_t alignment) {
  alignment = std::max(alignment, 1u);
  return ((count + alignment - 1u) / alignment) * alignment;
}

[[nodiscard]] uint16_t accum_vgpr_base(const KD &desc, rj_code_arch_t guest_arch) {
  if (!uses_gfx90a_accum_offset(guest_arch) || !arch_has_accvgpr(guest_arch))
    return 0;

  const uint32_t encoded =
      AMDHSA_BITS_GET(desc.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET);
  // The descriptor encoding is one less than the actual first AccVGPR offset:
  // field value 0 means acc0 starts at unified VGPR index 4, value 1 means 8,
  // and so on in groups of four registers.
  return static_cast<uint16_t>((encoded + 1) * 4);
}

[[nodiscard]] uint16_t ttmp_scalar_operand(uint16_t ttmp) {
  return static_cast<uint16_t>(kScalarOperandTtmpBase + ttmp);
}

// -----------------------------------------------------------------------------
// Kernel-entry prologue construction.
// -----------------------------------------------------------------------------

struct UserSgprLayout {
  uint32_t fixed_count = 0;
  std::optional<uint16_t> kernarg_segment_ptr_sgpr;
};

[[nodiscard]] UserSgprLayout user_sgpr_layout(const KD &desc) {
  const uint32_t kcp = desc.kernel_code_properties;
  UserSgprLayout layout;
  auto append = [&layout](uint32_t count) { layout.fixed_count += count; };

  if (AMDHSA_BITS_GET(kcp, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) != 0)
    append(4);
  if (AMDHSA_BITS_GET(kcp, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR) != 0)
    append(2);
  if (AMDHSA_BITS_GET(kcp, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR) != 0)
    append(2);
  if (AMDHSA_BITS_GET(kcp, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR) != 0) {
    layout.kernarg_segment_ptr_sgpr = static_cast<uint16_t>(layout.fixed_count);
    append(2);
  }
  if (AMDHSA_BITS_GET(kcp, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID) != 0)
    append(2);
  if (AMDHSA_BITS_GET(kcp, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT) != 0)
    append(2);
  if (AMDHSA_BITS_GET(kcp, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE) != 0)
    append(1);

  return layout;
}

[[nodiscard]] uint32_t smem_load_alignment(uint32_t dwords) { return dwords >= 8 ? 8 : dwords; }

[[nodiscard]] uint8_t smem_s_load_op_for_dwords(uint32_t dwords) {
  switch (dwords) {
  case 1:
    return 0;
  case 2:
    return 1;
  case 4:
    return 2;
  case 8:
    return 3;
  case 16:
    return 4;
  default:
    return 0;
  }
}

[[nodiscard]] uint32_t choose_smem_preload_dwords(uint32_t dst_sgpr, uint32_t remaining) {
  constexpr std::array<uint32_t, 5> candidates = {16, 8, 4, 2, 1};
  for (uint32_t dwords : candidates) {
    if (remaining < dwords)
      continue;
    if (dst_sgpr % smem_load_alignment(dwords) == 0)
      return dwords;
  }
  return 1;
}

void append_rdna4_smem_load(std::vector<uint32_t> &words, uint16_t sdata, uint16_t sbase,
                            uint32_t dwords, uint32_t byte_offset) {
  const uint8_t op = smem_s_load_op_for_dwords(dwords);
  words.push_back(
      (static_cast<uint32_t>(sbase & 0x3Fu)) | (static_cast<uint32_t>(sdata & 0x7Fu) << 6) |
      (static_cast<uint32_t>(op & 0x3Fu) << 13) | (static_cast<uint32_t>(kSmemEncoding) << 26));
  words.push_back((byte_offset & 0x00FF'FFFFu) | (static_cast<uint32_t>(kSmemSoffsetNone) << 25));
}

void append_rdna4_kernarg_preload_prologue(std::vector<uint32_t> &words, const KD &desc) {
  const uint32_t preload_length = kernarg_preload_length(desc);
  if (preload_length == 0)
    return;

  const UserSgprLayout layout = user_sgpr_layout(desc);
  if (!layout.kernarg_segment_ptr_sgpr ||
      layout.fixed_count + preload_length > user_sgpr_count(desc))
    return;

  uint32_t loaded = 0;
  while (loaded < preload_length) {
    const uint32_t dst_sgpr = layout.fixed_count + loaded;
    const uint32_t dwords = choose_smem_preload_dwords(dst_sgpr, preload_length - loaded);
    const uint32_t byte_offset = (kernarg_preload_offset(desc) + loaded) * sizeof(uint32_t);
    append_rdna4_smem_load(words, static_cast<uint16_t>(dst_sgpr), *layout.kernarg_segment_ptr_sgpr,
                           dwords, byte_offset);
    loaded += dwords;
  }
  words.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
  words.push_back(pack_sopp(kSoppWaitKmcnt, 0));
}

[[nodiscard]] bool can_materialize_rdna4_kernarg_preload(const KD &desc) {
  const uint32_t preload_length = kernarg_preload_length(desc);
  if (preload_length == 0)
    return false;

  const UserSgprLayout layout = user_sgpr_layout(desc);
  return layout.kernarg_segment_ptr_sgpr &&
         layout.fixed_count + preload_length <= user_sgpr_count(desc);
}

void append_salu_write(std::vector<uint32_t> &words, uint32_t word, rj_code_arch_t host_arch) {
  words.push_back(word);
  // The prologue feeds the original kernel entry, whose first few instructions
  // may immediately consume these SGPRs. GFX12 needs an explicit ALU delay for
  // scalar producer/consumer dependencies; entry prologues bypass the normal
  // instruction-level HazardTracker, so serialize each injected scalar write
  // before the patcher appends the branch back to the original entry.
  words.push_back(build_s_delay_alu(kDelayAluSaluDep1, host_arch));
}

void append_rdna4_workgroup_grid_prologue(std::vector<uint32_t> &words, const KD &desc,
                                          rj_code_arch_t host_arch, int16_t rdna4_grid_x_sgpr,
                                          int16_t rdna4_grid_yz_sgpr) {
  const uint16_t shift16 = scalar_positive_inline_u32(16);
  const int16_t sgpr_wg_id_x = workgroup_id_sgpr(desc, 0);
  const int16_t sgpr_wg_id_y = workgroup_id_sgpr(desc, 1);
  const int16_t sgpr_wg_id_z = workgroup_id_sgpr(desc, 2);
  const uint16_t rdna4_grid_yz_src = rdna4_grid_yz_sgpr >= 0
                                         ? static_cast<uint16_t>(rdna4_grid_yz_sgpr)
                                         : ttmp_scalar_operand(kTtmpRdna4GridYz);

  if (sgpr_wg_id_x >= 0) {
    append_salu_write(words,
                      build_s_mov_b32(static_cast<uint16_t>(sgpr_wg_id_x),
                                      ttmp_scalar_operand(kTtmpRdna4GridX), host_arch),
                      host_arch);
  }
  if (rdna4_grid_x_sgpr >= 0 && rdna4_grid_x_sgpr != sgpr_wg_id_x) {
    append_salu_write(words,
                      build_s_mov_b32(static_cast<uint16_t>(rdna4_grid_x_sgpr),
                                      ttmp_scalar_operand(kTtmpRdna4GridX), host_arch),
                      host_arch);
  }
  if (rdna4_grid_yz_sgpr >= 0) {
    append_salu_write(words,
                      build_s_mov_b32(static_cast<uint16_t>(rdna4_grid_yz_sgpr),
                                      ttmp_scalar_operand(kTtmpRdna4GridYz), host_arch),
                      host_arch);
  }

  if (sgpr_wg_id_y >= 0) {
    const auto sgpr_y = static_cast<uint16_t>(sgpr_wg_id_y);
    // RDNA4 packs GridY into TTMP7[15:0]. Preserve CDNA's 32-bit SGPR contract
    // by zero-extending the low half without needing an extra temporary SGPR.
    append_salu_write(words, build_s_mov_b32(sgpr_y, rdna4_grid_yz_src, host_arch), host_arch);
    append_salu_write(words, build_s_lshl_b32(sgpr_y, sgpr_y, shift16, host_arch), host_arch);
    append_salu_write(words, build_s_lshr_b32(sgpr_y, sgpr_y, shift16, host_arch), host_arch);
  }

  if (sgpr_wg_id_z >= 0) {
    const auto sgpr_z = static_cast<uint16_t>(sgpr_wg_id_z);
    // RDNA4 packs GridZ into TTMP7[31:16]. CDNA code expects that value in its
    // descriptor-selected workgroup_id_z SGPR.
    append_salu_write(words, build_s_lshr_b32(sgpr_z, rdna4_grid_yz_src, shift16, host_arch),
                      host_arch);
  }
}

void append_rdna4_vgpr_msb_entry_reset(std::vector<uint32_t> &words) {
  constexpr uint8_t kOpSSetregImm32B32 = 19;
  constexpr uint8_t kHwregMode = 1;
  constexpr uint8_t kVgprMsbModeSize = 8;
  const uint16_t hwreg = build_hwreg(kHwregMode, amdgpu::VGPR_MSB_MODE_SHIFT, kVgprMsbModeSize);
  words.push_back(build_sopk(kOpSSetregImm32B32, hwreg));
  words.push_back(0);
}

[[nodiscard]] std::vector<uint32_t>
build_kernel_entry_prologue(const KD &src, rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                            int16_t rdna4_grid_x_sgpr, int16_t rdna4_grid_yz_sgpr) {
  std::vector<uint32_t> words;

  // Kernel-entry register initialization ABI notes:
  // - CDNA and gfx1250 descriptors may request workgroup_id_x/y/z as SGPRs
  //   immediately after the user-SGPR block. Source hardware initializes those
  //   SGPRs before entering the kernel.
  // - GFX1250 hardware initializes VGPR_MSB mode to zero for a fresh kernel
  //   wave. RDNA4 exposes the equivalent through MODE[19:12], which can persist
  //   across translated dispatches unless each kernel resets it at entry.
  // - RDNA1-RDNA3 still have descriptor-controlled workgroup-id SGPR setup for
  //   the cases translated here, so no prologue is needed for those targets.
  // - RDNA4 provides the current workgroup-grid payload through TTMP registers
  //   instead: GridX in TTMP9, GridY in TTMP7[15:0], and GridZ in TTMP7[31:16].
  //   When translating CDNA or gfx1250 to RDNA4, materialize the guest-selected
  //   SGPRs from that payload once at kernel entry so the instruction stream can
  //   keep using the original guest SGPR numbering.
  // - gfx1250 descriptors can request AMDHSA kernarg preload, which places
  //   selected kernarg dwords into user SGPRs after the fixed ABI prefix. RDNA4
  //   hardware does not provide the same preload contents for translated gfx1250
  //   launches, so rebuild that payload from the kernarg pointer before the
  //   original entry observes it.
  // - Scratch/private-segment initialization is descriptor-driven today. If a
  //   future target needs SGPR-based scratch setup, it should be appended here
  //   and represented in KdTranslation::prologue_words, not hidden in the patcher.
  if (is_gfx1250_arch(guest_arch) && host_arch == ROCJITSU_CODE_ARCH_RDNA4)
    append_rdna4_kernarg_preload_prologue(words, src);
  if ((is_cdna_arch(guest_arch) &&
       (host_arch == ROCJITSU_CODE_ARCH_RDNA4 || host_arch == ROCJITSU_CODE_ARCH_GFX1250)) ||
      (is_gfx1250_arch(guest_arch) && host_arch == ROCJITSU_CODE_ARCH_RDNA4))
    append_rdna4_workgroup_grid_prologue(words, src, host_arch, rdna4_grid_x_sgpr,
                                         rdna4_grid_yz_sgpr);
  if (is_gfx1250_arch(guest_arch) && host_arch == ROCJITSU_CODE_ARCH_RDNA4)
    append_rdna4_vgpr_msb_entry_reset(words);

  return words;
}

// -----------------------------------------------------------------------------
// Descriptor translation.
// -----------------------------------------------------------------------------

void append_descriptor_error(KdTranslation &result, std::string message) {
  result.diagnostics.push_back({.severity = DiagnosticSeverity::Error,
                                .kind = DiagnosticKind::KernelDescriptor,
                                .guest_offset = std::nullopt,
                                .mnemonic = {},
                                .message = std::move(message),
                                .required_work = {}});
  result.supported = false;
}

[[nodiscard]] uint32_t clamp_granulated(uint32_t value, uint32_t max_value, KdTranslation &result,
                                        const char *field_name) {
  if (value <= max_value)
    return value;

  append_descriptor_error(
      result, std::string(field_name) +
                  " exceeds descriptor field width; resource virtualization is not implemented");
  return max_value;
}

[[nodiscard]] KdTranslation
translate_one_descriptor(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                         uint64_t descriptor_file_offset, uint64_t entry_text_offset,
                         std::string_view symbol_name, const KD &src,
                         const KernelDescriptorTranslationOptions &options) {
  KdTranslation result;
  result.descriptor_file_offset = descriptor_file_offset;
  result.entry_text_offset = entry_text_offset;
  result.symbol_name = std::string(symbol_name);

  // The source descriptor encodes the guest launch wave size. The target
  // descriptor must request a wave size the host can actually launch. We do not
  // emulate wave-size mismatches in the instruction stream yet, so a mismatch is
  // reported as unsupported even though the best-effort descriptor fields are
  // still computed for diagnostics and patching experiments.
  result.guest_wavefront_size = kernel_wavefront_size(guest_arch, src);
  result.host_wavefront_size = choose_host_wavefront_size(result.guest_wavefront_size, host_arch);
  result.target_wave_size = result.host_wavefront_size;
  result.force_wave64 = is_rdna_arch(host_arch) && result.target_wave_size == 64;
  if (result.guest_wavefront_size != result.host_wavefront_size) {
    append_descriptor_error(result,
                            "guest and host wavefront sizes differ; descriptor was translated "
                            "but instruction-level wave-size emulation is not implemented");
  }

  // CDNA MFMA kernels may address accumulator registers through a separate
  // AccVGPR file. RDNA targets do not expose that file in the same way, so the
  // semantic translator remaps AccVGPR references into the unified VGPR space.
  // The descriptor translator records the first unified VGPR index that must be
  // reserved for those remapped accumulator registers.
  result.accvgpr_base = accum_vgpr_base(src, guest_arch);
  result.target_accvgpr_base = result.accvgpr_base;

  // VGPR descriptor fields do not store raw register counts. They store
  // "granulated count", meaning (register_count / architecture_granularity) - 1
  // rounded up. The granularity depends on both ISA family and wave size, so the
  // source count must be decoded with the guest granularity and re-encoded with
  // the host granularity.
  const uint32_t guest_vgpr_granularity =
      descriptor_vgpr_granularity_for_wavefront(guest_arch, result.guest_wavefront_size);
  const uint32_t host_vgpr_granularity =
      descriptor_vgpr_granularity_for_wavefront(host_arch, result.host_wavefront_size);

  const uint32_t guest_vgpr_granulated =
      AMDHSA_BITS_GET(src.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  result.guest_vgpr_allocation_count =
      granulated_count_to_registers(guest_vgpr_granulated, guest_vgpr_granularity);
  if (arch_has_accvgpr(guest_arch) && result.accvgpr_base != 0 &&
      result.guest_vgpr_allocation_count > result.accvgpr_base) {
    result.guest_agpr_count = result.guest_vgpr_allocation_count - result.accvgpr_base;
  }
  result.guest_vgpr_count = result.guest_vgpr_allocation_count - result.guest_agpr_count;

  // Start from the guest's ordinary VGPR count. Keep the descriptor allocation
  // count separate: CDNA kernels with AccVGPRs encode a unified allocation end
  // in COMPUTE_PGM_RSRC1, but liveness and scratch allocation need the ordinary
  // VGPR count only.
  //
  // - CDNA-to-RDNA unifies AccVGPRs into the ordinary VGPR namespace, so the
  //   target needs enough VGPRs to cover the remapped accumulator window.
  //
  // - CDNA-to-CDNA keeps a real AccVGPR bank. On GFX90A/GFX942/GFX950 the
  //   ACCUM_OFFSET descriptor field selects where that bank starts in the
  //   unified register file. If a semantic lowering needs new ordinary VGPRs
  //   at or above the original offset, merely increasing ordinary VGPRs is not enough:
  //   those temporary VGPRs would alias a0, a1, ... at runtime. Move the
  //   accumulator base up and preserve the original accumulator-window size.
  //
  // - instruction lowering may request additional scratch temporaries through
  //   options.minimum_vgprs.
  uint32_t required_vgprs = result.guest_vgpr_count;
  if (options.target_vgpr_count_override != 0) {
    if (options.target_vgpr_count_override < result.guest_vgpr_count &&
        !(is_gfx1250_arch(guest_arch) && host_arch == ROCJITSU_CODE_ARCH_RDNA4)) {
      append_descriptor_error(result,
                              "target VGPR count override below guest allocation requires "
                              "unsupported VGPR virtualization");
    }
    required_vgprs = options.target_vgpr_count_override;
  }
  result.target_agpr_count = arch_has_accvgpr(host_arch) ? result.guest_agpr_count : 0;
  uint32_t required_vgpr_allocation = result.guest_vgpr_allocation_count;
  if (arch_has_accvgpr(guest_arch) && !arch_has_accvgpr(host_arch) && result.accvgpr_base != 0)
    required_vgprs = std::max(required_vgprs, result.accvgpr_base + result.guest_agpr_count);
  required_vgprs = std::max(required_vgprs, options.minimum_vgprs);
  if (arch_has_accvgpr(guest_arch) && arch_has_accvgpr(host_arch) &&
      uses_gfx90a_accum_offset(host_arch) && result.accvgpr_base != 0) {
    if (result.target_agpr_count != 0 && options.minimum_vgprs > result.accvgpr_base) {
      result.target_accvgpr_base = util::align_up(options.minimum_vgprs, 4u);
      if (result.target_accvgpr_base > 256) {
        append_descriptor_error(result,
                                "required AccVGPR offset exceeds GFX90A descriptor field range; "
                                "AccVGPR base virtualization is not implemented");
      }
    }
  }
  if (arch_has_accvgpr(host_arch) && result.target_agpr_count != 0) {
    required_vgpr_allocation =
        std::max(required_vgprs, result.target_accvgpr_base + result.target_agpr_count);
  } else {
    required_vgpr_allocation = required_vgprs;
  }
  result.host_vgpr_count = required_vgprs;
  result.host_vgpr_allocation_count = required_vgpr_allocation;
  result.target_vgpr_count = required_vgprs;
  result.target_vgpr_allocation_count = required_vgpr_allocation;
  const uint32_t max_host_vgprs = arch_max_vgprs(host_arch);
  if (max_host_vgprs != 0 && required_vgpr_allocation > max_host_vgprs) {
    std::string message = "required VGPR allocation " + std::to_string(required_vgpr_allocation) +
                          " exceeds target limit " + std::to_string(max_host_vgprs) +
                          " for kernel " + result.symbol_name +
                          "; guest_vgpr_count=" + std::to_string(result.guest_vgpr_count) +
                          ", target_vgpr_count=" + std::to_string(result.target_vgpr_count);
    if (is_gfx1250_arch(guest_arch) && host_arch == ROCJITSU_CODE_ARCH_RDNA4)
      message += "; gfx1250 high-bank virtualization above the RDNA4 launch limit is not "
                 "implemented for this descriptor";
    else
      message += "; spill tiers are not implemented for this descriptor";
    append_descriptor_error(result, std::move(message));
  }

  const uint32_t target_vgpr_granulated =
      register_count_to_granulated(required_vgpr_allocation, host_vgpr_granularity);
  const uint32_t max_target_vgpr_granulated =
      max_vgpr_granulated_field(host_arch, result.host_wavefront_size);
  if (host_arch == ROCJITSU_CODE_ARCH_RDNA4 && result.host_wavefront_size == 32) {
    result.target_vgpr_granulated = std::min(target_vgpr_granulated, max_target_vgpr_granulated);
  } else {
    result.target_vgpr_granulated =
        clamp_granulated(target_vgpr_granulated, max_target_vgpr_granulated, result,
                         "GRANULATED_WORKITEM_VGPR_COUNT");
  }

  // SGPR counts are stored as a granulated value in COMPUTE_PGM_RSRC1. The
  // descriptor granularity is fixed at eight SGPRs for the architectures
  // handled here.
  const uint32_t guest_sgpr_granulated =
      AMDHSA_BITS_GET(src.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  result.guest_sgpr_count = granulated_count_to_registers(guest_sgpr_granulated, 8);
  uint32_t abi_sgpr_floor = user_sgpr_count(src);
  for (uint32_t dim = 0; dim < 3; ++dim) {
    if (const int16_t wg_id_sgpr = workgroup_id_sgpr(src, dim); wg_id_sgpr >= 0)
      abi_sgpr_floor = std::max(abi_sgpr_floor, static_cast<uint32_t>(wg_id_sgpr) + 1u);
  }
  result.target_abi_sgpr_count = abi_sgpr_floor;
  result.host_sgpr_count =
      std::max({result.guest_sgpr_count, options.minimum_sgprs, abi_sgpr_floor});
  result.target_source_sgpr_count = result.host_sgpr_count;
  if (is_gfx1250_arch(guest_arch) && host_arch == ROCJITSU_CODE_ARCH_RDNA4) {
    auto allocate_capture_sgpr = [&](int16_t &sgpr, const char *payload_name) {
      if (result.host_sgpr_count < arch_max_sgprs(host_arch)) {
        sgpr = static_cast<int16_t>(result.host_sgpr_count);
        ++result.host_sgpr_count;
        return;
      }
      append_descriptor_error(result, "required SGPR count leaves no room for RDNA4 " +
                                          std::string(payload_name) + " capture SGPR");
    };
    allocate_capture_sgpr(result.rdna4_grid_x_sgpr, "grid-x");
    allocate_capture_sgpr(result.rdna4_grid_yz_sgpr, "grid-yz");
    if (result.supported) {
      constexpr uint32_t kScratchSgprs = 8;
      const uint32_t kScratchSgprLimit = arch_max_sgprs(host_arch);
      const uint32_t scratch_base = align_up_register_count(result.host_sgpr_count, 2);
      result.host_sgpr_count = std::max(result.host_sgpr_count,
                                        std::min(kScratchSgprLimit, scratch_base + kScratchSgprs));
    }
  }
  result.target_sgpr_count = result.host_sgpr_count;
  if (arch_max_sgprs(host_arch) != 0 && result.host_sgpr_count > arch_max_sgprs(host_arch)) {
    append_descriptor_error(result, "required SGPR count exceeds target limit; spill tiers are not "
                                    "implemented for this descriptor");
  }

  result.target_sgpr_granulated = register_count_to_granulated(result.host_sgpr_count, 8);
  result.target_sgpr_granulated =
      clamp_granulated(result.target_sgpr_granulated, kMaxSgprGranulatedField, result,
                       "GRANULATED_WAVEFRONT_SGPR_COUNT");

  // LDS/private sizes are copied from the source descriptor and extended by
  // explicit lowering addends. The addend ranges are exposed to semantic
  // lowerings as spill zones.
  result.target_private_size = src.private_segment_fixed_size;
  if (options.private_segment_fixed_size_addend != 0) {
    result.private_spill_zone_base = (src.private_segment_fixed_size + 3u) & ~3u;
    result.private_spill_zone_bytes = options.private_segment_fixed_size_addend;
    result.target_private_size =
        result.private_spill_zone_base + options.private_segment_fixed_size_addend;
  }
  result.target_lds_size = src.group_segment_fixed_size + options.group_segment_fixed_size_addend;
  if (options.group_segment_fixed_size_addend != 0) {
    result.lds_spill_zone_base = src.group_segment_fixed_size;
    result.lds_spill_zone_bytes = options.group_segment_fixed_size_addend;
  }

  // The fixed user-SGPR prefix is preserved. Workgroup-id SGPRs are derived
  // separately from the enable bits because they are allocated immediately after
  // this prefix and are not included in USER_SGPR_COUNT.
  result.target_user_sgpr_count = user_sgpr_count(src);

  if (is_gfx1250_arch(guest_arch) && host_arch == ROCJITSU_CODE_ARCH_RDNA4 &&
      kernarg_preload_length(src) != 0) {
    const UserSgprLayout layout = user_sgpr_layout(src);
    if (!layout.kernarg_segment_ptr_sgpr) {
      append_descriptor_error(
          result, "gfx1250 kernarg preload DBT requires ENABLE_SGPR_KERNARG_SEGMENT_PTR");
    } else if (layout.fixed_count + kernarg_preload_length(src) > user_sgpr_count(src)) {
      append_descriptor_error(result, "gfx1250 kernarg preload exceeds declared user SGPR count");
    }
  }
  result.clears_kernarg_preload = is_gfx1250_arch(guest_arch) &&
                                  host_arch == ROCJITSU_CODE_ARCH_RDNA4 &&
                                  can_materialize_rdna4_kernarg_preload(src);

  // Descriptor ABI fixes that require instructions, not bitfield changes, are
  // emitted as prologue words. CodeObjectPatcher decides where to place them and
  // redirects the kernel descriptor entry point if this vector is non-empty.
  result.prologue_words = build_kernel_entry_prologue(
      src, guest_arch, host_arch, result.rdna4_grid_x_sgpr, result.rdna4_grid_yz_sgpr);
  return result;
}

[[nodiscard]] KernelDescriptorTranslationOptions
options_for_entry(const KernelDescriptorTranslationOptions &options, uint64_t entry_text_offset) {
  KernelDescriptorTranslationOptions entry_options = options;
  for (const KernelDescriptorResourceOverride &override : options.kernel_overrides) {
    if (override.entry_text_offset != entry_text_offset)
      continue;
    entry_options.minimum_vgprs = std::max(entry_options.minimum_vgprs, override.minimum_vgprs);
    entry_options.target_vgpr_count_override =
        std::max(entry_options.target_vgpr_count_override, override.target_vgpr_count_override);
    entry_options.minimum_sgprs = std::max(entry_options.minimum_sgprs, override.minimum_sgprs);
    entry_options.group_segment_fixed_size_addend = std::max(
        entry_options.group_segment_fixed_size_addend, override.group_segment_fixed_size_addend);
    entry_options.private_segment_fixed_size_addend =
        std::max(entry_options.private_segment_fixed_size_addend,
                 override.private_segment_fixed_size_addend);
  }
  entry_options.kernel_overrides = {};
  return entry_options;
}

} // namespace

KernelDescriptorTranslator::KernelDescriptorTranslator(rj_code_arch_t guest_arch,
                                                       rj_code_arch_t host_arch)
    : guest_arch_(guest_arch), host_arch_(host_arch) {}

std::vector<KdTranslation> KernelDescriptorTranslator::translate_image(
    std::span<const uint8_t> image, uint64_t text_offset, uint64_t text_size,
    const KernelDescriptorTranslationOptions &options) const {
  std::vector<KdTranslation> translations;

  visit_kernel_descriptors(image, text_offset, text_size,
                           [&](uint64_t descriptor_file_offset, uint64_t entry_text_offset,
                               std::string_view symbol_name, const KD &src) {
                             KD desc{};
                             std::memcpy(&desc, &src, sizeof(desc));
                             translations.push_back(translate_one_descriptor(
                                 guest_arch_, host_arch_, descriptor_file_offset, entry_text_offset,
                                 symbol_name, desc, options_for_entry(options, entry_text_offset)));
                           });

  return translations;
}

std::optional<KdTranslation> KernelDescriptorTranslator::translate_descriptor(
    std::span<const uint8_t> image, uint64_t descriptor_file_offset, uint64_t entry_text_offset,
    const KernelDescriptorTranslationOptions &options) const {
  if (descriptor_file_offset > image.size() || image.size() - descriptor_file_offset < sizeof(KD))
    return std::nullopt;

  KD desc{};
  std::memcpy(&desc, image.data() + descriptor_file_offset, sizeof(desc));
  return translate_one_descriptor(guest_arch_, host_arch_, descriptor_file_offset,
                                  entry_text_offset, {}, desc, options);
}

} // namespace rocjitsu
