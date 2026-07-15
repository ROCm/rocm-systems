// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/isa_traits.h"
#include "util/bit.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

namespace rocjitsu {

namespace {

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

static_assert(sizeof(KD) == 64, "AMDHSA kernel descriptor size changed");

constexpr uint32_t kMaxVgprGranulatedField = 63;
constexpr uint32_t kMaxSgprGranulatedField = 15;
constexpr uint64_t kKernargPreloadSkipBytes = 256;
constexpr uint16_t kScalarOperandTtmpBase = 108;
constexpr uint16_t kTtmpRdna4GridYz = 7;
constexpr uint16_t kTtmpRdna4GridX = 9;

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

using KernelDescriptorVisitor = std::function<void(uint64_t descriptor_file_offset,
                                                   uint64_t entry_text_offset, const KD &desc)>;

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
  constexpr uint64_t max_u64 = std::numeric_limits<uint64_t>::max();
  if (*text_vaddr > max_u64 - text_size)
    return;
  const uint64_t text_end = *text_vaddr + text_size;

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

      KD desc;
      std::memcpy(&desc, image.data() + file_off, sizeof(desc));
      const int64_t entry_vaddr_signed =
          static_cast<int64_t>(symtab[j].st_value) + desc.kernel_code_entry_byte_offset;

      if (entry_vaddr_signed < 0)
        continue;
      const uint64_t entry_vaddr = static_cast<uint64_t>(entry_vaddr_signed);
      if (entry_vaddr < *text_vaddr || entry_vaddr >= text_end)
        continue;

      const uint64_t entry_text_offset = entry_vaddr - *text_vaddr;
      callback(file_off, entry_text_offset, desc);
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

[[nodiscard]] bool uses_gfx10_plus_rsrc3(rj_code_arch_t arch) { return is_rdna_arch(arch); }

[[nodiscard]] uint32_t descriptor_vgpr_granularity_for_wavefront(rj_code_arch_t arch,
                                                                 uint32_t wavefront_size) {
  // This is the AMDHSA kernel-descriptor encoding granularity for
  // COMPUTE_PGM_RSRC1.GRANULATED_WORKITEM_VGPR_COUNT, not the physical VGPR
  // allocation block from the ISA manuals. For example, RDNA3/RDNA4 manuals
  // describe Wave64 physical allocation in blocks of 8 VGPRs (or 12 on
  // 1536-VGPR/SIMD parts), while the AMDHSA descriptor table encodes
  // GFX10-GFX12 Wave64 as max(0, ceil(vgprs_used / 4) - 1).
  //
  // If/when occupancy modeling needs the physical allocation block size, add a
  // separate helper for that policy. Reusing this descriptor helper for
  // occupancy would mix two different hardware contracts.
  if (arch == ROCJITSU_CODE_ARCH_CDNA1)
    return 4;
  if (is_cdna_arch(arch))
    return 8;
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
                                          rj_code_arch_t host_arch) {
  const uint16_t shift16 = scalar_positive_inline_u32(16);
  const int16_t sgpr_wg_id_x = workgroup_id_sgpr(desc, 0);
  const int16_t sgpr_wg_id_y = workgroup_id_sgpr(desc, 1);
  const int16_t sgpr_wg_id_z = workgroup_id_sgpr(desc, 2);

  if (sgpr_wg_id_x >= 0) {
    append_salu_write(words,
                      build_s_mov_b32(static_cast<uint16_t>(sgpr_wg_id_x),
                                      ttmp_scalar_operand(kTtmpRdna4GridX), host_arch),
                      host_arch);
  }

  if (sgpr_wg_id_y >= 0) {
    const auto sgpr_y = static_cast<uint16_t>(sgpr_wg_id_y);
    // RDNA4 packs GridY into TTMP7[15:0]. Preserve CDNA's 32-bit SGPR contract
    // by zero-extending the low half without needing an extra temporary SGPR.
    append_salu_write(words,
                      build_s_mov_b32(sgpr_y, ttmp_scalar_operand(kTtmpRdna4GridYz), host_arch),
                      host_arch);
    append_salu_write(words, build_s_lshl_b32(sgpr_y, sgpr_y, shift16, host_arch), host_arch);
    append_salu_write(words, build_s_lshr_b32(sgpr_y, sgpr_y, shift16, host_arch), host_arch);
  }

  if (sgpr_wg_id_z >= 0) {
    const auto sgpr_z = static_cast<uint16_t>(sgpr_wg_id_z);
    // RDNA4 packs GridZ into TTMP7[31:16]. CDNA code expects that value in its
    // descriptor-selected workgroup_id_z SGPR.
    append_salu_write(
        words, build_s_lshr_b32(sgpr_z, ttmp_scalar_operand(kTtmpRdna4GridYz), shift16, host_arch),
        host_arch);
  }
}

[[nodiscard]] std::vector<uint32_t>
build_kernel_entry_prologue(const KD &src, rj_code_arch_t guest_arch, rj_code_arch_t host_arch) {
  std::vector<uint32_t> words;

  // Kernel-entry register initialization ABI notes:
  // - CDNA descriptors may request workgroup_id_x/y/z as SGPRs immediately
  //   after the user-SGPR block. CDNA hardware initializes those SGPRs before
  //   entering the kernel.
  // - RDNA1-RDNA3 still have descriptor-controlled workgroup-id SGPR setup for
  //   the cases translated here, so no prologue is needed for those targets.
  // - RDNA4 provides the current workgroup-grid payload through TTMP registers
  //   instead: GridX in TTMP9, GridY in TTMP7[15:0], and GridZ in TTMP7[31:16].
  //   When translating CDNA to RDNA4, materialize the guest-selected SGPRs from
  //   that payload once at kernel entry so the instruction stream can keep using
  //   the original CDNA SGPR numbering.
  // - Scratch/private-segment initialization is descriptor-driven today. If a
  //   future target needs SGPR-based scratch setup, it should be appended here
  //   and represented in KdTranslation::prologue_words, not hidden in the patcher.
  if (is_cdna_arch(guest_arch) && host_arch == ROCJITSU_CODE_ARCH_RDNA4)
    append_rdna4_workgroup_grid_prologue(words, src, host_arch);

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
                         uint64_t descriptor_file_offset, uint64_t entry_text_offset, const KD &src,
                         const KernelDescriptorTranslationOptions &options) {
  KdTranslation result;
  result.descriptor_file_offset = descriptor_file_offset;
  result.entry_text_offset = entry_text_offset;
  result.target_entry_text_offset = entry_text_offset;
  result.target_body_entry_text_offset = entry_text_offset;
  result.has_kernarg_preload = kernarg_preload_length(src) != 0;
  result.kernarg_preload_entry_text_offset =
      result.has_kernarg_preload ? entry_text_offset + kKernargPreloadSkipBytes : entry_text_offset;

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

  // Descriptor ABI fixes that require instructions, not bitfield changes, are
  // emitted as prologue words. BinaryTranslator places those words in the
  // kernel-local .text cave and records the final redirected entry offset.
  result.prologue_words = build_kernel_entry_prologue(src, guest_arch, host_arch);

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
    // CDNA encodes a unified VGPR allocation endpoint while ACCUM_OFFSET
    // carves the trailing AccVGPR window out of that allocation. Liveness and
    // semantic scratch allocation need the ordinary VGPR floor, not the unified
    // endpoint, or they will unnecessarily force new scratch above the
    // accumulator window and move ACCUM_OFFSET. That descriptor growth changed
    // dynamic Triton matmul results on gfx942 because the translated MFMA/DS
    // transpose sequence depends on the original ordinary/accumulator split.
    result.guest_agpr_count = result.guest_vgpr_allocation_count - result.accvgpr_base;
  }
  result.guest_vgpr_count = result.guest_vgpr_allocation_count - result.guest_agpr_count;

  // Start from the guest's ordinary VGPR count. Keep the descriptor allocation
  // count separate for target reporting and descriptor encoding. CDNA
  // descriptors encode the ordinary VGPRs and AccVGPRs as one allocation range,
  // but scratch allocation must reason about only the ordinary part so it can
  // reuse registers that are dead at a given expansion site.
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
  if (arch_max_vgprs(host_arch) != 0 && required_vgpr_allocation > arch_max_vgprs(host_arch)) {
    append_descriptor_error(result,
                            "required VGPR allocation exceeds target limit; spill tiers are "
                            "not implemented for this descriptor");
  }

  result.target_vgpr_granulated = clamp_granulated(
      register_count_to_granulated(required_vgpr_allocation, host_vgpr_granularity),
      kMaxVgprGranulatedField, result, "GRANULATED_WORKITEM_VGPR_COUNT");

  // SGPR counts are also stored as a granulated value, but the descriptor
  // granularity is fixed at eight SGPRs for the architectures handled here.
  // GFX10+ targets keep SGPR allocation in RSRC1/RSRC3 fields managed by the
  // patcher, so this translator only re-encodes the legacy SGPR field when the
  // host descriptor format actually uses it.
  const uint32_t guest_sgpr_granulated =
      AMDHSA_BITS_GET(src.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  result.guest_sgpr_count = granulated_count_to_registers(guest_sgpr_granulated, 8);
  result.host_sgpr_count = std::max(result.guest_sgpr_count, options.minimum_sgprs);
  result.target_sgpr_count = result.host_sgpr_count;
  if (arch_max_sgprs(host_arch) != 0 && result.host_sgpr_count > arch_max_sgprs(host_arch)) {
    append_descriptor_error(result, "required SGPR count exceeds target limit; spill tiers are not "
                                    "implemented for this descriptor");
  }

  if (!uses_gfx10_plus_rsrc3(host_arch)) {
    result.target_sgpr_granulated = register_count_to_granulated(result.host_sgpr_count, 8);
    result.target_sgpr_granulated =
        clamp_granulated(result.target_sgpr_granulated, kMaxSgprGranulatedField, result,
                         "GRANULATED_WAVEFRONT_SGPR_COUNT");
  }

  // LDS/private sizes are copied from the source descriptor and extended by
  // explicit lowering addends. Spill-zone allocation and Zorua-style LDS
  // overflow are represented in KdTranslation but are not implemented yet.
  result.target_private_size =
      src.private_segment_fixed_size + options.private_segment_fixed_size_addend;
  result.target_lds_size = src.group_segment_fixed_size + options.group_segment_fixed_size_addend;

  // The fixed user-SGPR prefix is preserved. Workgroup-id SGPRs are derived
  // separately from the enable bits because they are allocated immediately after
  // this prefix and are not included in USER_SGPR_COUNT.
  result.target_user_sgpr_count = user_sgpr_count(src);
  return result;
}

} // namespace

KernelDescriptorTranslator::KernelDescriptorTranslator(rj_code_arch_t guest_arch,
                                                       rj_code_arch_t host_arch)
    : guest_arch_(guest_arch), host_arch_(host_arch) {}

std::vector<KdTranslation> KernelDescriptorTranslator::translate_image(
    std::span<const uint8_t> image, uint64_t text_offset, uint64_t text_size,
    const KernelDescriptorTranslationOptions &options) const {
  std::vector<KdTranslation> translations;

  visit_kernel_descriptors(
      image, text_offset, text_size,
      [&](uint64_t descriptor_file_offset, uint64_t entry_text_offset, const KD &src) {
        KD desc{};
        std::memcpy(&desc, &src, sizeof(desc));
        translations.push_back(translate_one_descriptor(
            guest_arch_, host_arch_, descriptor_file_offset, entry_text_offset, desc, options));
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
                                  entry_text_offset, desc, options);
}

} // namespace rocjitsu
