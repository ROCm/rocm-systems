// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/code_object_patcher.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <cassert>
#include <cstring>
#include <elf.h>

namespace rocjitsu {
namespace {

struct Wave64DescriptorPatchInfo {
  uint32_t vgpr_granularity;  // rsrc1 encodes (actual_vgprs / granularity) - 1.
  bool clear_rsrc1_mode_bits; // GFX12 deprecates DX10_CLAMP and IEEE_MODE.
};

Wave64DescriptorPatchInfo wave64_descriptor_patch_info(rj_code_arch_t target_arch) {
  // Keep per-generation descriptor conventions in one place so adding a new
  // RDNA target does not require scattering magic constants through the patcher.
  switch (target_arch) {
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return {8u, false};
  case ROCJITSU_CODE_ARCH_RDNA4:
    return {12u, true};
  default:
    // Unknown targets keep the older GFX10/GFX11-compatible behavior: 8-VGPR
    // granularity and preserved floating-point mode bits.
    return {8u, false};
  }
}

} // namespace

CodeObjectPatcher::CodeObjectPatcher(const AmdGpuCodeObject &obj)
    : image_(obj.image_data(), obj.image_data() + obj.image_size()), text_offset_(0),
      text_size_(0) {
  auto &text_secs = obj.text_sections();
  if (!text_secs.empty()) {
    text_offset_ = text_secs[0]->sectionOffset();
    text_size_ = text_secs[0]->size();
  }
}

std::span<uint8_t> CodeObjectPatcher::text_bytes() {
  return {image_.data() + text_offset_, text_size_};
}

std::span<const uint8_t> CodeObjectPatcher::text_bytes() const {
  return {image_.data() + text_offset_, text_size_};
}

void CodeObjectPatcher::overwrite_text(std::span<const uint8_t> new_text) {
  assert(new_text.size() == text_size_ && "text size mismatch — cave body must fit in NOP padding");
  std::memcpy(image_.data() + text_offset_, new_text.data(), new_text.size());
}

void CodeObjectPatcher::update_elf_flags(uint32_t new_mach) {
  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image_.data());
  // Preserve upper bits (XNACK, SRAMECC feature flags); only replace EF_AMDGPU_MACH in low byte.
  ehdr->e_flags = (ehdr->e_flags & ~0xFFu) | (new_mach & 0xFFu);
}

void CodeObjectPatcher::patch_kernel_descriptors_for_wave64(rj_code_arch_t target_arch) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  namespace kd = rocr::llvm::amdhsa;

  // Select target-specific descriptor policy once; the loop below only applies
  // the policy while walking each kernel descriptor in the code object.
  const Wave64DescriptorPatchInfo target_info = wave64_descriptor_patch_info(target_arch);

  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image_.data());
  if (ehdr->e_shoff + static_cast<uint64_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr) > image_.size())
    return;
  auto *shdr = reinterpret_cast<Elf64_Shdr *>(image_.data() + ehdr->e_shoff);

  for (int i = 0; i < ehdr->e_shnum; ++i) {
    if (shdr[i].sh_type != SHT_SYMTAB)
      continue;
    if (shdr[i].sh_offset + shdr[i].sh_size > image_.size())
      continue;
    auto *symtab = reinterpret_cast<Elf64_Sym *>(image_.data() + shdr[i].sh_offset);
    int nsyms = shdr[i].sh_size / sizeof(Elf64_Sym);
    if (shdr[i].sh_link >= ehdr->e_shnum)
      continue;
    auto *strtab_shdr = &shdr[shdr[i].sh_link];
    if (strtab_shdr->sh_offset + strtab_shdr->sh_size > image_.size())
      continue;
    auto *strtab = reinterpret_cast<const char *>(image_.data() + strtab_shdr->sh_offset);

    for (int j = 0; j < nsyms; ++j) {
      if (symtab[j].st_size != sizeof(KD))
        continue;
      if (strtab_shdr->sh_size > 0 && symtab[j].st_name > 0) {
        if (symtab[j].st_name >= strtab_shdr->sh_size)
          continue;
        const char *name = strtab + symtab[j].st_name;
        size_t len = strlen(name);
        if (len >= 3 && strcmp(name + len - 3, ".kd") != 0)
          continue;
      }
      uint16_t sec_idx = symtab[j].st_shndx;
      if (sec_idx >= ehdr->e_shnum)
        continue;
      uint64_t file_off = shdr[sec_idx].sh_offset + (symtab[j].st_value - shdr[sec_idx].sh_addr);
      if (file_off + sizeof(KD) > image_.size())
        continue;

      auto *kd = reinterpret_cast<KD *>(image_.data() + file_off);

      // --- compute_pgm_rsrc1: translate GFX9 → GFX10+ ---

      // VGPR granularity: GFX9/GFX11 wave64 use 8, RDNA4 wave64 uses 12.
      // On CDNA3/4, AccVGPRs occupy the upper portion of the unified VGPR file.
      // ACCUM_OFFSET in rsrc3 indicates where AccVGPRs start: (offset+1)*4.
      // The translated code must allocate enough VGPRs for both ranges.
      uint32_t gfx9_vgpr_gran = AMDHSA_BITS_GET(
          kd->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
      uint32_t actual_vgprs = (gfx9_vgpr_gran + 1) * 8;

      uint32_t accum_offset =
          AMDHSA_BITS_GET(kd->compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET);
      if (accum_offset > 0)
        actual_vgprs = (accum_offset + 1) * 4 + actual_vgprs;

      // Semantic lowering (e.g., MFMA→WMMA) injects instructions that need
      // temp VGPRs found via liveness analysis. Ensure enough headroom.
      actual_vgprs = std::max(actual_vgprs, 128u);
      uint32_t target_vgpr_gran =
          (actual_vgprs + target_info.vgpr_granularity - 1) / target_info.vgpr_granularity - 1;
      AMDHSA_BITS_SET(kd->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                      target_vgpr_gran);

      // SGPR granularity: on GFX10+, the field is ignored by hardware but must
      // be set for the runtime. Use 0 (8 SGPRs) which matches native compilers.
      AMDHSA_BITS_SET(kd->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                      0);

      // DX10_CLAMP and IEEE_MODE are deprecated on GFX12. Preserve them for
      // GFX11, where they still affect floating-point behavior.
      if (target_info.clear_rsrc1_mode_bits) {
        AMDHSA_BITS_SET(kd->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_ENABLE_DX10_CLAMP, 0);
        AMDHSA_BITS_SET(kd->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_ENABLE_IEEE_MODE, 0);
      }

      // Set GFX10+ required mode bits.
      AMDHSA_BITS_SET(kd->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_WGP_MODE, 1);
      AMDHSA_BITS_SET(kd->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_MEM_ORDERED, 1);
      AMDHSA_BITS_SET(kd->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_FWD_PROGRESS, 1);

      // --- compute_pgm_rsrc3: translate GFX90A layout → GFX10+ layout ---
      // GFX90A: [0:5]=ACCUM_OFFSET, [16]=TG_SPLIT
      // GFX10+: [0:3]=SHARED_VGPR_COUNT, [4:9]=INST_PREF_SIZE
      kd->compute_pgm_rsrc3 = 0;
      AMDHSA_BITS_SET(kd->compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX10_PLUS_INST_PREF_SIZE, 2);

      // --- kernel_code_properties: ensure Wave64 ---
      AMDHSA_BITS_SET(kd->kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
                      0);
    }
  }
}

std::vector<KernelDescriptorInfo> CodeObjectPatcher::kernel_descriptor_info() const {
  return collect_kernel_descriptor_info(image_, text_offset_, text_size_);
}

void CodeObjectPatcher::append_cave_body(std::span<const uint32_t> words) {
  auto *bytes = reinterpret_cast<const uint8_t *>(words.data());
  cave_body_.insert(cave_body_.end(), bytes, bytes + words.size() * 4);
}

std::vector<uint8_t> CodeObjectPatcher::emit() const { return image_; }

} // namespace rocjitsu
