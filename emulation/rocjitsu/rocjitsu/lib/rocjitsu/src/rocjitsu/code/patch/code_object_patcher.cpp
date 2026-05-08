// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/code_object_patcher.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/patch/instruction_builder.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <cassert>
#include <climits>
#include <cstddef>
#include <cstring>
#include <elf.h>
#include <optional>

namespace rocjitsu {
namespace {

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

[[nodiscard]] bool target_supports_wave32(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4;
}

[[nodiscard]] bool target_uses_gfx10_plus_mode_bits(rj_code_arch_t arch) {
  return target_supports_wave32(arch);
}

[[nodiscard]] bool target_clears_rsrc1_mode_bits(rj_code_arch_t arch) {
  // DX10_CLAMP and IEEE_MODE are deprecated on GFX12. Preserve them for GFX10
  // and GFX11 targets where they still affect floating-point behavior.
  return arch == ROCJITSU_CODE_ARCH_RDNA4;
}

[[nodiscard]] uint32_t target_default_inst_pref_size(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
                 arch == ROCJITSU_CODE_ARCH_RDNA4
             ? 2
             : 0;
}

[[nodiscard]] bool image_contains_range(size_t image_size, uint64_t file_offset, uint64_t size) {
  const uint64_t limit = static_cast<uint64_t>(image_size);
  return file_offset <= limit && size <= limit - file_offset;
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

bool CodeObjectPatcher::patch_kernel_descriptor(uint64_t file_offset,
                                                std::span<const uint8_t> descriptor) {
  if (!image_contains_range(image_.size(), file_offset, descriptor.size()))
    return false;

  std::memcpy(image_.data() + file_offset, descriptor.data(), descriptor.size());
  return true;
}

bool CodeObjectPatcher::apply_kernel_descriptor_translation(const KdTranslation &translation,
                                                            rj_code_arch_t target_arch) {
  if (!image_contains_range(image_.size(), translation.descriptor_file_offset, sizeof(KD)))
    return false;

  std::optional<uint64_t> prologue_entry;
  if (!translation.prologue_words.empty()) {
    prologue_entry = append_kernel_entry_prologue(translation.entry_text_offset,
                                                  translation.prologue_words, target_arch);
    if (!prologue_entry)
      return false;
  }

  auto *desc = reinterpret_cast<KD *>(image_.data() + translation.descriptor_file_offset);

  AMDHSA_BITS_SET(desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  translation.target_vgpr_granulated);
  AMDHSA_BITS_SET(desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  translation.target_sgpr_granulated);

  if (target_uses_gfx10_plus_mode_bits(target_arch)) {
    if (target_clears_rsrc1_mode_bits(target_arch)) {
      AMDHSA_BITS_SET(desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_ENABLE_DX10_CLAMP, 0);
      AMDHSA_BITS_SET(desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_ENABLE_IEEE_MODE, 0);
    }
    AMDHSA_BITS_SET(desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_WGP_MODE, 1);
    AMDHSA_BITS_SET(desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_MEM_ORDERED, 1);
    AMDHSA_BITS_SET(desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_FWD_PROGRESS, 1);
  }

  if (target_supports_wave32(target_arch)) {
    AMDHSA_BITS_SET(desc->kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
                    translation.target_wave_size == 32 ? 1 : 0);
  } else {
    AMDHSA_BITS_SET(desc->kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
                    0);
  }

  if (target_uses_gfx10_plus_mode_bits(target_arch)) {
    desc->compute_pgm_rsrc3 = 0;
    if (const uint32_t inst_pref = target_default_inst_pref_size(target_arch); inst_pref != 0) {
      AMDHSA_BITS_SET(desc->compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX10_PLUS_INST_PREF_SIZE,
                      inst_pref);
    }
  }

  desc->private_segment_fixed_size = translation.target_private_size;
  desc->group_segment_fixed_size = translation.target_lds_size;
  AMDHSA_BITS_SET(desc->compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT,
                  translation.target_user_sgpr_count);
  AMDHSA_BITS_SET(desc->compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT,
                  translation.target_private_size != 0 ? 1 : 0);

  if (prologue_entry) {
    if (!redirect_kernel_entry(translation.descriptor_file_offset, translation.entry_text_offset,
                               *prologue_entry))
      return false;
  }
  return true;
}

std::optional<uint64_t> CodeObjectPatcher::append_kernel_entry_prologue(
    uint64_t entry_text_offset, std::span<const uint32_t> prologue_words, rj_code_arch_t arch) {
  assert(!prologue_words.empty() && "empty kernel entry prologue");

  // A kernel descriptor entry point is a hardware launch address, not an
  // ordinary branch target. CP expects that instruction address to be 256-byte
  // aligned. The patcher works in .text-relative offsets, so preserve the
  // original entry's 256-byte residue; if the original virtual address was
  // aligned, the redirected virtual address stays aligned too.
  const uint64_t current_offset = cave_start_ + cave_body_size();
  const uint64_t required_residue = entry_text_offset % 256;
  const uint64_t alignment_padding = (required_residue + 256 - (current_offset % 256)) % 256;
  assert(alignment_padding % sizeof(uint32_t) == 0 && "unaligned cave padding");

  std::vector<uint32_t> cave_words(prologue_words.begin(), prologue_words.end());
  const uint64_t cave_byte_offset = current_offset + alignment_padding;
  assert(cave_byte_offset % 256 == required_residue &&
         "kernel descriptor entry lost its 256-byte alignment");

  // The descriptor now enters the cave directly. The only control-flow fixup is
  // a final branch from the prologue body to the original, untouched entry.
  const int64_t branch_pc = static_cast<int64_t>(cave_byte_offset + cave_words.size() * 4);
  const int64_t target = static_cast<int64_t>(entry_text_offset);
  const int64_t target_delta_bytes = target - (branch_pc + 4);
  if (target_delta_bytes % static_cast<int64_t>(sizeof(uint32_t)) != 0)
    return std::nullopt;

  const int64_t target_dwords = target_delta_bytes / static_cast<int64_t>(sizeof(uint32_t));
  if (target_dwords < INT16_MIN || target_dwords > INT16_MAX)
    return std::nullopt;

  if (alignment_padding != 0) {
    std::vector<uint32_t> padding(alignment_padding / sizeof(uint32_t), build_s_nop(0, arch));
    append_cave_body(padding);
  }
  cave_words.push_back(build_s_branch(static_cast<int16_t>(target_dwords), arch));

  append_cave_body(cave_words);
  return cave_byte_offset;
}

bool CodeObjectPatcher::redirect_kernel_entry(uint64_t descriptor_file_offset,
                                              uint64_t old_entry_text_offset,
                                              uint64_t new_entry_text_offset) {
  if (!image_contains_range(image_.size(), descriptor_file_offset, sizeof(KD)))
    return false;

  auto *desc = reinterpret_cast<KD *>(image_.data() + descriptor_file_offset);
  const int64_t delta =
      static_cast<int64_t>(new_entry_text_offset) - static_cast<int64_t>(old_entry_text_offset);
  const int64_t redirected = static_cast<int64_t>(desc->kernel_code_entry_byte_offset) + delta;
  // The descriptor field is signed because the entry point may be before or
  // after the descriptor in virtual address order. Preserve that signed value
  // when applying the text-relative delta.
  desc->kernel_code_entry_byte_offset = redirected;
  return true;
}

void CodeObjectPatcher::append_cave_body(std::span<const uint32_t> words) {
  auto *bytes = reinterpret_cast<const uint8_t *>(words.data());
  cave_body_.insert(cave_body_.end(), bytes, bytes + words.size() * 4);
}

std::vector<uint8_t> CodeObjectPatcher::emit() const { return image_; }

} // namespace rocjitsu
