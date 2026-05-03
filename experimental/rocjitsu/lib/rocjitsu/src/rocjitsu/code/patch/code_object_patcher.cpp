// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/code_object_patcher.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/patch/instruction_builder.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstring>
#include <elf.h>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

[[nodiscard]] std::vector<Elf64_Shdr> read_section_headers(const std::vector<uint8_t> &image,
                                                           const Elf64_Ehdr &ehdr) {
  assert(ehdr.e_shentsize == sizeof(Elf64_Shdr) && "unsupported section header size");
  assert(ehdr.e_shoff + static_cast<uint64_t>(ehdr.e_shnum) * sizeof(Elf64_Shdr) <= image.size() &&
         "section header table out of bounds");

  std::vector<Elf64_Shdr> shdrs(ehdr.e_shnum);
  std::memcpy(shdrs.data(), image.data() + ehdr.e_shoff, shdrs.size() * sizeof(Elf64_Shdr));
  return shdrs;
}

[[nodiscard]] std::vector<Elf64_Phdr> read_program_headers(const std::vector<uint8_t> &image,
                                                           const Elf64_Ehdr &ehdr) {
  if (ehdr.e_phoff == 0 || ehdr.e_phnum == 0)
    return {};

  assert(ehdr.e_phentsize == sizeof(Elf64_Phdr) && "unsupported program header size");
  assert(ehdr.e_phoff + static_cast<uint64_t>(ehdr.e_phnum) * sizeof(Elf64_Phdr) <= image.size() &&
         "program header table out of bounds");

  std::vector<Elf64_Phdr> phdrs(ehdr.e_phnum);
  std::memcpy(phdrs.data(), image.data() + ehdr.e_phoff, phdrs.size() * sizeof(Elf64_Phdr));
  return phdrs;
}

void write_elf_tables(std::vector<uint8_t> &image, const Elf64_Ehdr &ehdr,
                      std::span<const Elf64_Shdr> shdrs, std::span<const Elf64_Phdr> phdrs) {
  assert(image.size() >= sizeof(Elf64_Ehdr) && "ELF header out of bounds");
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  assert(ehdr.e_shoff + shdrs.size() * sizeof(Elf64_Shdr) <= image.size() &&
         "section header table write out of bounds");
  std::memcpy(image.data() + ehdr.e_shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  if (!phdrs.empty()) {
    assert(ehdr.e_phoff + phdrs.size() * sizeof(Elf64_Phdr) <= image.size() &&
           "program header table write out of bounds");
    std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), phdrs.size() * sizeof(Elf64_Phdr));
  }
}

void insert_file_bytes(std::vector<uint8_t> &image, Elf64_Ehdr &ehdr,
                       std::vector<Elf64_Shdr> &shdrs, std::vector<Elf64_Phdr> &phdrs,
                       uint64_t file_offset, std::span<const uint8_t> bytes,
                       int grown_section_index, bool grow_load_at_segment_end) {
  assert(file_offset <= image.size() && "ELF insertion offset out of bounds");
  if (bytes.empty())
    return;

  const uint64_t delta = bytes.size();
  image.insert(image.begin() + static_cast<std::ptrdiff_t>(file_offset), bytes.begin(),
               bytes.end());

  if (ehdr.e_shoff >= file_offset)
    ehdr.e_shoff += delta;
  if (ehdr.e_phoff != 0 && ehdr.e_phoff >= file_offset)
    ehdr.e_phoff += delta;

  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (static_cast<int>(i) == grown_section_index)
      continue;
    if (shdrs[i].sh_type == SHT_NULL)
      continue;
    if (shdrs[i].sh_offset >= file_offset)
      shdrs[i].sh_offset += delta;
  }

  for (Elf64_Phdr &phdr : phdrs) {
    const uint64_t old_end = phdr.p_offset + phdr.p_filesz;
    if (phdr.p_offset >= file_offset) {
      phdr.p_offset += delta;
      continue;
    }

    // If bytes are inserted inside a loaded segment, keep the segment covering
    // the shifted contents. When inserting exactly at the end, only the caller
    // that is adding executable cave bytes should grow the segment.
    const bool inside_segment = file_offset < old_end;
    const bool at_segment_end = grow_load_at_segment_end && file_offset == old_end;
    if (phdr.p_type == PT_LOAD && (inside_segment || at_segment_end)) {
      phdr.p_filesz += delta;
      phdr.p_memsz += delta;
    }
  }
}

[[nodiscard]] uint64_t gcd_u64(uint64_t lhs, uint64_t rhs) {
  while (rhs != 0) {
    const uint64_t rem = lhs % rhs;
    lhs = rhs;
    rhs = rem;
  }
  return lhs;
}

[[nodiscard]] uint64_t lcm_u64(uint64_t lhs, uint64_t rhs) {
  if (lhs == 0 || rhs == 0)
    return std::max(lhs, rhs);
  const uint64_t gcd = gcd_u64(lhs, rhs);
  assert(lhs / gcd <= UINT64_MAX / rhs && "ELF load alignment LCM overflow");
  return (lhs / gcd) * rhs;
}

[[nodiscard]] uint64_t align_up(uint64_t value, uint64_t alignment) {
  if (alignment <= 1)
    return value;
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

[[nodiscard]] uint64_t shifted_load_delta_alignment(std::span<const Elf64_Phdr> phdrs,
                                                    uint64_t file_offset) {
  uint64_t alignment = 1;
  for (const Elf64_Phdr &phdr : phdrs) {
    if (phdr.p_type != PT_LOAD || phdr.p_align <= 1)
      continue;
    if (phdr.p_offset >= file_offset)
      alignment = lcm_u64(alignment, phdr.p_align);
  }
  return alignment;
}

[[nodiscard]] uint32_t append_section_name(std::vector<uint8_t> &image, Elf64_Ehdr &ehdr,
                                           std::vector<Elf64_Shdr> &shdrs,
                                           std::vector<Elf64_Phdr> &phdrs,
                                           std::string_view section_name) {
  assert(ehdr.e_shstrndx < shdrs.size() && "invalid section-name string table index");

  auto &shstrtab = shdrs[ehdr.e_shstrndx];
  const uint32_t name_offset = static_cast<uint32_t>(shstrtab.sh_size);

  std::vector<uint8_t> name_bytes(section_name.begin(), section_name.end());
  name_bytes.push_back('\0');

  insert_file_bytes(image, ehdr, shdrs, phdrs, shstrtab.sh_offset + shstrtab.sh_size, name_bytes,
                    ehdr.e_shstrndx, false);
  shstrtab.sh_size += name_bytes.size();
  return name_offset;
}

[[nodiscard]] size_t find_text_section(std::span<const Elf64_Shdr> shdrs, uint64_t text_offset,
                                       uint64_t text_size) {
  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (shdrs[i].sh_offset == text_offset && shdrs[i].sh_size == text_size)
      return i;
  }
  assert(false && "text section header not found");
  return 0;
}

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
  assert(new_text.size() == text_size_ && "text size mismatch");
  std::memcpy(image_.data() + text_offset_, new_text.data(), new_text.size());
}

void CodeObjectPatcher::update_elf_flags(uint32_t new_mach) {
  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image_.data());
  // Preserve upper bits (XNACK, SRAMECC feature flags); only replace EF_AMDGPU_MACH in low byte.
  ehdr->e_flags = (ehdr->e_flags & ~0xFFu) | (new_mach & 0xFFu);
}

void CodeObjectPatcher::patch_kernel_descriptor(uint64_t file_offset,
                                                std::span<const uint8_t> descriptor) {
  assert(file_offset + descriptor.size() <= image_.size() &&
         "kernel descriptor patch out of bounds");
  std::memcpy(image_.data() + file_offset, descriptor.data(), descriptor.size());
}

void CodeObjectPatcher::apply_kernel_descriptor_translation(const KdTranslation &translation,
                                                            rj_code_arch_t target_arch) {
  assert(translation.descriptor_file_offset + sizeof(KD) <= image_.size() &&
         "kernel descriptor translation out of bounds");

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

  if (!translation.prologue_words.empty()) {
    const uint64_t prologue_entry = append_kernel_entry_prologue(
        translation.entry_text_offset, translation.prologue_words, target_arch);
    redirect_kernel_entry(translation.descriptor_file_offset, translation.entry_text_offset,
                          prologue_entry);
  }
}

uint64_t CodeObjectPatcher::append_kernel_entry_prologue(uint64_t entry_text_offset,
                                                         std::span<const uint32_t> prologue_words,
                                                         rj_code_arch_t arch) {
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
  if (alignment_padding != 0) {
    std::vector<uint32_t> padding(alignment_padding / sizeof(uint32_t), build_s_nop(0, arch));
    append_cave_body(padding);
  }

  std::vector<uint32_t> cave_words(prologue_words.begin(), prologue_words.end());
  const uint64_t cave_byte_offset = cave_start_ + cave_body_size();
  assert(cave_byte_offset % 256 == required_residue &&
         "kernel descriptor entry lost its 256-byte alignment");

  // The descriptor now enters the cave directly. The only control-flow fixup is
  // a final branch from the prologue body to the original, untouched entry.
  const int64_t branch_pc = static_cast<int64_t>(cave_byte_offset + cave_words.size() * 4);
  const int64_t target = static_cast<int64_t>(entry_text_offset);
  const int64_t target_dwords = (target - (branch_pc + 4)) / 4;
  assert(target_dwords >= INT16_MIN && target_dwords <= INT16_MAX &&
         "kernel entry prologue branch exceeds simm16 range");
  cave_words.push_back(build_s_branch(static_cast<int16_t>(target_dwords), arch));

  append_cave_body(cave_words);
  return cave_byte_offset;
}

void CodeObjectPatcher::redirect_kernel_entry(uint64_t descriptor_file_offset,
                                              uint64_t old_entry_text_offset,
                                              uint64_t new_entry_text_offset) {
  assert(descriptor_file_offset + sizeof(KD) <= image_.size() &&
         "kernel descriptor redirect out of bounds");

  auto *desc = reinterpret_cast<KD *>(image_.data() + descriptor_file_offset);
  const int64_t delta =
      static_cast<int64_t>(new_entry_text_offset) - static_cast<int64_t>(old_entry_text_offset);
  const int64_t redirected = static_cast<int64_t>(desc->kernel_code_entry_byte_offset) + delta;
  // The descriptor field is signed because the entry point may be before or
  // after the descriptor in virtual address order. Preserve that signed value
  // when applying the text-relative delta.
  desc->kernel_code_entry_byte_offset = redirected;
}

void CodeObjectPatcher::append_cave_body(std::span<const uint32_t> words) {
  auto *bytes = reinterpret_cast<const uint8_t *>(words.data());
  cave_body_.insert(cave_body_.end(), bytes, bytes + words.size() * 4);
}

void CodeObjectPatcher::append_cave_section(std::string_view section_name) {
  if (cave_body_.empty())
    return;

  assert(cave_start_ == text_size_ &&
         "separate cave sections must start immediately after original .text");
  assert(text_offset_ + text_size_ <= image_.size() && "text section out of bounds");
  assert(cave_body_.size() % sizeof(uint32_t) == 0 && "cave body must be word-aligned");

  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image_.data());
  auto header = *ehdr;
  auto shdrs = read_section_headers(image_, header);
  auto phdrs = read_program_headers(image_, header);

  const size_t text_index = find_text_section(shdrs, text_offset_, text_size_);
  const auto text_header = shdrs[text_index];
  const uint64_t cave_file_offset = text_offset_ + text_size_;

  // Insert the executable bytes at the exact address assumed by branch stub
  // construction: .text-relative offset text_size_. Any later PT_LOAD segment
  // must keep p_offset congruent with p_vaddr modulo p_align, so the total file
  // delta is padded up to the required load alignment. The padding is part of
  // the RX LOAD segment but not part of the .rj_translations section.
  const uint64_t file_delta_alignment = shifted_load_delta_alignment(phdrs, cave_file_offset);
  const uint64_t padded_file_delta = align_up(cave_body_.size(), file_delta_alignment);
  std::vector<uint8_t> cave_file_bytes(cave_body_.begin(), cave_body_.end());
  cave_file_bytes.resize(padded_file_delta, 0);
  insert_file_bytes(image_, header, shdrs, phdrs, cave_file_offset, cave_file_bytes, -1, true);

  const uint32_t name_offset = append_section_name(image_, header, shdrs, phdrs, section_name);

  Elf64_Shdr cave_header{};
  cave_header.sh_name = name_offset;
  cave_header.sh_type = SHT_PROGBITS;
  cave_header.sh_flags = text_header.sh_flags;
  cave_header.sh_addr = text_header.sh_addr + text_size_;
  cave_header.sh_offset = cave_file_offset;
  cave_header.sh_size = cave_body_.size();
  cave_header.sh_addralign = sizeof(uint32_t);

  const uint64_t new_shdr_offset =
      header.e_shoff + static_cast<uint64_t>(shdrs.size()) * sizeof(Elf64_Shdr);
  const std::array<uint8_t, sizeof(Elf64_Shdr)> blank_header{};
  insert_file_bytes(image_, header, shdrs, phdrs, new_shdr_offset, blank_header, -1, false);

  shdrs.push_back(cave_header);
  header.e_shnum = static_cast<uint16_t>(shdrs.size());
  write_elf_tables(image_, header, shdrs, phdrs);
}

std::vector<uint8_t> CodeObjectPatcher::emit() const { return image_; }

} // namespace rocjitsu
