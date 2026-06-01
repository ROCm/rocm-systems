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
#include <cstddef>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
// Standard library
#include <span>
#include <string_view>
#include <unordered_set>
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
                       std::optional<size_t> grown_section_index, bool grow_load_at_segment_end) {
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

  // Sections at or after the insertion point move forward, except for the one
  // section that the caller is explicitly extending in place.
  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (grown_section_index && i == *grown_section_index)
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

[[nodiscard]] uint64_t checked_lcm_u64(uint64_t lhs, uint64_t rhs) {
  const uint64_t gcd = std::gcd(lhs, rhs);
  if (gcd == 0)
    return 0;
  assert(lhs / gcd <= std::numeric_limits<uint64_t>::max() / rhs &&
         "ELF load alignment LCM overflow");
  return std::lcm(lhs, rhs);
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
  // Later LOAD segments remain loader-valid only when their file shift keeps
  // p_offset % p_align congruent with p_vaddr % p_align.
  for (const Elf64_Phdr &phdr : phdrs) {
    if (phdr.p_type != PT_LOAD || phdr.p_align <= 1)
      continue;
    if (phdr.p_offset >= file_offset)
      alignment = checked_lcm_u64(alignment, phdr.p_align);
  }
  return alignment;
}

[[nodiscard]] uint32_t append_section_name(std::vector<uint8_t> &image, Elf64_Ehdr &ehdr,
                                           std::vector<Elf64_Shdr> &shdrs,
                                           std::vector<Elf64_Phdr> &phdrs,
                                           std::string_view section_name) {
  assert(ehdr.e_shstrndx < shdrs.size() && "invalid section-name string table index");

  auto &shstrtab = shdrs[ehdr.e_shstrndx];
  // sh_name values are offsets into .shstrtab before appending the new bytes.
  const uint32_t name_offset = static_cast<uint32_t>(shstrtab.sh_size);

  std::vector<uint8_t> name_bytes(section_name.begin(), section_name.end());
  name_bytes.push_back('\0');

  insert_file_bytes(image, ehdr, shdrs, phdrs, shstrtab.sh_offset + shstrtab.sh_size, name_bytes,
                    static_cast<size_t>(ehdr.e_shstrndx), false);
  shstrtab.sh_size += name_bytes.size();
  return name_offset;
}

[[nodiscard]] std::optional<size_t> find_text_section(std::span<const Elf64_Shdr> shdrs,
                                                      uint64_t text_offset, uint64_t text_size) {
  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (shdrs[i].sh_offset == text_offset && shdrs[i].sh_size == text_size)
      return i;
  }
  return std::nullopt;
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

[[nodiscard]] bool image_contains_range(size_t image_size, uint64_t file_offset, uint64_t size) {
  const uint64_t limit = static_cast<uint64_t>(image_size);
  return file_offset <= limit && size <= limit - file_offset;
}

void shift_symbols_in_moved_sections(std::vector<uint8_t> &image, const Elf64_Ehdr &ehdr,
                                     std::span<const Elf64_Shdr> shdrs,
                                     const std::vector<bool> &shift_section_vaddr, uint64_t delta) {
  assert(shdrs.size() == shift_section_vaddr.size() && "section shift map size mismatch");
  if (delta == 0 || ehdr.e_type == ET_REL)
    return;

  // ET_DYN symbol values are virtual addresses. Symbols bound to moved
  // allocated sections must track those sections' new virtual addresses.
  for (const Elf64_Shdr &symtab : shdrs) {
    if (symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM)
      continue;
    if (symtab.sh_entsize != sizeof(Elf64_Sym))
      continue;
    if (!image_contains_range(image.size(), symtab.sh_offset, symtab.sh_size))
      continue;

    const size_t count = symtab.sh_size / sizeof(Elf64_Sym);
    for (size_t i = 0; i < count; ++i) {
      const uint64_t symbol_offset = symtab.sh_offset + i * sizeof(Elf64_Sym);
      Elf64_Sym symbol{};
      std::memcpy(&symbol, image.data() + symbol_offset, sizeof(symbol));

      if (symbol.st_shndx == SHN_UNDEF || symbol.st_shndx == SHN_ABS ||
          symbol.st_shndx >= shdrs.size())
        continue;
      if (!shift_section_vaddr[symbol.st_shndx])
        continue;

      assert(symbol.st_value <= std::numeric_limits<uint64_t>::max() - delta &&
             "ELF symbol value overflow");
      symbol.st_value += delta;
      std::memcpy(image.data() + symbol_offset, &symbol, sizeof(symbol));
    }
  }
}

[[nodiscard]] bool relocation_place_was_moved(uint64_t place, std::span<const Elf64_Shdr> shdrs,
                                              const std::vector<bool> &shift_section_vaddr,
                                              uint64_t delta) {
  // shdrs already contains the new addresses, so recover the previous range
  // before deciding whether an ET_DYN relocation place moved.
  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (!shift_section_vaddr[i])
      continue;

    // Shifted sections were moved forward by exactly delta, so their new
    // address must be large enough to recover the pre-insert range.
    assert(shdrs[i].sh_addr >= delta && "moved section address underflow");
    const uint64_t old_addr = shdrs[i].sh_addr - delta;
    if (place >= old_addr && place < old_addr + shdrs[i].sh_size)
      return true;
  }
  return false;
}

void shift_relocation_offsets_in_moved_sections(std::vector<uint8_t> &image, const Elf64_Ehdr &ehdr,
                                                std::span<const Elf64_Shdr> shdrs,
                                                const std::vector<bool> &shift_section_vaddr,
                                                uint64_t delta) {
  assert(shdrs.size() == shift_section_vaddr.size() && "section shift map size mismatch");
  if (delta == 0 || ehdr.e_type != ET_DYN)
    return;

  // AMDHSA uses Elf64_Rela records, and ET_DYN relocation r_offset is the
  // relocated storage address. Any relocation whose place is inside a section we
  // moved must track that section's new virtual address.
  for (const Elf64_Shdr &relocs : shdrs) {
    if (relocs.sh_type != SHT_RELA && relocs.sh_type != SHT_REL)
      continue;
    if (!image_contains_range(image.size(), relocs.sh_offset, relocs.sh_size))
      continue;

    if (relocs.sh_type == SHT_RELA) {
      if (relocs.sh_entsize != sizeof(Elf64_Rela))
        continue;
      const size_t count = relocs.sh_size / sizeof(Elf64_Rela);
      for (size_t i = 0; i < count; ++i) {
        const uint64_t offset = relocs.sh_offset + i * sizeof(Elf64_Rela);
        Elf64_Rela rela{};
        std::memcpy(&rela, image.data() + offset, sizeof(rela));
        if (!relocation_place_was_moved(rela.r_offset, shdrs, shift_section_vaddr, delta))
          continue;
        assert(rela.r_offset <= std::numeric_limits<uint64_t>::max() - delta &&
               "ELF relocation offset overflow");
        rela.r_offset += delta;
        std::memcpy(image.data() + offset, &rela, sizeof(rela));
      }
      continue;
    }

    if (relocs.sh_entsize != sizeof(Elf64_Rel))
      continue;
    const size_t count = relocs.sh_size / sizeof(Elf64_Rel);
    for (size_t i = 0; i < count; ++i) {
      const uint64_t offset = relocs.sh_offset + i * sizeof(Elf64_Rel);
      Elf64_Rel rel{};
      std::memcpy(&rel, image.data() + offset, sizeof(rel));
      if (!relocation_place_was_moved(rel.r_offset, shdrs, shift_section_vaddr, delta))
        continue;
      assert(rel.r_offset <= std::numeric_limits<uint64_t>::max() - delta &&
             "ELF relocation offset overflow");
      rel.r_offset += delta;
      std::memcpy(image.data() + offset, &rel, sizeof(rel));
    }
  }
}

[[nodiscard]] bool kernel_descriptor_symbol(const Elf64_Sym &symbol, const char *strtab,
                                            size_t strtab_size) {
  if (symbol.st_size != sizeof(KD))
    return false;

  // AMDHSA kernel descriptors are global object symbols. This keeps other
  // sizeof(KD) data objects from being treated as descriptors by accident.
  if (elf_symbol_type(symbol.st_info) != kElfSymbolTypeObject ||
      elf_symbol_bind(symbol.st_info) != kElfSymbolBindGlobal)
    return false;

  // AMDHSA descriptors are named "<kernel>.kd". An unnamed 64-byte global
  // object is ambiguous, so require the ABI suffix instead of treating stripped
  // or minimized symbol records as descriptors.
  if (strtab == nullptr || strtab_size == 0 || symbol.st_name == 0)
    return false;
  if (symbol.st_name >= strtab_size)
    return false;

  const char *name = strtab + symbol.st_name;
  const size_t len = strnlen(name, strtab_size - symbol.st_name);
  return len > 3 && std::strcmp(name + len - 3, ".kd") == 0;
}

void adjust_kernel_descriptor_entry_offsets_in_moved_sections(
    std::vector<uint8_t> &image, std::span<const Elf64_Shdr> shdrs,
    const std::vector<bool> &shift_section_vaddr, uint64_t delta) {
  assert(shdrs.size() == shift_section_vaddr.size() && "section shift map size mismatch");
  if (delta == 0)
    return;

  // KERNEL_CODE_ENTRY_BYTE_OFFSET is relative to the descriptor address, not to
  // .text. When a .kd object lives in a shifted allocated section, preserve the
  // same absolute entry PC by subtracting the descriptor section's VA delta.
  std::unordered_set<uint64_t> adjusted_file_offsets;
  for (const Elf64_Shdr &symtab : shdrs) {
    if (symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM)
      continue;
    if (symtab.sh_entsize != sizeof(Elf64_Sym))
      continue;
    if (!image_contains_range(image.size(), symtab.sh_offset, symtab.sh_size))
      continue;

    const char *strtab = nullptr;
    size_t strtab_size = 0;
    if (symtab.sh_link < shdrs.size()) {
      const Elf64_Shdr &strtab_shdr = shdrs[symtab.sh_link];
      if (image_contains_range(image.size(), strtab_shdr.sh_offset, strtab_shdr.sh_size)) {
        strtab = reinterpret_cast<const char *>(image.data() + strtab_shdr.sh_offset);
        strtab_size = strtab_shdr.sh_size;
      }
    }

    const size_t count = symtab.sh_size / sizeof(Elf64_Sym);
    for (size_t i = 0; i < count; ++i) {
      const uint64_t symbol_offset = symtab.sh_offset + i * sizeof(Elf64_Sym);
      Elf64_Sym symbol{};
      std::memcpy(&symbol, image.data() + symbol_offset, sizeof(symbol));

      if (!kernel_descriptor_symbol(symbol, strtab, strtab_size))
        continue;
      if (symbol.st_shndx == SHN_UNDEF || symbol.st_shndx == SHN_ABS ||
          symbol.st_shndx >= shdrs.size())
        continue;
      if (!shift_section_vaddr[symbol.st_shndx])
        continue;

      const Elf64_Shdr &section = shdrs[symbol.st_shndx];
      if (symbol.st_value < section.sh_addr)
        continue;

      const uint64_t file_offset = section.sh_offset + (symbol.st_value - section.sh_addr);
      if (!image_contains_range(image.size(), file_offset, sizeof(KD)))
        continue;
      if (!adjusted_file_offsets.insert(file_offset).second)
        continue;

      KD desc{};
      std::memcpy(&desc, image.data() + file_offset, sizeof(desc));
      assert(delta <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
             "kernel descriptor shift too large");
      desc.kernel_code_entry_byte_offset -= static_cast<int64_t>(delta);
      std::memcpy(image.data() + file_offset, &desc, sizeof(desc));
    }
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
  assert(new_text.size() == text_size_ && "text size mismatch");
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
  if (target_dwords < std::numeric_limits<int16_t>::min() ||
      target_dwords > std::numeric_limits<int16_t>::max())
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

bool CodeObjectPatcher::append_cave_section(std::string_view section_name) {
  if (cave_body_.empty())
    return true;
  if (text_size_ == 0)
    return false;

  assert(cave_start_ == text_size_ &&
         "separate cave sections must start immediately after original .text");
  assert(text_offset_ + text_size_ <= image_.size() && "text section out of bounds");
  assert(cave_body_.size() % sizeof(uint32_t) == 0 && "cave body must be word-aligned");

  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image_.data());
  auto header = *ehdr;
  auto shdrs = read_section_headers(image_, header);
  auto phdrs = read_program_headers(image_, header);

  const auto text_index = find_text_section(shdrs, text_offset_, text_size_);
  if (!text_index) {
    assert(false && "text section header not found");
    return false;
  }

  const auto text_header = shdrs[*text_index];
  const uint64_t cave_file_offset = text_offset_ + text_size_;
  const uint64_t cave_vaddr = text_header.sh_addr + text_size_;

  // Insert the executable bytes at the exact address assumed by branch stub
  // construction: .text-relative offset text_size_. Any later PT_LOAD segment
  // must keep p_offset congruent with p_vaddr modulo p_align, so the total file
  // delta is padded up to the required load alignment. Later allocated sections
  // also move forward in virtual address space; otherwise a large cave can
  // overlap the following LOAD segment in memory. The padding is part of the RX
  // LOAD segment but not part of the .rj_translations section.
  const uint64_t file_delta_alignment = shifted_load_delta_alignment(phdrs, cave_file_offset);
  const uint64_t padded_file_delta = align_up(cave_body_.size(), file_delta_alignment);
  assert(padded_file_delta >= cave_body_.size() && "aligned cave delta underflowed");
  assert((file_delta_alignment <= 1 || padded_file_delta % file_delta_alignment == 0) &&
         "padded cave delta is not load-aligned");
  assert(padded_file_delta % sizeof(uint32_t) == 0 && "padded cave delta must stay word-aligned");
  std::vector<uint8_t> cave_file_bytes(cave_body_.begin(), cave_body_.end());
  cave_file_bytes.resize(padded_file_delta, 0);

  // insert_file_bytes handles file offsets. These side maps record which
  // already-existing allocated objects also need virtual-address adjustments.
  std::vector<bool> shift_section_vaddr(shdrs.size(), false);
  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (i == *text_index || shdrs[i].sh_type == SHT_NULL)
      continue;
    if ((shdrs[i].sh_flags & SHF_ALLOC) != 0 && shdrs[i].sh_addr >= cave_vaddr)
      shift_section_vaddr[i] = true;
  }

  std::vector<bool> shift_segment_vaddr(phdrs.size(), false);
  for (size_t i = 0; i < phdrs.size(); ++i) {
    if (phdrs[i].p_vaddr >= cave_vaddr && phdrs[i].p_offset >= cave_file_offset)
      shift_segment_vaddr[i] = true;
  }

  insert_file_bytes(image_, header, shdrs, phdrs, cave_file_offset, cave_file_bytes, std::nullopt,
                    true);

  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (shift_section_vaddr[i]) {
      [[maybe_unused]] const uint64_t old_addr = shdrs[i].sh_addr;
      shdrs[i].sh_addr += padded_file_delta;
      assert((shdrs[i].sh_addralign <= 1 ||
              shdrs[i].sh_addr % shdrs[i].sh_addralign == old_addr % shdrs[i].sh_addralign) &&
             "shifted allocated section lost its address alignment residue");
    }
  }
  shift_symbols_in_moved_sections(image_, header, shdrs, shift_section_vaddr, padded_file_delta);
  shift_relocation_offsets_in_moved_sections(image_, header, shdrs, shift_section_vaddr,
                                             padded_file_delta);
  adjust_kernel_descriptor_entry_offsets_in_moved_sections(image_, shdrs, shift_section_vaddr,
                                                           padded_file_delta);

  // The text LOAD grows over the inserted cave bytes; later LOADs move forward
  // in both file offset and virtual address.
  for (size_t i = 0; i < phdrs.size(); ++i) {
    if (!shift_segment_vaddr[i])
      continue;
    assert((phdrs[i].p_align <= 1 || padded_file_delta % phdrs[i].p_align == 0) &&
           "cave padding does not preserve shifted LOAD alignment");
    phdrs[i].p_vaddr += padded_file_delta;
    phdrs[i].p_paddr += padded_file_delta;
    assert((phdrs[i].p_align <= 1 ||
            phdrs[i].p_offset % phdrs[i].p_align == phdrs[i].p_vaddr % phdrs[i].p_align) &&
           "shifted LOAD lost file/virtual address congruence");
  }

  const uint32_t name_offset = append_section_name(image_, header, shdrs, phdrs, section_name);

  Elf64_Shdr cave_header{};
  cave_header.sh_name = name_offset;
  cave_header.sh_type = SHT_PROGBITS;
  cave_header.sh_flags = text_header.sh_flags;
  cave_header.sh_addr = cave_vaddr;
  cave_header.sh_offset = cave_file_offset;
  cave_header.sh_size = cave_body_.size();
  cave_header.sh_addralign = sizeof(uint32_t);
  assert(cave_header.sh_offset % cave_header.sh_addralign == 0 &&
         "cave section file offset is not word-aligned");
  assert(cave_header.sh_addr % cave_header.sh_addralign == 0 &&
         "cave section virtual address is not word-aligned");

  const uint64_t section_header_table_end =
      header.e_shoff + static_cast<uint64_t>(shdrs.size()) * sizeof(Elf64_Shdr);
  assert(section_header_table_end <= image_.size() && "section header table end out of bounds");

  // Reserve the next contiguous section-header slot, not necessarily EOF:
  //
  //   old table: [e_shoff, e_shoff + N * sizeof(Elf64_Shdr))
  //   new slot:                         ^ here
  //
  // insert_file_bytes shifts any later file contents to make room; then
  // write_elf_tables rewrites the expanded N + 1 section-header table at
  // e_shoff.
  const uint64_t new_shdr_offset = section_header_table_end;
  const std::array<uint8_t, sizeof(Elf64_Shdr)> blank_header{};
  insert_file_bytes(image_, header, shdrs, phdrs, new_shdr_offset, blank_header, std::nullopt,
                    false);
  assert(header.e_shoff + (static_cast<uint64_t>(shdrs.size()) + 1) * sizeof(Elf64_Shdr) <=
             image_.size() &&
         "expanded section header table out of bounds");

  shdrs.push_back(cave_header);
  assert(shdrs.size() <= std::numeric_limits<uint16_t>::max() && "ELF section count overflow");
  for (const Elf64_Phdr &phdr : phdrs) {
    if (phdr.p_type != PT_LOAD || phdr.p_align <= 1)
      continue;
    assert(phdr.p_offset % phdr.p_align == phdr.p_vaddr % phdr.p_align &&
           "patched LOAD lost file/virtual address congruence");
  }
  header.e_shnum = static_cast<uint16_t>(shdrs.size());
  write_elf_tables(image_, header, shdrs, phdrs);
  return true;
}

std::vector<uint8_t> CodeObjectPatcher::emit() const { return image_; }

} // namespace rocjitsu
