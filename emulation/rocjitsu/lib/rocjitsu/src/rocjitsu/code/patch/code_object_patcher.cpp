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
// Standard library
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
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

[[nodiscard]] bool image_contains_range(size_t image_size, uint64_t file_offset, uint64_t size);

[[nodiscard]] std::optional<std::string_view> section_name(std::span<const uint8_t> image,
                                                           const Elf64_Ehdr &ehdr,
                                                           std::span<const Elf64_Shdr> shdrs,
                                                           size_t section_index) {
  if (section_index >= shdrs.size() || ehdr.e_shstrndx >= shdrs.size())
    return std::nullopt;

  const Elf64_Shdr &shstrtab = shdrs[ehdr.e_shstrndx];
  const Elf64_Shdr &section = shdrs[section_index];
  if (!image_contains_range(image.size(), shstrtab.sh_offset, shstrtab.sh_size))
    return std::nullopt;
  if (section.sh_name >= shstrtab.sh_size)
    return std::nullopt;

  const char *base = reinterpret_cast<const char *>(image.data() + shstrtab.sh_offset);
  const char *name = base + section.sh_name;
  const size_t max_len = static_cast<size_t>(shstrtab.sh_size - section.sh_name);
  const size_t len = strnlen(name, max_len);
  if (len == max_len)
    return std::nullopt;
  return std::string_view(name, len);
}

[[nodiscard]] bool target_supports_wave32(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250;
}

[[nodiscard]] bool target_uses_gfx10_plus_mode_bits(rj_code_arch_t arch) {
  return target_supports_wave32(arch);
}

[[nodiscard]] bool target_uses_gfx90a_accum_offset(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA2 || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
         arch == ROCJITSU_CODE_ARCH_CDNA4;
}

[[nodiscard]] bool target_clears_rsrc1_mode_bits(rj_code_arch_t arch) {
  // DX10_CLAMP and IEEE_MODE are deprecated on GFX12. Preserve them for GFX10
  // and GFX11 targets where they still affect floating-point behavior.
  return arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250;
}

[[nodiscard]] uint32_t target_default_inst_pref_size(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
                 arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250
             ? 2
             : 0;
}

[[nodiscard]] std::optional<uint16_t>
kernel_entry_prologue_long_branch_sgpr_pair(const KdTranslation &translation,
                                            rj_code_arch_t target_arch) {
  if (target_arch != ROCJITSU_CODE_ARCH_RDNA4)
    return std::nullopt;

  uint32_t first_scratch_sgpr = translation.target_source_sgpr_count;
  if (translation.rdna4_grid_x_sgpr >= 0) {
    first_scratch_sgpr =
        std::max(first_scratch_sgpr, static_cast<uint32_t>(translation.rdna4_grid_x_sgpr) + 1u);
  }
  if (translation.rdna4_grid_yz_sgpr >= 0) {
    first_scratch_sgpr =
        std::max(first_scratch_sgpr, static_cast<uint32_t>(translation.rdna4_grid_yz_sgpr) + 1u);
  }
  first_scratch_sgpr = align_up(first_scratch_sgpr, 2);
  if (first_scratch_sgpr + 1u >= translation.target_sgpr_count)
    return std::nullopt;
  if (first_scratch_sgpr >= 127u)
    return std::nullopt;
  return static_cast<uint16_t>(first_scratch_sgpr);
}

[[nodiscard]] bool image_contains_range(size_t image_size, uint64_t file_offset, uint64_t size) {
  const uint64_t limit = static_cast<uint64_t>(image_size);
  return file_offset <= limit && size <= limit - file_offset;
}

bool patch_msgpack_uint(std::vector<uint8_t> &image, size_t offset, uint32_t value) {
  if (offset >= image.size())
    return false;
  const uint8_t tag = image[offset];
  if (tag <= 0x7Fu) {
    if (value > 0x7Fu)
      return false;
    image[offset] = static_cast<uint8_t>(value);
    return true;
  }
  if (tag == 0xCCu) {
    if (value > 0xFFu || !image_contains_range(image.size(), offset + 1, 1))
      return false;
    image[offset + 1] = static_cast<uint8_t>(value);
    return true;
  }
  if (tag == 0xCDu) {
    if (value > 0xFFFFu || !image_contains_range(image.size(), offset + 1, 2))
      return false;
    image[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFFu);
    image[offset + 2] = static_cast<uint8_t>(value & 0xFFu);
    return true;
  }
  if (tag == 0xCEu) {
    if (!image_contains_range(image.size(), offset + 1, 4))
      return false;
    image[offset + 1] = static_cast<uint8_t>((value >> 24) & 0xFFu);
    image[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xFFu);
    image[offset + 3] = static_cast<uint8_t>((value >> 8) & 0xFFu);
    image[offset + 4] = static_cast<uint8_t>(value & 0xFFu);
    return true;
  }
  return false;
}

[[nodiscard]] std::vector<uint8_t> encode_msgpack_uint(uint32_t value) {
  if (value <= 0x7Fu)
    return {static_cast<uint8_t>(value)};
  if (value <= 0xFFu)
    return {0xCCu, static_cast<uint8_t>(value)};
  if (value <= 0xFFFFu) {
    return {0xCDu, static_cast<uint8_t>((value >> 8) & 0xFFu), static_cast<uint8_t>(value & 0xFFu)};
  }
  return {0xCEu, static_cast<uint8_t>((value >> 24) & 0xFFu),
          static_cast<uint8_t>((value >> 16) & 0xFFu), static_cast<uint8_t>((value >> 8) & 0xFFu),
          static_cast<uint8_t>(value & 0xFFu)};
}

[[nodiscard]] std::optional<size_t> msgpack_uint_size(std::span<const uint8_t> image,
                                                      size_t offset) {
  if (offset >= image.size())
    return std::nullopt;
  const uint8_t tag = image[offset];
  if (tag <= 0x7Fu)
    return 1;
  if (tag == 0xCCu)
    return image_contains_range(image.size(), offset, 2) ? std::optional<size_t>{2} : std::nullopt;
  if (tag == 0xCDu)
    return image_contains_range(image.size(), offset, 3) ? std::optional<size_t>{3} : std::nullopt;
  if (tag == 0xCEu)
    return image_contains_range(image.size(), offset, 5) ? std::optional<size_t>{5} : std::nullopt;
  return std::nullopt;
}

std::optional<uint32_t> read_msgpack_uint(std::span<const uint8_t> image, size_t offset) {
  if (offset >= image.size())
    return std::nullopt;
  const uint8_t tag = image[offset];
  if (tag <= 0x7Fu)
    return tag;
  if (tag == 0xCCu) {
    if (!image_contains_range(image.size(), offset + 1, 1))
      return std::nullopt;
    return image[offset + 1];
  }
  if (tag == 0xCDu) {
    if (!image_contains_range(image.size(), offset + 1, 2))
      return std::nullopt;
    return (static_cast<uint32_t>(image[offset + 1]) << 8) |
           static_cast<uint32_t>(image[offset + 2]);
  }
  if (tag == 0xCEu) {
    if (!image_contains_range(image.size(), offset + 1, 4))
      return std::nullopt;
    return (static_cast<uint32_t>(image[offset + 1]) << 24) |
           (static_cast<uint32_t>(image[offset + 2]) << 16) |
           (static_cast<uint32_t>(image[offset + 3]) << 8) |
           static_cast<uint32_t>(image[offset + 4]);
  }
  return std::nullopt;
}

std::optional<std::pair<std::string_view, size_t>>
read_msgpack_string(std::span<const uint8_t> image, size_t offset) {
  if (offset >= image.size())
    return std::nullopt;

  const uint8_t tag = image[offset];
  size_t length = 0;
  size_t data_offset = offset + 1;
  if ((tag & 0xE0u) == 0xA0u) {
    length = tag & 0x1Fu;
  } else if (tag == 0xD9u) {
    if (!image_contains_range(image.size(), offset + 1, 1))
      return std::nullopt;
    length = image[offset + 1];
    data_offset = offset + 2;
  } else if (tag == 0xDAu) {
    if (!image_contains_range(image.size(), offset + 1, 2))
      return std::nullopt;
    length = (static_cast<size_t>(image[offset + 1]) << 8) | static_cast<size_t>(image[offset + 2]);
    data_offset = offset + 3;
  } else if (tag == 0xDBu) {
    if (!image_contains_range(image.size(), offset + 1, 4))
      return std::nullopt;
    length = (static_cast<size_t>(image[offset + 1]) << 24) |
             (static_cast<size_t>(image[offset + 2]) << 16) |
             (static_cast<size_t>(image[offset + 3]) << 8) | static_cast<size_t>(image[offset + 4]);
    data_offset = offset + 5;
  } else {
    return std::nullopt;
  }

  if (!image_contains_range(image.size(), data_offset, length))
    return std::nullopt;
  return std::pair{
      std::string_view(reinterpret_cast<const char *>(image.data() + data_offset), length),
      data_offset + length};
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

void shift_dynamic_pointers_in_moved_sections(std::vector<uint8_t> &image,
                                              std::span<const Elf64_Shdr> shdrs,
                                              const std::vector<bool> &shift_section_vaddr,
                                              uint64_t delta) {
  assert(shdrs.size() == shift_section_vaddr.size() && "section shift map size mismatch");
  if (delta == 0)
    return;

  constexpr uint32_t kShtDynamic = 6;
  constexpr int64_t kDtGnuHash = 0x6FFFFEF5;
  const auto is_pointer_tag = [](int64_t tag) {
    return tag == DT_HASH || tag == DT_STRTAB || tag == DT_SYMTAB || tag == kDtGnuHash;
  };

  for (const Elf64_Shdr &dynamic : shdrs) {
    if (dynamic.sh_type != kShtDynamic || dynamic.sh_entsize != sizeof(Elf64_Dyn))
      continue;
    if (!image_contains_range(image.size(), dynamic.sh_offset, dynamic.sh_size))
      continue;

    const size_t count = dynamic.sh_size / sizeof(Elf64_Dyn);
    for (size_t i = 0; i < count; ++i) {
      const uint64_t offset = dynamic.sh_offset + i * sizeof(Elf64_Dyn);
      Elf64_Dyn dyn{};
      std::memcpy(&dyn, image.data() + offset, sizeof(dyn));
      if (!is_pointer_tag(dyn.d_tag))
        continue;
      if (!relocation_place_was_moved(dyn.d_un.d_ptr, shdrs, shift_section_vaddr, delta))
        continue;
      assert(dyn.d_un.d_ptr <= std::numeric_limits<uint64_t>::max() - delta &&
             "ELF dynamic pointer overflow");
      dyn.d_un.d_ptr += delta;
      std::memcpy(image.data() + offset, &dyn, sizeof(dyn));
    }
  }
}

void insert_file_bytes_shifting_alloc_vaddrs(std::vector<uint8_t> &image, Elf64_Ehdr &ehdr,
                                             std::vector<Elf64_Shdr> &shdrs,
                                             std::vector<Elf64_Phdr> &phdrs, uint64_t file_offset,
                                             std::span<const uint8_t> bytes,
                                             size_t grown_section_index) {
  assert(file_offset <= image.size() && "ELF insertion offset out of bounds");
  if (bytes.empty())
    return;

  const uint64_t delta = bytes.size();
  std::vector<uint64_t> old_section_offsets;
  old_section_offsets.reserve(shdrs.size());
  std::vector<bool> shift_section_vaddr(shdrs.size(), false);
  for (size_t i = 0; i < shdrs.size(); ++i) {
    old_section_offsets.push_back(shdrs[i].sh_offset);
    if (i != grown_section_index && shdrs[i].sh_type != SHT_NULL &&
        (shdrs[i].sh_flags & SHF_ALLOC) != 0 && shdrs[i].sh_offset >= file_offset) {
      shift_section_vaddr[i] = true;
    }
  }

  struct OldProgramHeaderRange {
    uint64_t offset = 0;
    uint64_t end = 0;
  };
  std::vector<OldProgramHeaderRange> old_program_ranges;
  old_program_ranges.reserve(phdrs.size());
  for (const Elf64_Phdr &phdr : phdrs)
    old_program_ranges.push_back({phdr.p_offset, phdr.p_offset + phdr.p_filesz});

  image.insert(image.begin() + static_cast<std::ptrdiff_t>(file_offset), bytes.begin(),
               bytes.end());

  if (ehdr.e_shoff >= file_offset)
    ehdr.e_shoff += delta;
  if (ehdr.e_phoff != 0 && ehdr.e_phoff >= file_offset)
    ehdr.e_phoff += delta;

  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (i == grown_section_index) {
      shdrs[i].sh_size += delta;
      continue;
    }
    if (shdrs[i].sh_type == SHT_NULL)
      continue;
    if (old_section_offsets[i] >= file_offset) {
      shdrs[i].sh_offset += delta;
      if ((shdrs[i].sh_flags & SHF_ALLOC) != 0)
        shdrs[i].sh_addr += delta;
    }
  }

  for (size_t i = 0; i < phdrs.size(); ++i) {
    Elf64_Phdr &phdr = phdrs[i];
    const auto [old_offset, old_end] = old_program_ranges[i];
    if (old_offset >= file_offset) {
      phdr.p_offset += delta;
      phdr.p_vaddr += delta;
      phdr.p_paddr += delta;
      continue;
    }
    if (file_offset < old_end && (phdr.p_type == PT_LOAD || phdr.p_type == PT_NOTE)) {
      phdr.p_filesz += delta;
      phdr.p_memsz += delta;
    }
  }

  shift_symbols_in_moved_sections(image, ehdr, shdrs, shift_section_vaddr, delta);
  shift_relocation_offsets_in_moved_sections(image, ehdr, shdrs, shift_section_vaddr, delta);
  shift_dynamic_pointers_in_moved_sections(image, shdrs, shift_section_vaddr, delta);
  write_elf_tables(image, ehdr, shdrs, phdrs);
}

struct NoteLocation {
  size_t section_index = 0;
  uint64_t header_offset = 0;
  uint64_t desc_offset = 0;
  uint32_t desc_size = 0;
  uint64_t desc_aligned_size = 0;
};

[[nodiscard]] std::optional<NoteLocation>
find_note_containing_offset(std::span<const uint8_t> image, std::span<const Elf64_Shdr> shdrs,
                            size_t offset) {
  for (size_t section_index = 0; section_index < shdrs.size(); ++section_index) {
    const Elf64_Shdr &section = shdrs[section_index];
    if (section.sh_type != SHT_NOTE)
      continue;
    if (!image_contains_range(image.size(), section.sh_offset, section.sh_size))
      continue;

    uint64_t cursor = section.sh_offset;
    const uint64_t section_end = section.sh_offset + section.sh_size;
    while (cursor + sizeof(Elf64_Nhdr) <= section_end) {
      Elf64_Nhdr nhdr{};
      std::memcpy(&nhdr, image.data() + cursor, sizeof(nhdr));
      const uint64_t name_offset = cursor + sizeof(Elf64_Nhdr);
      const uint64_t desc_offset = name_offset + align_up(nhdr.n_namesz, 4);
      const uint64_t desc_aligned_size = align_up(nhdr.n_descsz, 4);
      const uint64_t desc_end = desc_offset + nhdr.n_descsz;
      const uint64_t note_end = desc_offset + desc_aligned_size;
      if (desc_end > image.size() || note_end > section_end)
        return std::nullopt;
      if (offset >= desc_offset && offset < desc_end) {
        return NoteLocation{section_index, cursor, desc_offset, nhdr.n_descsz, desc_aligned_size};
      }
      cursor = note_end;
    }
  }
  return std::nullopt;
}

bool patch_msgpack_uint_resizable(std::vector<uint8_t> &image, size_t offset, uint32_t value) {
  const auto old_size = msgpack_uint_size(image, offset);
  if (!old_size)
    return false;
  const std::vector<uint8_t> encoded = encode_msgpack_uint(value);
  if (encoded.size() <= *old_size)
    return patch_msgpack_uint(image, offset, value);

  Elf64_Ehdr header{};
  if (!image_contains_range(image.size(), 0, sizeof(header)))
    return false;
  std::memcpy(&header, image.data(), sizeof(header));
  auto shdrs = read_section_headers(image, header);
  auto phdrs = read_program_headers(image, header);
  const auto note = find_note_containing_offset(image, shdrs, offset);
  if (!note)
    return false;

  const uint64_t old_desc_end = note->desc_offset + note->desc_size;
  if (offset + *old_size > old_desc_end)
    return false;
  const uint64_t old_aligned_end = note->desc_offset + note->desc_aligned_size;
  if (!image_contains_range(image.size(), old_desc_end, old_aligned_end - old_desc_end))
    return false;

  const size_t encoding_delta = encoded.size() - *old_size;
  const uint32_t new_desc_size = note->desc_size + static_cast<uint32_t>(encoding_delta);
  const uint64_t new_desc_aligned_size = align_up(new_desc_size, 4);
  const uint64_t note_delta = new_desc_aligned_size - note->desc_aligned_size;
  if (note_delta != 0) {
    std::vector<uint8_t> padding(note_delta, 0);
    insert_file_bytes_shifting_alloc_vaddrs(image, header, shdrs, phdrs, old_aligned_end, padding,
                                            note->section_index);
  }

  const size_t tail_src = offset + *old_size;
  const size_t tail_size = static_cast<size_t>(old_desc_end - tail_src);
  std::memmove(image.data() + tail_src + encoding_delta, image.data() + tail_src, tail_size);
  std::memcpy(image.data() + offset, encoded.data(), encoded.size());

  const uint64_t new_desc_end = note->desc_offset + new_desc_size;
  const uint64_t new_aligned_end = note->desc_offset + new_desc_aligned_size;
  if (!image_contains_range(image.size(), new_desc_end, new_aligned_end - new_desc_end))
    return false;
  std::fill(image.begin() + static_cast<std::ptrdiff_t>(new_desc_end),
            image.begin() + static_cast<std::ptrdiff_t>(new_aligned_end), 0);

  Elf64_Nhdr nhdr{};
  std::memcpy(&nhdr, image.data() + note->header_offset, sizeof(nhdr));
  nhdr.n_descsz = new_desc_size;
  std::memcpy(image.data() + note->header_offset, &nhdr, sizeof(nhdr));
  return true;
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
    : image_(obj.image_data(), obj.image_data() + obj.image_size()), text_offset_(0), text_size_(0),
      text_vaddr_(0), text_tail_size_(0) {
  auto &text_secs = obj.text_sections();
  if (!text_secs.empty()) {
    text_offset_ = text_secs[0]->sectionOffset();
    text_size_ = text_secs[0]->size();
    text_vaddr_ = text_secs[0]->vaddr();
  }
}

bool CodeObjectPatcher::refresh_text_section_cache() {
  if (text_size_ == 0)
    return true;

  if (!image_contains_range(image_.size(), 0, sizeof(Elf64_Ehdr)))
    return false;
  Elf64_Ehdr header{};
  std::memcpy(&header, image_.data(), sizeof(header));
  const auto shdrs = read_section_headers(image_, header);

  std::optional<size_t> executable_candidate;
  for (size_t i = 0; i < shdrs.size(); ++i) {
    const Elf64_Shdr &shdr = shdrs[i];
    if (shdr.sh_type != SHT_PROGBITS || (shdr.sh_flags & SHF_EXECINSTR) == 0 ||
        shdr.sh_size != text_size_)
      continue;
    if (const auto name = section_name(image_, header, shdrs, i); name && *name == ".text") {
      text_offset_ = shdr.sh_offset;
      return true;
    }
    if (!executable_candidate)
      executable_candidate = i;
  }

  if (const auto text_index = find_text_section(shdrs, text_offset_, text_size_)) {
    text_offset_ = shdrs[*text_index].sh_offset;
    return true;
  }

  if (executable_candidate) {
    text_offset_ = shdrs[*executable_candidate].sh_offset;
    return true;
  }
  return false;
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

bool CodeObjectPatcher::replace_text(std::span<const uint8_t> new_text) {
  if (text_size_ == 0)
    return false;
  if (new_text.size() < text_size_)
    return false;
  if (!image_contains_range(image_.size(), text_offset_, text_size_))
    return false;

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
  const uint64_t old_text_end_file = text_offset_ + text_size_;
  const uint64_t old_text_end_vaddr = text_header.sh_addr + text_size_;
  const uint64_t growth = new_text.size() - text_size_;

  if (growth != 0) {
    // Growing .text can shift later LOAD segments. Pad the inserted file range
    // so shifted LOADs keep p_offset % p_align congruent with p_vaddr % p_align.
    const uint64_t file_delta_alignment = shifted_load_delta_alignment(phdrs, old_text_end_file);
    const uint64_t padded_file_delta = align_up(growth, file_delta_alignment);
    assert(padded_file_delta >= growth && "aligned text growth underflowed");
    assert(padded_file_delta % sizeof(uint32_t) == 0 && "text growth must stay word-aligned");

    std::vector<uint8_t> inserted(padded_file_delta, 0);

    std::vector<bool> shift_section_vaddr(shdrs.size(), false);
    for (size_t i = 0; i < shdrs.size(); ++i) {
      if (i == *text_index || shdrs[i].sh_type == SHT_NULL)
        continue;
      if ((shdrs[i].sh_flags & SHF_ALLOC) != 0 && shdrs[i].sh_addr >= old_text_end_vaddr)
        shift_section_vaddr[i] = true;
    }

    std::vector<bool> shift_segment_vaddr(phdrs.size(), false);
    for (size_t i = 0; i < phdrs.size(); ++i) {
      if (phdrs[i].p_vaddr >= old_text_end_vaddr && phdrs[i].p_offset >= old_text_end_file)
        shift_segment_vaddr[i] = true;
    }

    insert_file_bytes(image_, header, shdrs, phdrs, old_text_end_file, inserted, *text_index, true);

    for (size_t i = 0; i < shdrs.size(); ++i) {
      if (!shift_section_vaddr[i])
        continue;
      [[maybe_unused]] const uint64_t old_addr = shdrs[i].sh_addr;
      shdrs[i].sh_addr += padded_file_delta;
      assert((shdrs[i].sh_addralign <= 1 ||
              shdrs[i].sh_addr % shdrs[i].sh_addralign == old_addr % shdrs[i].sh_addralign) &&
             "shifted allocated section lost its address alignment residue");
    }

    shift_symbols_in_moved_sections(image_, header, shdrs, shift_section_vaddr, padded_file_delta);
    shift_relocation_offsets_in_moved_sections(image_, header, shdrs, shift_section_vaddr,
                                               padded_file_delta);
    shift_dynamic_pointers_in_moved_sections(image_, shdrs, shift_section_vaddr, padded_file_delta);
    adjust_kernel_descriptor_entry_offsets_in_moved_sections(image_, shdrs, shift_section_vaddr,
                                                             padded_file_delta);

    for (size_t i = 0; i < phdrs.size(); ++i) {
      if (!shift_segment_vaddr[i])
        continue;
      assert((phdrs[i].p_align <= 1 || padded_file_delta % phdrs[i].p_align == 0) &&
             "text padding does not preserve shifted LOAD alignment");
      phdrs[i].p_vaddr += padded_file_delta;
      phdrs[i].p_paddr += padded_file_delta;
      assert((phdrs[i].p_align <= 1 ||
              phdrs[i].p_offset % phdrs[i].p_align == phdrs[i].p_vaddr % phdrs[i].p_align) &&
             "shifted LOAD lost file/virtual address congruence");
    }
  }

  std::memcpy(image_.data() + text_offset_, new_text.data(), new_text.size());
  shdrs[*text_index].sh_size = new_text.size();
  text_size_ = new_text.size();

  for (const Elf64_Phdr &phdr : phdrs) {
    if (phdr.p_type != PT_LOAD || phdr.p_align <= 1)
      continue;
    assert(phdr.p_offset % phdr.p_align == phdr.p_vaddr % phdr.p_align &&
           "patched LOAD lost file/virtual address congruence");
  }

  write_elf_tables(image_, header, shdrs, phdrs);
  return true;
}

void CodeObjectPatcher::update_elf_flags(uint32_t new_mach) {
  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image_.data());
  // XNACK and SRAMECC selections describe target hardware capabilities, not
  // source-code semantics. Normalize them along with the exact target MACH.
  ehdr->e_flags = elf_flags_for_target(ehdr->e_flags, new_mach);
}

bool CodeObjectPatcher::patch_metadata_target_isa(std::string_view old_isa,
                                                  std::string_view new_isa) {
  if (old_isa.empty() || old_isa.size() != new_isa.size())
    return false;

  auto search_start = image_.begin();
  while (true) {
    const auto it = std::search(search_start, image_.end(), old_isa.begin(), old_isa.end());
    if (it == image_.end())
      return true;

    std::copy(new_isa.begin(), new_isa.end(), it);
    search_start = it + new_isa.size();
  }
}

bool CodeObjectPatcher::patch_kernel_descriptor(uint64_t file_offset,
                                                std::span<const uint8_t> descriptor) {
  if (!image_contains_range(image_.size(), file_offset, descriptor.size()))
    return false;

  std::memcpy(image_.data() + file_offset, descriptor.data(), descriptor.size());
  return true;
}

bool CodeObjectPatcher::patch_metadata_vgpr_count(uint32_t vgpr_count) {
  static constexpr std::array<uint8_t, 12> kVgprCountKey = {0xAB, '.', 'v', 'g', 'p', 'r',
                                                            '_',  'c', 'o', 'u', 'n', 't'};

  bool patched_any = false;
  auto search_start = image_.begin();
  while (true) {
    const auto it =
        std::search(search_start, image_.end(), kVgprCountKey.begin(), kVgprCountKey.end());
    if (it == image_.end())
      return true;

    const size_t key_offset = static_cast<size_t>(std::distance(image_.begin(), it));
    const size_t value_offset = key_offset + kVgprCountKey.size();
    const auto current_value = read_msgpack_uint(image_, value_offset);
    if (!current_value)
      return false;
    if (*current_value < vgpr_count && !patch_msgpack_uint(image_, value_offset, vgpr_count))
      return false;
    patched_any = true;
    search_start = it + kVgprCountKey.size();
  }
  return patched_any;
}

bool CodeObjectPatcher::patch_metadata_sgpr_count(uint32_t sgpr_count) {
  static constexpr std::array<uint8_t, 12> kSgprCountKey = {0xAB, '.', 's', 'g', 'p', 'r',
                                                            '_',  'c', 'o', 'u', 'n', 't'};

  bool patched_any = false;
  auto search_start = image_.begin();
  while (true) {
    const auto it =
        std::search(search_start, image_.end(), kSgprCountKey.begin(), kSgprCountKey.end());
    if (it == image_.end())
      return true;

    const size_t key_offset = static_cast<size_t>(std::distance(image_.begin(), it));
    const size_t value_offset = key_offset + kSgprCountKey.size();
    const auto current_value = read_msgpack_uint(image_, value_offset);
    if (!current_value)
      return false;
    if (*current_value < sgpr_count && !patch_msgpack_uint(image_, value_offset, sgpr_count))
      return false;
    patched_any = true;
    search_start = it + kSgprCountKey.size();
  }
  return patched_any;
}

bool CodeObjectPatcher::patch_metadata_private_segment_fixed_sizes(
    std::span<const KdTranslation> translations) {
  static constexpr std::array<uint8_t, 28> kPrivateSizeKey = {
      0xBB, '.', 'p', 'r', 'i', 'v', 'a', 't', 'e', '_', 's', 'e', 'g', 'm',
      'e',  'n', 't', '_', 'f', 'i', 'x', 'e', 'd', '_', 's', 'i', 'z', 'e'};
  static constexpr std::array<uint8_t, 8> kSymbolKey = {0xA7, '.', 's', 'y', 'm', 'b', 'o', 'l'};

  struct Patch {
    std::string_view symbol_name;
    uint32_t current_size = 0;
    uint32_t target_size = 0;
  };
  std::vector<Patch> pending;
  pending.reserve(translations.size());
  for (const KdTranslation &translation : translations) {
    if (translation.private_spill_zone_bytes == 0)
      continue;
    if (translation.symbol_name.empty())
      return false;
    pending.push_back({translation.symbol_name, translation.private_spill_zone_base,
                       translation.target_private_size});
  }
  if (pending.empty())
    return true;

  auto search_start = image_.begin();
  bool saw_key = false;
  struct PrivateCandidate {
    size_t value_offset = 0;
    uint32_t current_size = 0;
  };
  struct KernelMetadataCandidate {
    std::string symbol;
    PrivateCandidate private_size;
  };
  std::vector<KernelMetadataCandidate> candidates;
  std::optional<PrivateCandidate> pending_private;
  std::optional<std::string> pending_symbol;
  const auto record_candidate = [&](std::string symbol, const PrivateCandidate &candidate) {
    candidates.push_back({std::move(symbol), candidate});
  };

  while (true) {
    const auto private_it =
        std::search(search_start, image_.end(), kPrivateSizeKey.begin(), kPrivateSizeKey.end());
    const auto symbol_it =
        std::search(search_start, image_.end(), kSymbolKey.begin(), kSymbolKey.end());
    if (private_it == image_.end() && symbol_it == image_.end())
      break;

    saw_key = true;
    const bool use_private =
        private_it != image_.end() && (symbol_it == image_.end() || private_it < symbol_it);
    if (use_private) {
      const size_t key_offset = static_cast<size_t>(std::distance(image_.begin(), private_it));
      const size_t value_offset = key_offset + kPrivateSizeKey.size();
      const auto current_value = read_msgpack_uint(image_, value_offset);
      if (!current_value)
        return false;
      PrivateCandidate candidate{value_offset, *current_value};
      if (pending_symbol) {
        record_candidate(std::move(*pending_symbol), candidate);
        pending_symbol.reset();
      } else {
        pending_private = candidate;
      }
      search_start = private_it + kPrivateSizeKey.size();
      continue;
    }

    const size_t key_offset = static_cast<size_t>(std::distance(image_.begin(), symbol_it));
    const size_t value_offset = key_offset + kSymbolKey.size();
    const auto symbol = read_msgpack_string(image_, value_offset);
    if (!symbol)
      return false;
    if (pending_private) {
      record_candidate(std::string(symbol->first), *pending_private);
      pending_private.reset();
    } else {
      pending_symbol = std::string(symbol->first);
    }
    search_start = symbol_it + kSymbolKey.size();
  }
  if (!saw_key)
    return true;

  struct ValuePatch {
    size_t value_offset = 0;
    uint32_t target_size = 0;
  };
  std::vector<ValuePatch> value_patches;
  value_patches.reserve(pending.size());
  for (const Patch &patch : pending) {
    bool matched = false;
    for (const KernelMetadataCandidate &candidate : candidates) {
      if (candidate.symbol != patch.symbol_name)
        continue;
      if (candidate.private_size.current_size != patch.current_size)
        return false;
      value_patches.push_back({candidate.private_size.value_offset, patch.target_size});
      matched = true;
    }
    if (!matched)
      return false;
  }

  std::ranges::sort(value_patches, std::greater<>{}, &ValuePatch::value_offset);
  for (const ValuePatch &patch : value_patches) {
    if (!patch_msgpack_uint_resizable(image_, patch.value_offset, patch.target_size))
      return false;
  }
  return refresh_text_section_cache();
}

bool CodeObjectPatcher::apply_kernel_descriptor_translation(const KdTranslation &translation,
                                                            rj_code_arch_t target_arch) {
  if (!image_contains_range(image_.size(), translation.descriptor_file_offset, sizeof(KD)))
    return false;

  std::optional<uint64_t> prologue_entry;
  if (!translation.prologue_words.empty()) {
    prologue_entry = append_kernel_entry_prologue(
        translation.entry_text_offset, translation.prologue_words, target_arch,
        kernel_entry_prologue_long_branch_sgpr_pair(translation, target_arch));
    if (!prologue_entry)
      return false;
  }

  auto *desc = reinterpret_cast<KD *>(image_.data() + translation.descriptor_file_offset);

  AMDHSA_BITS_SET(desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  translation.target_vgpr_granulated);
  AMDHSA_BITS_SET(desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  translation.target_sgpr_granulated);

  if (target_uses_gfx90a_accum_offset(target_arch) && translation.target_accvgpr_base != 0) {
    // GFX90A-style descriptors encode the first AccVGPR as (field + 1) * 4.
    // KernelDescriptorTranslator may move this base upward when semantic
    // lowering needs ordinary VGPR scratch above the source AccVGPR window, so
    // the patcher must write the recomputed base alongside the VGPR allocation.
    const uint32_t encoded_accum_offset = (translation.target_accvgpr_base / 4) - 1;
    AMDHSA_BITS_SET(desc->compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                    encoded_accum_offset);
  }

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
    const uint32_t wave32 = translation.target_wave_size == 32 ? 1u : 0u;
    AMDHSA_BITS_SET(desc->kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
                    wave32);
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
  } else if (target_uses_gfx90a_accum_offset(target_arch) && translation.target_accvgpr_base != 0) {
    // On GFX90A/GFX942/GFX950, AccVGPRs are placed by ACCUM_OFFSET rather than
    // by the ordinary VGPR count. KernelDescriptorTranslator decides whether the
    // base must move up to make room for semantic-lowering scratch; the patcher
    // only materializes that already-translated target base.
    assert(translation.target_accvgpr_base >= 4 &&
           "ACCUM_OFFSET base must encode at least 4 VGPRs");
    assert(translation.target_accvgpr_base % 4 == 0 && "ACCUM_OFFSET base must be 4-VGPR aligned");
    AMDHSA_BITS_SET(desc->compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                    (translation.target_accvgpr_base / 4 - 1));
  }

  desc->private_segment_fixed_size = translation.target_private_size;
  desc->group_segment_fixed_size = translation.target_lds_size;
  if (translation.clears_kernarg_preload)
    desc->kernarg_preload = 0;
  AMDHSA_BITS_SET(desc->compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT,
                  translation.target_user_sgpr_count);
  const uint32_t enable_private_segment = translation.target_private_size != 0 ? 1u : 0u;
  AMDHSA_BITS_SET(desc->compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT,
                  enable_private_segment);

  if (prologue_entry) {
    if (!redirect_kernel_entry(translation.descriptor_file_offset, translation.entry_text_offset,
                               *prologue_entry))
      return false;
  }
  return true;
}

std::optional<uint64_t> CodeObjectPatcher::append_kernel_entry_prologue(
    uint64_t entry_text_offset, std::span<const uint32_t> prologue_words, rj_code_arch_t arch,
    std::optional<uint16_t> long_branch_sgpr_pair) {
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
  std::vector<uint32_t> branch_words;
  if (target_dwords >= std::numeric_limits<int16_t>::min() &&
      target_dwords <= std::numeric_limits<int16_t>::max()) {
    branch_words.push_back(build_s_branch(static_cast<int16_t>(target_dwords), arch));
  } else {
    if (!long_branch_sgpr_pair)
      return std::nullopt;
    branch_words = build_s_setpc_long_branch(branch_pc, entry_text_offset, *long_branch_sgpr_pair);
    if (branch_words.empty())
      return std::nullopt;
  }

  if (alignment_padding != 0) {
    std::vector<uint32_t> padding(alignment_padding / sizeof(uint32_t), build_s_nop(0, arch));
    append_cave_body(padding);
  }
  cave_words.insert(cave_words.end(), branch_words.begin(), branch_words.end());

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

void CodeObjectPatcher::overwrite_cave_body(uint64_t offset, std::span<const uint32_t> words) {
  const uint64_t byte_size = words.size() * sizeof(uint32_t);
  assert(offset % sizeof(uint32_t) == 0 && "cave body writes must be word-aligned");
  assert(offset + byte_size <= cave_body_.size() && "cave body overwrite out of bounds");
  std::memcpy(cave_body_.data() + offset, words.data(), byte_size);
}

void CodeObjectPatcher::truncate_cave_body(uint64_t size) {
  assert(size <= cave_body_.size() && "cannot grow cave body through truncate");
  assert(size % sizeof(uint32_t) == 0 && "cave body must remain word-aligned");
  cave_body_.resize(size);
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

std::vector<uint8_t> CodeObjectPatcher::emit() const & { return image_; }

std::vector<uint8_t> CodeObjectPatcher::emit() && { return std::move(image_); }

} // namespace rocjitsu
