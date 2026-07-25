// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/code_object_patcher.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/patch/error_report.h"
#include "rocjitsu/isa/arch/amdgpu/isa_properties.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <numeric>
// Standard library
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
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

[[nodiscard]] std::optional<size_t> find_text_section(std::span<const Elf64_Shdr> shdrs,
                                                      uint64_t text_offset, uint64_t text_size) {
  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (shdrs[i].sh_offset == text_offset && shdrs[i].sh_size == text_size)
      return i;
  }
  return std::nullopt;
}

[[nodiscard]] bool image_contains_range(size_t image_size, uint64_t file_offset, uint64_t size);

[[nodiscard]] std::optional<std::string_view> section_name(const std::vector<uint8_t> &image,
                                                           const Elf64_Ehdr &ehdr,
                                                           std::span<const Elf64_Shdr> shdrs,
                                                           size_t index) {
  if (ehdr.e_shstrndx >= shdrs.size() || index >= shdrs.size())
    return std::nullopt;
  const Elf64_Shdr &strings = shdrs[ehdr.e_shstrndx];
  if (strings.sh_type != SHT_STRTAB ||
      !image_contains_range(image.size(), strings.sh_offset, strings.sh_size) ||
      shdrs[index].sh_name >= strings.sh_size) {
    return std::nullopt;
  }

  const uint64_t name_offset = strings.sh_offset + shdrs[index].sh_name;
  const size_t available = static_cast<size_t>(strings.sh_size - shdrs[index].sh_name);
  const char *name = reinterpret_cast<const char *>(image.data() + name_offset);
  const size_t length = strnlen(name, available);
  if (length == available)
    return std::nullopt;
  return std::string_view(name, length);
}

[[nodiscard]] std::optional<uint64_t> read_uleb128(std::span<const uint8_t> bytes, size_t &cursor,
                                                   size_t end) {
  uint64_t value = 0;
  unsigned shift = 0;
  while (cursor < end) {
    const uint8_t byte = bytes[cursor++];
    const uint64_t payload = byte & 0x7fu;
    if (shift >= 64 || (payload << shift) >> shift != payload)
      return std::nullopt;
    value |= payload << shift;
    if ((byte & 0x80u) == 0)
      return value;
    shift += 7;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<int64_t> read_sleb128(std::span<const uint8_t> bytes, size_t &cursor,
                                                  size_t end) {
  uint64_t value = 0;
  unsigned shift = 0;
  uint8_t byte = 0;
  do {
    if (cursor >= end || shift >= 64)
      return std::nullopt;
    byte = bytes[cursor++];
    const uint64_t payload = byte & 0x7fu;
    if ((payload << shift) >> shift != payload)
      return std::nullopt;
    value |= payload << shift;
    shift += 7;
  } while ((byte & 0x80u) != 0);

  if ((byte & 0x40u) != 0 && shift < 64)
    value |= std::numeric_limits<uint64_t>::max() << shift;
  return static_cast<int64_t>(value);
}

[[nodiscard]] bool skip_uleb128(std::span<const uint8_t> bytes, size_t &cursor, size_t end) {
  return read_uleb128(bytes, cursor, end).has_value();
}

[[nodiscard]] bool skip_sleb128(std::span<const uint8_t> bytes, size_t &cursor, size_t end) {
  return read_sleb128(bytes, cursor, end).has_value();
}

[[nodiscard]] bool skip_cfi_expression(std::span<const uint8_t> bytes, size_t &cursor, size_t end) {
  const auto length = read_uleb128(bytes, cursor, end);
  if (!length || *length > end - cursor)
    return false;
  cursor += static_cast<size_t>(*length);
  return true;
}

/// @brief Return whether a CFI program describes one state for the whole FDE.
///
/// Descriptorless DBT may change instruction offsets inside a callable. CFI
/// state that advances at particular PCs would need its own instruction-level
/// relocation map; range-wide AMDGPU rules can be preserved by relocating only
/// the FDE start and end. Parse operands rather than byte-scanning so opcodes
/// embedded in DWARF expressions do not cause false decisions.
[[nodiscard]] bool cfi_program_is_location_invariant(std::span<const uint8_t> bytes) {
  size_t cursor = 0;
  while (cursor < bytes.size()) {
    const uint8_t opcode = bytes[cursor++];
    const uint8_t primary = opcode & 0xc0u;
    if (primary == 0x40u)
      return false; // DW_CFA_advance_loc
    if (primary == 0x80u) {
      if (!skip_uleb128(bytes, cursor, bytes.size()))
        return false; // DW_CFA_offset
      continue;
    }
    if (primary == 0xc0u)
      continue; // DW_CFA_restore

    switch (opcode) {
    case 0x00: // DW_CFA_nop
    case 0x0a: // DW_CFA_remember_state
    case 0x0b: // DW_CFA_restore_state
    case 0x2d: // DW_CFA_GNU_window_save / AArch64 negate_ra_state
      break;
    case 0x01: // DW_CFA_set_loc
    case 0x02: // DW_CFA_advance_loc1
    case 0x03: // DW_CFA_advance_loc2
    case 0x04: // DW_CFA_advance_loc4
      return false;
    case 0x05: // DW_CFA_offset_extended
    case 0x09: // DW_CFA_register
    case 0x0c: // DW_CFA_def_cfa
    case 0x14: // DW_CFA_val_offset
    case 0x2f: // DW_CFA_GNU_negative_offset_extended
      if (!skip_uleb128(bytes, cursor, bytes.size()) ||
          !skip_uleb128(bytes, cursor, bytes.size())) {
        return false;
      }
      break;
    case 0x06: // DW_CFA_restore_extended
    case 0x07: // DW_CFA_undefined
    case 0x08: // DW_CFA_same_value
    case 0x0d: // DW_CFA_def_cfa_register
    case 0x0e: // DW_CFA_def_cfa_offset
    case 0x2e: // DW_CFA_GNU_args_size
      if (!skip_uleb128(bytes, cursor, bytes.size()))
        return false;
      break;
    case 0x0f: // DW_CFA_def_cfa_expression
      if (!skip_cfi_expression(bytes, cursor, bytes.size()))
        return false;
      break;
    case 0x10: // DW_CFA_expression
    case 0x16: // DW_CFA_val_expression
      if (!skip_uleb128(bytes, cursor, bytes.size()) ||
          !skip_cfi_expression(bytes, cursor, bytes.size())) {
        return false;
      }
      break;
    case 0x11: // DW_CFA_offset_extended_sf
    case 0x12: // DW_CFA_def_cfa_sf
    case 0x15: // DW_CFA_val_offset_sf
      if (!skip_uleb128(bytes, cursor, bytes.size()) ||
          !skip_sleb128(bytes, cursor, bytes.size())) {
        return false;
      }
      break;
    case 0x13: // DW_CFA_def_cfa_offset_sf
      if (!skip_sleb128(bytes, cursor, bytes.size()))
        return false;
      break;
    default:
      return false;
    }
  }
  return true;
}

struct PlacedTextOffset {
  uint64_t target_offset = 0;
  bool is_instruction_start = false;
};
using PlacedTextOffsets = std::unordered_map<uint64_t, PlacedTextOffset>;

/// @brief Relocate the PC transitions in one FDE CFI program in place.
///
/// The linked AMDGPU form uses a four-byte code-alignment factor and compact
/// advance_loc opcodes. Re-encode each transition through DBT's exact text
/// offset map, preserving the original operand width so the `.eh_frame`
/// section itself does not change size.
[[nodiscard]] bool
relocate_fde_cfi_program(std::span<const uint8_t> bytes, uint64_t source_start, uint64_t source_end,
                         uint64_t target_start, uint64_t target_end, uint64_t text_vaddr,
                         uint64_t code_alignment_factor, const PlacedTextOffsets &targets_by_source,
                         std::vector<uint8_t> &relocated_out, std::string *error_out) {
  const auto fail = [error_out](const char *message) {
    report(error_out, message);
    return false;
  };
  if (code_alignment_factor == 0)
    return fail(".eh_frame CIE has a zero code-alignment factor");

  uint64_t source_location = source_start;
  uint64_t target_location = target_start;
  std::vector<uint8_t> relocated;
  relocated.reserve(bytes.size());
  const auto emit_advance = [&](uint64_t units) -> bool {
    if (units <= 0x3fu) {
      relocated.push_back(static_cast<uint8_t>(0x40u | units));
      return true;
    }
    if (units <= std::numeric_limits<uint8_t>::max()) {
      relocated.push_back(0x02);
      relocated.push_back(static_cast<uint8_t>(units));
      return true;
    }
    if (units <= std::numeric_limits<uint16_t>::max()) {
      relocated.push_back(0x03);
      const uint16_t encoded = static_cast<uint16_t>(units);
      const auto *data = reinterpret_cast<const uint8_t *>(&encoded);
      relocated.insert(relocated.end(), data, data + sizeof(encoded));
      return true;
    }
    if (units <= std::numeric_limits<uint32_t>::max()) {
      relocated.push_back(0x04);
      const uint32_t encoded = static_cast<uint32_t>(units);
      const auto *data = reinterpret_cast<const uint8_t *>(&encoded);
      relocated.insert(relocated.end(), data, data + sizeof(encoded));
      return true;
    }
    return fail("relocated .eh_frame CFI advance_loc4 no longer fits");
  };
  const auto relocate_advance = [&](uint64_t source_units, uint64_t &target_units) -> bool {
    if (source_units >
        (std::numeric_limits<uint64_t>::max() - source_location) / code_alignment_factor) {
      return fail(".eh_frame CFI source location overflows");
    }
    const uint64_t next_source = source_location + source_units * code_alignment_factor;
    if (next_source > source_end)
      return fail(".eh_frame CFI location has no exact text relocation");
    const auto targets_it = targets_by_source.find(next_source);
    if (targets_it == targets_by_source.end())
      return fail(".eh_frame CFI location has no monotonic text relocation");
    const uint64_t target = targets_it->second.target_offset;
    if (target < target_location || target > target_end)
      return fail(".eh_frame CFI location has no monotonic text relocation");
    const uint64_t target_delta = target - target_location;
    if (target_delta % code_alignment_factor != 0)
      return fail(".eh_frame CFI advance is not code-aligned after relocation");
    target_units = target_delta / code_alignment_factor;
    source_location = next_source;
    target_location = target;
    return true;
  };
  const auto relocate_set_location = [&](uint64_t source_vaddr, uint64_t &target_vaddr) -> bool {
    if (source_vaddr < text_vaddr)
      return fail(".eh_frame CFI set_loc points before .text");
    const uint64_t source_offset = source_vaddr - text_vaddr;
    if (source_offset < source_location || source_offset > source_end) {
      return fail(".eh_frame CFI set_loc has no exact text relocation");
    }
    const auto targets_it = targets_by_source.find(source_offset);
    if (targets_it == targets_by_source.end())
      return fail(".eh_frame CFI set_loc has no representable monotonic relocation");
    const uint64_t target = targets_it->second.target_offset;
    if (target < target_location || target > target_end ||
        text_vaddr > std::numeric_limits<uint64_t>::max() - target)
      return fail(".eh_frame CFI set_loc has no representable monotonic relocation");
    source_location = source_offset;
    target_location = target;
    target_vaddr = text_vaddr + target_location;
    return true;
  };

  size_t cursor = 0;
  while (cursor < bytes.size()) {
    const size_t opcode_offset = cursor;
    const uint8_t opcode = bytes[cursor++];
    const uint8_t primary = opcode & 0xc0u;
    if (primary == 0x40u) {
      uint64_t target_units = 0;
      if (!relocate_advance(opcode & 0x3fu, target_units))
        return false;
      if (!emit_advance(target_units))
        return false;
      continue;
    }
    if (primary == 0x80u) {
      if (!skip_uleb128(bytes, cursor, bytes.size()))
        return fail("allocated .eh_frame has malformed CFI");
      relocated.insert(relocated.end(), bytes.begin() + opcode_offset, bytes.begin() + cursor);
      continue;
    }
    if (primary == 0xc0u) {
      relocated.push_back(opcode);
      continue;
    }

    switch (opcode) {
    case 0x00: // DW_CFA_nop
      continue;
    case 0x0a: // DW_CFA_remember_state
    case 0x0b: // DW_CFA_restore_state
    case 0x2d: // DW_CFA_GNU_window_save / AArch64 negate_ra_state
      break;
    case 0x01: { // DW_CFA_set_loc
      if (bytes.size() - cursor < sizeof(uint64_t))
        return fail("allocated .eh_frame has a truncated CFI set_loc");
      uint64_t source_vaddr = 0;
      std::memcpy(&source_vaddr, bytes.data() + cursor, sizeof(source_vaddr));
      uint64_t target_vaddr = 0;
      if (!relocate_set_location(source_vaddr, target_vaddr))
        return false;
      relocated.push_back(opcode);
      const auto *data = reinterpret_cast<const uint8_t *>(&target_vaddr);
      relocated.insert(relocated.end(), data, data + sizeof(target_vaddr));
      cursor += sizeof(target_vaddr);
      continue;
    }
    case 0x02: { // DW_CFA_advance_loc1
      if (cursor == bytes.size())
        return fail("allocated .eh_frame has a truncated CFI advance_loc1");
      uint64_t target_units = 0;
      if (!relocate_advance(bytes[cursor], target_units))
        return false;
      ++cursor;
      if (!emit_advance(target_units))
        return false;
      continue;
    }
    case 0x03: { // DW_CFA_advance_loc2
      if (bytes.size() - cursor < sizeof(uint16_t))
        return fail("allocated .eh_frame has a truncated CFI advance_loc2");
      uint16_t source_units = 0;
      std::memcpy(&source_units, bytes.data() + cursor, sizeof(source_units));
      uint64_t target_units = 0;
      if (!relocate_advance(source_units, target_units))
        return false;
      cursor += sizeof(source_units);
      if (!emit_advance(target_units))
        return false;
      continue;
    }
    case 0x04: { // DW_CFA_advance_loc4
      if (bytes.size() - cursor < sizeof(uint32_t))
        return fail("allocated .eh_frame has a truncated CFI advance_loc4");
      uint32_t source_units = 0;
      std::memcpy(&source_units, bytes.data() + cursor, sizeof(source_units));
      uint64_t target_units = 0;
      if (!relocate_advance(source_units, target_units))
        return false;
      cursor += sizeof(source_units);
      if (!emit_advance(target_units))
        return false;
      continue;
    }
    case 0x05: // DW_CFA_offset_extended
    case 0x09: // DW_CFA_register
    case 0x0c: // DW_CFA_def_cfa
    case 0x14: // DW_CFA_val_offset
    case 0x2f: // DW_CFA_GNU_negative_offset_extended
      if (!skip_uleb128(bytes, cursor, bytes.size()) ||
          !skip_uleb128(bytes, cursor, bytes.size())) {
        return fail("allocated .eh_frame has malformed CFI");
      }
      break;
    case 0x06: // DW_CFA_restore_extended
    case 0x07: // DW_CFA_undefined
    case 0x08: // DW_CFA_same_value
    case 0x0d: // DW_CFA_def_cfa_register
    case 0x0e: // DW_CFA_def_cfa_offset
    case 0x2e: // DW_CFA_GNU_args_size
      if (!skip_uleb128(bytes, cursor, bytes.size()))
        return fail("allocated .eh_frame has malformed CFI");
      break;
    case 0x0f: // DW_CFA_def_cfa_expression
      if (!skip_cfi_expression(bytes, cursor, bytes.size()))
        return fail("allocated .eh_frame has malformed CFI");
      break;
    case 0x10: // DW_CFA_expression
    case 0x16: // DW_CFA_val_expression
      if (!skip_uleb128(bytes, cursor, bytes.size()) ||
          !skip_cfi_expression(bytes, cursor, bytes.size())) {
        return fail("allocated .eh_frame has malformed CFI");
      }
      break;
    case 0x11: // DW_CFA_offset_extended_sf
    case 0x12: // DW_CFA_def_cfa_sf
    case 0x15: // DW_CFA_val_offset_sf
      if (!skip_uleb128(bytes, cursor, bytes.size()) ||
          !skip_sleb128(bytes, cursor, bytes.size())) {
        return fail("allocated .eh_frame has malformed CFI");
      }
      break;
    case 0x13: // DW_CFA_def_cfa_offset_sf
      if (!skip_sleb128(bytes, cursor, bytes.size()))
        return fail("allocated .eh_frame has malformed CFI");
      break;
    default:
      return fail("allocated .eh_frame uses an unsupported CFI opcode");
    }
    relocated.insert(relocated.end(), bytes.begin() + opcode_offset, bytes.begin() + cursor);
  }
  relocated_out = std::move(relocated);
  return true;
}

/// @brief Decoded CIE properties required to relocate one `.eh_frame` FDE.
struct EhFrameCie {
  uint8_t fde_encoding = 0;
  bool has_augmentation_length = false;
  bool initial_program_location_invariant = true;
  uint64_t code_alignment_factor = 0;
  size_t relocated_offset = 0;
};

void shift_symbols_in_moved_sections(std::vector<uint8_t> &image, const Elf64_Ehdr &ehdr,
                                     std::span<const Elf64_Shdr> shdrs,
                                     const std::vector<bool> &shift_section_vaddr, uint64_t delta);
void shift_relocation_offsets_in_moved_sections(std::vector<uint8_t> &image, const Elf64_Ehdr &ehdr,
                                                std::span<const Elf64_Shdr> shdrs,
                                                const std::vector<bool> &shift_section_vaddr,
                                                uint64_t delta);

/// @brief Relocate allocated `.eh_frame` FDEs through DBT's text offset map.
///
/// The supported AMDGPU form is the linked-HSACO `zR` CIE with
/// pcrel+sdata4 FDE starts. Function ranges and FDE CFI location advances are
/// relocated through the exact instruction offset map. Unsupported forms fail
/// closed instead of leaving unwind PCs pointing at stale B0 code.
[[nodiscard]] bool relocate_eh_frame_fdes(std::vector<uint8_t> &image, Elf64_Ehdr &ehdr,
                                          std::vector<Elf64_Shdr> &shdrs,
                                          std::vector<Elf64_Phdr> &phdrs, size_t text_index,
                                          uint64_t old_text_size, uint64_t new_text_size,
                                          std::span<const TextOffsetRelocation> relocations,
                                          std::vector<bool> &shift_section_vaddr,
                                          uint64_t future_text_delta, std::string *error_out) {
  const auto fail = [error_out](const char *message) {
    report(error_out, message);
    return false;
  };
  for (const TextOffsetRelocation &relocation : relocations) {
    if (relocation.source_offset > old_text_size || relocation.target_offset > new_text_size)
      return fail(".eh_frame relocation map contains an out-of-range text offset");
  }

  const bool layout_changed = old_text_size != new_text_size ||
                              std::ranges::any_of(relocations, [](const auto &relocation) {
                                return relocation.source_offset != relocation.target_offset;
                              });
  if (!layout_changed)
    return true;
  if (shift_section_vaddr.size() != shdrs.size())
    return fail(".eh_frame relocation has incomplete final section placement");

  std::vector<size_t> eh_frame_sections;
  for (size_t section_index = 0; section_index < shdrs.size(); ++section_index) {
    const auto name = section_name(image, ehdr, shdrs, section_index);
    if (!name)
      return fail("ELF section names are malformed while locating .eh_frame");
    if (*name == ".eh_frame_hdr" && (shdrs[section_index].sh_flags & SHF_ALLOC) != 0)
      return fail("allocated .eh_frame_hdr relocation is unsupported");
    if (*name == ".eh_frame" && (shdrs[section_index].sh_flags & SHF_ALLOC) != 0)
      eh_frame_sections.push_back(section_index);
  }
  if (eh_frame_sections.empty())
    return true;

  struct RelocationPlacement {
    PlacedTextOffsets offsets;
  };
  std::vector<RelocationPlacement> placements;
  std::unordered_map<uint64_t, size_t> placement_index_by_id;
  for (const TextOffsetRelocation &relocation : relocations) {
    auto [placement_it, inserted] =
        placement_index_by_id.try_emplace(relocation.placement_id, placements.size());
    if (inserted)
      placements.emplace_back();
    PlacedTextOffsets &offsets = placements[placement_it->second].offsets;
    auto [offset_it, offset_inserted] = offsets.try_emplace(
        relocation.source_offset,
        PlacedTextOffset{.target_offset = relocation.target_offset,
                         .is_instruction_start = relocation.source_is_instruction_start});
    if (!offset_inserted) {
      if (offset_it->second.target_offset != relocation.target_offset)
        return fail(".eh_frame relocation placement maps one source offset to multiple targets");
      offset_it->second.is_instruction_start |= relocation.source_is_instruction_start;
    }
  }

  const Elf64_Shdr &text = shdrs[text_index];
  if (text.sh_addr > std::numeric_limits<uint64_t>::max() - old_text_size)
    return fail(".text address range overflows while relocating .eh_frame");
  const uint64_t old_text_end = text.sh_addr + old_text_size;

  for (const size_t section_index : eh_frame_sections) {
    const Elf64_Shdr section = shdrs[section_index];
    if (section.sh_type != SHT_PROGBITS ||
        !image_contains_range(image.size(), section.sh_offset, section.sh_size)) {
      return fail("allocated .eh_frame section is malformed");
    }
    const uint64_t section_future_delta =
        shift_section_vaddr[section_index] ? future_text_delta : 0;
    if (section.sh_addr > std::numeric_limits<uint64_t>::max() - section_future_delta) {
      return fail("relocated .eh_frame section address overflows");
    }
    // Parse pcrel fields at their source address, then encode them relative to
    // the section's final address after text growth shifts later LOADs.
    const uint64_t target_section_vaddr = section.sh_addr + section_future_delta;
    const auto bytes =
        std::span<uint8_t>(image.data() + section.sh_offset, static_cast<size_t>(section.sh_size));
    const auto const_bytes = std::span<const uint8_t>(bytes.data(), bytes.size());
    std::vector<uint8_t> relocated_section;
    relocated_section.reserve(bytes.size());
    std::vector<size_t> relocated_initial_location_fields;
    bool had_terminator = false;
    std::unordered_map<uint64_t, EhFrameCie> cies;
    size_t cursor = 0;
    while (cursor < bytes.size()) {
      const size_t entry_start = cursor;
      if (bytes.size() - cursor < sizeof(uint32_t))
        return fail("allocated .eh_frame has a truncated entry length");
      uint32_t length = 0;
      std::memcpy(&length, bytes.data() + cursor, sizeof(length));
      cursor += sizeof(length);
      if (length == 0) {
        if (!std::ranges::all_of(bytes.subspan(cursor), [](uint8_t byte) { return byte == 0; }))
          return fail("allocated .eh_frame has nonzero data after its terminator");
        had_terminator = true;
        break;
      }
      if (length == std::numeric_limits<uint32_t>::max() || length < sizeof(uint32_t) ||
          length > bytes.size() - cursor) {
        return fail("allocated .eh_frame has an unsupported or invalid entry length");
      }
      const size_t entry_end = cursor + length;
      const size_t id_field = cursor;
      uint32_t cie_pointer = 0;
      std::memcpy(&cie_pointer, bytes.data() + cursor, sizeof(cie_pointer));
      cursor += sizeof(cie_pointer);

      if (cie_pointer == 0) {
        if (cursor >= entry_end)
          return fail("allocated .eh_frame has a truncated CIE");
        const uint8_t version = bytes[cursor++];
        if (version != 1)
          return fail("unsupported .eh_frame CIE version");
        const size_t augmentation_start = cursor;
        while (cursor < entry_end && bytes[cursor] != 0)
          ++cursor;
        if (cursor == entry_end)
          return fail("allocated .eh_frame CIE has an unterminated augmentation string");
        const std::string_view augmentation(
            reinterpret_cast<const char *>(bytes.data() + augmentation_start),
            cursor - augmentation_start);
        ++cursor;
        const auto code_alignment_factor = read_uleb128(const_bytes, cursor, entry_end);
        if (!code_alignment_factor || *code_alignment_factor == 0 ||
            !read_sleb128(const_bytes, cursor, entry_end)) {
          return fail("allocated .eh_frame CIE has malformed alignment fields");
        }
        if (cursor >= entry_end)
          return fail("allocated .eh_frame CIE is missing its return-address register");
        ++cursor;

        EhFrameCie cie;
        if (!augmentation.empty()) {
          if (augmentation.front() != 'z')
            return fail("allocated .eh_frame CIE uses an unsupported augmentation format");
          cie.has_augmentation_length = true;
          const auto augmentation_size = read_uleb128(const_bytes, cursor, entry_end);
          if (!augmentation_size || *augmentation_size > entry_end - cursor)
            return fail("allocated .eh_frame CIE has malformed augmentation data");
          const size_t augmentation_end = cursor + static_cast<size_t>(*augmentation_size);
          for (const char kind : augmentation.substr(1)) {
            switch (kind) {
            case 'R':
              if (cursor >= augmentation_end)
                return fail("allocated .eh_frame CIE is missing its FDE encoding");
              cie.fde_encoding = bytes[cursor++];
              break;
            case 'L':
              return fail("allocated .eh_frame CIE uses unsupported LSDA augmentation");
            case 'S':
              break;
            default:
              // Personality encodings and vendor augmentation data need a
              // complete encoded-pointer parser before their FDE layout can be
              // trusted.
              return fail("allocated .eh_frame CIE uses an unsupported augmentation");
            }
          }
          if (cursor > augmentation_end)
            return fail("allocated .eh_frame CIE augmentation exceeds its declared size");
          cursor = augmentation_end;
        }
        cie.code_alignment_factor = *code_alignment_factor;
        cie.initial_program_location_invariant =
            cfi_program_is_location_invariant(const_bytes.subspan(cursor, entry_end - cursor));
        cie.relocated_offset = relocated_section.size();
        cies.emplace(entry_start, cie);
        relocated_section.insert(relocated_section.end(), bytes.begin() + entry_start,
                                 bytes.begin() + entry_end);
        cursor = entry_end;
        continue;
      }

      if (cie_pointer > id_field)
        return fail("allocated .eh_frame FDE has an invalid CIE pointer");
      const uint64_t cie_start = id_field - cie_pointer;
      const auto cie_it = cies.find(cie_start);
      if (cie_it == cies.end())
        return fail("allocated .eh_frame FDE references an unknown CIE");
      const EhFrameCie &cie = cie_it->second;
      constexpr uint8_t kDwEhPePcrelSdata4 = 0x1b;
      if (cie.fde_encoding != kDwEhPePcrelSdata4 || entry_end - cursor < 2 * sizeof(uint32_t)) {
        return fail("allocated .eh_frame FDE uses an unsupported encoding or is truncated");
      }

      const size_t initial_location_field = cursor;
      int32_t initial_location_delta = 0;
      std::memcpy(&initial_location_delta, bytes.data() + cursor, sizeof(initial_location_delta));
      cursor += sizeof(initial_location_delta);
      int32_t address_range = 0;
      const size_t address_range_field = cursor;
      std::memcpy(&address_range, bytes.data() + cursor, sizeof(address_range));
      cursor += sizeof(address_range);
      if (address_range < 0 ||
          section.sh_addr >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - initial_location_field) {
        return fail("allocated .eh_frame FDE has an invalid address range");
      }
      const int64_t field_vaddr = static_cast<int64_t>(section.sh_addr + initial_location_field);
      if ((initial_location_delta > 0 &&
           field_vaddr > std::numeric_limits<int64_t>::max() - initial_location_delta) ||
          (initial_location_delta < 0 &&
           field_vaddr < std::numeric_limits<int64_t>::min() - initial_location_delta)) {
        return fail("allocated .eh_frame FDE initial location overflows");
      }
      const int64_t initial_location = field_vaddr + initial_location_delta;
      if (initial_location < 0)
        return fail("allocated .eh_frame FDE has a negative initial location");
      const uint64_t source_vaddr = static_cast<uint64_t>(initial_location);
      const uint64_t source_range = static_cast<uint32_t>(address_range);
      if (source_vaddr > std::numeric_limits<uint64_t>::max() - source_range)
        return fail("allocated .eh_frame FDE source range overflows");
      const uint64_t source_vaddr_end = source_vaddr + source_range;

      if (cie.has_augmentation_length) {
        const auto augmentation_size = read_uleb128(const_bytes, cursor, entry_end);
        if (!augmentation_size || *augmentation_size > entry_end - cursor)
          return fail("allocated .eh_frame FDE has malformed augmentation data");
        cursor += static_cast<size_t>(*augmentation_size);
      }

      const bool starts_in_text = source_vaddr >= text.sh_addr && source_vaddr < old_text_end;
      const bool overlaps_text = source_vaddr < old_text_end && source_vaddr_end > text.sh_addr;
      struct RelocatedFde {
        uint64_t target_vaddr = 0;
        uint64_t target_range_size = 0;
        std::vector<uint8_t> cfi;
      };
      std::vector<RelocatedFde> relocated_fdes;
      if (!starts_in_text) {
        if (overlaps_text)
          return fail("allocated .eh_frame FDE partially overlaps .text");
        relocated_fdes.push_back(
            {.target_vaddr = source_vaddr,
             .target_range_size = source_range,
             .cfi = std::vector<uint8_t>(bytes.begin() + cursor, bytes.begin() + entry_end)});
      } else {
        if (source_vaddr_end > old_text_end)
          return fail("allocated .eh_frame FDE extends beyond .text");
        const uint64_t source_start = source_vaddr - text.sh_addr;
        const uint64_t source_end = source_vaddr_end - text.sh_addr;
        // A helper may be copied into several kernel-local translated bodies.
        // Preserve an FDE for every copy: ELF can retain only one symbol value,
        // but unwind consumers must be able to cover traps and samples in all
        // emitted clones. Pair start/end mappings only within the same emitted
        // placement, and require a real instruction at the source start. This
        // excludes the block-end mapping from the preceding kernel when two
        // source functions meet at the same offset.
        for (const RelocationPlacement &placement : placements) {
          const auto target_start_it = placement.offsets.find(source_start);
          const auto target_end_it = placement.offsets.find(source_end);
          if (target_start_it == placement.offsets.end() ||
              !target_start_it->second.is_instruction_start ||
              target_end_it == placement.offsets.end()) {
            continue;
          }

          const uint64_t target_start = target_start_it->second.target_offset;
          const uint64_t target_end = target_end_it->second.target_offset;
          if (target_end < target_start ||
              (source_end != source_start && target_end == target_start) ||
              target_end > new_text_size) {
            continue;
          }

          bool affine = true;
          for (const auto &[source, target] : placement.offsets) {
            if (source < source_start || source > source_end)
              continue;
            if (source - source_start > target_end - target_start ||
                target.target_offset != target_start + (source - source_start)) {
              affine = false;
              break;
            }
          }
          if (!affine && !cie.initial_program_location_invariant)
            return fail("non-affine .eh_frame CIE has location-sensitive CFI");

          std::vector<uint8_t> relocated_cfi;
          if (!relocate_fde_cfi_program(const_bytes.subspan(cursor, entry_end - cursor),
                                        source_start, source_end, target_start, target_end,
                                        text.sh_addr, cie.code_alignment_factor, placement.offsets,
                                        relocated_cfi, error_out)) {
            return false;
          }
          if (text.sh_addr > std::numeric_limits<uint64_t>::max() - target_start)
            return fail("relocated .eh_frame FDE target address overflows");
          relocated_fdes.push_back({.target_vaddr = text.sh_addr + target_start,
                                    .target_range_size = target_end - target_start,
                                    .cfi = std::move(relocated_cfi)});
        }
        if (relocated_fdes.empty())
          return fail("allocated .eh_frame FDE lacks an exact instruction relocation range");
      }

      const size_t initial_location_in_entry = initial_location_field - entry_start;
      const size_t address_range_in_entry = address_range_field - entry_start;
      const size_t id_field_in_entry = id_field - entry_start;
      for (const RelocatedFde &relocated_fde : relocated_fdes) {
        const size_t relocated_entry_start = relocated_section.size();
        if (relocated_entry_start >
                std::numeric_limits<uint64_t>::max() - initial_location_in_entry ||
            target_section_vaddr > std::numeric_limits<uint64_t>::max() -
                                       (relocated_entry_start + initial_location_in_entry)) {
          return fail("relocated .eh_frame FDE field address overflows");
        }
        const uint64_t relocated_initial_location_field =
            relocated_entry_start + initial_location_in_entry;
        const uint64_t target_field_vaddr = target_section_vaddr + relocated_initial_location_field;
        if (relocated_fde.target_vaddr >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            target_field_vaddr > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          return fail("relocated .eh_frame FDE target is not representable");
        }
        const int64_t target_delta = static_cast<int64_t>(relocated_fde.target_vaddr) -
                                     static_cast<int64_t>(target_field_vaddr);
        if (target_delta < std::numeric_limits<int32_t>::min() ||
            target_delta > std::numeric_limits<int32_t>::max() ||
            relocated_fde.target_range_size >
                static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
          return fail("relocated .eh_frame FDE does not fit its pcrel+sdata4 encoding");
        }

        std::vector<uint8_t> relocated_entry(bytes.begin() + entry_start, bytes.begin() + cursor);
        relocated_entry.insert(relocated_entry.end(), relocated_fde.cfi.begin(),
                               relocated_fde.cfi.end());
        while (relocated_entry.size() % sizeof(uint32_t) != 0)
          relocated_entry.push_back(0);
        if (relocated_entry.size() < 2 * sizeof(uint32_t) ||
            relocated_entry.size() - sizeof(uint32_t) > std::numeric_limits<uint32_t>::max()) {
          return fail("relocated .eh_frame FDE length is not representable");
        }
        if (relocated_entry_start > std::numeric_limits<uint64_t>::max() - id_field_in_entry ||
            relocated_entry_start + id_field_in_entry < cie.relocated_offset ||
            relocated_entry_start + id_field_in_entry - cie.relocated_offset >
                std::numeric_limits<uint32_t>::max()) {
          return fail("relocated .eh_frame FDE CIE pointer is not representable");
        }
        const uint32_t encoded_length =
            static_cast<uint32_t>(relocated_entry.size() - sizeof(uint32_t));
        const uint32_t encoded_cie_pointer =
            static_cast<uint32_t>(relocated_entry_start + id_field_in_entry - cie.relocated_offset);
        const int32_t encoded_target_delta = static_cast<int32_t>(target_delta);
        const int32_t encoded_target_range = static_cast<int32_t>(relocated_fde.target_range_size);
        std::memcpy(relocated_entry.data(), &encoded_length, sizeof(encoded_length));
        std::memcpy(relocated_entry.data() + id_field_in_entry, &encoded_cie_pointer,
                    sizeof(encoded_cie_pointer));
        std::memcpy(relocated_entry.data() + initial_location_in_entry, &encoded_target_delta,
                    sizeof(encoded_target_delta));
        std::memcpy(relocated_entry.data() + address_range_in_entry, &encoded_target_range,
                    sizeof(encoded_target_range));
        relocated_initial_location_fields.push_back(relocated_initial_location_field);
        relocated_section.insert(relocated_section.end(), relocated_entry.begin(),
                                 relocated_entry.end());
      }
      cursor = entry_end;
    }
    if (had_terminator)
      relocated_section.resize(relocated_section.size() + sizeof(uint32_t), 0);
    if (relocated_section.size() > bytes.size()) {
      const uint64_t insertion_file_offset = section.sh_offset + section.sh_size;
      const uint64_t required_growth = relocated_section.size() - bytes.size();
      const uint64_t file_alignment = shifted_load_delta_alignment(phdrs, insertion_file_offset);
      const uint64_t padded_growth = align_up(required_growth, file_alignment);
      if (padded_growth < required_growth || padded_growth > std::numeric_limits<size_t>::max() ||
          section.sh_size > std::numeric_limits<uint64_t>::max() - padded_growth) {
        return fail("relocated .eh_frame section growth overflows");
      }

      std::optional<size_t> owner_index;
      for (size_t phdr_index = 0; phdr_index < phdrs.size(); ++phdr_index) {
        const Elf64_Phdr &phdr = phdrs[phdr_index];
        if (phdr.p_type != PT_LOAD ||
            phdr.p_offset > std::numeric_limits<uint64_t>::max() - phdr.p_filesz)
          continue;
        if (section.sh_offset >= phdr.p_offset &&
            phdr.p_offset + phdr.p_filesz == insertion_file_offset) {
          owner_index = phdr_index;
          break;
        }
      }

      bool can_grow_in_place = false;
      if (owner_index &&
          section.sh_addr <= std::numeric_limits<uint64_t>::max() - section.sh_size) {
        const Elf64_Phdr &owner = phdrs[*owner_index];
        if (owner.p_vaddr <= std::numeric_limits<uint64_t>::max() - owner.p_memsz &&
            section.sh_addr + section.sh_size <= owner.p_vaddr + owner.p_memsz) {
          const uint64_t owner_vaddr_end = owner.p_vaddr + owner.p_memsz;
          if (owner_vaddr_end <= std::numeric_limits<uint64_t>::max() - padded_growth) {
            can_grow_in_place = std::ranges::none_of(phdrs, [&](const Elf64_Phdr &phdr) {
              return phdr.p_type == PT_LOAD && phdr.p_vaddr >= owner_vaddr_end &&
                     phdr.p_vaddr < owner_vaddr_end + padded_growth;
            });
          }
        }
      }

      if (can_grow_in_place) {
        std::vector<uint8_t> inserted(static_cast<size_t>(padded_growth), 0);
        insert_file_bytes(image, ehdr, shdrs, phdrs, insertion_file_offset, inserted, section_index,
                          true);
        shdrs[section_index].sh_size += padded_growth;
      } else {
        // Linked GPU objects commonly place .eh_frame immediately before the
        // executable LOAD. When cloning a shared helper requires more unwind
        // entries than that virtual gap can hold, move the rebuilt section to
        // the tail LOAD instead of shifting the executable image or dropping
        // coverage for a clone.
        std::optional<size_t> tail_load_index;
        uint64_t greatest_file_end = 0;
        uint64_t greatest_memory_end = 0;
        for (size_t phdr_index = 0; phdr_index < phdrs.size(); ++phdr_index) {
          const Elf64_Phdr &phdr = phdrs[phdr_index];
          if (phdr.p_type != PT_LOAD ||
              phdr.p_offset > std::numeric_limits<uint64_t>::max() - phdr.p_filesz ||
              phdr.p_vaddr > std::numeric_limits<uint64_t>::max() - phdr.p_memsz) {
            continue;
          }
          const uint64_t file_end = phdr.p_offset + phdr.p_filesz;
          const uint64_t memory_end = phdr.p_vaddr + phdr.p_memsz;
          greatest_file_end = std::max(greatest_file_end, file_end);
          if (!tail_load_index || memory_end > greatest_memory_end ||
              (memory_end == greatest_memory_end &&
               file_end > phdrs[*tail_load_index].p_offset + phdrs[*tail_load_index].p_filesz)) {
            tail_load_index = phdr_index;
            greatest_memory_end = memory_end;
          }
        }
        if (!tail_load_index)
          return fail("allocated .eh_frame has no LOAD available for expanded unwind data");

        const Elf64_Phdr tail_source = phdrs[*tail_load_index];
        if (tail_source.p_offset > std::numeric_limits<uint64_t>::max() - tail_source.p_filesz ||
            tail_source.p_vaddr > std::numeric_limits<uint64_t>::max() - tail_source.p_memsz ||
            tail_source.p_offset + tail_source.p_filesz != greatest_file_end ||
            tail_source.p_vaddr + tail_source.p_memsz != greatest_memory_end ||
            tail_source.p_memsz < tail_source.p_filesz) {
          return fail("allocated .eh_frame cannot identify a non-overlapping tail LOAD");
        }

        const uint64_t section_alignment = std::max<uint64_t>(section.sh_addralign, 1);
        const uint64_t tail_memory_end = tail_source.p_vaddr + tail_source.p_memsz;
        if (tail_memory_end > std::numeric_limits<uint64_t>::max() - (section_alignment - 1)) {
          return fail("allocated .eh_frame tail alignment overflows");
        }
        const uint64_t destination_vaddr = align_up(tail_memory_end, section_alignment);
        const uint64_t section_load_offset = destination_vaddr - tail_source.p_vaddr;
        if (section_load_offset < tail_source.p_filesz ||
            section_load_offset > std::numeric_limits<uint64_t>::max() - relocated_section.size()) {
          return fail("allocated .eh_frame tail placement overflows");
        }
        const uint64_t inserted_size =
            section_load_offset - tail_source.p_filesz + relocated_section.size();
        if (inserted_size > std::numeric_limits<size_t>::max() ||
            tail_source.p_offset > std::numeric_limits<uint64_t>::max() - tail_source.p_filesz ||
            tail_source.p_offset > std::numeric_limits<uint64_t>::max() - section_load_offset ||
            tail_source.p_offset + tail_source.p_filesz > image.size() ||
            tail_source.p_vaddr > std::numeric_limits<uint64_t>::max() - section_load_offset) {
          return fail("allocated .eh_frame tail insertion is not representable");
        }

        const uint64_t tail_insertion_file_offset = tail_source.p_offset + tail_source.p_filesz;
        std::vector<std::pair<size_t, uint64_t>> preserved_nobits_offsets;
        for (size_t other_index = 0; other_index < shdrs.size(); ++other_index) {
          const Elf64_Shdr &other = shdrs[other_index];
          if (other.sh_type != SHT_NOBITS || (other.sh_flags & SHF_ALLOC) == 0 ||
              other.sh_offset < tail_insertion_file_offset) {
            continue;
          }
          const bool remains_in_place = std::ranges::any_of(phdrs, [&](const Elf64_Phdr &phdr) {
            return phdr.p_type == PT_LOAD &&
                   phdr.p_vaddr <= std::numeric_limits<uint64_t>::max() - phdr.p_memsz &&
                   other.sh_addr >= phdr.p_vaddr && other.sh_addr - phdr.p_vaddr < phdr.p_memsz;
          });
          if (remains_in_place)
            preserved_nobits_offsets.emplace_back(other_index, other.sh_offset);
        }

        std::vector<uint8_t> inserted(static_cast<size_t>(inserted_size), 0);
        insert_file_bytes(image, ehdr, shdrs, phdrs, tail_insertion_file_offset, inserted,
                          std::nullopt, false);
        Elf64_Phdr &tail = phdrs[*tail_load_index];
        tail.p_offset = tail_source.p_offset;
        tail.p_filesz = section_load_offset + relocated_section.size();
        tail.p_memsz = tail.p_filesz;
        for (const auto &[other_index, old_offset] : preserved_nobits_offsets)
          shdrs[other_index].sh_offset = old_offset;

        const uint64_t destination_file_offset = tail_source.p_offset + section_load_offset;
        if (destination_file_offset % section_alignment != 0)
          return fail("allocated .eh_frame tail LOAD cannot preserve section alignment");
        if (destination_vaddr < section.sh_addr)
          return fail("allocated .eh_frame tail LOAD precedes its source address");
        tail.p_flags |= PF_R;
        const uint64_t move_delta = destination_vaddr - section.sh_addr;
        shdrs[section_index].sh_addr = destination_vaddr;
        shdrs[section_index].sh_offset = destination_file_offset;
        shdrs[section_index].sh_size = relocated_section.size();

        std::vector<bool> moved_section(shdrs.size(), false);
        moved_section[section_index] = true;
        shift_symbols_in_moved_sections(image, ehdr, shdrs, moved_section, move_delta);
        shift_relocation_offsets_in_moved_sections(image, ehdr, shdrs, moved_section, move_delta);

        if (text.sh_offset > std::numeric_limits<uint64_t>::max() - old_text_size)
          return fail("source .text file range overflows while moving .eh_frame");
        const uint64_t old_text_end_file = text.sh_offset + old_text_size;
        const bool destination_shifts_with_text = future_text_delta != 0 &&
                                                  destination_vaddr >= old_text_end &&
                                                  destination_file_offset >= old_text_end_file;
        shift_section_vaddr[section_index] = destination_shifts_with_text;
        const uint64_t destination_future_delta =
            destination_shifts_with_text ? future_text_delta : 0;
        if (destination_vaddr > std::numeric_limits<uint64_t>::max() - destination_future_delta) {
          return fail("relocated .eh_frame tail address overflows");
        }
        const uint64_t final_destination_vaddr = destination_vaddr + destination_future_delta;
        if (target_section_vaddr > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            final_destination_vaddr > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          return fail("relocated .eh_frame tail address is not representable");
        }
        const int64_t base_adjustment = static_cast<int64_t>(target_section_vaddr) -
                                        static_cast<int64_t>(final_destination_vaddr);
        for (const size_t field_offset : relocated_initial_location_fields) {
          if (field_offset > relocated_section.size() ||
              sizeof(int32_t) > relocated_section.size() - field_offset) {
            return fail("relocated .eh_frame FDE field lies outside the rebuilt section");
          }
          int32_t encoded_delta = 0;
          std::memcpy(&encoded_delta, relocated_section.data() + field_offset,
                      sizeof(encoded_delta));
          if ((encoded_delta > 0 &&
               base_adjustment > std::numeric_limits<int64_t>::max() - encoded_delta) ||
              (encoded_delta < 0 &&
               base_adjustment < std::numeric_limits<int64_t>::min() - encoded_delta)) {
            return fail("relocated .eh_frame tail PC adjustment overflows");
          }
          const int64_t adjusted_delta = base_adjustment + encoded_delta;
          if (adjusted_delta < std::numeric_limits<int32_t>::min() ||
              adjusted_delta > std::numeric_limits<int32_t>::max()) {
            return fail("relocated .eh_frame tail PC does not fit pcrel+sdata4");
          }
          const int32_t adjusted = static_cast<int32_t>(adjusted_delta);
          std::memcpy(relocated_section.data() + field_offset, &adjusted, sizeof(adjusted));
        }
      }
    }
    const auto relocated_bytes =
        std::span<uint8_t>(image.data() + shdrs[section_index].sh_offset,
                           static_cast<size_t>(shdrs[section_index].sh_size));
    std::ranges::fill(relocated_bytes, 0);
    std::ranges::copy(relocated_section, relocated_bytes.begin());
  }
  return true;
}

[[nodiscard]] bool target_supports_wave32(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250;
}

[[nodiscard]] bool target_uses_gfx10_plus_mode_bits(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250;
}

[[nodiscard]] bool target_uses_wgp_mode(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4;
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

[[nodiscard]] bool image_contains_range(size_t image_size, uint64_t file_offset, uint64_t size) {
  const uint64_t limit = static_cast<uint64_t>(image_size);
  return file_offset <= limit && size <= limit - file_offset;
}

void apply_kernel_descriptor_resource_translation(KD &desc, const KdTranslation &translation,
                                                  rj_code_arch_t target_arch) {
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  translation.target_vgpr_granulated);
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  translation.target_sgpr_granulated);

  if (target_uses_gfx90a_accum_offset(target_arch) && translation.target_accvgpr_base != 0) {
    // GFX90A-style descriptors encode the first AccVGPR as (field + 1) * 4.
    // KernelDescriptorTranslator may move this base upward when semantic
    // lowering needs ordinary VGPR scratch above the source AccVGPR window, so
    // the patcher must write the recomputed base alongside the VGPR allocation.
    const uint32_t encoded_accum_offset = (translation.target_accvgpr_base / 4) - 1;
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                    encoded_accum_offset);
  }

  if (target_uses_gfx10_plus_mode_bits(target_arch)) {
    if (target_clears_rsrc1_mode_bits(target_arch)) {
      AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_ENABLE_DX10_CLAMP, 0);
      AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_ENABLE_IEEE_MODE, 0);
    }
    const uint32_t wgp_mode = target_uses_wgp_mode(target_arch) ? 1u : 0u;
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_WGP_MODE, wgp_mode);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_MEM_ORDERED, 1);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_FWD_PROGRESS, 1);
  }

  if (target_supports_wave32(target_arch)) {
    const uint32_t wave32 = translation.target_wave_size == 32 ? 1u : 0u;
    AMDHSA_BITS_SET(desc.kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
                    wave32);
  } else {
    AMDHSA_BITS_SET(desc.kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
                    0);
  }

  if (target_uses_gfx10_plus_mode_bits(target_arch)) {
    desc.compute_pgm_rsrc3 = 0;
    if (const uint32_t inst_pref = target_default_inst_pref_size(target_arch); inst_pref != 0) {
      AMDHSA_BITS_SET(desc.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX10_PLUS_INST_PREF_SIZE,
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
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                    (translation.target_accvgpr_base / 4 - 1));
  }

  desc.private_segment_fixed_size = translation.target_private_size;
  desc.group_segment_fixed_size = translation.target_lds_size;
  desc.kernarg_size = translation.target_kernarg_size;
  // `kernel_code_properties` is a 16-bit descriptor field. Keep this target ABI
  // bit update explicit so adding the DBT-only kernarg segment pointer cannot
  // be lost through integer-promotion surprises in the AMDHSA field macro.
  constexpr uint16_t kKernargSegmentPtrMask =
      static_cast<uint16_t>(kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR);
  if (translation.has_kernarg_segment_ptr) {
    desc.kernel_code_properties =
        static_cast<uint16_t>(desc.kernel_code_properties | kKernargSegmentPtrMask);
  } else {
    desc.kernel_code_properties =
        static_cast<uint16_t>(desc.kernel_code_properties & ~kKernargSegmentPtrMask);
  }
  // The AMDHSA descriptor must not carry a preprogrammed LDS_SIZE in RSRC2.
  // CP derives COMPUTE_PGM_RSRC2.LDS_SIZE from the dispatch packet's group
  // segment size, which already includes descriptor fixed LDS plus dispatch-time
  // dynamic LDS. Leaving a guest value here is especially bad for virtual-LDS
  // sidecars: their packet LDS is zero, but stale descriptor bits can still be
  // ORed into hardware command streams on some runtime paths.
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE, 0);
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT,
                  translation.target_user_sgpr_count);
  // Fixed private size can be zero for a kernel that requests its call stack
  // dynamically through the AQL packet. Preserve an existing scratch-enable
  // requirement and also enable it whenever DBT introduces fixed spill space.
  // The skipped-kernel path below intentionally clears the failed guest ABI.
  const uint32_t enable_private_segment =
      AMDHSA_BITS_GET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT) != 0 ||
              translation.target_private_size != 0
          ? 1u
          : 0u;
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT,
                  enable_private_segment);
  if (translation.skipped) {
    // Skipped kernels are target-ISA no-op stubs. Do not preserve the
    // failed guest kernel's descriptor ABI inputs: ROCR validates the resource
    // request before the stub can run, and stale kernarg/LDS/private settings
    // can prevent the stub from being loaded because resource validation fails.
    //
    // Keep the same non-resource defaults that amdclang emits for an empty
    // gfx942 kernel: IEEE/DX10 mode with both FP denorm modes enabled, plus
    // SGPR_WORKGROUP_ID_X. A completely zeroed RSRC1/RSRC2 looks smaller, but
    // it is not a reliable dispatchable descriptor on the runtime path used by
    // skipped-kernel stubs.
    desc.kernarg_size = 0;
    desc.compute_pgm_rsrc1 = 0;
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32, 3);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64, 3);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_ENABLE_DX10_CLAMP, 1);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_ENABLE_IEEE_MODE, 1);
    desc.compute_pgm_rsrc3 = 0;
    desc.compute_pgm_rsrc2 = 0;
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1);
    desc.kernel_code_properties = 0;
    desc.kernarg_preload = 0;
  }
}

[[nodiscard]] bool set_kernel_entry_from_vaddr(KD &desc, uint64_t descriptor_vaddr,
                                               uint64_t text_vaddr,
                                               uint64_t new_entry_text_offset) {
  if (descriptor_vaddr > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return false;
  if (text_vaddr > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      new_entry_text_offset >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - text_vaddr)
    return false;

  const int64_t entry_vaddr = static_cast<int64_t>(text_vaddr + new_entry_text_offset);
  const int64_t descriptor = static_cast<int64_t>(descriptor_vaddr);
  desc.kernel_code_entry_byte_offset = entry_vaddr - descriptor;
  return true;
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

void grow_text_function_symbols(std::vector<uint8_t> &image, const Elf64_Ehdr &ehdr,
                                std::span<const Elf64_Shdr> shdrs, size_t text_index,
                                uint64_t old_text_size, uint64_t new_text_size) {
  if (new_text_size <= old_text_size)
    return;

  const Elf64_Shdr &text = shdrs[text_index];
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

      if (symbol.st_shndx != text_index || elf_symbol_type(symbol.st_info) != kElfSymbolTypeFunc)
        continue;

      // ET_DYN symbols use virtual addresses; ET_REL symbols use section
      // offsets.  Normalize both forms so the range test below is independent
      // of the input code-object kind.
      uint64_t symbol_text_offset = symbol.st_value;
      if (ehdr.e_type != ET_REL) {
        if (symbol.st_value < text.sh_addr)
          continue;
        symbol_text_offset = symbol.st_value - text.sh_addr;
      }
      if (symbol_text_offset > old_text_size)
        continue;

      // The translator appends executable cave code after the original .text
      // contents and branches into it from relocated instructions.  AMDGPU code
      // objects often leave padding after the kernel function, so a valid input
      // function symbol may not reach the old section end.  Any non-empty
      // function body that starts in the translated .text can now reach appended
      // cave code, so keep its extent covering those bytes for loaders and
      // runtime tooling that consult FUNC ranges.
      if (symbol.st_size == 0)
        continue;

      const uint64_t grown_size = new_text_size - symbol_text_offset;
      if (symbol.st_size >= grown_size)
        continue;
      symbol.st_size = grown_size;
      std::memcpy(image.data() + symbol_offset, &symbol, sizeof(symbol));
    }
  }
}

[[nodiscard]] bool relocate_text_symbols(std::vector<uint8_t> &image, const Elf64_Ehdr &ehdr,
                                         std::span<const Elf64_Shdr> shdrs, size_t text_index,
                                         uint64_t old_text_size, uint64_t new_text_size,
                                         std::span<const TextOffsetRelocation> relocations,
                                         bool require_exact_function_ranges) {
  if (relocations.empty())
    return true;

  std::unordered_map<uint64_t, uint64_t> target_by_source;
  target_by_source.reserve(relocations.size());
  for (const TextOffsetRelocation &relocation : relocations) {
    if (relocation.source_offset > old_text_size || relocation.target_offset > new_text_size)
      return false;
    // A helper block can be copied into more than one kernel-local body. ELF
    // has only one value for its local label, so retain the first deterministic
    // placement. Control-flow fixups remain kernel-local and do not depend on
    // this tooling/debug symbol choice.
    target_by_source.try_emplace(relocation.source_offset, relocation.target_offset);
  }

  const Elf64_Shdr &text = shdrs[text_index];
  // Only referenced symbols are correctness-critical. Debug and tooling symbol
  // tables may contain labels in padding or unreachable bytes that DBT does not
  // emit, so requiring an offset-map entry for every text symbol would reject
  // otherwise translatable code objects. Index referenced symbols once here to
  // avoid an O(symbols * relocations) search below.
  std::unordered_map<size_t, std::unordered_set<uint32_t>> referenced_by_symtab;
  for (const Elf64_Shdr &relocs : shdrs) {
    if (relocs.sh_type != SHT_RELA || relocs.sh_entsize != sizeof(Elf64_Rela) ||
        relocs.sh_link >= shdrs.size() ||
        !image_contains_range(image.size(), relocs.sh_offset, relocs.sh_size)) {
      continue;
    }
    const Elf64_Shdr &symtab = shdrs[relocs.sh_link];
    if (symtab.sh_entsize != sizeof(Elf64_Sym) ||
        !image_contains_range(image.size(), symtab.sh_offset, symtab.sh_size)) {
      continue;
    }
    const size_t count = relocs.sh_size / sizeof(Elf64_Rela);
    for (size_t i = 0; i < count; ++i) {
      Elf64_Rela rela{};
      std::memcpy(&rela, image.data() + relocs.sh_offset + i * sizeof(rela), sizeof(rela));
      const uint32_t symbol_index = elf_reloc_sym(rela.r_info);
      if (symbol_index == 0 ||
          static_cast<uint64_t>(symbol_index) * sizeof(Elf64_Sym) + sizeof(Elf64_Sym) >
              symtab.sh_size) {
        continue;
      }
      Elf64_Sym symbol{};
      std::memcpy(&symbol, image.data() + symtab.sh_offset + symbol_index * sizeof(symbol),
                  sizeof(symbol));
      if (symbol.st_shndx == text_index &&
          elf_symbol_type(symbol.st_info) != kElfSymbolTypeSection) {
        referenced_by_symtab[relocs.sh_link].insert(symbol_index);
      }
    }
  }

  for (size_t symtab_index = 0; symtab_index < shdrs.size(); ++symtab_index) {
    const Elf64_Shdr &symtab = shdrs[symtab_index];
    if (symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM)
      continue;
    if (symtab.sh_entsize != sizeof(Elf64_Sym) ||
        !image_contains_range(image.size(), symtab.sh_offset, symtab.sh_size))
      continue;

    const size_t count = symtab.sh_size / sizeof(Elf64_Sym);
    for (size_t i = 0; i < count; ++i) {
      const uint64_t symbol_offset = symtab.sh_offset + i * sizeof(Elf64_Sym);
      Elf64_Sym symbol{};
      std::memcpy(&symbol, image.data() + symbol_offset, sizeof(symbol));
      if (symbol.st_shndx != text_index || elf_symbol_type(symbol.st_info) == kElfSymbolTypeSection)
        continue;

      const auto referenced = referenced_by_symtab.find(symtab_index);
      const bool must_relocate =
          referenced != referenced_by_symtab.end() && referenced->second.contains(i);

      uint64_t source_text_offset = symbol.st_value;
      if (ehdr.e_type != ET_REL) {
        if (symbol.st_value < text.sh_addr) {
          if (must_relocate)
            return false;
          continue;
        }
        source_text_offset = symbol.st_value - text.sh_addr;
      }
      if (source_text_offset > old_text_size) {
        if (must_relocate)
          return false;
        continue;
      }
      const auto relocated_start = target_by_source.find(source_text_offset);
      if (relocated_start == target_by_source.end()) {
        if (must_relocate)
          return false;
        continue;
      }

      const uint64_t old_size = symbol.st_size;
      const uint8_t symbol_type = elf_symbol_type(symbol.st_info);
      const bool sized_function = old_size != 0 && (symbol_type == kElfSymbolTypeFunc ||
                                                    symbol_type == kElfSymbolTypeAmdGpuHsaKernel);
      if (old_size > old_text_size - source_text_offset) {
        if (must_relocate || (require_exact_function_ranges && sized_function))
          return false;
      } else {
        const auto relocated_end = target_by_source.find(source_text_offset + old_size);
        if (relocated_end == target_by_source.end() ||
            relocated_end->second < relocated_start->second) {
          if (must_relocate || (require_exact_function_ranges && sized_function))
            return false;
        } else {
          symbol.st_size = relocated_end->second - relocated_start->second;
        }
      }
      symbol.st_value =
          ehdr.e_type == ET_REL ? relocated_start->second : text.sh_addr + relocated_start->second;
      std::memcpy(image.data() + symbol_offset, &symbol, sizeof(symbol));
    }
  }
  return true;
}

[[nodiscard]] bool
relocate_relative_text_addends(std::vector<uint8_t> &image, const Elf64_Ehdr &ehdr,
                               std::span<const Elf64_Shdr> shdrs, size_t text_index,
                               uint64_t old_text_size, uint64_t new_text_size,
                               std::span<const TextOffsetRelocation> relocations) {
  if (relocations.empty() || ehdr.e_type != ET_DYN)
    return true;

  std::unordered_map<uint64_t, uint64_t> target_by_source;
  target_by_source.reserve(relocations.size());
  // A shared helper block is emitted once per kernel-local scope, so the same
  // source offset can appear here with DIFFERENT target placements — and the
  // scopes are not interchangeable (e.g. a hardware-LDS clone vs a virtual-LDS
  // sidecar clone with different LDS lowering, liveness, and resources). For a
  // RELATIVE64 addend (a RUNTIME-DEREFERENCED function pointer) we cannot know
  // which clone a given dispatcher belongs to from the source offset alone, so
  // collapsing to one clone would let a sidecar dispatch jump into the wrong one.
  // Record which source offsets have conflicting placements and fail closed below
  // if any such offset is actually referenced by a function-table addend. Source
  // offsets that are copied to multiple scopes but never used as a RELATIVE64
  // pointer are harmless (control-flow fixups stay kernel-local), so a conflict
  // that is never dereferenced does not reject the translation.
  std::unordered_set<uint64_t> conflicting_sources;
  for (const TextOffsetRelocation &relocation : relocations) {
    if (relocation.source_offset > old_text_size || relocation.target_offset > new_text_size)
      return false;
    auto [it, inserted] =
        target_by_source.try_emplace(relocation.source_offset, relocation.target_offset);
    if (!inserted && it->second != relocation.target_offset)
      conflicting_sources.insert(relocation.source_offset);
  }

  const Elf64_Shdr &text = shdrs[text_index];
  for (const Elf64_Shdr &relocs : shdrs) {
    if (relocs.sh_type != SHT_RELA || relocs.sh_entsize != sizeof(Elf64_Rela) ||
        !image_contains_range(image.size(), relocs.sh_offset, relocs.sh_size)) {
      continue;
    }

    const size_t count = relocs.sh_size / sizeof(Elf64_Rela);
    for (size_t i = 0; i < count; ++i) {
      const uint64_t rela_offset = relocs.sh_offset + i * sizeof(Elf64_Rela);
      Elf64_Rela rela{};
      std::memcpy(&rela, image.data() + rela_offset, sizeof(rela));
      if (elf_reloc_type(rela.r_info) != R_AMDGPU_RELATIVE64 || rela.r_addend < 0)
        continue;

      const uint64_t addend = static_cast<uint64_t>(rela.r_addend);
      if (addend < text.sh_addr)
        continue;
      const uint64_t source_offset = addend - text.sh_addr;
      if (source_offset >= old_text_size)
        continue;

      // This addend IS dereferenced as a function pointer. If its source block
      // was emitted at conflicting placements across scopes, no single rewrite is
      // correct — fail closed rather than pick an arbitrary clone.
      if (conflicting_sources.contains(source_offset))
        return false;
      const auto relocated = target_by_source.find(source_offset);
      // Leaving an in-text addend unchanged would silently preserve a stale PC.
      // Compatibility was established from the relocation form, but final
      // materialization must also prove that this exact target was emitted.
      if (relocated == target_by_source.end())
        return false;
      if (relocated->second > std::numeric_limits<uint64_t>::max() - text.sh_addr)
        return false;
      const uint64_t target_addend = text.sh_addr + relocated->second;
      if (target_addend > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return false;
      rela.r_addend = static_cast<int64_t>(target_addend);
      std::memcpy(image.data() + rela_offset, &rela, sizeof(rela));
    }
  }
  return true;
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
    : image_(obj.image_data(), obj.image_data() + obj.image_size()), text_offset_(0), text_size_(0),
      text_vaddr_(0), text_tail_size_(0) {
  auto &text_secs = obj.text_sections();
  if (!text_secs.empty()) {
    text_offset_ = text_secs[0]->sectionOffset();
    text_size_ = text_secs[0]->size();
    text_vaddr_ = text_secs[0]->vaddr();
  }
}

std::span<uint8_t> CodeObjectPatcher::text_bytes() {
  return {image_.data() + text_offset_, text_size_};
}

std::span<const uint8_t> CodeObjectPatcher::text_bytes() const {
  return {image_.data() + text_offset_, text_size_};
}

bool CodeObjectPatcher::has_relocations_within_text() const {
  if (text_size_ == 0)
    return false;
  if (image_.size() < sizeof(Elf64_Ehdr))
    return false;

  auto header = *reinterpret_cast<const Elf64_Ehdr *>(image_.data());
  const auto shdrs = read_section_headers(image_, header);
  const auto text_index = find_text_section(shdrs, text_offset_, text_size_);
  if (!text_index)
    return false;

  // A relocation's place (r_offset) is interpreted differently by ELF type:
  //   - ET_DYN/ET_EXEC: r_offset is a virtual address. An in-.text place is one
  //     inside [text.sh_addr, text.sh_addr + size). The reloc section applies
  //     across the whole image, so every record must be range-checked.
  //   - ET_REL: r_offset is relative to the section the reloc section applies to,
  //     named by the reloc section's sh_info. Only records whose sh_info is the
  //     text section can land in .text, and there any r_offset < text.sh_size is
  //     in-text. Comparing an ET_REL r_offset against a file offset (text.sh_offset)
  //     is wrong and would miss e.g. an .rela.text record with r_offset == 0.
  const Elf64_Shdr &text = shdrs[*text_index];
  const bool is_rel_object = header.e_type == ET_REL;

  for (const Elf64_Shdr &relocs : shdrs) {
    if (relocs.sh_type != SHT_RELA && relocs.sh_type != SHT_REL)
      continue;
    if (!image_contains_range(image_.size(), relocs.sh_offset, relocs.sh_size))
      continue;
    if (is_rel_object && relocs.sh_info != *text_index)
      continue; // ET_REL: only relocations that apply to .text matter.
    const bool is_rela = relocs.sh_type == SHT_RELA;
    const size_t entsize = is_rela ? sizeof(Elf64_Rela) : sizeof(Elf64_Rel);
    if (relocs.sh_entsize != entsize)
      continue;
    const size_t count = relocs.sh_size / entsize;
    for (size_t i = 0; i < count; ++i) {
      const uint64_t offset = relocs.sh_offset + i * entsize;
      uint64_t r_offset = 0;
      if (is_rela) {
        Elf64_Rela rela{};
        std::memcpy(&rela, image_.data() + offset, sizeof(rela));
        r_offset = rela.r_offset;
      } else {
        Elf64_Rel rel{};
        std::memcpy(&rel, image_.data() + offset, sizeof(rel));
        r_offset = rel.r_offset;
      }
      const bool in_text = is_rel_object
                               ? r_offset < text_size_
                               : (r_offset >= text.sh_addr && r_offset < text.sh_addr + text_size_);
      if (in_text)
        return true;
    }
  }
  return false;
}

bool CodeObjectPatcher::has_unsupported_relocation_to_text() const {
  if (text_size_ == 0)
    return false;
  if (image_.size() < sizeof(Elf64_Ehdr))
    return false;

  auto header = *reinterpret_cast<const Elf64_Ehdr *>(image_.data());
  const auto shdrs = read_section_headers(image_, header);
  const auto text_index = find_text_section(shdrs, text_offset_, text_size_);
  if (!text_index)
    return false;

  // The virtual-address interval identifies symbol-less RELATIVE64 addends that
  // refer to translated code. Those references are supported because
  // replace_text() rewrites the explicit addend through the final offset map.
  const Elf64_Shdr &text = shdrs[*text_index];
  const uint64_t text_addr_lo = text.sh_addr;
  const uint64_t text_addr_hi = text.sh_addr + text_size_;

  // Return the referenced symbol only when it is defined in .text. Keeping the
  // complete symbol is necessary because ordinary named symbols and STT_SECTION
  // symbols require different relocation strategies.
  auto text_symbol = [&](const Elf64_Shdr &symtab, uint32_t sym_index) -> std::optional<Elf64_Sym> {
    if (symtab.sh_entsize != sizeof(Elf64_Sym))
      return std::nullopt;
    if (!image_contains_range(image_.size(), symtab.sh_offset, symtab.sh_size))
      return std::nullopt;
    if (static_cast<uint64_t>(sym_index) * sizeof(Elf64_Sym) + sizeof(Elf64_Sym) > symtab.sh_size)
      return std::nullopt;
    Elf64_Sym symbol{};
    std::memcpy(&symbol, image_.data() + symtab.sh_offset + sym_index * sizeof(Elf64_Sym),
                sizeof(symbol));
    if (symbol.st_shndx != *text_index)
      return std::nullopt;
    return symbol;
  };

  for (const Elf64_Shdr &relocs : shdrs) {
    if (relocs.sh_type != SHT_RELA && relocs.sh_type != SHT_REL)
      continue;
    if (!image_contains_range(image_.size(), relocs.sh_offset, relocs.sh_size))
      continue;
    // sh_link names the symbol table this relocation section indexes into.
    if (relocs.sh_link >= shdrs.size())
      continue;
    const Elf64_Shdr &symtab = shdrs[relocs.sh_link];
    const bool is_rela = relocs.sh_type == SHT_RELA;
    const size_t entsize = is_rela ? sizeof(Elf64_Rela) : sizeof(Elf64_Rel);
    if (relocs.sh_entsize != entsize)
      continue;
    const size_t count = relocs.sh_size / entsize;
    for (size_t i = 0; i < count; ++i) {
      const uint64_t offset = relocs.sh_offset + i * entsize;
      uint64_t r_info = 0;
      int64_t r_addend = 0;
      if (is_rela) {
        Elf64_Rela rela{};
        std::memcpy(&rela, image_.data() + offset, sizeof(rela));
        r_info = rela.r_info;
        r_addend = rela.r_addend;
      } else {
        Elf64_Rel rel{};
        std::memcpy(&rel, image_.data() + offset, sizeof(rel));
        r_info = rel.r_info;
      }
      const uint32_t sym_index = elf_reloc_sym(r_info);
      if (sym_index != 0) {
        const auto symbol = text_symbol(symtab, sym_index);
        if (!symbol)
          continue;

        // A zero-addend RELA reference to an ordinary text symbol follows the
        // relocated symbol value written by relocate_text_symbols(). A section
        // symbol leaves the source offset in the addend, while REL keeps an
        // implicit addend at the relocation place; neither form can be repaired
        // safely without interpreting the individual relocation type.
        if (!is_rela || elf_symbol_type(symbol->st_info) == kElfSymbolTypeSection ||
            r_addend != 0) {
          return true;
        }
        continue;
      }

      // Symbol index 0 has no st_value to update. RELATIVE64 is the one form for
      // which the target can be derived generically: load bias plus r_addend.
      // replace_text() remaps an in-text addend exactly. Other symbol-zero forms
      // provide no generic way to identify a .text target here.
      if (is_rela && elf_reloc_type(r_info) == R_AMDGPU_RELATIVE64) {
        const uint64_t target = static_cast<uint64_t>(r_addend);
        if (target >= text_addr_lo && target < text_addr_hi)
          continue;
      }
    }
  }
  return false;
}

bool CodeObjectPatcher::replace_text(std::span<const uint8_t> new_text,
                                     std::span<const TextOffsetRelocation> text_relocations,
                                     std::span<const PcRelativeDataRelocation> data_relocations,
                                     std::string *error_out, std::vector<std::string> *warnings_out,
                                     bool require_exact_function_ranges) {
  const auto fail = [error_out](const char *message) {
    report(error_out, message);
    return false;
  };
  // Keep fail-closed behavior for callers that assume word-aligned executable
  // sections; accepting a non-word-aligned replacement can break downstream
  // PC-relative patching and branch-distance checks.
  if ((new_text.size() % sizeof(uint32_t)) != 0)
    return fail("replacement .text is not word-aligned");

  if (text_size_ == 0)
    return fail("code object has no replaceable .text payload");
  if (new_text.size() < text_size_)
    return fail("replacement .text is smaller than the source allocation");
  if (!image_contains_range(image_.size(), text_offset_, text_size_))
    return fail("source .text extends beyond the ELF image");
  if (!text_relocations.empty() && has_unsupported_relocation_to_text())
    return fail("ELF contains an unsupported relocation referencing .text");

  // replace_text() is a public patch transaction. Several late validation
  // paths run after unwind relocation or file insertion, so retain the exact
  // original state until every ELF table update succeeds.
  struct Rollback {
    CodeObjectPatcher &patcher;
    std::vector<uint8_t> image;
    uint64_t text_offset;
    uint64_t text_size;
    uint64_t text_tail_size;
    bool committed = false;
    ~Rollback() {
      if (committed)
        return;
      patcher.image_ = std::move(image);
      patcher.text_offset_ = text_offset;
      patcher.text_size_ = text_size;
      patcher.text_tail_size_ = text_tail_size;
    }
  } rollback{*this, image_, text_offset_, text_size_, text_tail_size_};

  auto *ehdr = reinterpret_cast<Elf64_Ehdr *>(image_.data());
  auto header = *ehdr;
  auto shdrs = read_section_headers(image_, header);
  auto phdrs = read_program_headers(image_, header);

  const auto text_index = find_text_section(shdrs, text_offset_, text_size_);
  if (!text_index) {
    assert(false && "text section header not found");
    return fail("source .text section header could not be found");
  }
  const auto text_header = shdrs[*text_index];
  uint64_t old_text_end_file = text_offset_ + text_size_;
  const uint64_t old_text_end_vaddr = text_header.sh_addr + text_size_;
  const uint64_t growth = new_text.size() - text_size_;
  uint64_t padded_file_delta = 0;
  std::vector<bool> shift_section_vaddr(shdrs.size(), false);
  std::vector<bool> shift_segment_vaddr(phdrs.size(), false);
  if (growth != 0) {
    // Growing .text can shift later LOAD segments. Pad the inserted file range
    // so every shifted LOAD keeps p_offset % p_align congruent with p_vaddr %
    // p_align. The padding is executable segment filler, not part of .text.
    const uint64_t file_delta_alignment = shifted_load_delta_alignment(phdrs, old_text_end_file);
    padded_file_delta = align_up(growth, file_delta_alignment);
    assert(padded_file_delta >= growth && "aligned text growth underflowed");
    assert(padded_file_delta % sizeof(uint32_t) == 0 && "text growth must stay word-aligned");

    for (size_t i = 0; i < shdrs.size(); ++i) {
      if (i == *text_index || shdrs[i].sh_type == SHT_NULL)
        continue;
      if ((shdrs[i].sh_flags & SHF_ALLOC) != 0 && shdrs[i].sh_addr >= old_text_end_vaddr) {
        shift_section_vaddr[i] = true;
      }
    }
    for (size_t i = 0; i < phdrs.size(); ++i) {
      if (phdrs[i].p_vaddr >= old_text_end_vaddr && phdrs[i].p_offset >= old_text_end_file)
        shift_segment_vaddr[i] = true;
    }
  }

  std::vector<std::string> warnings;
  const bool layout_changed =
      new_text.size() != text_size_ ||
      std::ranges::any_of(text_relocations, [](const TextOffsetRelocation &relocation) {
        return relocation.source_offset != relocation.target_offset;
      });
  if (layout_changed) {
    bool dropped_debug = false;
    for (size_t section_index = 0; section_index < shdrs.size(); ++section_index) {
      const auto name = section_name(image_, header, shdrs, section_index);
      if (!name)
        return fail("ELF section names are malformed while locating debug metadata");
      if ((shdrs[section_index].sh_flags & SHF_ALLOC) == 0 &&
          (name->starts_with(".debug_") || name->starts_with(".zdebug_")) &&
          shdrs[section_index].sh_size != 0) {
        shdrs[section_index].sh_size = 0;
        dropped_debug = true;
      }
    }
    if (dropped_debug) {
      warnings.emplace_back(
          "dropped non-allocated DWARF sections whose PC ranges became stale after .text "
          "relocation");
    }
  }
  if (!relocate_eh_frame_fdes(image_, header, shdrs, phdrs, *text_index, text_size_,
                              new_text.size(), text_relocations, shift_section_vaddr,
                              padded_file_delta, error_out)) {
    return false;
  }
  text_offset_ = shdrs[*text_index].sh_offset;
  old_text_end_file = text_offset_ + text_size_;

  struct ResolvedDataRelocation {
    PcRelativeDataRelocation relocation;
    size_t target_section = 0;
    uint64_t target_section_offset = 0;
  };
  std::vector<ResolvedDataRelocation> resolved_data_relocations;
  resolved_data_relocations.reserve(data_relocations.size());
  for (const PcRelativeDataRelocation &relocation : data_relocations) {
    if (relocation.target_getpc_offset > new_text.size() ||
        sizeof(uint32_t) > new_text.size() - relocation.target_getpc_offset ||
        relocation.target_literal_offset > new_text.size() ||
        sizeof(uint64_t) > new_text.size() - relocation.target_literal_offset) {
      return fail("PC-relative data relocation lies outside replacement .text");
    }
    const auto section = std::ranges::find_if(shdrs, [&](const Elf64_Shdr &candidate) {
      return (candidate.sh_flags & SHF_ALLOC) != 0 && (candidate.sh_flags & SHF_EXECINSTR) == 0 &&
             relocation.source_target_vaddr >= candidate.sh_addr &&
             relocation.source_target_vaddr - candidate.sh_addr < candidate.sh_size;
    });
    if (section == shdrs.end())
      return fail("PC-relative data relocation target is not in allocated non-executable data");
    resolved_data_relocations.push_back(
        {.relocation = relocation,
         .target_section = static_cast<size_t>(section - shdrs.begin()),
         .target_section_offset = relocation.source_target_vaddr - section->sh_addr});
  }
  if (growth != 0) {
    std::vector<uint8_t> inserted(padded_file_delta, 0);
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
  for (const ResolvedDataRelocation &resolved : resolved_data_relocations) {
    if (resolved.target_section >= shdrs.size() ||
        resolved.target_section_offset >= shdrs[resolved.target_section].sh_size) {
      return fail("PC-relative data relocation target moved outside its section");
    }
    const uint64_t target_vaddr =
        shdrs[resolved.target_section].sh_addr + resolved.target_section_offset;
    if (text_header.sh_addr >
            std::numeric_limits<uint64_t>::max() - resolved.relocation.target_getpc_offset ||
        text_header.sh_addr + resolved.relocation.target_getpc_offset >
            std::numeric_limits<uint64_t>::max() - sizeof(uint32_t)) {
      return fail("PC-relative data relocation address overflows");
    }
    const uint64_t getpc_result =
        text_header.sh_addr + resolved.relocation.target_getpc_offset + sizeof(uint32_t);
    const uint64_t delta = target_vaddr - getpc_result;
    std::memcpy(image_.data() + text_offset_ + resolved.relocation.target_literal_offset, &delta,
                sizeof(delta));
  }
  shdrs[*text_index].sh_size = new_text.size();
  if (!relocate_text_symbols(image_, header, shdrs, *text_index, text_size_, new_text.size(),
                             text_relocations, require_exact_function_ranges)) {
    return fail("text symbols cannot be mapped exactly into replacement .text");
  }
  if (!relocate_relative_text_addends(image_, header, shdrs, *text_index, text_size_,
                                      new_text.size(), text_relocations)) {
    return fail("R_AMDGPU_RELATIVE64 addends cannot be mapped exactly into replacement .text");
  }
  // Instrumentation appends a cave without relocating the original body and
  // therefore supplies no offset map. Preserve its historical function-range
  // growth. DBT supplies exact starts/ends and gets precise relocated sizes.
  if (text_relocations.empty())
    grow_text_function_symbols(image_, header, shdrs, *text_index, text_size_, new_text.size());
  text_size_ = new_text.size();

  for (const Elf64_Phdr &phdr : phdrs) {
    if (phdr.p_type != PT_LOAD || phdr.p_align <= 1)
      continue;
    assert(phdr.p_offset % phdr.p_align == phdr.p_vaddr % phdr.p_align &&
           "patched LOAD lost file/virtual address congruence");
  }
  write_elf_tables(image_, header, shdrs, phdrs);
  rollback.committed = true;
  if (warnings_out != nullptr)
    warnings_out->insert(warnings_out->end(), warnings.begin(), warnings.end());
  return true;
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

  KD desc;
  std::memcpy(&desc, image_.data() + translation.descriptor_file_offset, sizeof(desc));
  apply_kernel_descriptor_resource_translation(desc, translation, target_arch);

  std::memcpy(image_.data() + translation.descriptor_file_offset, &desc, sizeof(desc));
  if (!redirect_kernel_entry(translation.descriptor_file_offset, translation.entry_text_offset,
                             translation.target_entry_text_offset))
    return false;
  return true;
}

std::optional<std::vector<AppendedSidecarDescriptor>>
CodeObjectPatcher::append_sidecar_descriptor_translations(
    std::span<const KdTranslation> translations, rj_code_arch_t target_arch, uint64_t alignment,
    std::string *error_out) {
  if (translations.empty())
    return std::vector<AppendedSidecarDescriptor>{};
  if (alignment == 0 || (alignment & (alignment - 1)) != 0)
    return std::nullopt;
  if (text_size_ == 0 || text_vaddr_ == 0)
    return std::nullopt;
  if (image_.size() < sizeof(Elf64_Ehdr))
    return std::nullopt;

  auto header = *reinterpret_cast<const Elf64_Ehdr *>(image_.data());
  auto shdrs = read_section_headers(image_, header);
  auto phdrs = read_program_headers(image_, header);

  const auto text_index = find_text_section(shdrs, text_offset_, text_size_);
  if (!text_index)
    return std::nullopt;

  const uint64_t insertion_file_offset = text_offset_ + text_size_ + text_tail_size_;
  const uint64_t insertion_vaddr = text_vaddr_ + text_size_ + text_tail_size_;
  if (insertion_file_offset > image_.size())
    return std::nullopt;

  const bool insertion_is_loaded = std::ranges::any_of(phdrs, [&](const Elf64_Phdr &phdr) {
    if (phdr.p_type != PT_LOAD)
      return false;
    const uint64_t segment_end = phdr.p_offset + phdr.p_filesz;
    return phdr.p_offset <= insertion_file_offset && insertion_file_offset <= segment_end;
  });
  if (!insertion_is_loaded)
    return std::nullopt;

  for (size_t section_index = 0; section_index < shdrs.size(); ++section_index) {
    const auto name = section_name(image_, header, shdrs, section_index);
    if (!name) {
      report(error_out, "ELF section names are malformed while locating unwind metadata");
      return std::nullopt;
    }
    if ((shdrs[section_index].sh_flags & SHF_ALLOC) != 0 &&
        (*name == ".eh_frame" || *name == ".eh_frame_hdr") &&
        shdrs[section_index].sh_addr >= insertion_vaddr) {
      report(error_out,
             "sidecar insertion would move allocated unwind metadata without relocating it");
      return std::nullopt;
    }
  }

  std::vector<uint8_t> inserted;
  std::vector<AppendedSidecarDescriptor> appended;
  appended.reserve(translations.size());
  uint64_t cursor_vaddr = insertion_vaddr;
  for (const KdTranslation &translation : translations) {
    const uint64_t descriptor_vaddr = align_up(cursor_vaddr, alignment);
    if (descriptor_vaddr < cursor_vaddr)
      return std::nullopt;
    const uint64_t padding = descriptor_vaddr - cursor_vaddr;
    if (padding > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) - inserted.size())
      return std::nullopt;
    inserted.resize(inserted.size() + static_cast<size_t>(padding), 0);

    // Copy the sidecar template from the source-descriptor snapshot, NOT from
    // translation.descriptor_file_offset: this runs after .text growth, which
    // shifts a descriptor section that follows .text, leaving that offset
    // pointing at relocated instruction bytes.
    static_assert(sizeof(KD) == 64, "sidecar descriptor snapshot size mismatch");
    KD desc{};
    std::memcpy(&desc, translation.source_descriptor_bytes.data(), sizeof(desc));
    apply_kernel_descriptor_resource_translation(desc, translation, target_arch);
    if (!set_kernel_entry_from_vaddr(desc, descriptor_vaddr, text_vaddr_,
                                     translation.target_entry_text_offset))
      return std::nullopt;

    const uint64_t descriptor_file_offset = insertion_file_offset + inserted.size();
    const auto *descriptor_bytes = reinterpret_cast<const uint8_t *>(&desc);
    inserted.insert(inserted.end(), descriptor_bytes, descriptor_bytes + sizeof(desc));
    appended.push_back({.file_offset = descriptor_file_offset, .vaddr = descriptor_vaddr});
    cursor_vaddr = descriptor_vaddr + sizeof(desc);
  }

  // Later LOAD segments remain valid only if the file shift preserves their
  // p_offset/p_vaddr congruence. Any padding after the descriptor payload is
  // intentionally unsectioned loaded data; runtime metadata points only at the
  // descriptor starts recorded above.
  const uint64_t file_delta_alignment = shifted_load_delta_alignment(phdrs, insertion_file_offset);
  const uint64_t padded_size = align_up(inserted.size(), file_delta_alignment);
  if (padded_size < inserted.size() ||
      padded_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    return std::nullopt;
  inserted.resize(static_cast<size_t>(padded_size), 0);

  std::vector<bool> shift_section_vaddr(shdrs.size(), false);
  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (i == *text_index || shdrs[i].sh_type == SHT_NULL)
      continue;
    if ((shdrs[i].sh_flags & SHF_ALLOC) != 0 && shdrs[i].sh_addr >= insertion_vaddr)
      shift_section_vaddr[i] = true;
  }

  std::vector<bool> shift_segment_vaddr(phdrs.size(), false);
  for (size_t i = 0; i < phdrs.size(); ++i) {
    if (phdrs[i].p_vaddr >= insertion_vaddr && phdrs[i].p_offset >= insertion_file_offset)
      shift_segment_vaddr[i] = true;
  }

  insert_file_bytes(image_, header, shdrs, phdrs, insertion_file_offset, inserted, std::nullopt,
                    true);

  const uint64_t delta = inserted.size();
  for (size_t i = 0; i < shdrs.size(); ++i) {
    if (!shift_section_vaddr[i])
      continue;
    [[maybe_unused]] const uint64_t old_addr = shdrs[i].sh_addr;
    shdrs[i].sh_addr += delta;
    assert((shdrs[i].sh_addralign <= 1 ||
            shdrs[i].sh_addr % shdrs[i].sh_addralign == old_addr % shdrs[i].sh_addralign) &&
           "shifted allocated section lost its address alignment residue");
  }

  shift_symbols_in_moved_sections(image_, header, shdrs, shift_section_vaddr, delta);
  shift_relocation_offsets_in_moved_sections(image_, header, shdrs, shift_section_vaddr, delta);
  adjust_kernel_descriptor_entry_offsets_in_moved_sections(image_, shdrs, shift_section_vaddr,
                                                           delta);

  for (size_t i = 0; i < phdrs.size(); ++i) {
    if (!shift_segment_vaddr[i])
      continue;
    assert((phdrs[i].p_align <= 1 || delta % phdrs[i].p_align == 0) &&
           "descriptor tail padding does not preserve shifted LOAD alignment");
    phdrs[i].p_vaddr += delta;
    phdrs[i].p_paddr += delta;
    assert((phdrs[i].p_align <= 1 ||
            phdrs[i].p_offset % phdrs[i].p_align == phdrs[i].p_vaddr % phdrs[i].p_align) &&
           "shifted LOAD lost file/virtual address congruence");
  }

  for (const Elf64_Phdr &phdr : phdrs) {
    if (phdr.p_type != PT_LOAD || phdr.p_align <= 1)
      continue;
    assert(phdr.p_offset % phdr.p_align == phdr.p_vaddr % phdr.p_align &&
           "patched LOAD lost file/virtual address congruence");
  }

  text_tail_size_ += delta;
  write_elf_tables(image_, header, shdrs, phdrs);
  return appended;
}

bool CodeObjectPatcher::append_nonalloc_section(std::string_view name,
                                                std::span<const uint8_t> contents,
                                                uint64_t alignment) {
  if (name.empty() || contents.empty())
    return false;
  if (alignment == 0 || (alignment & (alignment - 1)) != 0)
    return false;
  if (image_.size() < sizeof(Elf64_Ehdr))
    return false;

  auto header = *reinterpret_cast<const Elf64_Ehdr *>(image_.data());
  if (header.e_shoff == 0 || header.e_shentsize != sizeof(Elf64_Shdr) || header.e_shnum == 0)
    return false;
  if (header.e_shstrndx == SHN_UNDEF || header.e_shstrndx >= header.e_shnum)
    return false;
  if (!image_contains_range(image_.size(), header.e_shoff,
                            static_cast<uint64_t>(header.e_shnum) * sizeof(Elf64_Shdr)))
    return false;

  auto shdrs = read_section_headers(image_, header);
  auto phdrs = read_program_headers(image_, header);
  Elf64_Shdr &shstrtab = shdrs[header.e_shstrndx];
  if (shstrtab.sh_type != SHT_STRTAB)
    return false;
  if (!image_contains_range(image_.size(), shstrtab.sh_offset, shstrtab.sh_size))
    return false;
  if (shstrtab.sh_size > std::numeric_limits<uint32_t>::max())
    return false;
  if (name.size() > std::numeric_limits<uint32_t>::max() - shstrtab.sh_size - 1)
    return false;
  if (shdrs.size() >= std::numeric_limits<uint16_t>::max())
    return false;

  std::vector<uint8_t> new_shstrtab(
      image_.begin() + static_cast<std::ptrdiff_t>(shstrtab.sh_offset),
      image_.begin() + static_cast<std::ptrdiff_t>(shstrtab.sh_offset + shstrtab.sh_size));
  const uint32_t section_name_offset = static_cast<uint32_t>(new_shstrtab.size());
  new_shstrtab.insert(new_shstrtab.end(), name.begin(), name.end());
  new_shstrtab.push_back('\0');

  const uint64_t payload_offset = align_up(image_.size(), alignment);
  if (payload_offset < image_.size())
    return false;
  image_.resize(static_cast<size_t>(payload_offset), 0);
  image_.insert(image_.end(), contents.begin(), contents.end());

  const uint64_t shstrtab_offset = image_.size();
  image_.insert(image_.end(), new_shstrtab.begin(), new_shstrtab.end());

  constexpr uint64_t kSectionHeaderAlignment = 8;
  const uint64_t new_shoff = align_up(image_.size(), kSectionHeaderAlignment);
  if (new_shoff < image_.size())
    return false;
  image_.resize(static_cast<size_t>(new_shoff), 0);

  shstrtab.sh_offset = shstrtab_offset;
  shstrtab.sh_size = new_shstrtab.size();
  shstrtab.sh_addralign = 1;

  Elf64_Shdr metadata_section{};
  metadata_section.sh_name = section_name_offset;
  metadata_section.sh_type = SHT_PROGBITS;
  metadata_section.sh_flags = 0;
  metadata_section.sh_addr = 0;
  metadata_section.sh_offset = payload_offset;
  metadata_section.sh_size = contents.size();
  metadata_section.sh_addralign = alignment;
  shdrs.push_back(metadata_section);

  header.e_shoff = new_shoff;
  header.e_shnum = static_cast<uint16_t>(shdrs.size());
  image_.resize(static_cast<size_t>(new_shoff + shdrs.size() * sizeof(Elf64_Shdr)), 0);
  write_elf_tables(image_, header, shdrs, phdrs);
  return true;
}

bool CodeObjectPatcher::redirect_kernel_entry(uint64_t descriptor_file_offset,
                                              uint64_t old_entry_text_offset,
                                              uint64_t new_entry_text_offset) {
  if (!image_contains_range(image_.size(), descriptor_file_offset, sizeof(KD)))
    return false;

  KD desc;
  std::memcpy(&desc, image_.data() + descriptor_file_offset, sizeof(desc));
  const int64_t delta =
      static_cast<int64_t>(new_entry_text_offset) - static_cast<int64_t>(old_entry_text_offset);
  const int64_t redirected = static_cast<int64_t>(desc.kernel_code_entry_byte_offset) + delta;
  // The descriptor field is signed because the entry point may be before or
  // after the descriptor in virtual address order. Preserve that signed value
  // when applying the text-relative delta.
  desc.kernel_code_entry_byte_offset = redirected;
  std::memcpy(image_.data() + descriptor_file_offset, &desc, sizeof(desc));
  return true;
}

std::vector<uint8_t> CodeObjectPatcher::emit() const & { return image_; }

std::vector<uint8_t> CodeObjectPatcher::emit() && { return std::move(image_); }

} // namespace rocjitsu
