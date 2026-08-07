// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/dwarf_relocator.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace rocjitsu {
namespace {

constexpr uint8_t DW_LNS_copy = 1;
constexpr uint8_t DW_LNS_advance_pc = 2;
constexpr uint8_t DW_LNS_advance_line = 3;
constexpr uint8_t DW_LNS_set_file = 4;
constexpr uint8_t DW_LNS_set_column = 5;
constexpr uint8_t DW_LNS_negate_stmt = 6;
constexpr uint8_t DW_LNS_set_basic_block = 7;
constexpr uint8_t DW_LNS_const_add_pc = 8;
constexpr uint8_t DW_LNS_fixed_advance_pc = 9;
constexpr uint8_t DW_LNS_set_prologue_end = 10;
constexpr uint8_t DW_LNS_set_epilogue_begin = 11;
constexpr uint8_t DW_LNS_set_isa = 12;
constexpr uint8_t DW_LNE_end_sequence = 1;
constexpr uint8_t DW_LNE_set_address = 2;
constexpr uint8_t DW_LNE_set_discriminator = 4;

constexpr uint64_t DW_AT_stmt_list = 0x10;
constexpr uint64_t DW_AT_low_pc = 0x11;
constexpr uint64_t DW_AT_high_pc = 0x12;
constexpr uint64_t DW_AT_addr_base = 0x73;
constexpr uint64_t DW_AT_rnglists_base = 0x74;
constexpr uint64_t DW_AT_loclists_base = 0x8c;

constexpr uint64_t DW_FORM_addr = 0x01;
constexpr uint64_t DW_FORM_block2 = 0x03;
constexpr uint64_t DW_FORM_block4 = 0x04;
constexpr uint64_t DW_FORM_data2 = 0x05;
constexpr uint64_t DW_FORM_data4 = 0x06;
constexpr uint64_t DW_FORM_data8 = 0x07;
constexpr uint64_t DW_FORM_string = 0x08;
constexpr uint64_t DW_FORM_block = 0x09;
constexpr uint64_t DW_FORM_block1 = 0x0a;
constexpr uint64_t DW_FORM_data1 = 0x0b;
constexpr uint64_t DW_FORM_flag = 0x0c;
constexpr uint64_t DW_FORM_sdata = 0x0d;
constexpr uint64_t DW_FORM_strp = 0x0e;
constexpr uint64_t DW_FORM_udata = 0x0f;
constexpr uint64_t DW_FORM_ref_addr = 0x10;
constexpr uint64_t DW_FORM_ref1 = 0x11;
constexpr uint64_t DW_FORM_ref2 = 0x12;
constexpr uint64_t DW_FORM_ref4 = 0x13;
constexpr uint64_t DW_FORM_ref8 = 0x14;
constexpr uint64_t DW_FORM_ref_udata = 0x15;
constexpr uint64_t DW_FORM_indirect = 0x16;
constexpr uint64_t DW_FORM_sec_offset = 0x17;
constexpr uint64_t DW_FORM_exprloc = 0x18;
constexpr uint64_t DW_FORM_flag_present = 0x19;
constexpr uint64_t DW_FORM_strx = 0x1a;
constexpr uint64_t DW_FORM_addrx = 0x1b;
constexpr uint64_t DW_FORM_ref_sup4 = 0x1c;
constexpr uint64_t DW_FORM_strp_sup = 0x1d;
constexpr uint64_t DW_FORM_data16 = 0x1e;
constexpr uint64_t DW_FORM_line_strp = 0x1f;
constexpr uint64_t DW_FORM_ref_sig8 = 0x20;
constexpr uint64_t DW_FORM_implicit_const = 0x21;
constexpr uint64_t DW_FORM_loclistx = 0x22;
constexpr uint64_t DW_FORM_rnglistx = 0x23;
constexpr uint64_t DW_FORM_ref_sup8 = 0x24;
constexpr uint64_t DW_FORM_strx1 = 0x25;
constexpr uint64_t DW_FORM_strx2 = 0x26;
constexpr uint64_t DW_FORM_strx3 = 0x27;
constexpr uint64_t DW_FORM_strx4 = 0x28;
constexpr uint64_t DW_FORM_addrx1 = 0x29;
constexpr uint64_t DW_FORM_addrx2 = 0x2a;
constexpr uint64_t DW_FORM_addrx3 = 0x2b;
constexpr uint64_t DW_FORM_addrx4 = 0x2c;

constexpr uint8_t DW_RLE_end_of_list = 0x00;
constexpr uint8_t DW_RLE_base_addressx = 0x01;
constexpr uint8_t DW_RLE_startx_endx = 0x02;
constexpr uint8_t DW_RLE_startx_length = 0x03;
constexpr uint8_t DW_RLE_offset_pair = 0x04;
constexpr uint8_t DW_RLE_base_address = 0x05;
constexpr uint8_t DW_RLE_start_end = 0x06;
constexpr uint8_t DW_RLE_start_length = 0x07;

constexpr uint8_t DW_LLE_end_of_list = 0x00;
constexpr uint8_t DW_LLE_base_addressx = 0x01;
constexpr uint8_t DW_LLE_startx_endx = 0x02;
constexpr uint8_t DW_LLE_startx_length = 0x03;
constexpr uint8_t DW_LLE_offset_pair = 0x04;
constexpr uint8_t DW_LLE_default_location = 0x05;
constexpr uint8_t DW_LLE_base_address = 0x06;
constexpr uint8_t DW_LLE_start_end = 0x07;
constexpr uint8_t DW_LLE_start_length = 0x08;

constexpr uint8_t DW_CFA_nop = 0x00;
constexpr uint8_t DW_CFA_set_loc = 0x01;
constexpr uint8_t DW_CFA_advance_loc1 = 0x02;
constexpr uint8_t DW_CFA_advance_loc2 = 0x03;
constexpr uint8_t DW_CFA_advance_loc4 = 0x04;
constexpr uint8_t DW_CFA_offset_extended = 0x05;
constexpr uint8_t DW_CFA_restore_extended = 0x06;
constexpr uint8_t DW_CFA_undefined = 0x07;
constexpr uint8_t DW_CFA_same_value = 0x08;
constexpr uint8_t DW_CFA_register = 0x09;
constexpr uint8_t DW_CFA_remember_state = 0x0a;
constexpr uint8_t DW_CFA_restore_state = 0x0b;
constexpr uint8_t DW_CFA_def_cfa = 0x0c;
constexpr uint8_t DW_CFA_def_cfa_register = 0x0d;
constexpr uint8_t DW_CFA_def_cfa_offset = 0x0e;
constexpr uint8_t DW_CFA_def_cfa_expression = 0x0f;
constexpr uint8_t DW_CFA_expression = 0x10;
constexpr uint8_t DW_CFA_offset_extended_sf = 0x11;
constexpr uint8_t DW_CFA_def_cfa_sf = 0x12;
constexpr uint8_t DW_CFA_def_cfa_offset_sf = 0x13;
constexpr uint8_t DW_CFA_val_offset = 0x14;
constexpr uint8_t DW_CFA_val_offset_sf = 0x15;
constexpr uint8_t DW_CFA_val_expression = 0x16;

class Cursor {
public:
  Cursor(std::span<const uint8_t> bytes, size_t offset = 0) : bytes_(bytes), offset_(offset) {}

  size_t offset() const { return offset_; }
  size_t remaining() const { return offset_ <= bytes_.size() ? bytes_.size() - offset_ : 0; }
  bool seek(size_t offset) {
    if (offset > bytes_.size())
      return false;
    offset_ = offset;
    return true;
  }
  bool skip(size_t count) {
    if (count > remaining())
      return false;
    offset_ += count;
    return true;
  }
  template <typename T> bool read(T &value) {
    if (sizeof(T) > remaining())
      return false;
    std::memcpy(&value, bytes_.data() + offset_, sizeof(T));
    offset_ += sizeof(T);
    return true;
  }
  bool read_uint(size_t size, uint64_t &value) {
    if (size == 0 || size > sizeof(value) || size > remaining())
      return false;
    value = 0;
    std::memcpy(&value, bytes_.data() + offset_, size);
    offset_ += size;
    return true;
  }
  bool read_uleb(uint64_t &value) {
    value = 0;
    unsigned shift = 0;
    for (unsigned i = 0; i < 10; ++i) {
      uint8_t byte = 0;
      if (!read(byte) || (shift == 63 && (byte & 0xfe) != 0))
        return false;
      value |= static_cast<uint64_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0)
        return true;
      shift += 7;
    }
    return false;
  }
  bool read_sleb(int64_t &value) {
    value = 0;
    unsigned shift = 0;
    uint8_t byte = 0;
    for (unsigned i = 0; i < 10; ++i) {
      if (!read(byte))
        return false;
      value |= static_cast<int64_t>(byte & 0x7f) << shift;
      shift += 7;
      if ((byte & 0x80) == 0) {
        if (shift < 64 && (byte & 0x40) != 0)
          value |= -(int64_t{1} << shift);
        return true;
      }
    }
    return false;
  }

private:
  std::span<const uint8_t> bytes_;
  size_t offset_ = 0;
};

template <typename T> void append_value(std::vector<uint8_t> &out, T value) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
  out.insert(out.end(), bytes, bytes + sizeof(value));
}

void append_uleb(std::vector<uint8_t> &out, uint64_t value) {
  do {
    uint8_t byte = static_cast<uint8_t>(value & 0x7f);
    value >>= 7;
    if (value != 0)
      byte |= 0x80;
    out.push_back(byte);
  } while (value != 0);
}

void append_sleb(std::vector<uint8_t> &out, int64_t value) {
  bool more = true;
  while (more) {
    uint8_t byte = static_cast<uint8_t>(value & 0x7f);
    const bool sign = (byte & 0x40) != 0;
    value >>= 7;
    more = !((value == 0 && !sign) || (value == -1 && sign));
    if (more)
      byte |= 0x80;
    out.push_back(byte);
  }
}

bool write_uint(std::vector<uint8_t> &bytes, size_t offset, size_t size, uint64_t value) {
  if (size == 0 || size > sizeof(value) || offset > bytes.size() || size > bytes.size() - offset)
    return false;
  if (size < sizeof(value) && value >= (uint64_t{1} << (size * 8)))
    return false;
  std::memcpy(bytes.data() + offset, &value, size);
  return true;
}

struct AddressMap {
  uint64_t text_address = 0;
  uint64_t old_size = 0;
  uint64_t new_size = 0;
  std::unordered_map<uint64_t, uint64_t> exact;
  std::vector<std::pair<uint64_t, uint64_t>> sorted;

  AddressMap(uint64_t address, uint64_t old_text_size, uint64_t new_text_size,
             std::span<const TextOffsetRelocation> relocations)
      : text_address(address), old_size(old_text_size), new_size(new_text_size) {
    exact.reserve(relocations.size());
    sorted.reserve(relocations.size());
    for (const auto &relocation : relocations) {
      exact.try_emplace(relocation.source_offset, relocation.target_offset);
    }
    for (const auto &[source, target] : exact)
      sorted.emplace_back(source, target);
    std::ranges::sort(sorted);
  }

  bool inside(uint64_t address) const {
    return address >= text_address && address - text_address <= old_size;
  }
  std::optional<uint64_t> translate(uint64_t address) const {
    if (!inside(address))
      return address;
    const auto found = exact.find(address - text_address);
    if (found == exact.end())
      return std::nullopt;
    return text_address + found->second;
  }
  std::optional<uint64_t> translate_boundary(uint64_t address) const {
    if (const auto exact_address = translate(address))
      return exact_address;
    if (!inside(address))
      return address;

    // Debug producers may extend a function range through alignment padding,
    // for which DBT intentionally emits no instruction mapping. The first
    // mapped source location after that padding is the same boundary. Using it
    // is safe only as an endpoint; callers still verify that it follows the
    // relocated range start.
    const uint64_t source_offset = address - text_address;
    const auto next =
        std::lower_bound(sorted.begin(), sorted.end(), source_offset,
                         [](const auto &entry, uint64_t value) { return entry.first < value; });
    if (next == sorted.end())
      return std::nullopt;
    return text_address + next->second;
  }
  std::vector<std::pair<uint64_t, uint64_t>> translate_range(uint64_t begin, uint64_t end) const {
    if (begin >= end)
      return {};
    const uint64_t lo = begin <= text_address ? 0 : begin - text_address;
    const uint64_t hi = end <= text_address ? 0 : std::min(end - text_address, old_size);
    auto it =
        std::lower_bound(sorted.begin(), sorted.end(), lo,
                         [](const auto &entry, uint64_t value) { return entry.first < value; });
    if (it == sorted.end() || it->first > hi)
      return {};
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    uint64_t range_begin = it->second;
    uint64_t range_end = it->second;
    for (++it; it != sorted.end() && it->first <= hi; ++it) {
      const uint64_t target = it->second;
      // Translation can reorder independently placed code regions. Start a
      // new range only at such a discontinuity. A forward jump is growth
      // inside the source range and must remain covered so inserted lowering
      // instructions belong to the same debug scope.
      if (target < range_end) {
        ranges.emplace_back(text_address + range_begin, text_address + range_end);
        range_begin = target;
      }
      range_end = target;
    }
    ranges.emplace_back(text_address + range_begin, text_address + range_end);
    return ranges;
  }
};

std::optional<size_t> find_section(std::span<const uint8_t> image, const Elf64_Ehdr &header,
                                   std::span<const Elf64_Shdr> sections, std::string_view name) {
  if (header.e_shstrndx >= sections.size())
    return std::nullopt;
  const auto &strings = sections[header.e_shstrndx];
  if (strings.sh_offset > image.size() || strings.sh_size > image.size() - strings.sh_offset)
    return std::nullopt;
  const char *names = reinterpret_cast<const char *>(image.data() + strings.sh_offset);
  for (size_t i = 0; i < sections.size(); ++i) {
    if (sections[i].sh_name >= strings.sh_size)
      continue;
    const char *candidate = names + sections[i].sh_name;
    const size_t available = strings.sh_size - sections[i].sh_name;
    const void *terminator = std::memchr(candidate, 0, available);
    if (terminator != nullptr && std::string_view(candidate) == name)
      return i;
  }
  return std::nullopt;
}

bool replace_section(std::vector<uint8_t> &image, std::vector<Elf64_Shdr> &sections, size_t index,
                     std::span<const uint8_t> payload) {
  if (index >= sections.size() || (sections[index].sh_flags & SHF_ALLOC) != 0)
    return false;
  Elf64_Shdr &section = sections[index];
  if (payload.size() <= section.sh_size && section.sh_offset <= image.size() &&
      section.sh_size <= image.size() - section.sh_offset) {
    std::memcpy(image.data() + section.sh_offset, payload.data(), payload.size());
    std::fill(image.begin() + static_cast<std::ptrdiff_t>(section.sh_offset + payload.size()),
              image.begin() + static_cast<std::ptrdiff_t>(section.sh_offset + section.sh_size), 0);
    section.sh_size = payload.size();
    return true;
  }
  const uint64_t alignment = std::max<uint64_t>(section.sh_addralign, 1);
  const uint64_t padding = (alignment - image.size() % alignment) % alignment;
  image.insert(image.end(), padding, 0);
  section.sh_offset = image.size();
  section.sh_size = payload.size();
  image.insert(image.end(), payload.begin(), payload.end());
  return true;
}

struct AddressEntry {
  uint64_t source = 0;
  size_t field_offset = 0;
  uint8_t size = 0;
};

struct AddressTables {
  std::unordered_map<uint64_t, AddressEntry> by_section_offset;
};

bool relocate_debug_addr(std::vector<uint8_t> &image, const Elf64_Shdr &section,
                         const AddressMap &addresses, AddressTables &tables) {
  if (section.sh_offset > image.size() || section.sh_size > image.size() - section.sh_offset)
    return false;
  const auto bytes = std::span<const uint8_t>(image).subspan(section.sh_offset, section.sh_size);
  Cursor cursor(bytes);
  while (cursor.remaining() != 0) {
    const size_t unit_start = cursor.offset();
    uint32_t length = 0;
    uint16_t version = 0;
    uint8_t address_size = 0, segment_size = 0;
    if (!cursor.read(length) || length == 0xffffffffu || length > cursor.remaining() ||
        !cursor.read(version) || !cursor.read(address_size) || !cursor.read(segment_size) ||
        version != 5 || segment_size != 0 || address_size == 0 || address_size > 8) {
      return false;
    }
    const size_t unit_end = unit_start + sizeof(length) + length;
    while (cursor.offset() < unit_end) {
      const size_t entry_offset = cursor.offset();
      uint64_t source = 0;
      if (!cursor.read_uint(address_size, source))
        return false;
      tables.by_section_offset.emplace(
          entry_offset, AddressEntry{source, section.sh_offset + entry_offset, address_size});
      if (addresses.inside(source)) {
        const uint64_t target = addresses.translate(source).value_or(0);
        if (!write_uint(image, section.sh_offset + entry_offset, address_size, target))
          return false;
      }
    }
    if (cursor.offset() != unit_end)
      return false;
  }
  return true;
}

struct LineState {
  uint64_t address = 0;
  uint64_t op_index = 0;
  uint64_t file = 1;
  int64_t line = 1;
  uint64_t column = 0;
  bool is_stmt = false;
  bool basic_block = false;
  bool end_sequence = false;
  bool prologue_end = false;
  bool epilogue_begin = false;
  uint64_t isa = 0;
  uint64_t discriminator = 0;
};

void reset_line_state(LineState &state, bool default_is_stmt) {
  state = {};
  state.file = 1;
  state.line = 1;
  state.is_stmt = default_is_stmt;
}

void advance_line_address(LineState &state, uint64_t operation_advance, uint8_t minimum_length,
                          uint8_t maximum_operations) {
  const uint64_t operations = state.op_index + operation_advance;
  state.address += minimum_length * (operations / maximum_operations);
  state.op_index = operations % maximum_operations;
}

void append_extended(std::vector<uint8_t> &out, uint8_t opcode,
                     std::span<const uint8_t> operand = {}) {
  out.push_back(0);
  append_uleb(out, operand.size() + 1);
  out.push_back(opcode);
  out.insert(out.end(), operand.begin(), operand.end());
}

void append_set_address(std::vector<uint8_t> &out, uint64_t address, uint8_t address_size) {
  std::vector<uint8_t> operand(address_size);
  std::memcpy(operand.data(), &address, address_size);
  append_extended(out, DW_LNE_set_address, operand);
}

bool rewrite_line_unit(std::span<const uint8_t> unit, const AddressMap &addresses,
                       std::vector<uint8_t> &output) {
  Cursor cursor(unit);
  uint32_t old_length = 0;
  uint16_t version = 0;
  uint8_t address_size = 0, segment_size = 0;
  uint32_t header_length = 0;
  if (!cursor.read(old_length) || old_length == 0xffffffffu || old_length + 4 != unit.size() ||
      !cursor.read(version) || version != 5 || !cursor.read(address_size) ||
      !cursor.read(segment_size) || segment_size != 0 || address_size == 0 || address_size > 8 ||
      !cursor.read(header_length) || header_length > cursor.remaining()) {
    return false;
  }
  const size_t prologue_start = cursor.offset();
  const size_t program_start = prologue_start + header_length;
  uint8_t minimum_length = 0, maximum_operations = 0, default_is_stmt = 0;
  int8_t line_base = 0;
  uint8_t line_range = 0, opcode_base = 0;
  if (!cursor.read(minimum_length) || !cursor.read(maximum_operations) || maximum_operations == 0 ||
      !cursor.read(default_is_stmt) || !cursor.read(line_base) || !cursor.read(line_range) ||
      line_range == 0 || !cursor.read(opcode_base) || opcode_base == 0 ||
      static_cast<size_t>(opcode_base - 1) > cursor.remaining() || !cursor.seek(program_start)) {
    return false;
  }

  std::vector<LineState> rows;
  LineState state;
  reset_line_state(state, default_is_stmt != 0);
  auto emit_row = [&] {
    rows.push_back(state);
    state.basic_block = false;
    state.prologue_end = false;
    state.epilogue_begin = false;
    state.discriminator = 0;
  };
  while (cursor.remaining() != 0) {
    uint8_t opcode = 0;
    if (!cursor.read(opcode))
      return false;
    if (opcode == 0) {
      uint64_t length = 0;
      if (!cursor.read_uleb(length) || length == 0 || length > cursor.remaining())
        return false;
      const size_t extended_end = cursor.offset() + length;
      uint8_t extended = 0;
      if (!cursor.read(extended))
        return false;
      if (extended == DW_LNE_end_sequence) {
        state.end_sequence = true;
        emit_row();
        reset_line_state(state, default_is_stmt != 0);
      } else if (extended == DW_LNE_set_address) {
        if (length != static_cast<uint64_t>(address_size) + 1 ||
            !cursor.read_uint(address_size, state.address))
          return false;
        state.op_index = 0;
      } else if (extended == DW_LNE_set_discriminator) {
        if (!cursor.read_uleb(state.discriminator))
          return false;
      }
      if (!cursor.seek(extended_end))
        return false;
      continue;
    }
    if (opcode >= opcode_base) {
      const uint8_t adjusted = opcode - opcode_base;
      advance_line_address(state, adjusted / line_range, minimum_length, maximum_operations);
      state.line += line_base + adjusted % line_range;
      emit_row();
      continue;
    }
    uint64_t operand = 0;
    int64_t signed_operand = 0;
    switch (opcode) {
    case DW_LNS_copy:
      emit_row();
      break;
    case DW_LNS_advance_pc:
      if (!cursor.read_uleb(operand))
        return false;
      advance_line_address(state, operand, minimum_length, maximum_operations);
      break;
    case DW_LNS_advance_line:
      if (!cursor.read_sleb(signed_operand))
        return false;
      state.line += signed_operand;
      break;
    case DW_LNS_set_file:
      if (!cursor.read_uleb(state.file))
        return false;
      break;
    case DW_LNS_set_column:
      if (!cursor.read_uleb(state.column))
        return false;
      break;
    case DW_LNS_negate_stmt:
      state.is_stmt = !state.is_stmt;
      break;
    case DW_LNS_set_basic_block:
      state.basic_block = true;
      break;
    case DW_LNS_const_add_pc:
      advance_line_address(state, (255 - opcode_base) / line_range, minimum_length,
                           maximum_operations);
      break;
    case DW_LNS_fixed_advance_pc: {
      uint16_t advance = 0;
      if (!cursor.read(advance))
        return false;
      state.address += advance;
      state.op_index = 0;
      break;
    }
    case DW_LNS_set_prologue_end:
      state.prologue_end = true;
      break;
    case DW_LNS_set_epilogue_begin:
      state.epilogue_begin = true;
      break;
    case DW_LNS_set_isa:
      if (!cursor.read_uleb(state.isa))
        return false;
      break;
    default:
      // The prologue says how many ULEB operands an unknown standard opcode has.
      if (prologue_start + 6 + opcode > program_start)
        return false;
      for (uint8_t i = 0; i < unit[prologue_start + 6 + opcode - 1]; ++i) {
        if (!cursor.read_uleb(operand))
          return false;
      }
      break;
    }
  }

  std::vector<uint8_t> program;
  LineState encoded;
  reset_line_state(encoded, default_is_stmt != 0);
  bool sequence_has_rows = false;
  std::vector<uint8_t> sequence;
  auto flush_sequence = [&] {
    if (sequence_has_rows)
      program.insert(program.end(), sequence.begin(), sequence.end());
    sequence.clear();
    sequence_has_rows = false;
    reset_line_state(encoded, default_is_stmt != 0);
  };
  for (const LineState &source_row : rows) {
    LineState row = source_row;
    const auto target = addresses.translate(row.address);
    if (!target) {
      if (row.end_sequence) {
        if (sequence_has_rows) {
          append_set_address(sequence, encoded.address, address_size);
          append_extended(sequence, DW_LNE_end_sequence);
        }
        flush_sequence();
      }
      continue;
    }
    row.address = *target;
    if (row.end_sequence) {
      append_set_address(sequence, row.address, address_size);
      append_extended(sequence, DW_LNE_end_sequence);
      flush_sequence();
      continue;
    }
    sequence_has_rows = true;
    append_set_address(sequence, row.address, address_size);
    encoded.address = row.address;
    encoded.op_index = 0;
    if (row.file != encoded.file) {
      sequence.push_back(DW_LNS_set_file);
      append_uleb(sequence, row.file);
      encoded.file = row.file;
    }
    if (row.column != encoded.column) {
      sequence.push_back(DW_LNS_set_column);
      append_uleb(sequence, row.column);
      encoded.column = row.column;
    }
    if (row.line != encoded.line) {
      sequence.push_back(DW_LNS_advance_line);
      append_sleb(sequence, row.line - encoded.line);
      encoded.line = row.line;
    }
    if (row.is_stmt != encoded.is_stmt) {
      sequence.push_back(DW_LNS_negate_stmt);
      encoded.is_stmt = row.is_stmt;
    }
    if (row.basic_block)
      sequence.push_back(DW_LNS_set_basic_block);
    if (row.prologue_end)
      sequence.push_back(DW_LNS_set_prologue_end);
    if (row.epilogue_begin)
      sequence.push_back(DW_LNS_set_epilogue_begin);
    if (row.isa != encoded.isa) {
      sequence.push_back(DW_LNS_set_isa);
      append_uleb(sequence, row.isa);
      encoded.isa = row.isa;
    }
    if (row.discriminator != 0) {
      std::vector<uint8_t> value;
      append_uleb(value, row.discriminator);
      append_extended(sequence, DW_LNE_set_discriminator, value);
    }
    sequence.push_back(DW_LNS_copy);
  }
  flush_sequence();

  output.assign(unit.begin(), unit.begin() + static_cast<std::ptrdiff_t>(program_start));
  output.insert(output.end(), program.begin(), program.end());
  if (output.size() - 4 > std::numeric_limits<uint32_t>::max())
    return false;
  const uint32_t new_length = output.size() - 4;
  std::memcpy(output.data(), &new_length, sizeof(new_length));
  return true;
}

bool relocate_debug_line(std::vector<uint8_t> &image, std::vector<Elf64_Shdr> &sections,
                         size_t index, const AddressMap &addresses,
                         std::unordered_map<uint64_t, uint64_t> &unit_offsets) {
  const Elf64_Shdr section = sections[index];
  if (section.sh_offset > image.size() || section.sh_size > image.size() - section.sh_offset)
    return false;
  const auto input = std::span<const uint8_t>(image).subspan(section.sh_offset, section.sh_size);
  Cursor cursor(input);
  std::vector<uint8_t> output;
  while (cursor.remaining() != 0) {
    const size_t old_offset = cursor.offset();
    uint32_t length = 0;
    if (!cursor.read(length) || length == 0xffffffffu || length > cursor.remaining())
      return false;
    const size_t unit_size = sizeof(length) + length;
    std::vector<uint8_t> rewritten;
    if (!rewrite_line_unit(input.subspan(old_offset, unit_size), addresses, rewritten))
      return false;
    unit_offsets.emplace(old_offset, output.size());
    output.insert(output.end(), rewritten.begin(), rewritten.end());
    if (!cursor.seek(old_offset + unit_size))
      return false;
  }
  return replace_section(image, sections, index, output);
}

struct AbbrevAttribute {
  uint64_t name = 0;
  uint64_t form = 0;
  int64_t implicit = 0;
};
struct Abbrev {
  std::vector<AbbrevAttribute> attributes;
};
using AbbrevTable = std::unordered_map<uint64_t, Abbrev>;

bool parse_abbrev_table(std::span<const uint8_t> bytes, size_t offset, AbbrevTable &table) {
  Cursor cursor(bytes, offset);
  while (true) {
    uint64_t code = 0, tag = 0;
    uint8_t children = 0;
    if (!cursor.read_uleb(code))
      return false;
    if (code == 0)
      return true;
    if (!cursor.read_uleb(tag) || !cursor.read(children))
      return false;
    (void)tag;
    (void)children;
    Abbrev abbrev;
    while (true) {
      AbbrevAttribute attribute;
      if (!cursor.read_uleb(attribute.name) || !cursor.read_uleb(attribute.form))
        return false;
      if (attribute.name == 0 && attribute.form == 0)
        break;
      if (attribute.form == DW_FORM_implicit_const && !cursor.read_sleb(attribute.implicit))
        return false;
      abbrev.attributes.push_back(attribute);
    }
    if (!table.emplace(code, std::move(abbrev)).second)
      return false;
  }
}

struct FormValue {
  uint64_t value = 0;
  size_t field_offset = 0;
  size_t field_size = 0;
  uint64_t form = 0;
};

bool read_form(Cursor &cursor, uint64_t form, uint8_t address_size, uint8_t offset_size,
               int64_t implicit, FormValue &result) {
  result = {.field_offset = cursor.offset(), .form = form};
  auto fixed = [&](size_t size) {
    result.field_size = size;
    return cursor.read_uint(size, result.value);
  };
  switch (form) {
  case DW_FORM_addr:
    return fixed(address_size);
  case DW_FORM_data1:
  case DW_FORM_flag:
  case DW_FORM_ref1:
  case DW_FORM_strx1:
  case DW_FORM_addrx1:
    return fixed(1);
  case DW_FORM_data2:
  case DW_FORM_ref2:
  case DW_FORM_strx2:
  case DW_FORM_addrx2:
    return fixed(2);
  case DW_FORM_strx3:
  case DW_FORM_addrx3:
    return fixed(3);
  case DW_FORM_data4:
  case DW_FORM_ref4:
  case DW_FORM_ref_sup4:
  case DW_FORM_strx4:
  case DW_FORM_addrx4:
    return fixed(4);
  case DW_FORM_data8:
  case DW_FORM_ref8:
  case DW_FORM_ref_sig8:
  case DW_FORM_ref_sup8:
    return fixed(8);
  case DW_FORM_data16:
    result.field_size = 16;
    return cursor.skip(16);
  case DW_FORM_strp:
  case DW_FORM_line_strp:
  case DW_FORM_strp_sup:
  case DW_FORM_sec_offset:
    return fixed(offset_size);
  case DW_FORM_ref_addr:
    return fixed(offset_size);
  case DW_FORM_udata:
  case DW_FORM_ref_udata:
  case DW_FORM_strx:
  case DW_FORM_addrx:
  case DW_FORM_loclistx:
  case DW_FORM_rnglistx: {
    result.field_offset = cursor.offset();
    if (!cursor.read_uleb(result.value))
      return false;
    result.field_size = cursor.offset() - result.field_offset;
    return true;
  }
  case DW_FORM_sdata: {
    int64_t value = 0;
    if (!cursor.read_sleb(value))
      return false;
    result.value = static_cast<uint64_t>(value);
    result.field_size = cursor.offset() - result.field_offset;
    return true;
  }
  case DW_FORM_string:
    while (true) {
      uint8_t byte = 0;
      if (!cursor.read(byte))
        return false;
      if (byte == 0)
        return true;
    }
  case DW_FORM_block1: {
    uint8_t size = 0;
    return cursor.read(size) && cursor.skip(size);
  }
  case DW_FORM_block2: {
    uint16_t size = 0;
    return cursor.read(size) && cursor.skip(size);
  }
  case DW_FORM_block4: {
    uint32_t size = 0;
    return cursor.read(size) && cursor.skip(size);
  }
  case DW_FORM_block:
  case DW_FORM_exprloc: {
    uint64_t size = 0;
    return cursor.read_uleb(size) && size <= cursor.remaining() && cursor.skip(size);
  }
  case DW_FORM_flag_present:
    result.value = 1;
    return true;
  case DW_FORM_implicit_const:
    result.value = static_cast<uint64_t>(implicit);
    return true;
  case DW_FORM_indirect: {
    uint64_t actual = 0;
    return cursor.read_uleb(actual) &&
           read_form(cursor, actual, address_size, offset_size, 0, result);
  }
  default:
    return false;
  }
}

std::optional<uint64_t> source_address(const FormValue &value, uint64_t addr_base,
                                       const AddressTables &tables) {
  if (value.form == DW_FORM_addr)
    return value.value;
  bool indexed = value.form == DW_FORM_addrx || value.form == DW_FORM_addrx1 ||
                 value.form == DW_FORM_addrx2 || value.form == DW_FORM_addrx3 ||
                 value.form == DW_FORM_addrx4;
  if (!indexed)
    return std::nullopt;
  const auto first = tables.by_section_offset.find(addr_base);
  if (first == tables.by_section_offset.end())
    return std::nullopt;
  const uint64_t address_size = first->second.size;
  if (value.value > (std::numeric_limits<uint64_t>::max() - addr_base) / address_size)
    return std::nullopt;
  const auto found = tables.by_section_offset.find(addr_base + value.value * address_size);
  if (found == tables.by_section_offset.end())
    return std::nullopt;
  return found->second.source;
}

bool patch_form_uint(std::vector<uint8_t> &image, uint64_t section_offset, const FormValue &value,
                     uint64_t replacement) {
  if (value.form == DW_FORM_udata) {
    std::vector<uint8_t> encoded;
    append_uleb(encoded, replacement);
    if (encoded.size() != value.field_size)
      return false;
    std::memcpy(image.data() + section_offset + value.field_offset, encoded.data(), encoded.size());
    return true;
  }
  return value.field_size != 0 &&
         write_uint(image, section_offset + value.field_offset, value.field_size, replacement);
}

bool relocate_debug_info(std::vector<uint8_t> &image, const Elf64_Shdr &info,
                         const Elf64_Shdr &abbrev, const AddressMap &addresses,
                         const AddressTables &tables,
                         const std::unordered_map<uint64_t, uint64_t> &line_offsets,
                         const std::unordered_map<uint64_t, uint64_t> &rnglist_bases,
                         const std::unordered_map<uint64_t, uint64_t> &loclist_bases) {
  if (info.sh_offset > image.size() || info.sh_size > image.size() - info.sh_offset ||
      abbrev.sh_offset > image.size() || abbrev.sh_size > image.size() - abbrev.sh_offset)
    return false;
  const auto info_bytes = std::span<const uint8_t>(image).subspan(info.sh_offset, info.sh_size);
  const auto abbrev_bytes =
      std::span<const uint8_t>(image).subspan(abbrev.sh_offset, abbrev.sh_size);
  std::unordered_map<uint64_t, AbbrevTable> abbrev_tables;
  Cursor cursor(info_bytes);
  while (cursor.remaining() != 0) {
    const size_t unit_start = cursor.offset();
    uint32_t length = 0;
    uint16_t version = 0;
    uint8_t unit_type = 0, address_size = 0;
    uint32_t abbrev_offset = 0;
    if (!cursor.read(length) || length == 0xffffffffu || length > cursor.remaining() ||
        !cursor.read(version) || version != 5 || !cursor.read(unit_type) ||
        !cursor.read(address_size) || !cursor.read(abbrev_offset) || address_size == 0 ||
        address_size > 8) {
      return false;
    }
    // Other DWARF 5 unit kinds add fields to the unit header. Parse only the
    // ordinary compile-unit layout instead of mistaking those fields for DIEs.
    if (unit_type != 0x01)
      return false;
    const size_t unit_end = unit_start + sizeof(length) + length;
    auto [table_it, inserted] = abbrev_tables.try_emplace(abbrev_offset);
    if (inserted && !parse_abbrev_table(abbrev_bytes, abbrev_offset, table_it->second))
      return false;
    uint64_t addr_base = 0;
    while (cursor.offset() < unit_end) {
      uint64_t code = 0;
      if (!cursor.read_uleb(code))
        return false;
      if (code == 0)
        continue;
      const auto definition = table_it->second.find(code);
      if (definition == table_it->second.end())
        return false;
      std::optional<FormValue> low_pc;
      std::optional<FormValue> high_pc;
      for (const AbbrevAttribute &attribute : definition->second.attributes) {
        FormValue value;
        if (!read_form(cursor, attribute.form, address_size, 4, attribute.implicit, value))
          return false;
        if (attribute.name == DW_AT_addr_base)
          addr_base = value.value;
        else if (attribute.name == DW_AT_stmt_list) {
          if (const auto found = line_offsets.find(value.value);
              found != line_offsets.end() &&
              !patch_form_uint(image, info.sh_offset, value, found->second))
            return false;
        } else if (attribute.name == DW_AT_rnglists_base) {
          if (const auto found = rnglist_bases.find(value.value);
              found != rnglist_bases.end() &&
              !patch_form_uint(image, info.sh_offset, value, found->second))
            return false;
        } else if (attribute.name == DW_AT_loclists_base) {
          if (const auto found = loclist_bases.find(value.value);
              found != loclist_bases.end() &&
              !patch_form_uint(image, info.sh_offset, value, found->second))
            return false;
        } else if (attribute.name == DW_AT_low_pc) {
          low_pc = value;
        } else if (attribute.name == DW_AT_high_pc) {
          high_pc = value;
        }
      }
      if (!low_pc)
        continue;
      const auto source_low = source_address(*low_pc, addr_base, tables);
      if (!source_low || !addresses.inside(*source_low))
        continue;
      const auto target_low = addresses.translate(*source_low);
      if (low_pc->form == DW_FORM_addr &&
          !patch_form_uint(image, info.sh_offset, *low_pc, target_low.value_or(0)))
        return false;
      if (!high_pc)
        continue;
      const bool high_is_address =
          high_pc->form == DW_FORM_addr || high_pc->form == DW_FORM_addrx ||
          high_pc->form == DW_FORM_addrx1 || high_pc->form == DW_FORM_addrx2 ||
          high_pc->form == DW_FORM_addrx3 || high_pc->form == DW_FORM_addrx4;
      if (high_is_address) {
        if (high_pc->form == DW_FORM_addr) {
          const auto source_high = source_address(*high_pc, addr_base, tables);
          const uint64_t target_high =
              source_high ? addresses.translate(*source_high).value_or(0) : 0;
          if (!patch_form_uint(image, info.sh_offset, *high_pc, target_high))
            return false;
        }
        // Indexed high-PC values were already handled in `.debug_addr`.
        continue;
      }
      uint64_t new_length = 0;
      if (target_low) {
        if (high_pc->value > std::numeric_limits<uint64_t>::max() - *source_low)
          return false;
        const uint64_t source_high = *source_low + high_pc->value;
        auto target_high = addresses.translate_boundary(source_high);
        if (!target_high) {
          const auto ranges = addresses.translate_range(*source_low, source_high);
          if (ranges.size() == 1 && ranges.front().first == *target_low)
            target_high = ranges.front().second;
        }
        if (target_high && *target_high >= *target_low)
          new_length = *target_high - *target_low;
      }
      if (!patch_form_uint(image, info.sh_offset, *high_pc, new_length))
        return false;
    }
    if (cursor.offset() != unit_end)
      return false;
  }
  return true;
}

bool decode_rnglist(Cursor &cursor, size_t end, uint8_t address_size, uint64_t initial_base,
                    uint64_t addr_base, const AddressTables &tables,
                    std::vector<std::pair<uint64_t, uint64_t>> &ranges) {
  uint64_t base = initial_base;
  auto addrx = [&](uint64_t index) -> std::optional<uint64_t> {
    const auto found = tables.by_section_offset.find(addr_base + index * address_size);
    return found == tables.by_section_offset.end() ? std::nullopt
                                                   : std::optional(found->second.source);
  };
  while (cursor.offset() < end) {
    uint8_t encoding = 0;
    if (!cursor.read(encoding))
      return false;
    uint64_t first = 0, second = 0;
    switch (encoding) {
    case DW_RLE_end_of_list:
      return true;
    case DW_RLE_base_addressx: {
      if (!cursor.read_uleb(first))
        return false;
      const auto value = addrx(first);
      if (!value)
        return false;
      base = *value;
      break;
    }
    case DW_RLE_startx_endx: {
      if (!cursor.read_uleb(first) || !cursor.read_uleb(second))
        return false;
      const auto begin = addrx(first), finish = addrx(second);
      if (!begin || !finish)
        return false;
      ranges.emplace_back(*begin, *finish);
      break;
    }
    case DW_RLE_startx_length: {
      if (!cursor.read_uleb(first) || !cursor.read_uleb(second))
        return false;
      const auto begin = addrx(first);
      if (!begin || second > std::numeric_limits<uint64_t>::max() - *begin)
        return false;
      ranges.emplace_back(*begin, *begin + second);
      break;
    }
    case DW_RLE_offset_pair:
      if (!cursor.read_uleb(first) || !cursor.read_uleb(second) ||
          first > std::numeric_limits<uint64_t>::max() - base ||
          second > std::numeric_limits<uint64_t>::max() - base)
        return false;
      ranges.emplace_back(base + first, base + second);
      break;
    case DW_RLE_base_address:
      if (!cursor.read_uint(address_size, base))
        return false;
      break;
    case DW_RLE_start_end:
      if (!cursor.read_uint(address_size, first) || !cursor.read_uint(address_size, second))
        return false;
      ranges.emplace_back(first, second);
      break;
    case DW_RLE_start_length:
      if (!cursor.read_uint(address_size, first) || !cursor.read_uleb(second) ||
          second > std::numeric_limits<uint64_t>::max() - first)
        return false;
      ranges.emplace_back(first, first + second);
      break;
    default:
      return false;
    }
  }
  return false;
}

[[nodiscard]] std::vector<uint64_t> source_address_candidates(const AddressTables &tables) {
  std::vector<uint64_t> candidates;
  candidates.reserve(tables.by_section_offset.size());
  for (const auto &[offset, address] : tables.by_section_offset) {
    (void)offset;
    candidates.push_back(address.source);
  }
  std::ranges::sort(candidates);
  candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
  return candidates;
}

[[nodiscard]] bool range_touches_text(const AddressMap &addresses, uint64_t begin, uint64_t end) {
  return addresses.inside(begin) || addresses.inside(end);
}

bool relocate_debug_rnglists(std::vector<uint8_t> &image, std::vector<Elf64_Shdr> &sections,
                             size_t index, const AddressMap &addresses, const AddressTables &tables,
                             std::unordered_map<uint64_t, uint64_t> &base_offsets) {
  const Elf64_Shdr section = sections[index];
  if (section.sh_offset > image.size() || section.sh_size > image.size() - section.sh_offset)
    return false;
  const auto input = std::span<const uint8_t>(image).subspan(section.sh_offset, section.sh_size);
  Cursor cursor(input);
  std::vector<uint8_t> output;
  uint64_t inferred_addr_base = 0;
  if (!tables.by_section_offset.empty()) {
    inferred_addr_base =
        std::min_element(tables.by_section_offset.begin(), tables.by_section_offset.end(),
                         [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; })
            ->first;
  }
  const auto address_candidates = source_address_candidates(tables);
  while (cursor.remaining() != 0) {
    const size_t unit_start = cursor.offset();
    uint32_t length = 0, offset_count = 0;
    uint16_t version = 0;
    uint8_t address_size = 0, segment_size = 0;
    if (!cursor.read(length) || length == 0xffffffffu || length > cursor.remaining() ||
        !cursor.read(version) || version != 5 || !cursor.read(address_size) ||
        !cursor.read(segment_size) || segment_size != 0 || address_size == 0 || address_size > 8 ||
        !cursor.read(offset_count) || offset_count == 0 ||
        offset_count > cursor.remaining() / sizeof(uint32_t))
      return false;
    const size_t unit_end = unit_start + sizeof(length) + length;
    const size_t old_base = cursor.offset();
    std::vector<uint32_t> old_offsets(offset_count);
    for (uint32_t &offset : old_offsets) {
      if (!cursor.read(offset))
        return false;
    }
    const size_t lists_start = cursor.offset();
    std::vector<std::vector<uint8_t>> lists;
    lists.reserve(offset_count);
    for (uint32_t offset : old_offsets) {
      const size_t list_start = old_base + offset;
      if (list_start < lists_start || list_start >= unit_end || !cursor.seek(list_start))
        return false;
      std::vector<std::pair<uint64_t, uint64_t>> source_ranges;
      if (!decode_rnglist(cursor, unit_end, address_size, 0, inferred_addr_base, tables,
                          source_ranges))
        return false;
      const auto touches_text = [&](const auto &ranges) {
        return std::ranges::any_of(ranges, [&](const auto &range) {
          return range_touches_text(addresses, range.first, range.second);
        });
      };
      if (!source_ranges.empty() && !touches_text(source_ranges)) {
        std::optional<std::vector<std::pair<uint64_t, uint64_t>>> rebased_ranges;
        for (uint64_t candidate : address_candidates) {
          if (!cursor.seek(list_start))
            return false;
          std::vector<std::pair<uint64_t, uint64_t>> candidate_ranges;
          if (!decode_rnglist(cursor, unit_end, address_size, candidate, inferred_addr_base, tables,
                              candidate_ranges))
            return false;
          if (!touches_text(candidate_ranges))
            continue;
          if (rebased_ranges)
            return false;
          rebased_ranges = std::move(candidate_ranges);
        }
        if (rebased_ranges)
          source_ranges = std::move(*rebased_ranges);
      }
      std::vector<std::pair<uint64_t, uint64_t>> target_ranges;
      for (const auto &[begin, end] : source_ranges) {
        if (!addresses.inside(begin) && !addresses.inside(end)) {
          target_ranges.emplace_back(begin, end);
          continue;
        }
        auto translated = addresses.translate_range(begin, end);
        target_ranges.insert(target_ranges.end(), translated.begin(), translated.end());
      }
      std::ranges::sort(target_ranges);
      std::vector<uint8_t> encoded;
      for (const auto &[begin, end] : target_ranges) {
        if (begin >= end)
          continue;
        encoded.push_back(DW_RLE_start_end);
        const size_t old_size = encoded.size();
        encoded.resize(old_size + 2 * address_size);
        std::memcpy(encoded.data() + old_size, &begin, address_size);
        std::memcpy(encoded.data() + old_size + address_size, &end, address_size);
      }
      encoded.push_back(DW_RLE_end_of_list);
      lists.push_back(std::move(encoded));
    }

    const size_t new_unit_start = output.size();
    append_value<uint32_t>(output, 0);
    append_value<uint16_t>(output, version);
    output.push_back(address_size);
    output.push_back(segment_size);
    append_value<uint32_t>(output, offset_count);
    const size_t new_base = output.size();
    output.resize(output.size() + static_cast<size_t>(offset_count) * sizeof(uint32_t));
    for (size_t i = 0; i < lists.size(); ++i) {
      const uint64_t relative = output.size() - new_base;
      if (relative > std::numeric_limits<uint32_t>::max())
        return false;
      const uint32_t relative32 = relative;
      std::memcpy(output.data() + new_base + i * sizeof(uint32_t), &relative32, sizeof(relative32));
      output.insert(output.end(), lists[i].begin(), lists[i].end());
    }
    const uint64_t new_length = output.size() - new_unit_start - sizeof(uint32_t);
    if (new_length > std::numeric_limits<uint32_t>::max())
      return false;
    const uint32_t new_length32 = new_length;
    std::memcpy(output.data() + new_unit_start, &new_length32, sizeof(new_length32));
    base_offsets.emplace(old_base, new_base);
    if (!cursor.seek(unit_end))
      return false;
  }
  return replace_section(image, sections, index, output);
}

struct LocationRange {
  uint64_t begin = 0;
  uint64_t end = 0;
  std::vector<uint8_t> expression;
};

bool read_location_expression(Cursor &cursor, std::vector<uint8_t> &expression) {
  uint64_t size = 0;
  if (!cursor.read_uleb(size) || size > cursor.remaining())
    return false;
  expression.resize(size);
  for (uint8_t &byte : expression) {
    if (!cursor.read(byte))
      return false;
  }
  return true;
}

bool decode_loclist(Cursor &cursor, size_t end, uint8_t address_size, uint64_t initial_base,
                    uint64_t addr_base, const AddressTables &tables,
                    std::vector<LocationRange> &ranges,
                    std::vector<std::vector<uint8_t>> &default_locations) {
  uint64_t base = initial_base;
  auto addrx = [&](uint64_t index) -> std::optional<uint64_t> {
    const auto found = tables.by_section_offset.find(addr_base + index * address_size);
    return found == tables.by_section_offset.end() ? std::nullopt
                                                   : std::optional(found->second.source);
  };
  while (cursor.offset() < end) {
    uint8_t encoding = 0;
    uint64_t first = 0, second = 0;
    LocationRange range;
    if (!cursor.read(encoding))
      return false;
    switch (encoding) {
    case DW_LLE_end_of_list:
      return true;
    case DW_LLE_base_addressx: {
      if (!cursor.read_uleb(first))
        return false;
      const auto value = addrx(first);
      if (!value)
        return false;
      base = *value;
      continue;
    }
    case DW_LLE_startx_endx: {
      if (!cursor.read_uleb(first) || !cursor.read_uleb(second))
        return false;
      const auto begin = addrx(first), finish = addrx(second);
      if (!begin || !finish)
        return false;
      range.begin = *begin;
      range.end = *finish;
      break;
    }
    case DW_LLE_startx_length: {
      if (!cursor.read_uleb(first) || !cursor.read_uleb(second))
        return false;
      const auto begin = addrx(first);
      if (!begin || second > std::numeric_limits<uint64_t>::max() - *begin)
        return false;
      range.begin = *begin;
      range.end = *begin + second;
      break;
    }
    case DW_LLE_offset_pair:
      if (!cursor.read_uleb(first) || !cursor.read_uleb(second) ||
          first > std::numeric_limits<uint64_t>::max() - base ||
          second > std::numeric_limits<uint64_t>::max() - base)
        return false;
      range.begin = base + first;
      range.end = base + second;
      break;
    case DW_LLE_default_location: {
      std::vector<uint8_t> expression;
      if (!read_location_expression(cursor, expression))
        return false;
      default_locations.push_back(std::move(expression));
      continue;
    }
    case DW_LLE_base_address:
      if (!cursor.read_uint(address_size, base))
        return false;
      continue;
    case DW_LLE_start_end:
      if (!cursor.read_uint(address_size, range.begin) ||
          !cursor.read_uint(address_size, range.end))
        return false;
      break;
    case DW_LLE_start_length:
      if (!cursor.read_uint(address_size, range.begin) || !cursor.read_uleb(second) ||
          second > std::numeric_limits<uint64_t>::max() - range.begin)
        return false;
      range.end = range.begin + second;
      break;
    default:
      return false;
    }
    if (!read_location_expression(cursor, range.expression))
      return false;
    ranges.push_back(std::move(range));
  }
  return false;
}

bool relocate_debug_loclists(std::vector<uint8_t> &image, std::vector<Elf64_Shdr> &sections,
                             size_t index, const AddressMap &addresses, const AddressTables &tables,
                             std::unordered_map<uint64_t, uint64_t> &base_offsets) {
  const Elf64_Shdr section = sections[index];
  if (section.sh_offset > image.size() || section.sh_size > image.size() - section.sh_offset)
    return false;
  const auto input = std::span<const uint8_t>(image).subspan(section.sh_offset, section.sh_size);
  Cursor cursor(input);
  std::vector<uint8_t> output;
  uint64_t inferred_addr_base = 0;
  if (!tables.by_section_offset.empty()) {
    inferred_addr_base =
        std::min_element(tables.by_section_offset.begin(), tables.by_section_offset.end(),
                         [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; })
            ->first;
  }
  const auto address_candidates = source_address_candidates(tables);
  while (cursor.remaining() != 0) {
    const size_t unit_start = cursor.offset();
    uint32_t length = 0, offset_count = 0;
    uint16_t version = 0;
    uint8_t address_size = 0, segment_size = 0;
    if (!cursor.read(length) || length == 0xffffffffu || length > cursor.remaining() ||
        !cursor.read(version) || version != 5 || !cursor.read(address_size) ||
        !cursor.read(segment_size) || segment_size != 0 || address_size == 0 || address_size > 8 ||
        !cursor.read(offset_count) || offset_count == 0 ||
        offset_count > cursor.remaining() / sizeof(uint32_t))
      return false;
    const size_t unit_end = unit_start + sizeof(length) + length;
    const size_t old_base = cursor.offset();
    std::vector<uint32_t> old_offsets(offset_count);
    for (uint32_t &offset : old_offsets) {
      if (!cursor.read(offset))
        return false;
    }
    const size_t lists_start = cursor.offset();
    std::vector<std::vector<uint8_t>> lists;
    lists.reserve(offset_count);
    for (uint32_t offset : old_offsets) {
      const size_t list_start = old_base + offset;
      if (list_start < lists_start || list_start >= unit_end || !cursor.seek(list_start))
        return false;
      std::vector<LocationRange> source_ranges;
      std::vector<std::vector<uint8_t>> defaults;
      if (!decode_loclist(cursor, unit_end, address_size, 0, inferred_addr_base, tables,
                          source_ranges, defaults))
        return false;
      const auto touches_text = [&](const auto &ranges) {
        return std::ranges::any_of(ranges, [&](const LocationRange &range) {
          return range_touches_text(addresses, range.begin, range.end);
        });
      };
      if (!source_ranges.empty() && !touches_text(source_ranges)) {
        std::optional<std::vector<LocationRange>> rebased_ranges;
        std::vector<std::vector<uint8_t>> rebased_defaults;
        for (uint64_t candidate : address_candidates) {
          if (!cursor.seek(list_start))
            return false;
          std::vector<LocationRange> candidate_ranges;
          std::vector<std::vector<uint8_t>> candidate_defaults;
          if (!decode_loclist(cursor, unit_end, address_size, candidate, inferred_addr_base, tables,
                              candidate_ranges, candidate_defaults))
            return false;
          if (!touches_text(candidate_ranges))
            continue;
          if (rebased_ranges)
            return false;
          rebased_ranges = std::move(candidate_ranges);
          rebased_defaults = std::move(candidate_defaults);
        }
        if (rebased_ranges) {
          source_ranges = std::move(*rebased_ranges);
          defaults = std::move(rebased_defaults);
        }
      }
      std::vector<uint8_t> encoded;
      for (const LocationRange &source : source_ranges) {
        std::vector<std::pair<uint64_t, uint64_t>> target_ranges;
        if (!addresses.inside(source.begin) && !addresses.inside(source.end))
          target_ranges.emplace_back(source.begin, source.end);
        else
          target_ranges = addresses.translate_range(source.begin, source.end);
        for (const auto &[begin, end] : target_ranges) {
          if (begin >= end)
            continue;
          encoded.push_back(DW_LLE_start_end);
          const size_t old_size = encoded.size();
          encoded.resize(old_size + 2 * address_size);
          std::memcpy(encoded.data() + old_size, &begin, address_size);
          std::memcpy(encoded.data() + old_size + address_size, &end, address_size);
          append_uleb(encoded, source.expression.size());
          encoded.insert(encoded.end(), source.expression.begin(), source.expression.end());
        }
      }
      for (const auto &expression : defaults) {
        encoded.push_back(DW_LLE_default_location);
        append_uleb(encoded, expression.size());
        encoded.insert(encoded.end(), expression.begin(), expression.end());
      }
      encoded.push_back(DW_LLE_end_of_list);
      lists.push_back(std::move(encoded));
    }

    const size_t new_unit_start = output.size();
    append_value<uint32_t>(output, 0);
    append_value<uint16_t>(output, version);
    output.push_back(address_size);
    output.push_back(segment_size);
    append_value<uint32_t>(output, offset_count);
    const size_t new_base = output.size();
    output.resize(output.size() + static_cast<size_t>(offset_count) * sizeof(uint32_t));
    for (size_t i = 0; i < lists.size(); ++i) {
      const uint64_t relative = output.size() - new_base;
      if (relative > std::numeric_limits<uint32_t>::max())
        return false;
      const uint32_t relative32 = relative;
      std::memcpy(output.data() + new_base + i * sizeof(uint32_t), &relative32, sizeof(relative32));
      output.insert(output.end(), lists[i].begin(), lists[i].end());
    }
    const uint64_t new_length = output.size() - new_unit_start - sizeof(uint32_t);
    if (new_length > std::numeric_limits<uint32_t>::max())
      return false;
    const uint32_t new_length32 = new_length;
    std::memcpy(output.data() + new_unit_start, &new_length32, sizeof(new_length32));
    base_offsets.emplace(old_base, new_base);
    if (!cursor.seek(unit_end))
      return false;
  }
  return replace_section(image, sections, index, output);
}

struct FrameCie {
  uint64_t code_alignment = 0;
  uint8_t fde_encoding = 0;
  bool has_fde_augmentation = false;
};

bool skip_cfi_block(Cursor &cursor) {
  uint64_t size = 0;
  return cursor.read_uleb(size) && size <= cursor.remaining() && cursor.skip(size);
}

bool relocate_cfi_locations(std::vector<uint8_t> &image, uint64_t section_offset, Cursor &cursor,
                            size_t end, uint64_t source_start, uint64_t target_start,
                            uint64_t target_end, uint64_t code_alignment,
                            const AddressMap &addresses) {
  uint64_t source_location = source_start;
  uint64_t target_location = target_start;
  auto translated_advance = [&](uint64_t source_amount) -> std::optional<uint64_t> {
    if (source_amount > (std::numeric_limits<uint64_t>::max() - source_location) / code_alignment)
      return std::nullopt;
    source_location += source_amount * code_alignment;
    const auto target = addresses.translate_boundary(source_location);
    if (!target || *target < target_location || *target > target_end ||
        (*target - target_location) % code_alignment != 0)
      return std::nullopt;
    const uint64_t amount = (*target - target_location) / code_alignment;
    target_location = *target;
    return amount;
  };
  auto uleb = [&] {
    uint64_t ignored = 0;
    return cursor.read_uleb(ignored);
  };
  auto sleb = [&] {
    int64_t ignored = 0;
    return cursor.read_sleb(ignored);
  };
  while (cursor.offset() < end) {
    const size_t opcode_offset = cursor.offset();
    uint8_t opcode = 0;
    if (!cursor.read(opcode))
      return false;
    if ((opcode & 0xc0) == 0x40) {
      const auto amount = translated_advance(opcode & 0x3f);
      if (!amount || *amount > 0x3f)
        return false;
      image[section_offset + opcode_offset] = static_cast<uint8_t>(0x40 | *amount);
      continue;
    }
    if ((opcode & 0xc0) == 0x80) {
      if (!uleb())
        return false;
      continue;
    }
    if ((opcode & 0xc0) == 0xc0)
      continue;
    switch (opcode) {
    case DW_CFA_nop:
    case DW_CFA_remember_state:
    case DW_CFA_restore_state:
      break;
    case DW_CFA_set_loc:
      // A set-location operand uses the CIE's pointer encoding. It is uncommon
      // in AMDGPU FDEs; rejecting it avoids silently retaining an address whose
      // encoded width or application is not described by this scanner.
      return false;
    case DW_CFA_advance_loc1: {
      const size_t operand_offset = cursor.offset();
      uint8_t value = 0;
      if (!cursor.read(value))
        return false;
      const auto amount = translated_advance(value);
      if (!amount || *amount > std::numeric_limits<uint8_t>::max())
        return false;
      image[section_offset + operand_offset] = static_cast<uint8_t>(*amount);
      break;
    }
    case DW_CFA_advance_loc2: {
      const size_t operand_offset = cursor.offset();
      uint16_t value = 0;
      if (!cursor.read(value))
        return false;
      const auto amount = translated_advance(value);
      if (!amount || *amount > std::numeric_limits<uint16_t>::max())
        return false;
      const uint16_t encoded = static_cast<uint16_t>(*amount);
      std::memcpy(image.data() + section_offset + operand_offset, &encoded, sizeof(encoded));
      break;
    }
    case DW_CFA_advance_loc4: {
      const size_t operand_offset = cursor.offset();
      uint32_t value = 0;
      if (!cursor.read(value))
        return false;
      const auto amount = translated_advance(value);
      if (!amount || *amount > std::numeric_limits<uint32_t>::max())
        return false;
      const uint32_t encoded = static_cast<uint32_t>(*amount);
      std::memcpy(image.data() + section_offset + operand_offset, &encoded, sizeof(encoded));
      break;
    }
    case DW_CFA_offset_extended:
    case DW_CFA_register:
    case DW_CFA_def_cfa:
    case DW_CFA_val_offset:
      if (!uleb() || !uleb())
        return false;
      break;
    case DW_CFA_offset_extended_sf:
    case DW_CFA_def_cfa_sf:
    case DW_CFA_val_offset_sf:
      if (!uleb() || !sleb())
        return false;
      break;
    case DW_CFA_restore_extended:
    case DW_CFA_undefined:
    case DW_CFA_same_value:
    case DW_CFA_def_cfa_register:
    case DW_CFA_def_cfa_offset:
      if (!uleb())
        return false;
      break;
    case DW_CFA_def_cfa_offset_sf:
      if (!sleb())
        return false;
      break;
    case DW_CFA_def_cfa_expression:
      if (!skip_cfi_block(cursor))
        return false;
      break;
    case DW_CFA_expression:
    case DW_CFA_val_expression:
      if (!uleb() || !skip_cfi_block(cursor))
        return false;
      break;
    default:
      return false;
    }
  }
  return cursor.offset() == end;
}

bool relocate_eh_frame(std::vector<uint8_t> &image, const Elf64_Shdr &section,
                       const AddressMap &addresses) {
  if (section.sh_offset > image.size() || section.sh_size > image.size() - section.sh_offset)
    return false;
  const auto bytes = std::span<const uint8_t>(image).subspan(section.sh_offset, section.sh_size);
  Cursor cursor(bytes);
  std::unordered_map<size_t, FrameCie> cies;
  while (cursor.remaining() != 0) {
    const size_t entry_start = cursor.offset();
    uint32_t length = 0;
    if (!cursor.read(length) || length == 0xffffffffu || length > cursor.remaining())
      return false;
    if (length == 0)
      return cursor.remaining() == 0;
    const size_t entry_end = cursor.offset() + length;
    const size_t id_offset = cursor.offset();
    uint32_t cie_pointer = 0;
    if (!cursor.read(cie_pointer))
      return false;
    if (cie_pointer == 0) {
      uint8_t version = 0;
      if (!cursor.read(version) || version != 1)
        return false;
      std::string augmentation;
      while (true) {
        uint8_t byte = 0;
        if (!cursor.read(byte))
          return false;
        if (byte == 0)
          break;
        augmentation.push_back(static_cast<char>(byte));
      }
      FrameCie cie;
      int64_t data_alignment = 0;
      uint8_t return_register = 0;
      if (!cursor.read_uleb(cie.code_alignment) || cie.code_alignment == 0 ||
          !cursor.read_sleb(data_alignment) || !cursor.read(return_register))
        return false;
      (void)data_alignment;
      (void)return_register;
      if (augmentation == "zR") {
        uint64_t augmentation_size = 0;
        if (!cursor.read_uleb(augmentation_size) || augmentation_size != 1 ||
            !cursor.read(cie.fde_encoding))
          return false;
        cie.has_fde_augmentation = true;
      } else if (!augmentation.empty()) {
        return false;
      }
      if (cie.fde_encoding != 0x1b)
        return false;
      cies.emplace(entry_start, cie);
      if (!cursor.seek(entry_end))
        return false;
      continue;
    }
    if (cie_pointer > id_offset)
      return false;
    const auto cie = cies.find(id_offset - cie_pointer);
    if (cie == cies.end())
      return false;
    const size_t location_offset = cursor.offset();
    int32_t relative_location = 0;
    uint32_t source_range = 0;
    if (!cursor.read(relative_location) || !cursor.read(source_range))
      return false;
    const uint64_t location_address = section.sh_addr + location_offset;
    const int64_t signed_location = static_cast<int64_t>(location_address) + relative_location;
    if (signed_location < 0)
      return false;
    const uint64_t source_start = static_cast<uint64_t>(signed_location);
    if (cie->second.has_fde_augmentation) {
      uint64_t augmentation_size = 0;
      if (!cursor.read_uleb(augmentation_size) || augmentation_size > cursor.remaining() ||
          !cursor.skip(augmentation_size))
        return false;
    }
    if (addresses.inside(source_start)) {
      const auto target_start = addresses.translate(source_start);
      if (!target_start) {
        if (!write_uint(image, section.sh_offset + location_offset + sizeof(relative_location),
                        sizeof(source_range), 0))
          return false;
      } else {
        if (source_range > std::numeric_limits<uint64_t>::max() - source_start)
          return false;
        auto target_end = addresses.translate_boundary(source_start + source_range);
        if (!target_end) {
          const auto ranges = addresses.translate_range(source_start, source_start + source_range);
          if (ranges.size() == 1 && ranges.front().first == *target_start)
            target_end = ranges.front().second;
        }
        if (!target_end || *target_end < *target_start ||
            *target_end - *target_start > std::numeric_limits<uint32_t>::max())
          return false;
        if (!relocate_cfi_locations(image, section.sh_offset, cursor, entry_end, source_start,
                                    *target_start, *target_end, cie->second.code_alignment,
                                    addresses))
          return false;
        const int64_t target_relative =
            static_cast<int64_t>(*target_start) - static_cast<int64_t>(location_address);
        if (target_relative < std::numeric_limits<int32_t>::min() ||
            target_relative > std::numeric_limits<int32_t>::max())
          return false;
        const int32_t encoded_location = static_cast<int32_t>(target_relative);
        const uint32_t encoded_range = static_cast<uint32_t>(*target_end - *target_start);
        std::memcpy(image.data() + section.sh_offset + location_offset, &encoded_location,
                    sizeof(encoded_location));
        std::memcpy(image.data() + section.sh_offset + location_offset + sizeof(encoded_location),
                    &encoded_range, sizeof(encoded_range));
      }
    }
    if (!cursor.seek(entry_end))
      return false;
  }
  return true;
}

bool relocate_debug_frame(std::vector<uint8_t> &image, const Elf64_Shdr &section,
                          const AddressMap &addresses) {
  if (section.sh_offset > image.size() || section.sh_size > image.size() - section.sh_offset)
    return false;
  const auto bytes = std::span<const uint8_t>(image).subspan(section.sh_offset, section.sh_size);
  Cursor cursor(bytes);
  struct DebugCie {
    uint64_t code_alignment = 0;
    uint8_t address_size = 0;
  };
  std::unordered_map<size_t, DebugCie> cies;
  while (cursor.remaining() != 0) {
    const size_t entry_start = cursor.offset();
    uint32_t length = 0, cie_pointer = 0;
    if (!cursor.read(length) || length == 0xffffffffu || length > cursor.remaining())
      return false;
    if (length == 0)
      continue;
    const size_t entry_end = cursor.offset() + length;
    if (!cursor.read(cie_pointer))
      return false;
    if (cie_pointer == 0xffffffffu) {
      uint8_t version = 0;
      if (!cursor.read(version) || version != 4)
        return false;
      uint8_t augmentation = 0;
      if (!cursor.read(augmentation) || augmentation != 0)
        return false;
      DebugCie cie;
      uint8_t segment_size = 0;
      int64_t data_alignment = 0;
      uint64_t return_register = 0;
      if (!cursor.read(cie.address_size) || !cursor.read(segment_size) || segment_size != 0 ||
          cie.address_size == 0 || cie.address_size > 8 || !cursor.read_uleb(cie.code_alignment) ||
          cie.code_alignment == 0 || !cursor.read_sleb(data_alignment) ||
          !cursor.read_uleb(return_register))
        return false;
      (void)data_alignment;
      (void)return_register;
      cies.emplace(entry_start, cie);
      if (!cursor.seek(entry_end))
        return false;
      continue;
    }
    const auto cie = cies.find(cie_pointer);
    if (cie == cies.end())
      return false;
    const size_t location_offset = cursor.offset();
    uint64_t source_start = 0, source_range = 0;
    if (!cursor.read_uint(cie->second.address_size, source_start) ||
        !cursor.read_uint(cie->second.address_size, source_range))
      return false;
    if (addresses.inside(source_start)) {
      const auto target_start = addresses.translate(source_start);
      if (!target_start) {
        if (!write_uint(image, section.sh_offset + location_offset + cie->second.address_size,
                        cie->second.address_size, 0))
          return false;
      } else {
        if (source_range > std::numeric_limits<uint64_t>::max() - source_start)
          return false;
        auto target_end = addresses.translate_boundary(source_start + source_range);
        if (!target_end) {
          const auto ranges = addresses.translate_range(source_start, source_start + source_range);
          if (ranges.size() == 1 && ranges.front().first == *target_start)
            target_end = ranges.front().second;
        }
        if (!target_end || *target_end < *target_start ||
            !relocate_cfi_locations(image, section.sh_offset, cursor, entry_end, source_start,
                                    *target_start, *target_end, cie->second.code_alignment,
                                    addresses) ||
            !write_uint(image, section.sh_offset + location_offset, cie->second.address_size,
                        *target_start) ||
            !write_uint(image, section.sh_offset + location_offset + cie->second.address_size,
                        cie->second.address_size, *target_end - *target_start))
          return false;
      }
    }
    if (!cursor.seek(entry_end))
      return false;
  }
  return true;
}

} // namespace

bool relocate_dwarf(std::vector<uint8_t> &image, const Elf64_Ehdr &header,
                    std::vector<Elf64_Shdr> &sections, size_t text_index, uint64_t old_text_size,
                    uint64_t new_text_size, std::span<const TextOffsetRelocation> relocations) {
  if (relocations.empty())
    return true;
  if (text_index >= sections.size())
    return false;

  std::vector<uint8_t> updated_image = image;
  std::vector<Elf64_Shdr> updated_sections = sections;
  const AddressMap addresses(sections[text_index].sh_addr, old_text_size, new_text_size,
                             relocations);
  AddressTables address_tables;

  // These sections also carry code ranges or addresses. Support them before
  // accepting an object that contains them; retaining their source values
  // would make the otherwise relocated debug view internally inconsistent.
  constexpr std::array<std::string_view, 3> unsupported_address_sections = {
      ".debug_aranges", ".debug_loc", ".debug_ranges"};
  if (std::ranges::any_of(unsupported_address_sections, [&](std::string_view name) {
        return find_section(updated_image, header, updated_sections, name).has_value();
      }))
    return false;

  const auto eh_frame = find_section(updated_image, header, updated_sections, ".eh_frame");
  if (eh_frame && !relocate_eh_frame(updated_image, updated_sections[*eh_frame], addresses))
    return false;
  const auto debug_frame = find_section(updated_image, header, updated_sections, ".debug_frame");
  if (debug_frame &&
      !relocate_debug_frame(updated_image, updated_sections[*debug_frame], addresses))
    return false;

  const auto debug_addr = find_section(updated_image, header, updated_sections, ".debug_addr");
  if (debug_addr &&
      !relocate_debug_addr(updated_image, updated_sections[*debug_addr], addresses, address_tables))
    return false;

  std::unordered_map<uint64_t, uint64_t> line_offsets;
  const auto debug_line = find_section(updated_image, header, updated_sections, ".debug_line");
  if (debug_line &&
      !relocate_debug_line(updated_image, updated_sections, *debug_line, addresses, line_offsets))
    return false;

  std::unordered_map<uint64_t, uint64_t> rnglist_bases;
  const auto debug_rnglists =
      find_section(updated_image, header, updated_sections, ".debug_rnglists");
  if (debug_rnglists && !relocate_debug_rnglists(updated_image, updated_sections, *debug_rnglists,
                                                 addresses, address_tables, rnglist_bases))
    return false;

  std::unordered_map<uint64_t, uint64_t> loclist_bases;
  const auto debug_loclists =
      find_section(updated_image, header, updated_sections, ".debug_loclists");
  if (debug_loclists && !relocate_debug_loclists(updated_image, updated_sections, *debug_loclists,
                                                 addresses, address_tables, loclist_bases))
    return false;

  const auto debug_info = find_section(updated_image, header, updated_sections, ".debug_info");
  const auto debug_abbrev = find_section(updated_image, header, updated_sections, ".debug_abbrev");
  if (debug_info != debug_abbrev) {
    if (!debug_info || !debug_abbrev ||
        !relocate_debug_info(updated_image, updated_sections[*debug_info],
                             updated_sections[*debug_abbrev], addresses, address_tables,
                             line_offsets, rnglist_bases, loclist_bases))
      return false;
  }

  image = std::move(updated_image);
  sections = std::move(updated_sections);
  return true;
}

} // namespace rocjitsu
