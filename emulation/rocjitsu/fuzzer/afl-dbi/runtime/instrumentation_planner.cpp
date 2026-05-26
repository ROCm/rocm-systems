#include "instrumentation_planner.h"

#include "hsa/AMDHSAKernelDescriptor.h"
#include "rocjitsu/analysis/def_use_chain.h"
#include "instruction_relocator.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu::fuzzer::afl_dbi {
namespace {

using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
namespace kd = rocr::llvm::amdhsa;

template <typename T>
std::optional<T> read_struct(std::span<const uint8_t> image, uint64_t offset) {
  if (offset > image.size() || sizeof(T) > image.size() - offset)
    return std::nullopt;
  T out{};
  memcpy(&out, image.data() + offset, sizeof(T));
  return out;
}

struct ElfNoteHeader {
  uint32_t namesz = 0;
  uint32_t descsz = 0;
  uint32_t type = 0;
};

struct MsgpackValue {
  enum class Kind {
    Scalar,
    String,
    Array,
    Map,
  };

  Kind kind = Kind::Scalar;
  size_t offset = 0;
  size_t payload_offset = 0;
  size_t payload_size = 0;
  size_t next_offset = 0;
  uint32_t count = 0;
};

uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

std::optional<uint32_t> read_be_uint(std::span<const uint8_t> data, size_t offset,
                                     size_t width) {
  if (offset > data.size() || width > data.size() - offset)
    return std::nullopt;
  uint32_t value = 0;
  for (size_t i = 0; i < width; ++i)
    value = (value << 8) | data[offset + i];
  return value;
}

std::optional<MsgpackValue> read_msgpack_header(std::span<const uint8_t> data,
                                                size_t offset) {
  if (offset >= data.size())
    return std::nullopt;

  MsgpackValue value;
  value.offset = offset;
  const uint8_t tag = data[offset];
  if (tag <= 0x7f) {
    value.kind = MsgpackValue::Kind::Scalar;
    value.payload_offset = offset;
    value.payload_size = 1;
    value.next_offset = offset + 1;
    return value;
  }
  if ((tag & 0xf0u) == 0x80u) {
    value.kind = MsgpackValue::Kind::Map;
    value.count = tag & 0x0fu;
    value.next_offset = offset + 1;
    return value;
  }
  if ((tag & 0xf0u) == 0x90u) {
    value.kind = MsgpackValue::Kind::Array;
    value.count = tag & 0x0fu;
    value.next_offset = offset + 1;
    return value;
  }
  if ((tag & 0xe0u) == 0xa0u) {
    value.kind = MsgpackValue::Kind::String;
    value.payload_offset = offset + 1;
    value.payload_size = tag & 0x1fu;
    value.next_offset = value.payload_offset + value.payload_size;
    if (value.next_offset > data.size())
      return std::nullopt;
    return value;
  }

  switch (tag) {
  case 0xc0:
  case 0xc2:
  case 0xc3:
    value.kind = MsgpackValue::Kind::Scalar;
    value.payload_offset = offset;
    value.payload_size = 1;
    value.next_offset = offset + 1;
    return value;
  case 0xcc:
  case 0xd0:
    value.kind = MsgpackValue::Kind::Scalar;
    value.payload_offset = offset + 1;
    value.payload_size = 1;
    value.next_offset = offset + 2;
    break;
  case 0xcd:
  case 0xd1:
    value.kind = MsgpackValue::Kind::Scalar;
    value.payload_offset = offset + 1;
    value.payload_size = 2;
    value.next_offset = offset + 3;
    break;
  case 0xce:
  case 0xd2:
    value.kind = MsgpackValue::Kind::Scalar;
    value.payload_offset = offset + 1;
    value.payload_size = 4;
    value.next_offset = offset + 5;
    break;
  case 0xcf:
  case 0xd3:
    value.kind = MsgpackValue::Kind::Scalar;
    value.payload_offset = offset + 1;
    value.payload_size = 8;
    value.next_offset = offset + 9;
    break;
  case 0xd9:
    if (offset + 2 > data.size())
      return std::nullopt;
    value.kind = MsgpackValue::Kind::String;
    value.payload_offset = offset + 2;
    value.payload_size = data[offset + 1];
    value.next_offset = value.payload_offset + value.payload_size;
    break;
  case 0xda: {
    std::optional<uint32_t> size = read_be_uint(data, offset + 1, 2);
    if (!size)
      return std::nullopt;
    value.kind = MsgpackValue::Kind::String;
    value.payload_offset = offset + 3;
    value.payload_size = *size;
    value.next_offset = value.payload_offset + value.payload_size;
    break;
  }
  case 0xdb: {
    std::optional<uint32_t> size = read_be_uint(data, offset + 1, 4);
    if (!size)
      return std::nullopt;
    value.kind = MsgpackValue::Kind::String;
    value.payload_offset = offset + 5;
    value.payload_size = *size;
    value.next_offset = value.payload_offset + value.payload_size;
    break;
  }
  case 0xdc: {
    std::optional<uint32_t> count = read_be_uint(data, offset + 1, 2);
    if (!count)
      return std::nullopt;
    value.kind = MsgpackValue::Kind::Array;
    value.count = *count;
    value.next_offset = offset + 3;
    break;
  }
  case 0xdd: {
    std::optional<uint32_t> count = read_be_uint(data, offset + 1, 4);
    if (!count)
      return std::nullopt;
    value.kind = MsgpackValue::Kind::Array;
    value.count = *count;
    value.next_offset = offset + 5;
    break;
  }
  case 0xde: {
    std::optional<uint32_t> count = read_be_uint(data, offset + 1, 2);
    if (!count)
      return std::nullopt;
    value.kind = MsgpackValue::Kind::Map;
    value.count = *count;
    value.next_offset = offset + 3;
    break;
  }
  case 0xdf: {
    std::optional<uint32_t> count = read_be_uint(data, offset + 1, 4);
    if (!count)
      return std::nullopt;
    value.kind = MsgpackValue::Kind::Map;
    value.count = *count;
    value.next_offset = offset + 5;
    break;
  }
  default:
    return std::nullopt;
  }

  if (value.next_offset > data.size())
    return std::nullopt;
  return value;
}

bool skip_msgpack_value(std::span<const uint8_t> data, size_t offset, size_t *next_offset) {
  std::optional<MsgpackValue> value = read_msgpack_header(data, offset);
  if (!value)
    return false;

  size_t cursor = value->next_offset;
  if (value->kind == MsgpackValue::Kind::Array) {
    for (uint32_t i = 0; i < value->count; ++i) {
      if (!skip_msgpack_value(data, cursor, &cursor))
        return false;
    }
  } else if (value->kind == MsgpackValue::Kind::Map) {
    for (uint32_t i = 0; i < value->count; ++i) {
      if (!skip_msgpack_value(data, cursor, &cursor) ||
          !skip_msgpack_value(data, cursor, &cursor))
        return false;
    }
  }
  *next_offset = cursor;
  return true;
}

std::optional<std::string_view> msgpack_string_view(std::span<const uint8_t> data,
                                                    const MsgpackValue &value) {
  if (value.kind != MsgpackValue::Kind::String ||
      value.payload_offset > data.size() ||
      value.payload_size > data.size() - value.payload_offset)
    return std::nullopt;
  return std::string_view(reinterpret_cast<const char *>(data.data() + value.payload_offset),
                          value.payload_size);
}

std::optional<uint32_t> msgpack_uint32(std::span<const uint8_t> data,
                                       const MsgpackValue &value) {
  if (value.kind != MsgpackValue::Kind::Scalar || value.offset >= data.size())
    return std::nullopt;

  const uint8_t tag = data[value.offset];
  if (value.payload_offset == value.offset) {
    if (tag <= 0x7f)
      return tag;
    return std::nullopt;
  }

  size_t width = 0;
  switch (tag) {
  case 0xcc:
    width = 1;
    break;
  case 0xcd:
    width = 2;
    break;
  case 0xce:
    width = 4;
    break;
  default:
    return std::nullopt;
  }
  if (value.payload_offset > data.size() ||
      width > data.size() - value.payload_offset)
    return std::nullopt;
  std::optional<uint32_t> out = read_be_uint(data, value.payload_offset, width);
  if (!out)
    return std::nullopt;
  return *out;
}

struct MsgpackIntegerPatch {
  enum class Kind {
    InPlaceBytes,
    ReplaceEncodedValue,
  };

  Kind kind = Kind::InPlaceBytes;
  size_t offset = 0;
  size_t size = 0;
  std::array<uint8_t, 8> bytes{};
  std::vector<uint8_t> replacement;
};

std::vector<uint8_t> encode_msgpack_uint(uint32_t value) {
  if (value <= 0x7fu)
    return {static_cast<uint8_t>(value)};
  if (value <= 0xffu)
    return {0xcc, static_cast<uint8_t>(value)};
  if (value <= 0xffffu) {
    return {0xcd, static_cast<uint8_t>((value >> 8) & 0xffu),
            static_cast<uint8_t>(value & 0xffu)};
  }
  return {0xce, static_cast<uint8_t>((value >> 24) & 0xffu),
          static_cast<uint8_t>((value >> 16) & 0xffu),
          static_cast<uint8_t>((value >> 8) & 0xffu),
          static_cast<uint8_t>(value & 0xffu)};
}

std::optional<MsgpackIntegerPatch>
plan_msgpack_integer_patch(std::span<const uint8_t> metadata, const MsgpackValue &value,
                           uint32_t new_value, const char **failure_reason) {
  auto fail_optional = [&](const char *reason) -> std::optional<MsgpackIntegerPatch> {
    if (failure_reason != nullptr)
      *failure_reason = reason;
    return std::nullopt;
  };

  if (value.payload_size == 0 || value.payload_size > MsgpackIntegerPatch{}.bytes.size())
    return fail_optional("metadata resource integer encoding is unsupported");

  if (value.payload_offset == value.offset) {
    if (value.offset >= metadata.size() || metadata[value.offset] > 0x7f)
      return fail_optional("metadata resource value is not an unsigned integer");
    if (new_value <= 0x7fu) {
      MsgpackIntegerPatch patch;
      patch.kind = MsgpackIntegerPatch::Kind::InPlaceBytes;
      patch.offset = value.offset;
      patch.bytes[0] = static_cast<uint8_t>(new_value);
      patch.size = 1;
      return patch;
    }
    MsgpackIntegerPatch patch;
    patch.kind = MsgpackIntegerPatch::Kind::ReplaceEncodedValue;
    patch.offset = value.offset;
    patch.size = value.next_offset - value.offset;
    patch.replacement = encode_msgpack_uint(new_value);
    return patch;
  }

  if (value.offset >= metadata.size() || metadata[value.offset] < 0xcc ||
      metadata[value.offset] > 0xcf)
    return fail_optional("metadata resource value is not an unsigned integer");
  uint64_t max_value = 0;
  for (size_t i = 0; i < value.payload_size; ++i)
    max_value = (max_value << 8) | 0xffu;
  if (new_value > max_value) {
    MsgpackIntegerPatch patch;
    patch.kind = MsgpackIntegerPatch::Kind::ReplaceEncodedValue;
    patch.offset = value.offset;
    patch.size = value.next_offset - value.offset;
    patch.replacement = encode_msgpack_uint(new_value);
    return patch;
  }

  MsgpackIntegerPatch patch;
  patch.kind = MsgpackIntegerPatch::Kind::InPlaceBytes;
  patch.offset = value.payload_offset;
  patch.size = value.payload_size;
  const uint64_t wide_new_value = new_value;
  for (size_t i = 0; i < value.payload_size; ++i) {
    const size_t shift = 8 * (value.payload_size - i - 1);
    patch.bytes[i] = static_cast<uint8_t>((wide_new_value >> shift) & 0xffu);
  }
  return patch;
}

bool kernel_metadata_matches(std::string_view metadata_name, std::string_view metadata_symbol,
                             std::string_view kernel_name) {
  if (metadata_name == kernel_name || metadata_symbol == kernel_name)
    return true;
  std::string kd_symbol(kernel_name);
  kd_symbol += ".kd";
  return metadata_symbol == kd_symbol;
}

bool plan_metadata_integer_patch_from_kernel_map(
    std::span<const uint8_t> metadata, const MsgpackValue &kernel_map,
    std::string_view kernel_name, std::string_view field_name, uint32_t new_value,
    MsgpackIntegerPatch *patch,
    const char **failure_reason) {
  auto fail = [&](const char *reason) -> bool {
    if (failure_reason != nullptr)
      *failure_reason = reason;
    return false;
  };
  if (kernel_map.kind != MsgpackValue::Kind::Map)
    return fail("metadata kernel entry is not a map");

  size_t cursor = kernel_map.next_offset;
  std::string_view metadata_name;
  std::string_view metadata_symbol;
  std::optional<MsgpackValue> field_value;
  for (uint32_t i = 0; i < kernel_map.count; ++i) {
    std::optional<MsgpackValue> key = read_msgpack_header(metadata, cursor);
    if (!key)
      return fail("metadata kernel map is malformed");
    std::optional<std::string_view> key_name = msgpack_string_view(metadata, *key);
    if (!key_name)
      return fail("metadata kernel map key is not a string");
    cursor = key->next_offset;

    std::optional<MsgpackValue> value = read_msgpack_header(metadata, cursor);
    if (!value)
      return fail("metadata kernel map value is malformed");
    if (*key_name == ".name") {
      if (std::optional<std::string_view> name = msgpack_string_view(metadata, *value))
        metadata_name = *name;
    } else if (*key_name == ".symbol") {
      if (std::optional<std::string_view> symbol = msgpack_string_view(metadata, *value))
        metadata_symbol = *symbol;
    } else if (*key_name == field_name) {
      field_value = *value;
    }
    if (!skip_msgpack_value(metadata, cursor, &cursor))
      return fail("metadata kernel map value cannot be skipped");
  }

  if (!kernel_metadata_matches(metadata_name, metadata_symbol, kernel_name))
    return false;
  if (!field_value)
    return fail("metadata kernel resource field is missing");
  std::optional<MsgpackIntegerPatch> integer_patch =
      plan_msgpack_integer_patch(metadata, *field_value, new_value, failure_reason);
  if (!integer_patch)
    return false;
  *patch = std::move(*integer_patch);
  return true;
}

bool plan_metadata_integer_patch_from_metadata(std::span<const uint8_t> metadata,
                                               std::string_view kernel_name,
                                               std::string_view field_name,
                                               uint32_t new_value,
                                               MsgpackIntegerPatch *patch,
                                               const char **failure_reason) {
  auto fail = [&](const char *reason) -> bool {
    if (failure_reason != nullptr)
      *failure_reason = reason;
    return false;
  };

  std::optional<MsgpackValue> root = read_msgpack_header(metadata, 0);
  if (!root || root->kind != MsgpackValue::Kind::Map)
    return fail("AMDGPU metadata root is not a msgpack map");

  size_t cursor = root->next_offset;
  for (uint32_t i = 0; i < root->count; ++i) {
    std::optional<MsgpackValue> key = read_msgpack_header(metadata, cursor);
    if (!key)
      return fail("AMDGPU metadata root map is malformed");
    std::optional<std::string_view> key_name = msgpack_string_view(metadata, *key);
    if (!key_name)
      return fail("AMDGPU metadata root key is not a string");
    cursor = key->next_offset;

    std::optional<MsgpackValue> value = read_msgpack_header(metadata, cursor);
    if (!value)
      return fail("AMDGPU metadata root value is malformed");
    if (*key_name == "amdhsa.kernels") {
      if (value->kind != MsgpackValue::Kind::Array)
        return fail("AMDGPU metadata kernels entry is not an array");
      size_t kernel_cursor = value->next_offset;
      for (uint32_t kernel_index = 0; kernel_index < value->count; ++kernel_index) {
        std::optional<MsgpackValue> kernel = read_msgpack_header(metadata, kernel_cursor);
        if (!kernel)
          return fail("AMDGPU metadata kernel entry is malformed");
        if (plan_metadata_integer_patch_from_kernel_map(
                metadata, *kernel, kernel_name, field_name, new_value, patch,
                failure_reason)) {
          return true;
        }
        if (failure_reason != nullptr && *failure_reason != nullptr)
          return false;
        if (!skip_msgpack_value(metadata, kernel_cursor, &kernel_cursor))
          return fail("AMDGPU metadata kernel entry cannot be skipped");
      }
      return fail("AMDGPU metadata kernel entry was not found");
    }
    if (!skip_msgpack_value(metadata, cursor, &cursor))
      return fail("AMDGPU metadata root value cannot be skipped");
  }

  return fail("AMDGPU metadata kernels entry is missing");
}

std::optional<std::vector<uint8_t>> rebuild_note_section_with_metadata_patch(
    std::span<const uint8_t> image, const rocjitsu::Elf64_Shdr &section,
    uint64_t note_offset, const ElfNoteHeader &note, uint64_t name_offset,
    uint64_t desc_offset, uint64_t next_note, std::span<const uint8_t> metadata,
    const MsgpackIntegerPatch &integer_patch, const char **failure_reason) {
  auto fail = [&](const char *reason) -> std::optional<std::vector<uint8_t>> {
    if (failure_reason != nullptr)
      *failure_reason = reason;
    return std::nullopt;
  };

  if (integer_patch.kind != MsgpackIntegerPatch::Kind::ReplaceEncodedValue ||
      integer_patch.replacement.empty()) {
    return fail("metadata private segment replacement is malformed");
  }
  if (integer_patch.offset > metadata.size() ||
      integer_patch.size > metadata.size() - integer_patch.offset) {
    return fail("metadata private segment replacement is out of range");
  }

  std::vector<uint8_t> new_metadata;
  new_metadata.reserve(metadata.size() - integer_patch.size +
                       integer_patch.replacement.size());
  new_metadata.insert(new_metadata.end(), metadata.begin(),
                      metadata.begin() + integer_patch.offset);
  new_metadata.insert(new_metadata.end(), integer_patch.replacement.begin(),
                      integer_patch.replacement.end());
  new_metadata.insert(new_metadata.end(),
                      metadata.begin() + integer_patch.offset + integer_patch.size,
                      metadata.end());
  if (new_metadata.size() > std::numeric_limits<uint32_t>::max())
    return fail("rebuilt AMDGPU metadata note is too large");

  const uint64_t section_end = section.sh_offset + section.sh_size;
  if (section.sh_offset > image.size() || section.sh_size > image.size() - section.sh_offset ||
      note_offset < section.sh_offset || note_offset > section_end ||
      name_offset < note_offset || desc_offset < name_offset || next_note < desc_offset ||
      next_note > section_end) {
    return fail("AMDGPU metadata note rebuild range is out of bounds");
  }

  ElfNoteHeader rebuilt_header = note;
  rebuilt_header.descsz = static_cast<uint32_t>(new_metadata.size());

  std::vector<uint8_t> rebuilt_section;
  rebuilt_section.reserve(section.sh_size - note.descsz + new_metadata.size() + 4);
  rebuilt_section.insert(rebuilt_section.end(), image.begin() + section.sh_offset,
                         image.begin() + note_offset);
  const auto *header_bytes = reinterpret_cast<const uint8_t *>(&rebuilt_header);
  rebuilt_section.insert(rebuilt_section.end(), header_bytes,
                         header_bytes + sizeof(rebuilt_header));
  rebuilt_section.insert(rebuilt_section.end(), image.begin() + name_offset,
                         image.begin() + desc_offset);
  rebuilt_section.insert(rebuilt_section.end(), new_metadata.begin(), new_metadata.end());
  while (rebuilt_section.size() % 4 != 0)
    rebuilt_section.push_back(0);
  rebuilt_section.insert(rebuilt_section.end(), image.begin() + next_note,
                         image.begin() + section_end);
  return rebuilt_section;
}

struct ElfSections {
  std::vector<rocjitsu::Elf64_Shdr> sections;
  const rocjitsu::Elf64_Shdr *text = nullptr;
  std::span<const char> shstrtab;
};

std::optional<ElfSections> read_sections(std::span<const uint8_t> image) {
  auto ehdr = read_struct<rocjitsu::Elf64_Ehdr>(image, 0);
  if (!ehdr || ehdr->e_shentsize != sizeof(rocjitsu::Elf64_Shdr) || ehdr->e_shoff > image.size() ||
      static_cast<uint64_t>(ehdr->e_shnum) * sizeof(rocjitsu::Elf64_Shdr) >
          image.size() - ehdr->e_shoff) {
    return std::nullopt;
  }

  ElfSections out;
  out.sections.resize(ehdr->e_shnum);
  memcpy(out.sections.data(), image.data() + ehdr->e_shoff,
         out.sections.size() * sizeof(rocjitsu::Elf64_Shdr));
  if (ehdr->e_shstrndx >= out.sections.size())
    return std::nullopt;
  const auto &shstr = out.sections[ehdr->e_shstrndx];
  if (shstr.sh_offset > image.size() || shstr.sh_size > image.size() - shstr.sh_offset)
    return std::nullopt;
  out.shstrtab = {reinterpret_cast<const char *>(image.data() + shstr.sh_offset),
                  static_cast<size_t>(shstr.sh_size)};

  for (const auto &section : out.sections) {
    if (section.sh_name >= out.shstrtab.size())
      continue;
    const char *name = out.shstrtab.data() + section.sh_name;
    const size_t max_len = out.shstrtab.size() - section.sh_name;
    if (std::string_view(name, strnlen(name, max_len)) == ".text") {
      out.text = &section;
      break;
    }
  }
  if (out.text == nullptr)
    return std::nullopt;
  return out;
}

std::optional<uint64_t> va_to_file_offset(const ElfSections &sections, uint64_t va) {
  for (const auto &section : sections.sections) {
    if ((section.sh_flags & rocjitsu::SHF_ALLOC) == 0 || section.sh_size == 0)
      continue;
    if (va >= section.sh_addr && va < section.sh_addr + section.sh_size)
      return section.sh_offset + (va - section.sh_addr);
  }
  return std::nullopt;
}

std::optional<uint32_t> read_elf_mach(std::span<const uint8_t> image) {
  auto ehdr = read_struct<rocjitsu::Elf64_Ehdr>(image, 0);
  if (!ehdr)
    return std::nullopt;
  return ehdr->e_flags & rocjitsu::EF_AMDGPU_MACH;
}

bool descriptor_sgpr_count_is_effective(uint32_t elf_mach) {
  const ProbeTarget *target = probe_target_for_elf_mach(elf_mach);
  if (target == nullptr)
    return true;
  switch (target->arch) {
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return false;
  default:
    return true;
  }
}

struct KernelMetadataResourceInfo {
  std::optional<uint32_t> sgpr_count;
  std::optional<uint32_t> vgpr_count;
  std::optional<uint32_t> wavefront_size;
  std::optional<uint32_t> private_segment_fixed_size;
};

void add_kernel_metadata_alias(
    std::unordered_map<std::string, KernelMetadataResourceInfo> &resources,
    std::string_view alias, const KernelMetadataResourceInfo &info) {
  if (alias.empty())
    return;
  resources[std::string(alias)] = info;
  if (alias.size() > 3 && alias.substr(alias.size() - 3) == ".kd")
    resources[std::string(alias.substr(0, alias.size() - 3))] = info;
}

void parse_kernel_metadata_resource_map(
    std::span<const uint8_t> metadata, const MsgpackValue &kernel_map,
    std::unordered_map<std::string, KernelMetadataResourceInfo> &resources) {
  if (kernel_map.kind != MsgpackValue::Kind::Map)
    return;

  size_t cursor = kernel_map.next_offset;
  std::string_view metadata_name;
  std::string_view metadata_symbol;
  KernelMetadataResourceInfo info;
  for (uint32_t i = 0; i < kernel_map.count; ++i) {
    std::optional<MsgpackValue> key = read_msgpack_header(metadata, cursor);
    if (!key)
      return;
    std::optional<std::string_view> key_name = msgpack_string_view(metadata, *key);
    if (!key_name)
      return;
    cursor = key->next_offset;

    std::optional<MsgpackValue> value = read_msgpack_header(metadata, cursor);
    if (!value)
      return;
    if (*key_name == ".name") {
      if (std::optional<std::string_view> name = msgpack_string_view(metadata, *value))
        metadata_name = *name;
    } else if (*key_name == ".symbol") {
      if (std::optional<std::string_view> symbol = msgpack_string_view(metadata, *value))
        metadata_symbol = *symbol;
    } else if (*key_name == ".sgpr_count") {
      info.sgpr_count = msgpack_uint32(metadata, *value);
    } else if (*key_name == ".vgpr_count") {
      info.vgpr_count = msgpack_uint32(metadata, *value);
    } else if (*key_name == ".wavefront_size") {
      info.wavefront_size = msgpack_uint32(metadata, *value);
    } else if (*key_name == ".private_segment_fixed_size") {
      info.private_segment_fixed_size = msgpack_uint32(metadata, *value);
    }
    if (!skip_msgpack_value(metadata, cursor, &cursor))
      return;
  }

  add_kernel_metadata_alias(resources, metadata_name, info);
  add_kernel_metadata_alias(resources, metadata_symbol, info);
}

void parse_kernel_metadata_resources(
    std::span<const uint8_t> metadata,
    std::unordered_map<std::string, KernelMetadataResourceInfo> &resources) {
  std::optional<MsgpackValue> root = read_msgpack_header(metadata, 0);
  if (!root || root->kind != MsgpackValue::Kind::Map)
    return;

  size_t cursor = root->next_offset;
  for (uint32_t i = 0; i < root->count; ++i) {
    std::optional<MsgpackValue> key = read_msgpack_header(metadata, cursor);
    if (!key)
      return;
    std::optional<std::string_view> key_name = msgpack_string_view(metadata, *key);
    if (!key_name)
      return;
    cursor = key->next_offset;

    std::optional<MsgpackValue> value = read_msgpack_header(metadata, cursor);
    if (!value)
      return;
    if (*key_name == "amdhsa.kernels" && value->kind == MsgpackValue::Kind::Array) {
      size_t kernel_cursor = value->next_offset;
      for (uint32_t kernel_index = 0; kernel_index < value->count; ++kernel_index) {
        std::optional<MsgpackValue> kernel = read_msgpack_header(metadata, kernel_cursor);
        if (!kernel)
          return;
        parse_kernel_metadata_resource_map(metadata, *kernel, resources);
        if (!skip_msgpack_value(metadata, kernel_cursor, &kernel_cursor))
          return;
      }
    }
    if (!skip_msgpack_value(metadata, cursor, &cursor))
      return;
  }
}

std::unordered_map<std::string, KernelMetadataResourceInfo>
read_kernel_metadata_resources(std::span<const uint8_t> image,
                               const ElfSections &sections) {
  std::unordered_map<std::string, KernelMetadataResourceInfo> resources;
  for (const auto &shdr : sections.sections) {
    if (shdr.sh_type != rocjitsu::SHT_NOTE)
      continue;
    if (shdr.sh_offset > image.size() || shdr.sh_size > image.size() - shdr.sh_offset)
      continue;
    uint64_t note_offset = shdr.sh_offset;
    const uint64_t note_end = shdr.sh_offset + shdr.sh_size;
    while (note_offset + sizeof(ElfNoteHeader) <= note_end) {
      auto note = read_struct<ElfNoteHeader>(image, note_offset);
      if (!note)
        break;
      const uint64_t name_offset = note_offset + sizeof(ElfNoteHeader);
      const uint64_t desc_offset = align_up_u64(name_offset + note->namesz, 4);
      const uint64_t next_note = align_up_u64(desc_offset + note->descsz, 4);
      if (name_offset > note_end || note->namesz > note_end - name_offset ||
          desc_offset > note_end || note->descsz > note_end - desc_offset ||
          next_note > note_end) {
        break;
      }
      const bool amdgpu_note =
          note->type == rocjitsu::NT_AMDGPU_METADATA && note->namesz >= 6 &&
          memcmp(image.data() + name_offset, "AMDGPU", 6) == 0;
      if (amdgpu_note) {
        std::span<const uint8_t> metadata(image.data() + desc_offset, note->descsz);
        parse_kernel_metadata_resources(metadata, resources);
      }
      note_offset = next_note;
    }
  }
  return resources;
}

uint32_t planning_sgpr_count(uint32_t descriptor_count,
                             const KernelMetadataResourceInfo *metadata,
                             bool descriptor_count_effective) {
  if (!descriptor_count_effective)
    return metadata != nullptr && metadata->sgpr_count ? *metadata->sgpr_count : 0;
  if (metadata != nullptr && metadata->sgpr_count)
    return std::max(descriptor_count, *metadata->sgpr_count);
  return descriptor_count;
}

const rocjitsu::Instruction *first_instruction(const rocjitsu::BasicBlock &block) {
  auto &mutable_block = const_cast<rocjitsu::BasicBlock &>(block);
  auto it = mutable_block.instructions().begin();
  if (it == mutable_block.instructions().end())
    return nullptr;
  return &*it;
}

const rocjitsu::Instruction *second_instruction(const rocjitsu::BasicBlock &block) {
  auto &mutable_block = const_cast<rocjitsu::BasicBlock &>(block);
  auto it = mutable_block.instructions().begin();
  if (it == mutable_block.instructions().end())
    return nullptr;
  ++it;
  if (it == mutable_block.instructions().end())
    return nullptr;
  return &*it;
}

bool is_unconditional_direct_branch(const rocjitsu::Instruction &inst) {
  return (inst.flags() & rocjitsu::BRANCH) != 0 &&
         (inst.flags() & (rocjitsu::COND_BRANCH | rocjitsu::INDIRECT_BRANCH |
                          rocjitsu::INDIRECT_CALL | rocjitsu::PROGRAM_TERMINATOR)) == 0 &&
         inst.branch_offset_bytes().has_value();
}

bool is_conditional_direct_branch(const rocjitsu::Instruction &inst) {
  return (inst.flags() & rocjitsu::COND_BRANCH) != 0 &&
         (inst.flags() & (rocjitsu::INDIRECT_BRANCH | rocjitsu::INDIRECT_CALL |
                          rocjitsu::PROGRAM_TERMINATOR)) == 0 &&
         inst.branch_offset_bytes().has_value();
}

bool is_wait_counter_instruction(const rocjitsu::Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  return (inst.flags() & rocjitsu::WAITCNT) != 0 || mnemonic == "s_wait_alu" ||
         mnemonic == "s_wait_kmcnt" || mnemonic == "s_waitcnt";
}

bool is_delay_alu_instruction(const rocjitsu::Instruction &inst) {
  return inst.mnemonic() == "s_delay_alu";
}

bool is_exec_mask_transition_instruction(const rocjitsu::Instruction &inst) {
  return inst.mnemonic().find("saveexec") != std::string_view::npos;
}

std::string_view vopd_block_entry_skip_reason(const rocjitsu::Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw != nullptr && inst.size() >= 2 * static_cast<int>(sizeof(uint32_t)) &&
      decode_vopd64_liveness_model(raw[0], raw[1]).has_value()) {
    return {};
  }
  return "entry VOPD liveness is unmodeled and replay relocation is not enabled yet";
}

std::string_view branch_terminator_skip_reason(const rocjitsu::BasicBlock &block,
                                               const rocjitsu::Instruction *term) {
  if (term == nullptr)
    return "block has no instructions";
  if (!block.has_terminator())
    return "block falls through without a branch terminator";

  const uint64_t flags = term->flags();
  if ((flags & rocjitsu::PROGRAM_TERMINATOR) != 0)
    return "terminator exits kernel";
  if ((flags & rocjitsu::INDIRECT_CALL) != 0)
    return "terminator is indirect call and needs return-address-preserving coverage policy";
  if ((flags & rocjitsu::INDIRECT_BRANCH) != 0)
    return "terminator is indirect branch and needs dynamic-target coverage policy";
  if ((flags & (rocjitsu::BRANCH | rocjitsu::COND_BRANCH)) == 0)
    return "terminator is not branch control flow";
  if (!term->branch_offset_bytes().has_value())
    return "direct branch terminator has no PC-relative target";
  return "terminator is not a direct patchable branch";
}

uint64_t branch_terminator_skip_offset(const rocjitsu::BasicBlock &block,
                                       const rocjitsu::Instruction *term) {
  if (term == nullptr || term->size() <= 0)
    return block.end_offset();
  const uint64_t size = static_cast<uint64_t>(term->size());
  if (size > block.size())
    return block.end_offset();
  return block.end_offset() - size;
}

std::string_view block_entry_skip_reason(const rocjitsu::Instruction &inst) {
  if ((inst.flags() & (rocjitsu::BRANCH | rocjitsu::COND_BRANCH | rocjitsu::INDIRECT_BRANCH |
                       rocjitsu::INDIRECT_CALL | rocjitsu::PROGRAM_TERMINATOR)) != 0) {
    if (is_unconditional_direct_branch(inst) || is_conditional_direct_branch(inst))
      return {};
    return "entry instruction is control flow and is not PC-relocatable yet";
  }
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic == "s_mov_b32" && s_mov_b32_requires_state_preservation(inst)) {
    return "entry instruction may capture EXEC/SCC state before the probe mutates it";
  }
  if (is_exec_mask_transition_instruction(inst)) {
    return "entry instruction manipulates EXEC and needs mask-transition relocation support";
  }
  if (mnemonic == "s_or_b64" && s_or_b64_requires_state_preservation(inst)) {
    return "entry scalar OR may restore EXEC and needs operand-sensitive relocation support";
  }
  if (mnemonic == "vopd_opaque") {
    return vopd_block_entry_skip_reason(inst);
  }
  if (mnemonic == "unknown_opaque") {
    return "entry instruction is an unknown encoding and is not relocatable yet";
  }
  if (scc_compare_requires_state_preservation(inst)) {
    return "entry SCC compare reads special state and needs operand-sensitive relocation support";
  }
  return {};
}

std::string_view block_entry_skip_reason(const rocjitsu::BasicBlock &block,
                                         const rocjitsu::Instruction &first) {
  if (is_wait_counter_instruction(first)) {
    const rocjitsu::Instruction *second = second_instruction(block);
    if ((first.mnemonic() == "s_wait_kmcnt" || first.mnemonic() == "s_waitcnt") &&
        second != nullptr && is_exec_mask_transition_instruction(*second)) {
      return "entry memory wait guards an EXEC transition and must execute before injected probe";
    }
    return "entry wait-counter must execute before injected probe";
  }
  if (is_delay_alu_instruction(first))
    return "entry ALU-delay instruction must execute before injected probe";

  const std::string_view first_reason = block_entry_skip_reason(first);
  if (!first_reason.empty())
    return first_reason;

  return {};
}

struct BranchEdgeBudgetSelection {
  PreviousBbBranchAggregateBudget budget;
  uint32_t branch_candidate_edges = 0;
  std::string_view edge_reason;
  std::string_view site_reason;
  std::string_view fixed_counter_fallback_reason;
};

bool branch_opcode_depends_on_exec(uint16_t opcode);

BranchEdgeBudgetSelection select_branch_edge_budget(
    std::span<const rocjitsu::BasicBlock *const> blocks,
    const InstrumentationPlanOptions &options,
    bool previous_bb_branch_policy) {
  uint32_t branch_candidate_edges = 0;
  uint32_t branch_candidate_sites = 0;
  uint32_t previous_bb_candidate_edges = 0;
  uint32_t previous_bb_candidate_sites = 0;
  uint32_t edge_budget = options.branch_edge_site_limit;
  uint32_t site_budget = options.previous_bb_branch_site_limit;
  std::string_view edge_reason =
      options.branch_edge_site_limit_auto ? "auto-code-growth-cap" : "configured-limit";
  std::string_view site_reason = options.previous_bb_branch_site_limit_auto
                                     ? "auto-previous-bb-state-writer-cap"
                                     : "configured-limit";
  std::string_view fixed_counter_fallback_reason = "disabled";

  for (const rocjitsu::BasicBlock *block : blocks) {
    if (block == nullptr)
      continue;
    const rocjitsu::Instruction *term = block->terminator();
    const bool unconditional = term != nullptr && is_unconditional_direct_branch(*term);
    const bool conditional = term != nullptr && is_conditional_direct_branch(*term);
    if (!unconditional && !conditional)
      continue;

    const uint32_t edge_count = conditional ? 2u : 1u;
    if (branch_candidate_sites != std::numeric_limits<uint32_t>::max())
      ++branch_candidate_sites;
    branch_candidate_edges = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(branch_candidate_edges) + edge_count,
                           std::numeric_limits<uint32_t>::max()));

    const bool exec_conditioned_previous_bb_branch =
        previous_bb_branch_policy && conditional &&
        branch_opcode_depends_on_exec(term->opcode());
    if (exec_conditioned_previous_bb_branch)
      continue;

    if (previous_bb_candidate_sites != std::numeric_limits<uint32_t>::max())
      ++previous_bb_candidate_sites;
    previous_bb_candidate_edges = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(previous_bb_candidate_edges) +
                               edge_count,
                           std::numeric_limits<uint32_t>::max()));
  }

  const uint32_t aggregate_candidate_edges =
      previous_bb_branch_policy ? previous_bb_candidate_edges : branch_candidate_edges;
  const uint32_t aggregate_candidate_sites =
      previous_bb_branch_policy ? previous_bb_candidate_sites : branch_candidate_sites;

  if (options.previous_bb_branch_site_limit_auto &&
      aggregate_candidate_sites <= options.previous_bb_branch_site_limit) {
    site_budget = aggregate_candidate_sites;
    site_reason = "auto-candidate-count";
  }
  if (options.branch_edge_site_limit_auto) {
    if (aggregate_candidate_edges <= options.branch_edge_site_limit) {
      edge_budget = aggregate_candidate_edges;
      edge_reason = "auto-candidate-count";
    } else if (previous_bb_branch_policy &&
               options.previous_bb_branch_site_limit_auto) {
      const uint32_t site_derived_edge_budget =
          previous_bb_branch_site_derived_edge_budget(site_budget);
      if (aggregate_candidate_edges <= site_derived_edge_budget) {
        edge_budget = aggregate_candidate_edges;
        edge_reason = "auto-previous-bb-site-derived-candidate-count";
      } else if (site_derived_edge_budget > edge_budget) {
        edge_budget = site_derived_edge_budget;
        edge_reason = "auto-previous-bb-site-derived-cap";
      }
    }
  }
  uint32_t fixed_counter_fallback_budget = 0;
  if (!previous_bb_branch_policy) {
    fixed_counter_fallback_reason = "not-previous-bb-policy";
  } else if (options.fixed_counter_fallback_for_branch_budget) {
    fixed_counter_fallback_budget =
        options.fixed_counter_branch_edge_fallback_limit;
    fixed_counter_fallback_reason =
        options.branch_edge_site_limit_auto
            ? "auto-candidate-branch-edge-count"
            : "configured-limit";
    if (options.branch_edge_site_limit_auto) {
      const uint32_t candidate_budget =
          adaptive_fixed_counter_branch_edge_fallback_budget(branch_candidate_edges);
      fixed_counter_fallback_budget =
          std::min(candidate_budget,
                   options.fixed_counter_branch_edge_fallback_limit);
      if (branch_candidate_edges > kFixedCounterBranchEdgeFallbackMapBudget) {
        fixed_counter_fallback_reason = "auto-fixed-counter-map-cap";
      } else if (candidate_budget >
                 options.fixed_counter_branch_edge_fallback_limit) {
        fixed_counter_fallback_reason = "configured-limit";
      }
    }
  }

  return BranchEdgeBudgetSelection{
      make_previous_bb_branch_aggregate_budget(
          aggregate_candidate_edges, aggregate_candidate_sites, edge_budget, site_budget,
          options.branch_edge_site_limit_auto,
          options.previous_bb_branch_site_limit_auto, previous_bb_branch_policy,
          options.fixed_counter_fallback_for_branch_budget,
          fixed_counter_fallback_budget),
      branch_candidate_edges, edge_reason, site_reason,
      fixed_counter_fallback_reason};
}

const char *previous_bb_aggregate_safety_name(
    const PreviousBbBranchAggregateBudget &budget) {
  if (!budget.previous_bb_policy)
    return "not-previous-bb-policy";
  if (!budget.edge_budget_auto && !budget.site_budget_auto)
    return "debug-configured-budget";
  if (budget.edge_over_budget == 0 && budget.site_over_budget == 0)
    return "full-previous-bb-within-auto-budget";
  if (budget.fixed_counter_fallback_enabled)
    return "capped-previous-bb-with-fixed-fallback";
  return "capped-previous-bb";
}

std::string previous_bb_aggregate_safety_reason(
    const PreviousBbBranchAggregateBudget &budget) {
  if (!budget.previous_bb_policy)
    return "branch slot policy does not write previous-BB state";
  if (!budget.edge_budget_auto && !budget.site_budget_auto)
    return "debug/configured branch budget bypasses the product aggregate caps";
  if (budget.edge_over_budget == 0 && budget.site_over_budget == 0) {
    if (budget.site_budget == budget.candidate_sites &&
        budget.edge_budget == budget.candidate_edges) {
      return "all candidate previous-BB branch writers fit the candidate-derived budget; per-site liveness, resource, descriptor, scratch, and trampoline-placement checks still fail closed";
    }
    return "candidate previous-BB branch edges and writer sites fit the current aggregate caps";
  }

  std::string reason = "candidate previous-BB ";
  bool need_and = false;
  if (budget.edge_over_budget != 0) {
    reason += "logical edges exceed the current aggregate edge cap";
    need_and = true;
  }
  if (budget.site_over_budget != 0) {
    if (need_and)
      reason += " and ";
    reason += "writer sites exceed the current aggregate state-writer cap";
  }
  if (budget.fixed_counter_fallback_enabled)
    reason += "; excess sites use fixed counters when the smaller probe is safe";
  else
    reason += "; excess sites are skipped";
  return reason;
}

void record_block_entry_patch_skip(std::string_view *skip_reason,
                                   BlockEntryPatchSkip *skip,
                                   const rocjitsu::Instruction *inst,
                                   uint64_t text_offset,
                                   std::string_view reason) {
  if (skip_reason != nullptr)
    *skip_reason = reason;
  if (skip != nullptr)
    *skip = BlockEntryPatchSkip{inst, text_offset, reason};
}

std::optional<BlockEntryPatchPoint>
select_block_entry_patch_point_impl(const rocjitsu::BasicBlock &block,
                                    const rocjitsu::Instruction &first,
                                    std::string_view *skip_reason,
                                    BlockEntryPatchSkip *skip) {
  const std::string_view first_reason = block_entry_skip_reason(block, first);
  if (first_reason.empty())
    return BlockEntryPatchPoint{&first, block.start_offset()};

  // Wait-counter, delay-ALU, and saveexec-style EXEC-transition instructions
  // must execute before the probe. Keep the original prefix in place and patch
  // the first following relocatable instruction, so the block is recorded under
  // the EXEC mask that the original code selected for this divergent path.
  if (!is_wait_counter_instruction(first) && !is_delay_alu_instruction(first) &&
      !is_exec_mask_transition_instruction(first)) {
    record_block_entry_patch_skip(skip_reason, skip, &first, block.start_offset(),
                                  first_reason);
    return std::nullopt;
  }

  auto &mutable_block = const_cast<rocjitsu::BasicBlock &>(block);
  uint64_t text_offset = block.start_offset();
  bool saw_exec_transition_prefix = false;
  for (const rocjitsu::Instruction &inst : mutable_block.instructions()) {
    const uint64_t inst_offset = text_offset;
    if (inst.size() <= 0) {
      record_block_entry_patch_skip(skip_reason, skip, &inst, inst_offset,
                                    "entry instruction has invalid size");
      return std::nullopt;
    }
    text_offset += static_cast<uint64_t>(inst.size());
    if (is_wait_counter_instruction(inst) || is_delay_alu_instruction(inst))
      continue;
    if (is_exec_mask_transition_instruction(inst)) {
      saw_exec_transition_prefix = true;
      continue;
    }

    const std::string_view delayed_reason = block_entry_skip_reason(inst);
    if (!delayed_reason.empty()) {
      record_block_entry_patch_skip(skip_reason, skip, &inst, inst_offset,
                                    delayed_reason);
      return std::nullopt;
    }
    return BlockEntryPatchPoint{&inst, inst_offset};
  }

  record_block_entry_patch_skip(
      skip_reason, skip, nullptr, block.end_offset(),
      saw_exec_transition_prefix
          ? "entry EXEC transition has no following relocatable instruction"
          : "entry wait-counter has no following relocatable instruction");
  return std::nullopt;
}

bool is_unsupported_vopd_word(uint32_t word) {
  const uint32_t high = word >> 24;
  return high >= 0xc8u && high <= 0xcbu;
}

void add_all_tracked_register_uses(RegisterSet &uses) {
  for (uint16_t reg = 0; reg < REGISTER_SET_MAX_SGPRS; ++reg)
    uses.expand({RegClass::SGPR, reg, 1});
  for (uint16_t reg = 0; reg < REGISTER_SET_MAX_VGPRS; ++reg)
    uses.expand({RegClass::VGPR, reg, 1});
  for (uint16_t reg = 0; reg < REGISTER_SET_MAX_ACC_VGPRS; ++reg)
    uses.expand({RegClass::ACC_VGPR, reg, 1});
}

} // namespace

std::optional<BlockEntryPatchPoint>
select_block_entry_patch_point(const rocjitsu::BasicBlock &block,
                               const rocjitsu::Instruction &first,
                               std::string_view *skip_reason,
                               BlockEntryPatchSkip *skip) {
  return select_block_entry_patch_point_impl(block, first, skip_reason, skip);
}

namespace {

class OpaqueInstruction final : public rocjitsu::Instruction {
public:
  OpaqueInstruction(const char *mnemonic, const uint32_t *words, size_t word_count)
      : rocjitsu::Instruction(mnemonic, [](rocjitsu::Instruction &, void *) {}) {
    word_count = std::min(word_count, words_.size());
    for (size_t i = 0; i < word_count; ++i)
      words_[i] = words[i];
    size_ = word_count * sizeof(uint32_t);
    raw_encoding_ = words_.data();
    encoding_id_ = static_cast<uint16_t>(words_[0] >> 23);
  }

  void implicit_uses(RegisterSet &uses) const override {
    // Unknown encodings may read any original register. Model them as all-uses
    // rather than all-defs so liveness stays conservative before the opaque
    // instruction without blocking probes at later program points.
    add_all_tracked_register_uses(uses);
  }

private:
  std::array<uint32_t, 2> words_{};
};

class OpaqueVopdInstruction final : public rocjitsu::Instruction {
public:
  explicit OpaqueVopdInstruction(const uint32_t *words)
      : rocjitsu::Instruction("vopd_opaque", [](rocjitsu::Instruction &, void *) {}) {
    words_[0] = words[0];
    words_[1] = words[1];
    const uint32_t word_count = decode_vopd64_word_count(words_[0], words_[1]);
    if (word_count == 3)
      words_[2] = words[2];
    liveness_model_ = decode_vopd64_liveness_model(words_[0], words_[1]);
    size_ = static_cast<int>((word_count == 0 ? 2 : word_count) * sizeof(uint32_t));
    raw_encoding_ = words_.data();
    encoding_id_ = static_cast<uint16_t>(words_[0] >> 23);
  }

  void implicit_uses(RegisterSet &uses) const override {
    if (liveness_model_) {
      uses |= liveness_model_->uses;
      return;
    }
    add_all_tracked_register_uses(uses);
  }

  void implicit_defs(RegisterSet &defs) const override {
    if (liveness_model_)
      defs |= liveness_model_->defs;
  }

  bool has_liveness_model() const { return liveness_model_.has_value(); }

private:
  std::array<uint32_t, 3> words_{};
  std::optional<Vopd64LivenessModel> liveness_model_;
};

class CfgPlanningDecoder final : public rocjitsu::Decoder {
public:
  explicit CfgPlanningDecoder(std::unique_ptr<rocjitsu::Decoder> base) : base_(std::move(base)) {}

  rocjitsu::Instruction *decode(const rj_code_binary_inst_t *inst) override {
    try {
      return base_->decode(inst);
    } catch (const std::exception &) {
      if (is_unsupported_vopd_word(inst[0]))
        return new OpaqueVopdInstruction(inst);
      return new OpaqueInstruction("unknown_opaque", inst, 1);
    }
  }

private:
  std::unique_ptr<rocjitsu::Decoder> base_;
};

std::unique_ptr<rocjitsu::Decoder> create_cfg_planning_decoder(rj_code_arch_t arch) {
  auto base = rocjitsu::Decoder::create(arch);
  if (base == nullptr)
    return nullptr;
  return std::make_unique<CfgPlanningDecoder>(std::move(base));
}

bool is_opaque_instruction(const rocjitsu::Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  return mnemonic == "unknown_opaque" || mnemonic == "vopd_opaque";
}

bool opaque_liveness_modeled(const rocjitsu::Instruction &inst) {
  if (const auto *vopd = dynamic_cast<const OpaqueVopdInstruction *>(&inst))
    return vopd->has_liveness_model();
  return false;
}

void record_opaque_instruction_diagnostics(KernelEdgeSelectionSummary &summary,
                                           std::span<rocjitsu::BasicBlock *const> blocks) {
  for (rocjitsu::BasicBlock *block : blocks) {
    if (block == nullptr)
      continue;
    uint64_t text_offset = block->start_offset();
    for (const rocjitsu::Instruction &inst : block->instructions()) {
      if (is_opaque_instruction(inst)) {
        if (summary.opaque_instruction_count != std::numeric_limits<uint32_t>::max())
          ++summary.opaque_instruction_count;
        const bool liveness_modeled = opaque_liveness_modeled(inst);
        if (!liveness_modeled &&
            summary.unmodeled_opaque_instruction_count != std::numeric_limits<uint32_t>::max())
          ++summary.unmodeled_opaque_instruction_count;
        if (summary.sampled_opaque_instructions.size() < 16) {
          OpaqueInstructionSample sample;
          sample.mnemonic = std::string(inst.mnemonic());
          sample.text_offset = text_offset;
          sample.liveness_modeled = liveness_modeled;
          if (inst.raw_encoding() != nullptr) {
            const uint32_t word_count = static_cast<uint32_t>(std::min<int>(
                inst.size() / static_cast<int>(sizeof(uint32_t)), 4));
            sample.words.assign(inst.raw_encoding(), inst.raw_encoding() + word_count);
          }
          summary.sampled_opaque_instructions.push_back(std::move(sample));
        }
      }
      text_offset += static_cast<uint64_t>(std::max(inst.size(), 0));
    }
  }
}

bool range_overlaps(uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                    uint16_t rhs_count) {
  const uint32_t lhs_end = static_cast<uint32_t>(lhs_base) + lhs_count;
  const uint32_t rhs_end = static_cast<uint32_t>(rhs_base) + rhs_count;
  return lhs_base < rhs_end && rhs_base < lhs_end;
}

void add_capped(uint32_t &value, uint64_t amount) {
  value = static_cast<uint32_t>(
      std::min<uint64_t>(static_cast<uint64_t>(value) + amount,
                         std::numeric_limits<uint32_t>::max()));
}

void add_reason_count(std::vector<EdgeSiteSkipReasonCount> &counts,
                      std::string_view kind, std::string_view reason,
                      uint32_t amount = 1) {
  if (amount == 0)
    return;
  bool counted = false;
  for (EdgeSiteSkipReasonCount &count : counts) {
    if (count.kind == kind && count.reason == reason) {
      add_capped(count.count, amount);
      counted = true;
      break;
    }
  }
  if (!counted) {
    EdgeSiteSkipReasonCount count;
    count.kind = std::string(kind);
    count.reason = std::string(reason);
    count.count = amount;
    counts.push_back(std::move(count));
  }
}

void populate_skip_sample_instruction(EdgeSiteSkipSample &sample,
                                      const rocjitsu::Instruction *inst) {
  if (inst == nullptr)
    return;
  sample.mnemonic = std::string(inst->mnemonic());
  sample.instruction_size = static_cast<uint32_t>(std::max(inst->size(), 0));
  sample.instruction_flags = inst->flags();
  if (inst->raw_encoding() == nullptr || inst->size() <= 0)
    return;
  const uint32_t word_count = static_cast<uint32_t>(
      std::min<int>(inst->size() / static_cast<int>(sizeof(uint32_t)), 4));
  sample.words.assign(inst->raw_encoding(), inst->raw_encoding() + word_count);
}

enum class SkipSamplePolicy {
  FirstReason,
  DistinctSite,
};

void record_skip_sample(KernelEdgeSelectionSummary &summary, std::string_view kind,
                        uint64_t text_offset, std::string_view reason,
                        const rocjitsu::Instruction *inst = nullptr,
                        SkipSamplePolicy policy = SkipSamplePolicy::FirstReason) {
  add_reason_count(summary.skip_reason_counts, kind, reason);

  for (EdgeSiteSkipSample &sample : summary.sampled_skips) {
    if (std::string_view(sample.kind) != kind ||
        std::string_view(sample.reason) != reason)
      continue;
    if (policy == SkipSamplePolicy::DistinctSite &&
        sample.text_offset != text_offset)
      continue;
    if (sample.mnemonic.empty())
      populate_skip_sample_instruction(sample, inst);
    return;
  }

  if (summary.sampled_skips.size() >= 16)
    return;
  EdgeSiteSkipSample sample;
  sample.kind = std::string(kind);
  sample.text_offset = text_offset;
  sample.reason = std::string(reason);
  populate_skip_sample_instruction(sample, inst);
  summary.sampled_skips.push_back(std::move(sample));
}

void record_degradation_count(KernelEdgeSelectionSummary &summary,
                              std::string_view kind, std::string_view reason,
                              uint32_t edge_count) {
  add_reason_count(summary.degradation_reason_counts, kind, reason, edge_count);
}

std::string contextual_liveness_failure_reason(
    const char *failure_reason,
    const char *fresh_register_growth_disabled_reason) {
  const std::string_view base =
      failure_reason != nullptr ? std::string_view(failure_reason)
                                : std::string_view("no liveness-safe probe registers");
  if (fresh_register_growth_disabled_reason == nullptr ||
      base.find("allocated") == std::string_view::npos) {
    return std::string(base);
  }
  std::string reason(fresh_register_growth_disabled_reason);
  reason += "; ";
  reason += base;
  return reason;
}

constexpr uint16_t kGfx11PlusSoppCbranchExecz = 0x25;
constexpr uint16_t kGfx11PlusSoppCbranchExecnz = 0x26;

bool branch_opcode_depends_on_exec(uint16_t opcode) {
  return opcode == kGfx11PlusSoppCbranchExecz ||
         opcode == kGfx11PlusSoppCbranchExecnz;
}

bool branch_condition_depends_on_exec(const EdgeSite &site) {
  if (!edge_patch_kind_is_conditional_dispatch(site.kind))
    return false;
  return branch_opcode_depends_on_exec(site.branch_opcode);
}

bool fixed_counter_edge_has_exec_empty_outcome(const EdgeSite &site) {
  return site.slot_policy == EdgeSlotPolicyKind::FixedCounter &&
         site.kind == EdgePatchKind::ConditionalBranchTerminator &&
         branch_opcode_depends_on_exec(site.branch_opcode);
}

const char *kernel_coverage_strategy(const KernelEdgeSelectionSummary &summary,
                                     const InstrumentationPlanOptions &options) {
  if (summary.block_selected == 0 && summary.branch_edges_selected == 0)
    return "none";
  const bool mixed_branch_slots =
      summary.slot_policy_summary.hashed_edge_sites != 0 &&
      summary.slot_policy_summary.fixed_edge_sites != 0 &&
      summary.branch_edges_selected != 0;
  const bool fixed_only_branch_slots =
      summary.slot_policy_summary.hashed_edge_sites == 0 &&
      summary.slot_policy_summary.fixed_edge_sites != 0 &&
      summary.branch_edges_selected != 0;
  if (summary.self_contained_probe) {
    if (mixed_branch_slots)
      return "self-contained-hybrid-previous-bb-and-fixed-branch";
    if (fixed_only_branch_slots)
      return "self-contained-fixed-branch";
    return options.branch_terminator_slot_policy == EdgeSlotPolicyKind::PreviousBbHash
               ? "self-contained-previous-bb-branch"
               : "self-contained-fixed-branch";
  }
  if (summary.block_selected != 0 && summary.branch_edges_selected != 0) {
    if (options.branch_terminator_slot_policy == EdgeSlotPolicyKind::FixedCounter)
      return "entry-previous-bb-block-and-fixed-branch";
    if (mixed_branch_slots)
      return "entry-previous-bb-block-and-hybrid-branch";
    return "entry-previous-bb-block-and-previous-bb-branch";
  }
  if (summary.block_selected != 0)
    return "entry-previous-bb-block";
  return options.branch_terminator_slot_policy == EdgeSlotPolicyKind::PreviousBbHash
             ? "entry-backed-previous-bb-branch"
             : "entry-backed-fixed-branch";
}

void add_site_slot_summary(EdgeSlotPolicySummary &summary, const EdgeSite &site) {
  if (site.slot_policy == EdgeSlotPolicyKind::PreviousBbHash) {
    ++summary.hashed_edge_sites;
    return;
  }

  ++summary.fixed_edge_sites;
  const uint32_t edge_count = edge_count_for_site(site);
  summary.fixed_slot_requests += edge_count;
  summary.fixed_slots_reserved += edge_count;
  summary.fixed_slot_collisions += site.fixed_slot_collisions;
  summary.inline_slot_requests += edge_count;
  summary.inline_slots_reserved += edge_count;
}

void add_slot_summary(EdgeSlotPolicySummary &summary,
                      const EdgeSlotPolicySummary &delta) {
  add_capped(summary.hashed_edge_sites, delta.hashed_edge_sites);
  add_capped(summary.fixed_edge_sites, delta.fixed_edge_sites);
  add_capped(summary.fixed_slot_requests, delta.fixed_slot_requests);
  add_capped(summary.fixed_slots_reserved, delta.fixed_slots_reserved);
  add_capped(summary.fixed_slot_exhaustions, delta.fixed_slot_exhaustions);
  add_capped(summary.fixed_slot_collisions, delta.fixed_slot_collisions);
  add_capped(summary.inline_slot_requests, delta.inline_slot_requests);
  add_capped(summary.inline_slots_reserved, delta.inline_slots_reserved);
  add_capped(summary.inline_slot_exhaustions, delta.inline_slot_exhaustions);
}

void subtract_capped(uint32_t &value, uint32_t amount) {
  value = amount > value ? 0 : value - amount;
}

void subtract_slot_summary(EdgeSlotPolicySummary &summary,
                           const EdgeSlotPolicySummary &dropped) {
  subtract_capped(summary.hashed_edge_sites, dropped.hashed_edge_sites);
  subtract_capped(summary.fixed_edge_sites, dropped.fixed_edge_sites);
  subtract_capped(summary.fixed_slot_requests, dropped.fixed_slot_requests);
  subtract_capped(summary.fixed_slots_reserved, dropped.fixed_slots_reserved);
  subtract_capped(summary.fixed_slot_collisions, dropped.fixed_slot_collisions);
  subtract_capped(summary.inline_slot_requests, dropped.inline_slot_requests);
  subtract_capped(summary.inline_slots_reserved, dropped.inline_slots_reserved);
}

bool range_overlaps_used(std::span<const std::pair<uint16_t, uint16_t>> used,
                         uint16_t base, uint16_t count) {
  for (const auto &[used_base, used_count] : used) {
    if (range_overlaps(base, count, used_base, used_count))
      return true;
  }
  return false;
}

bool range_is_dead_at_all_points(const rocjitsu::LivenessAnalysis &liveness,
                                 std::span<const rocjitsu::Instruction *const> points,
                                 RegClass cls, uint16_t base, uint16_t count) {
  for (const rocjitsu::Instruction *point : points) {
    if (point == nullptr)
      return false;
    const RegisterSet &live = liveness.live_before(*point);
    for (uint16_t i = 0; i < count; ++i) {
      if (live.contains({cls, static_cast<uint16_t>(base + i), 1}))
        return false;
    }
  }
  return true;
}

std::optional<uint16_t>
find_common_dead_run(const rocjitsu::LivenessAnalysis &liveness,
                     std::span<const rocjitsu::Instruction *const> points, RegClass cls,
                     uint16_t count, uint16_t limit, bool even_aligned,
                     std::span<const std::pair<uint16_t, uint16_t>> used) {
  if (count == 0 || limit < count)
    return std::nullopt;
  for (uint32_t base = 0; base + count <= limit; base += even_aligned ? 2 : 1) {
    if (even_aligned && base % 2 != 0)
      continue;
    const uint16_t reg = static_cast<uint16_t>(base);
    if (range_overlaps_used(used, reg, count))
      continue;
    if (range_is_dead_at_all_points(liveness, points, cls, reg, count))
      return reg;
  }
  return std::nullopt;
}

std::optional<uint16_t>
find_dead_run_in_set(const RegisterSet &live, RegClass cls, uint16_t count,
                     uint16_t limit, bool even_aligned,
                     std::span<const std::pair<uint16_t, uint16_t>> used) {
  if (count == 0 || limit < count)
    return std::nullopt;
  for (uint32_t base = 0; base + count <= limit; base += even_aligned ? 2 : 1) {
    if (even_aligned && base % 2 != 0)
      continue;
    const uint16_t reg = static_cast<uint16_t>(base);
    if (range_overlaps_used(used, reg, count))
      continue;
    bool any_live = false;
    for (uint16_t i = 0; i < count; ++i) {
      if (live.contains({cls, static_cast<uint16_t>(base + i), 1})) {
        any_live = true;
        break;
      }
    }
    if (!any_live)
      return reg;
  }
  return std::nullopt;
}

std::optional<Rdna4ProbeRegisters>
select_probe_registers_from_live_set(const KernelSite &kernel, const RegisterSet &live,
                                     bool previous_bb_probe_registers,
                                     bool stable_state_sgpr,
                                     const Rdna4ProbeRegisters &base_registers) {
  const uint32_t sgpr_limit_u32 =
      std::min<uint32_t>(kernel.allocated_sgpr_count,
                         static_cast<uint32_t>(REGISTER_SET_ALLOCATABLE_SGPRS));
  const uint32_t vgpr_limit_u32 =
      std::min<uint32_t>(kernel.allocated_vgpr_count,
                         static_cast<uint32_t>(REGISTER_SET_MAX_VGPRS));
  if (sgpr_limit_u32 > std::numeric_limits<uint16_t>::max() ||
      vgpr_limit_u32 > std::numeric_limits<uint16_t>::max()) {
    return std::nullopt;
  }
  const uint16_t sgpr_limit = static_cast<uint16_t>(sgpr_limit_u32);
  const uint16_t vgpr_limit = static_cast<uint16_t>(vgpr_limit_u32);
  const uint16_t required_sgprs =
      previous_bb_probe_registers ? (stable_state_sgpr ? 5 : 7) : (stable_state_sgpr ? 0 : 2);
  const uint16_t required_vgprs = previous_bb_probe_registers ? 4 : 2;
  if (sgpr_limit < required_sgprs)
    return std::nullopt;
  if (vgpr_limit < required_vgprs)
    return std::nullopt;

  std::vector<std::pair<uint16_t, uint16_t>> used_sgprs;
  auto reserve_sgpr_run = [&](uint16_t count, bool even_aligned) -> std::optional<uint16_t> {
    auto reg = find_dead_run_in_set(live, RegClass::SGPR, count, sgpr_limit,
                                    even_aligned, used_sgprs);
    if (reg)
      used_sgprs.push_back({*reg, count});
    return reg;
  };
  auto reserve_sgpr_pair = [&]() -> std::optional<uint16_t> {
    return reserve_sgpr_run(/*count=*/2, /*even_aligned=*/true);
  };

  std::optional<uint16_t> state_sgpr;
  if (stable_state_sgpr) {
    if (base_registers.state_sgpr + 2 > sgpr_limit)
      return std::nullopt;
    state_sgpr = base_registers.state_sgpr;
    used_sgprs.push_back({base_registers.state_sgpr, 2});
  } else if (sgpr_limit >= 2) {
    state_sgpr = reserve_sgpr_pair();
  }
  std::optional<uint16_t> saved_exec_sgpr = state_sgpr;
  std::optional<uint16_t> tmp_sgpr = state_sgpr;
  std::optional<uint16_t> scc_sgpr = state_sgpr;
  if (previous_bb_probe_registers) {
    saved_exec_sgpr = reserve_sgpr_pair();
    tmp_sgpr = reserve_sgpr_pair();
    scc_sgpr = reserve_sgpr_run(/*count=*/1, /*even_aligned=*/false);
  }
  if (!state_sgpr && !previous_bb_probe_registers) {
    state_sgpr = base_registers.state_sgpr;
    saved_exec_sgpr = state_sgpr;
    tmp_sgpr = state_sgpr;
    scc_sgpr = state_sgpr;
  }
  if (!state_sgpr || !saved_exec_sgpr || !tmp_sgpr || !scc_sgpr)
    return std::nullopt;

  const std::vector<std::pair<uint16_t, uint16_t>> no_used_vgprs;
  std::optional<uint16_t> workitem_vgpr =
      find_dead_run_in_set(live, RegClass::VGPR, required_vgprs, vgpr_limit,
                           /*even_aligned=*/false, no_used_vgprs);
  if (!workitem_vgpr)
    return std::nullopt;

  Rdna4ProbeRegisters regs;
  regs.state_sgpr = static_cast<uint8_t>(*state_sgpr);
  regs.saved_exec_sgpr = static_cast<uint8_t>(*saved_exec_sgpr);
  regs.tmp0_sgpr = static_cast<uint8_t>(*tmp_sgpr);
  regs.tmp1_sgpr = static_cast<uint8_t>(*tmp_sgpr);
  regs.scc_sgpr = static_cast<uint8_t>(*scc_sgpr);
  regs.workitem_vgpr = static_cast<uint8_t>(*workitem_vgpr);
  regs.tmp0_vgpr = static_cast<uint8_t>(*workitem_vgpr + 1);
  regs.tmp1_vgpr =
      static_cast<uint8_t>(*workitem_vgpr + (previous_bb_probe_registers ? 2 : 1));
  regs.tmp2_vgpr =
      static_cast<uint8_t>(*workitem_vgpr + (previous_bb_probe_registers ? 3 : 1));
  return regs;
}

struct EntryProbeLivenessState {
  RegisterSet live_in;
  RegisterSet live_out;
  RegisterSet gen;
  RegisterSet kill;
};

RegisterSet entry_probe_kills(const rocjitsu::InstDefUse &du) {
  if (du.has_predicated_def)
    return {};
  return du.defs;
}

std::optional<RegisterSet>
entry_probe_live_in(const rocjitsu::BasicBlock *entry,
                    std::span<rocjitsu::BasicBlock *const> scope) {
  if (entry == nullptr || scope.empty())
    return std::nullopt;

  std::unordered_map<const rocjitsu::BasicBlock *, size_t> block_index;
  block_index.reserve(scope.size());
  for (size_t i = 0; i < scope.size(); ++i) {
    if (scope[i] != nullptr)
      block_index.emplace(scope[i], i);
  }
  auto entry_it = block_index.find(entry);
  if (entry_it == block_index.end())
    return std::nullopt;

  std::vector<EntryProbeLivenessState> states(scope.size());
  for (size_t i = 0; i < scope.size(); ++i) {
    rocjitsu::BasicBlock *block = scope[i];
    if (block == nullptr)
      continue;
    EntryProbeLivenessState &state = states[i];
    for (const rocjitsu::Instruction &inst : block->instructions()) {
      rocjitsu::InstDefUse du(inst);
      RegisterSet kills = entry_probe_kills(du);
      RegisterSet upward_uses = du.uses;
      upward_uses -= state.kill;
      state.gen |= upward_uses;
      state.kill |= kills;
    }
  }

  // Entry probes execute before compiler-generated EXEC manipulation. For this
  // one point, vector defs in the original kernel can be treated as kills for
  // the currently active lanes. General edge probes keep using rocjitsu's
  // conservative EXEC-masked liveness.
  const std::vector<const rocjitsu::BasicBlock *> rpo = rocjitsu::reverse_post_order(scope);
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto it = rpo.rbegin(); it != rpo.rend(); ++it) {
      const rocjitsu::BasicBlock *block = *it;
      auto idx_it = block_index.find(block);
      if (idx_it == block_index.end())
        continue;
      EntryProbeLivenessState &state = states[idx_it->second];
      RegisterSet live_out;
      for (const rocjitsu::BasicBlock *succ : block->successors()) {
        auto succ_it = block_index.find(succ);
        if (succ_it != block_index.end())
          live_out |= states[succ_it->second].live_in;
      }

      RegisterSet live_in = live_out;
      live_in -= state.kill;
      live_in |= state.gen;
      if (state.live_out != live_out || state.live_in != live_in) {
        state.live_out = live_out;
        state.live_in = live_in;
        changed = true;
      }
    }
  }

  return states[entry_it->second].live_in;
}

std::optional<Rdna4ProbeRegisters>
select_liveness_probe_registers_from_analysis(
    const KernelSite &kernel, const rocjitsu::LivenessAnalysis &liveness,
    std::span<const rocjitsu::Instruction *const> probe_points,
    bool previous_bb_probe_registers, bool stable_state_sgpr,
    const Rdna4ProbeRegisters &base_registers) {
  if (probe_points.empty())
    return std::nullopt;

  const uint32_t sgpr_limit_u32 =
      std::min<uint32_t>(kernel.allocated_sgpr_count,
                         static_cast<uint32_t>(REGISTER_SET_ALLOCATABLE_SGPRS));
  const uint32_t vgpr_limit_u32 =
      std::min<uint32_t>(kernel.allocated_vgpr_count,
                         static_cast<uint32_t>(REGISTER_SET_MAX_VGPRS));
  if (sgpr_limit_u32 > std::numeric_limits<uint16_t>::max() ||
      vgpr_limit_u32 > std::numeric_limits<uint16_t>::max()) {
    return std::nullopt;
  }
  const uint16_t sgpr_limit = static_cast<uint16_t>(sgpr_limit_u32);
  const uint16_t vgpr_limit = static_cast<uint16_t>(vgpr_limit_u32);
  const uint16_t required_sgprs =
      previous_bb_probe_registers ? (stable_state_sgpr ? 5 : 7) : (stable_state_sgpr ? 0 : 2);
  const uint16_t required_vgprs = previous_bb_probe_registers ? 4 : 2;
  if (sgpr_limit < required_sgprs || vgpr_limit < required_vgprs)
    return std::nullopt;

  std::vector<std::pair<uint16_t, uint16_t>> used_sgprs;
  auto reserve_sgpr_run = [&](uint16_t count, bool even_aligned) -> std::optional<uint16_t> {
    auto reg = find_common_dead_run(liveness, probe_points, RegClass::SGPR, count,
                                    sgpr_limit, even_aligned, used_sgprs);
    if (reg)
      used_sgprs.push_back({*reg, count});
    return reg;
  };
  auto reserve_sgpr_pair = [&]() -> std::optional<uint16_t> {
    return reserve_sgpr_run(/*count=*/2, /*even_aligned=*/true);
  };

  std::optional<uint16_t> state_sgpr;
  if (stable_state_sgpr) {
    if (base_registers.state_sgpr + 2 > sgpr_limit)
      return std::nullopt;
    state_sgpr = base_registers.state_sgpr;
    used_sgprs.push_back({base_registers.state_sgpr, 2});
  } else {
    state_sgpr = reserve_sgpr_pair();
  }
  std::optional<uint16_t> saved_exec_sgpr = state_sgpr;
  std::optional<uint16_t> tmp_sgpr = state_sgpr;
  std::optional<uint16_t> scc_sgpr = state_sgpr;
  if (previous_bb_probe_registers) {
    saved_exec_sgpr = reserve_sgpr_pair();
    tmp_sgpr = reserve_sgpr_pair();
    scc_sgpr = reserve_sgpr_run(/*count=*/1, /*even_aligned=*/false);
  }
  if (!state_sgpr && !previous_bb_probe_registers) {
    // Self-contained fixed-counter probes can safely keep the existing
    // high-register state pointer fallback while using liveness-selected
    // VGPRs. State-sharing probes also keep this fixed state SGPR because
    // the injected pointer is live from the entry hook to later edge hooks.
    state_sgpr = base_registers.state_sgpr;
    saved_exec_sgpr = state_sgpr;
    tmp_sgpr = state_sgpr;
    scc_sgpr = state_sgpr;
  }
  if (!state_sgpr || !saved_exec_sgpr || !tmp_sgpr || !scc_sgpr)
    return std::nullopt;

  const std::vector<std::pair<uint16_t, uint16_t>> no_used_vgprs;
  std::optional<uint16_t> workitem_vgpr =
      find_common_dead_run(liveness, probe_points, RegClass::VGPR, required_vgprs,
                           vgpr_limit, /*even_aligned=*/false, no_used_vgprs);
  if (!workitem_vgpr)
    return std::nullopt;

  Rdna4ProbeRegisters regs;
  regs.state_sgpr = static_cast<uint8_t>(*state_sgpr);
  regs.saved_exec_sgpr = static_cast<uint8_t>(*saved_exec_sgpr);
  regs.tmp0_sgpr = static_cast<uint8_t>(*tmp_sgpr);
  regs.tmp1_sgpr = static_cast<uint8_t>(*tmp_sgpr);
  regs.scc_sgpr = static_cast<uint8_t>(*scc_sgpr);
  regs.workitem_vgpr = static_cast<uint8_t>(*workitem_vgpr);
  regs.tmp0_vgpr = static_cast<uint8_t>(*workitem_vgpr + 1);
  regs.tmp1_vgpr =
      static_cast<uint8_t>(*workitem_vgpr + (previous_bb_probe_registers ? 2 : 1));
  regs.tmp2_vgpr =
      static_cast<uint8_t>(*workitem_vgpr + (previous_bb_probe_registers ? 3 : 1));
  return regs;
}

uint16_t allocatable_sgpr_limit_for_arch(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return static_cast<uint16_t>(REGISTER_SET_MAX_SGPRS);
  default:
    return static_cast<uint16_t>(REGISTER_SET_ALLOCATABLE_SGPRS);
  }
}

std::optional<uint16_t>
find_fresh_run(uint16_t start, uint16_t count, uint16_t limit, bool even_aligned,
               std::span<const std::pair<uint16_t, uint16_t>> used) {
  if (count == 0 || limit < count)
    return std::nullopt;
  uint32_t base = start;
  if (even_aligned && base % 2 != 0)
    ++base;
  for (; base + count <= limit; base += even_aligned ? 2 : 1) {
    const uint16_t reg = static_cast<uint16_t>(base);
    if (!range_overlaps_used(used, reg, count))
      return reg;
  }
  return std::nullopt;
}

struct ProbeSgprSelection {
  uint16_t state_sgpr = 0;
  uint16_t saved_exec_sgpr = 0;
  uint16_t tmp_sgpr = 0;
  uint16_t scc_sgpr = 0;
  bool uses_fresh_registers = false;
};

uint16_t required_probe_vgprs(bool previous_bb_probe_registers) {
  return previous_bb_probe_registers ? 4 : 2;
}

void assign_probe_sgprs(Rdna4ProbeRegisters &regs, const ProbeSgprSelection &sgprs) {
  regs.state_sgpr = static_cast<uint8_t>(sgprs.state_sgpr);
  regs.saved_exec_sgpr = static_cast<uint8_t>(sgprs.saved_exec_sgpr);
  regs.tmp0_sgpr = static_cast<uint8_t>(sgprs.tmp_sgpr);
  regs.tmp1_sgpr = static_cast<uint8_t>(sgprs.tmp_sgpr);
  regs.scc_sgpr = static_cast<uint8_t>(sgprs.scc_sgpr);
}

void assign_probe_vgpr_run(Rdna4ProbeRegisters &regs, uint16_t workitem_vgpr,
                           bool previous_bb_probe_registers) {
  regs.workitem_vgpr = static_cast<uint8_t>(workitem_vgpr);
  regs.tmp0_vgpr = static_cast<uint8_t>(workitem_vgpr + 1);
  regs.tmp1_vgpr =
      static_cast<uint8_t>(workitem_vgpr + (previous_bb_probe_registers ? 2 : 1));
  regs.tmp2_vgpr =
      static_cast<uint8_t>(workitem_vgpr + (previous_bb_probe_registers ? 3 : 1));
}

std::vector<uint8_t> probe_vgpr_run(uint16_t workitem_vgpr,
                                    bool previous_bb_probe_registers) {
  const uint16_t count = required_probe_vgprs(previous_bb_probe_registers);
  std::vector<uint8_t> regs;
  regs.reserve(count);
  for (uint16_t i = 0; i < count; ++i)
    regs.push_back(static_cast<uint8_t>(workitem_vgpr + i));
  return regs;
}

std::optional<uint16_t>
find_allocated_run(uint16_t count, uint16_t limit, bool even_aligned,
                   std::span<const std::pair<uint16_t, uint16_t>> used) {
  if (count == 0 || limit < count)
    return std::nullopt;
  for (uint32_t base = 0; base + count <= limit; base += even_aligned ? 2 : 1) {
    if (even_aligned && base % 2 != 0)
      continue;
    const uint16_t reg = static_cast<uint16_t>(base);
    if (!range_overlaps_used(used, reg, count))
      return reg;
  }
  return std::nullopt;
}

bool target_supports_vgpr_scratch_spills(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return true;
  default:
    return false;
  }
}

std::optional<uint32_t> round_up_power_of_two(uint32_t value, uint32_t limit) {
  if (value == 0)
    return std::nullopt;
  uint32_t rounded = 1;
  while (rounded < value) {
    if (rounded > std::numeric_limits<uint32_t>::max() / 2)
      return std::nullopt;
    rounded <<= 1;
  }
  if (rounded > limit)
    return std::nullopt;
  return rounded;
}

std::optional<ProbeSgprSelection>
select_safe_probe_sgprs_from_analysis(
    const KernelSite &kernel, rj_code_arch_t arch, const rocjitsu::LivenessAnalysis &liveness,
    std::span<const rocjitsu::Instruction *const> probe_points,
    bool previous_bb_probe_registers, bool stable_state_sgpr,
    const Rdna4ProbeRegisters &base_registers, bool allow_fresh_registers = true,
    const char **failure_reason = nullptr,
    bool force_saved_exec_sgpr_pair = false,
    bool force_fresh_sgprs = false) {
  auto fail = [&](const char *reason) -> std::optional<ProbeSgprSelection> {
    if (failure_reason != nullptr)
      *failure_reason = reason;
    return std::nullopt;
  };

  if (probe_points.empty())
    return fail("no probe points for liveness register selection");

  const uint32_t sgpr_arch_limit_u32 = allocatable_sgpr_limit_for_arch(arch);
  if (sgpr_arch_limit_u32 > std::numeric_limits<uint16_t>::max()) {
    return fail("probe register architecture limit overflow");
  }
  const uint16_t sgpr_arch_limit = static_cast<uint16_t>(sgpr_arch_limit_u32);
  const uint16_t allocated_sgprs = static_cast<uint16_t>(
      std::min<uint32_t>(kernel.allocated_sgpr_count, sgpr_arch_limit));
  const bool allow_fresh_sgprs =
      allow_fresh_registers && kernel.fresh_sgpr_growth_supported;
  if (force_fresh_sgprs && !allow_fresh_sgprs)
    return fail("fresh SGPR growth disabled by target resource model");
  const bool needs_saved_exec_sgpr_pair =
      previous_bb_probe_registers || force_saved_exec_sgpr_pair;
  const uint16_t required_sgprs =
      previous_bb_probe_registers ? (stable_state_sgpr ? 5 : 7)
                                  : (needs_saved_exec_sgpr_pair
                                         ? (stable_state_sgpr ? 2 : 4)
                                         : (stable_state_sgpr ? 0 : 2));
  if (sgpr_arch_limit < required_sgprs)
    return fail("target has too few SGPRs for probe");

  ProbeSgprSelection selection;
  std::vector<std::pair<uint16_t, uint16_t>> used_sgprs;
  auto reserve_sgpr_run = [&](uint16_t count, bool even_aligned,
                              const char *reason) -> std::optional<uint16_t> {
    std::optional<uint16_t> reg;
    if (!force_fresh_sgprs) {
      reg = find_common_dead_run(liveness, probe_points, RegClass::SGPR, count,
                                 allocated_sgprs, even_aligned, used_sgprs);
    }
    if (!reg && allow_fresh_sgprs) {
      reg = find_fresh_run(allocated_sgprs, count, sgpr_arch_limit, even_aligned,
                           used_sgprs);
      if (reg)
        selection.uses_fresh_registers = true;
    }
    if (reg)
      used_sgprs.push_back({*reg, count});
    else if (failure_reason != nullptr) {
      if (allow_fresh_registers && !allow_fresh_sgprs)
        *failure_reason = "no liveness-safe allocated SGPR run; fresh SGPR "
                          "growth disabled by target resource model";
      else
        *failure_reason = reason;
    }
    return reg;
  };
  auto reserve_sgpr_pair = [&](const char *reason) -> std::optional<uint16_t> {
    return reserve_sgpr_run(/*count=*/2, /*even_aligned=*/true, reason);
  };

  std::optional<uint16_t> state_sgpr;
  if (stable_state_sgpr) {
    const uint16_t state_sgpr_limit =
        kernel.fresh_sgpr_growth_supported ? sgpr_arch_limit : allocated_sgprs;
    if (base_registers.state_sgpr + 2 > state_sgpr_limit)
      return fail(kernel.fresh_sgpr_growth_supported
                      ? "fixed state SGPR pair exceeds target SGPR limit"
                      : "fixed state SGPR pair exceeds current SGPR allocation");
    state_sgpr = base_registers.state_sgpr;
    used_sgprs.push_back({base_registers.state_sgpr, 2});
  } else {
    state_sgpr = reserve_sgpr_pair(allow_fresh_registers
                                       ? "no liveness-safe state SGPR pair"
                                       : "no liveness-safe allocated state SGPR pair");
  }
  std::optional<uint16_t> saved_exec_sgpr = state_sgpr;
  std::optional<uint16_t> tmp_sgpr = state_sgpr;
  std::optional<uint16_t> scc_sgpr = state_sgpr;
  if (needs_saved_exec_sgpr_pair) {
    saved_exec_sgpr = reserve_sgpr_pair(
        allow_fresh_registers ? "no liveness-safe saved EXEC SGPR pair"
                              : "no liveness-safe allocated saved EXEC SGPR pair");
    tmp_sgpr = saved_exec_sgpr;
  }
  if (previous_bb_probe_registers) {
    tmp_sgpr = reserve_sgpr_pair(
        allow_fresh_registers ? "no liveness-safe previous-BB temp SGPR pair"
                              : "no liveness-safe allocated previous-BB temp SGPR pair");
    scc_sgpr = reserve_sgpr_run(
        /*count=*/1, /*even_aligned=*/false,
        allow_fresh_registers ? "no liveness-safe SCC save SGPR"
                              : "no liveness-safe allocated SCC save SGPR");
  }
  if (!state_sgpr || !saved_exec_sgpr || !tmp_sgpr || !scc_sgpr)
    return std::nullopt;

  selection.state_sgpr = *state_sgpr;
  selection.saved_exec_sgpr = *saved_exec_sgpr;
  selection.tmp_sgpr = *tmp_sgpr;
  selection.scc_sgpr = *scc_sgpr;
  return selection;
}

std::optional<EdgeProbeRegisterSelection>
select_safe_probe_registers_from_analysis(
    const KernelSite &kernel, rj_code_arch_t arch, const rocjitsu::LivenessAnalysis &liveness,
    std::span<const rocjitsu::Instruction *const> probe_points,
    bool previous_bb_probe_registers, bool stable_state_sgpr,
    const Rdna4ProbeRegisters &base_registers, bool allow_fresh_registers = true,
    const char **failure_reason = nullptr,
    bool force_fresh_sgprs = false, bool force_saved_exec_sgpr_pair = false,
    bool force_fresh_vgprs = false) {
  auto fail = [&](const char *reason) -> std::optional<EdgeProbeRegisterSelection> {
    if (failure_reason != nullptr)
      *failure_reason = reason;
    return std::nullopt;
  };

  const char *sgpr_failure_reason = nullptr;
  std::optional<ProbeSgprSelection> sgprs = select_safe_probe_sgprs_from_analysis(
      kernel, arch, liveness, probe_points, previous_bb_probe_registers, stable_state_sgpr,
      base_registers, allow_fresh_registers, &sgpr_failure_reason,
      force_saved_exec_sgpr_pair, force_fresh_sgprs);
  if (!sgprs)
    return fail(sgpr_failure_reason != nullptr ? sgpr_failure_reason
                                               : "no liveness-safe SGPRs for probe");

  const uint32_t vgpr_arch_limit_u32 = REGISTER_SET_MAX_VGPRS;
  if (vgpr_arch_limit_u32 > std::numeric_limits<uint16_t>::max())
    return fail("probe register architecture limit overflow");
  const uint16_t vgpr_arch_limit = static_cast<uint16_t>(vgpr_arch_limit_u32);
  const uint16_t allocated_vgprs = static_cast<uint16_t>(
      std::min<uint32_t>(kernel.allocated_vgpr_count, vgpr_arch_limit));
  const uint16_t required_vgprs = required_probe_vgprs(previous_bb_probe_registers);
  if (vgpr_arch_limit < required_vgprs)
    return fail("target has too few VGPRs for probe");
  if (force_fresh_vgprs && !allow_fresh_registers)
    return fail("fresh VGPR growth disabled by target resource model");

  EdgeProbeRegisterSelection selection;
  selection.uses_fresh_registers = sgprs->uses_fresh_registers;
  const std::vector<std::pair<uint16_t, uint16_t>> no_used_vgprs;
  std::optional<uint16_t> workitem_vgpr;
  if (!force_fresh_vgprs) {
    workitem_vgpr =
        find_common_dead_run(liveness, probe_points, RegClass::VGPR, required_vgprs,
                             allocated_vgprs, /*even_aligned=*/false, no_used_vgprs);
  }
  if (!workitem_vgpr && allow_fresh_registers) {
    workitem_vgpr = find_fresh_run(allocated_vgprs, required_vgprs, vgpr_arch_limit,
                                   /*even_aligned=*/false, no_used_vgprs);
    if (workitem_vgpr)
      selection.uses_fresh_registers = true;
  }
  if (!workitem_vgpr)
    return fail(force_fresh_vgprs ? "no fresh VGPR run"
                                  : (allow_fresh_registers
                                         ? "no liveness-safe VGPR run"
                                         : "no liveness-safe allocated VGPR run"));

  assign_probe_sgprs(selection.probe_registers, *sgprs);
  assign_probe_vgpr_run(selection.probe_registers, *workitem_vgpr,
                        previous_bb_probe_registers);
  return selection;
}

std::optional<EdgeProbeRegisterSelection>
select_vgpr_spill_probe_registers_from_analysis(
    const KernelSite &kernel, rj_code_arch_t arch, const rocjitsu::LivenessAnalysis &liveness,
    std::span<const rocjitsu::Instruction *const> probe_points,
    bool previous_bb_probe_registers, bool stable_state_sgpr,
    const Rdna4ProbeRegisters &base_registers, bool allow_fresh_registers,
    bool allow_direct_exec_fixed_counter_scratch_spills,
    bool allow_sgpr_scratch_spills,
    bool force_saved_exec_sgpr_pair,
    bool force_fresh_scratch_address_vgpr,
    const char **failure_reason = nullptr) {
  auto fail = [&](const char *reason) -> std::optional<EdgeProbeRegisterSelection> {
    if (failure_reason != nullptr)
      *failure_reason = reason;
    return std::nullopt;
  };

  if (!target_supports_vgpr_scratch_spills(arch))
    return fail("scratch-backed VGPR spills are not supported for target");

  const char *sgpr_failure_reason = nullptr;
  const bool needs_saved_exec_sgpr_pair =
      previous_bb_probe_registers || !allow_direct_exec_fixed_counter_scratch_spills ||
      force_saved_exec_sgpr_pair;
  std::optional<ProbeSgprSelection> sgprs = select_safe_probe_sgprs_from_analysis(
      kernel, arch, liveness, probe_points, previous_bb_probe_registers, stable_state_sgpr,
      base_registers, allow_fresh_registers, &sgpr_failure_reason,
      needs_saved_exec_sgpr_pair);
  bool spill_saved_exec_sgprs = false;
  std::vector<uint8_t> spilled_sgprs;
  if (!sgprs && previous_bb_probe_registers && allow_sgpr_scratch_spills) {
    const char *state_sgpr_failure_reason = nullptr;
    sgprs = select_safe_probe_sgprs_from_analysis(
        kernel, arch, liveness, probe_points,
        /*previous_bb_probe_registers=*/false, stable_state_sgpr, base_registers,
        allow_fresh_registers, &state_sgpr_failure_reason,
        /*force_saved_exec_sgpr_pair=*/false);
    if (sgprs) {
      const uint32_t sgpr_arch_limit_u32 = allocatable_sgpr_limit_for_arch(arch);
      if (sgpr_arch_limit_u32 > std::numeric_limits<uint16_t>::max())
        return fail("probe register architecture limit overflow");
      const uint16_t allocated_sgprs = static_cast<uint16_t>(
          std::min<uint32_t>(kernel.allocated_sgpr_count, sgpr_arch_limit_u32));
      std::vector<std::pair<uint16_t, uint16_t>> used_sgprs;
      if (sgprs->state_sgpr + 2 <= allocated_sgprs)
        used_sgprs.push_back({sgprs->state_sgpr, 2});
      std::optional<uint16_t> saved_exec_sgpr =
          find_allocated_run(/*count=*/2, allocated_sgprs,
                             /*even_aligned=*/true, used_sgprs);
      if (!saved_exec_sgpr)
        return fail("no allocated SGPR pair available for scratch-backed saved EXEC");
      used_sgprs.push_back({*saved_exec_sgpr, 2});
      std::optional<uint16_t> tmp_sgpr =
          find_allocated_run(/*count=*/2, allocated_sgprs,
                             /*even_aligned=*/true, used_sgprs);
      if (!tmp_sgpr)
        return fail("no allocated SGPR pair available for scratch-backed previous-BB temps");
      used_sgprs.push_back({*tmp_sgpr, 2});
      std::optional<uint16_t> scc_sgpr =
          find_allocated_run(/*count=*/1, allocated_sgprs,
                             /*even_aligned=*/false, used_sgprs);
      if (!scc_sgpr)
        return fail("no allocated SGPR available for scratch-backed SCC preservation");
      sgprs->saved_exec_sgpr = *saved_exec_sgpr;
      sgprs->tmp_sgpr = *tmp_sgpr;
      sgprs->scc_sgpr = *scc_sgpr;
      spilled_sgprs = {
          static_cast<uint8_t>(*saved_exec_sgpr),
          static_cast<uint8_t>(*saved_exec_sgpr + 1),
          static_cast<uint8_t>(*tmp_sgpr),
          static_cast<uint8_t>(*tmp_sgpr + 1),
          static_cast<uint8_t>(*scc_sgpr),
      };
      spill_saved_exec_sgprs = true;
    } else {
      sgpr_failure_reason = state_sgpr_failure_reason;
    }
  }
  if (!sgprs)
    return fail(sgpr_failure_reason != nullptr ? sgpr_failure_reason
                                               : "no liveness-safe SGPRs for probe");

  const uint32_t vgpr_arch_limit_u32 = REGISTER_SET_MAX_VGPRS;
  if (vgpr_arch_limit_u32 > std::numeric_limits<uint16_t>::max())
    return fail("probe register architecture limit overflow");
  const uint16_t vgpr_arch_limit = static_cast<uint16_t>(vgpr_arch_limit_u32);
  const uint16_t allocated_vgprs = static_cast<uint16_t>(
      std::min<uint32_t>(kernel.allocated_vgpr_count, vgpr_arch_limit));
  const uint16_t required_vgprs = required_probe_vgprs(previous_bb_probe_registers);
  if (allocated_vgprs <= required_vgprs)
    return fail("not enough allocated VGPRs for scratch-backed probe");

  std::vector<uint16_t> address_candidates;
  if (force_fresh_scratch_address_vgpr) {
    // Diagnostic override. Product forced-lane scratch probes use an allocated
    // liveness-dead address VGPR below; the wrapper restores the spilled
    // counter VGPRs but intentionally does not restore the address VGPR.
    if (!allow_fresh_registers)
      return fail("fresh scratch address VGPR growth disabled by target resource model");
    const std::vector<std::pair<uint16_t, uint16_t>> no_used_vgprs;
    std::optional<uint16_t> fresh_address =
        find_fresh_run(allocated_vgprs, /*count=*/1, vgpr_arch_limit,
                       /*even_aligned=*/false, no_used_vgprs);
    if (!fresh_address)
      return fail("no fresh scratch address VGPR for forced-lane probe");
    address_candidates.push_back(*fresh_address);
  } else {
    for (uint16_t address_vgpr = 0; address_vgpr < allocated_vgprs; ++address_vgpr) {
      if (!range_is_dead_at_all_points(liveness, probe_points, RegClass::VGPR, address_vgpr,
                                       /*count=*/1)) {
        continue;
      }
      address_candidates.push_back(address_vgpr);
    }
  }

  for (uint16_t address_vgpr : address_candidates) {

    const std::array<std::pair<uint16_t, uint16_t>, 1> used_vgprs = {
        std::pair<uint16_t, uint16_t>{address_vgpr, 1}};
    std::optional<uint16_t> workitem_vgpr =
        find_allocated_run(required_vgprs, allocated_vgprs, /*even_aligned=*/false,
                           used_vgprs);
    if (!workitem_vgpr)
      continue;

    std::vector<uint8_t> spilled_vgprs =
        probe_vgpr_run(*workitem_vgpr, previous_bb_probe_registers);
    std::optional<ProbeScratchSpillPlan> spill_plan =
        spill_saved_exec_sgprs
            ? plan_probe_scratch_spills(
                  arch, static_cast<uint8_t>(address_vgpr),
                  std::span<const uint8_t>(spilled_vgprs.data(), spilled_vgprs.size()),
                  std::span<const uint8_t>(spilled_sgprs.data(), spilled_sgprs.size()),
                  kernel.private_segment_fixed_size, kernel.wave32)
            : plan_vgpr_scratch_spills(
                  arch, static_cast<uint8_t>(address_vgpr),
                  std::span<const uint8_t>(spilled_vgprs.data(), spilled_vgprs.size()),
                  kernel.private_segment_fixed_size, kernel.wave32);
    if (!spill_plan)
      return fail("scratch spill slot planning failed");
    // Scratch growth is installed only after the descriptor and loader-visible
    // AMDGPU metadata can both be patched. If metadata is absent or not
    // in-place patchable, installation fails closed before dispatch.

    EdgeProbeRegisterSelection selection;
    selection.uses_fresh_registers =
        sgprs->uses_fresh_registers || force_fresh_scratch_address_vgpr;
    selection.scratch_spill_plan = std::move(*spill_plan);
    assign_probe_sgprs(selection.probe_registers, *sgprs);
    assign_probe_vgpr_run(selection.probe_registers, *workitem_vgpr,
                          previous_bb_probe_registers);
    return selection;
  }

  return fail("no liveness-safe scratch address VGPR");
}

} // namespace

std::optional<EdgeProbeRegisterSelection> select_edge_probe_registers_from_liveness(
    const KernelSite &kernel, rj_code_arch_t arch, const rocjitsu::LivenessAnalysis &liveness,
    std::span<const rocjitsu::Instruction *const> probe_points,
    bool previous_bb_probe_registers, bool stable_state_sgpr,
    const Rdna4ProbeRegisters &base_registers, bool allow_fresh_registers,
    bool allow_vgpr_scratch_spills, const char **failure_reason,
    bool allow_direct_exec_fixed_counter_scratch_spills,
    bool allow_sgpr_scratch_spills, bool force_fresh_sgprs,
    bool force_saved_exec_sgpr_pair, bool force_fresh_vgprs,
    bool force_fresh_scratch_address_vgpr) {
  const char *normal_failure_reason = nullptr;
  std::optional<EdgeProbeRegisterSelection> selection =
      select_safe_probe_registers_from_analysis(
          kernel, arch, liveness, probe_points, previous_bb_probe_registers, stable_state_sgpr,
          base_registers, allow_fresh_registers, &normal_failure_reason,
          force_fresh_sgprs, force_saved_exec_sgpr_pair, force_fresh_vgprs);
  if (selection)
    return selection;

  if (!allow_vgpr_scratch_spills) {
    if (failure_reason != nullptr)
      *failure_reason = normal_failure_reason;
    return std::nullopt;
  }

  const char *spill_failure_reason = nullptr;
  selection = select_vgpr_spill_probe_registers_from_analysis(
      kernel, arch, liveness, probe_points, previous_bb_probe_registers, stable_state_sgpr,
      base_registers, allow_fresh_registers,
      allow_direct_exec_fixed_counter_scratch_spills,
      allow_sgpr_scratch_spills, force_saved_exec_sgpr_pair,
      force_fresh_scratch_address_vgpr, &spill_failure_reason);
  if (selection)
    return selection;

  if (failure_reason != nullptr) {
    *failure_reason = spill_failure_reason != nullptr ? spill_failure_reason
                                                      : normal_failure_reason;
  }
  return std::nullopt;
}

namespace {

std::optional<Rdna4ProbeRegisters>
select_liveness_probe_registers(const KernelSite &kernel,
                                std::span<rocjitsu::BasicBlock *const> scope,
                                std::span<const rocjitsu::Instruction *const> probe_points,
                                bool previous_bb_probe_registers,
                                bool stable_state_sgpr,
                                const Rdna4ProbeRegisters &base_registers) {
  if (scope.empty() || probe_points.empty())
    return std::nullopt;

  try {
    rocjitsu::LivenessAnalysis liveness(scope);
    return select_liveness_probe_registers_from_analysis(
        kernel, liveness, probe_points, previous_bb_probe_registers, stable_state_sgpr,
        base_registers);
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

struct PerSiteLivenessResult {
  uint32_t kept_probe_points = 0;
  uint32_t fresh_register_probe_points = 0;
  uint32_t scratch_spill_probe_points = 0;
  uint32_t vgpr_scratch_spill_probe_points = 0;
  uint32_t sgpr_scratch_spill_probe_points = 0;
  uint32_t sgpr_scratch_spill_guarded_probe_points = 0;
  uint32_t sgpr_scratch_spill_exec_guarded_probe_points = 0;
  uint32_t direct_exec_fixed_scratch_guarded_probe_points = 0;
  uint32_t degraded_edges = 0;
  uint32_t degraded_previous_bb_branch_edges = 0;
  uint32_t degraded_previous_bb_branch_sites = 0;
  uint32_t fixed_counter_branch_edge_liveness_fallback_used = 0;
  uint32_t dropped_sites = 0;
  uint32_t dropped_edges = 0;
  uint32_t dropped_previous_bb_branch_edges = 0;
  uint32_t dropped_previous_bb_branch_sites = 0;
  EdgeSlotPolicySummary dropped_slot_summary;
};

bool failure_mentions_saved_exec_pressure(const char *reason) {
  return reason != nullptr &&
         std::string_view(reason).find("saved EXEC") != std::string_view::npos;
}

void record_liveness_result(KernelEdgeSelectionSummary &summary,
                            const PerSiteLivenessResult &result) {
  if (result.kept_probe_points != 0) {
    summary.liveness_registers = true;
    add_capped(summary.liveness_probe_points, result.kept_probe_points);
  }
  if (result.fresh_register_probe_points != 0) {
    summary.fresh_registers = true;
    add_capped(summary.fresh_register_probe_points, result.fresh_register_probe_points);
  }
  if (result.scratch_spill_probe_points != 0)
    add_capped(summary.scratch_spill_probe_points, result.scratch_spill_probe_points);
  if (result.vgpr_scratch_spill_probe_points != 0)
    add_capped(summary.vgpr_scratch_spill_probe_points,
               result.vgpr_scratch_spill_probe_points);
  if (result.sgpr_scratch_spill_probe_points != 0)
    add_capped(summary.sgpr_scratch_spill_probe_points,
               result.sgpr_scratch_spill_probe_points);
  if (result.sgpr_scratch_spill_guarded_probe_points != 0)
    add_capped(summary.sgpr_scratch_spill_disabled_by_opaque_probe_points,
               result.sgpr_scratch_spill_guarded_probe_points);
  if (result.sgpr_scratch_spill_exec_guarded_probe_points != 0)
    add_capped(summary.sgpr_scratch_spill_disabled_by_exec_condition_probe_points,
               result.sgpr_scratch_spill_exec_guarded_probe_points);
  if (result.direct_exec_fixed_scratch_guarded_probe_points != 0)
    add_capped(summary.direct_exec_fixed_scratch_disabled_by_opaque_probe_points,
               result.direct_exec_fixed_scratch_guarded_probe_points);
}

void record_opaque_fresh_register_candidate(
    KernelEdgeSelectionSummary &summary, const KernelSite &kernel, rj_code_arch_t arch,
    const rocjitsu::LivenessAnalysis &liveness,
    std::span<const rocjitsu::Instruction *const> probe_points,
    std::span<const size_t> site_indices, std::span<const EdgeSite> sites,
    bool previous_bb_probe_registers, bool stable_state_sgpr,
    const Rdna4ProbeRegisters &base_registers, std::string_view kind) {
  if (summary.opaque_instruction_count == 0 || probe_points.empty())
    return;

  std::optional<EdgeProbeRegisterSelection> candidate =
      select_safe_probe_registers_from_analysis(
          kernel, arch, liveness, probe_points, previous_bb_probe_registers,
          stable_state_sgpr, base_registers, /*allow_fresh_registers=*/true);
  if (!candidate || !candidate->uses_fresh_registers)
    return;

  const ProbeRegisterRequirements requirements =
      previous_bb_probe_registers
          ? previous_bb_probe_register_requirements(candidate->probe_registers)
          : flagless_counter_probe_register_requirements(candidate->probe_registers);
  const uint32_t probe_count =
      static_cast<uint32_t>(std::min<size_t>(probe_points.size(),
                                             std::numeric_limits<uint32_t>::max()));
  add_capped(summary.opaque_fresh_register_candidate_probe_points, probe_count);
  summary.opaque_fresh_register_candidate_required_sgprs =
      std::max(summary.opaque_fresh_register_candidate_required_sgprs,
               requirements.sgprs);
  summary.opaque_fresh_register_candidate_required_vgprs =
      std::max(summary.opaque_fresh_register_candidate_required_vgprs,
               requirements.vgprs);
  if (requirements.sgprs > kernel.allocated_sgpr_count) {
    add_capped(summary.opaque_fresh_register_candidate_sgpr_growth_probe_points,
               probe_count);
  }
  if (requirements.vgprs > kernel.allocated_vgpr_count) {
    add_capped(summary.opaque_fresh_register_candidate_vgpr_growth_probe_points,
               probe_count);
  }
  const bool sgpr_growth = requirements.sgprs > kernel.allocated_sgpr_count;
  const bool vgpr_growth = requirements.vgprs > kernel.allocated_vgpr_count;
  if (!sgpr_growth && !vgpr_growth)
    return;

  const size_t n = std::min(probe_points.size(), site_indices.size());
  for (size_t i = 0; i < n; ++i) {
    if (summary.sampled_opaque_fresh_register_candidates.size() >= 16)
      break;
    const size_t site_index = site_indices[i];
    if (site_index >= sites.size())
      continue;

    const EdgeSite &site = sites[site_index];
    OpaqueFreshRegisterCandidateSample sample;
    sample.kind = std::string(kind);
    sample.patch_text_offset = site.patch_text_offset;
    sample.required_sgprs = requirements.sgprs;
    sample.required_vgprs = requirements.vgprs;
    sample.allocated_sgprs = kernel.allocated_sgpr_count;
    sample.allocated_vgprs = kernel.allocated_vgpr_count;
    sample.sgpr_growth = sgpr_growth;
    sample.vgpr_growth = vgpr_growth;
    sample.previous_bb_probe_registers = previous_bb_probe_registers;
    sample.stable_state_sgpr = stable_state_sgpr;
    sample.slot_policy = edge_slot_policy_name(site.slot_policy);
    sample.state_sgpr = candidate->probe_registers.state_sgpr;
    sample.saved_exec_sgpr = candidate->probe_registers.saved_exec_sgpr;
    sample.tmp0_sgpr = candidate->probe_registers.tmp0_sgpr;
    sample.tmp1_sgpr = candidate->probe_registers.tmp1_sgpr;
    sample.scc_sgpr = candidate->probe_registers.scc_sgpr;
    sample.workitem_vgpr = candidate->probe_registers.workitem_vgpr;
    sample.tmp0_vgpr = candidate->probe_registers.tmp0_vgpr;
    sample.tmp1_vgpr = candidate->probe_registers.tmp1_vgpr;
    sample.tmp2_vgpr = candidate->probe_registers.tmp2_vgpr;
    if (probe_points[i] != nullptr) {
      sample.mnemonic = probe_points[i]->mnemonic();
      if (probe_points[i]->raw_encoding() != nullptr) {
        const uint32_t word_count = static_cast<uint32_t>(std::min<int>(
            probe_points[i]->size() / static_cast<int>(sizeof(uint32_t)), 4));
        sample.words.assign(probe_points[i]->raw_encoding(),
                            probe_points[i]->raw_encoding() + word_count);
      }
    }
    summary.sampled_opaque_fresh_register_candidates.push_back(std::move(sample));
  }
}

void record_scratch_spill_result(PerSiteLivenessResult &result,
                                 const ProbeScratchSpillPlan &plan) {
  ++result.scratch_spill_probe_points;
  if (!plan.vgpr_spills.empty())
    ++result.vgpr_scratch_spill_probe_points;
  if (!plan.sgpr_spills.empty())
    ++result.sgpr_scratch_spill_probe_points;
}

PerSiteLivenessResult assign_per_site_liveness_or_drop(
    const KernelSite &kernel, rj_code_arch_t arch, const rocjitsu::LivenessAnalysis &liveness,
    std::span<const rocjitsu::Instruction *const> probe_points,
    std::span<const size_t> site_indices, bool stable_state_sgpr,
    const Rdna4ProbeRegisters &base_registers,
    bool allow_fresh_registers, bool allow_fixed_counter_branch_fallback,
    bool allow_vgpr_scratch_spills, bool force_fresh_sgprs,
    bool force_fresh_vgprs,
    const char *fresh_register_growth_disabled_reason,
    EdgeSlotPolicy *edge_slots, std::vector<EdgeSite> &sites,
    KernelEdgeSelectionSummary &summary, std::string_view kind) {
  PerSiteLivenessResult result;
  std::vector<size_t> drop_indices;
  const size_t n = std::min(probe_points.size(), site_indices.size());

  for (size_t i = 0; i < n; ++i) {
    const size_t site_index = site_indices[i];
    if (site_index >= sites.size())
      continue;

    EdgeSite &site = sites[site_index];
    std::span<const rocjitsu::Instruction *const> single_point(&probe_points[i], 1);
    const bool site_previous_bb_probe_registers =
        site.slot_policy == EdgeSlotPolicyKind::PreviousBbHash;
    // An EXEC==0 edge has no active lane to observe the counter update. Try the
    // lane-0-forced fixed-counter probe first. If it needs scratch backing, the
    // wrapper spills the selected counter VGPRs and uses a liveness-dead
    // allocated address VGPR, so fully VGPR-saturated kernels can still be
    // handled when liveness proves one dead register.
    const bool force_lane0_fixed_counter =
        fixed_counter_edge_has_exec_empty_outcome(site);
    // Block-entry scratch spills still need a separate prologue-safe plan.
    // Branch scratch spills are allowed here; descriptor and AMDGPU metadata
    // patching will fail closed before dispatch if the private growth cannot
    // be installed in the loaded image.
    const bool block_entry_scratch_spills_blocked =
        allow_vgpr_scratch_spills && kind == "block";
    const bool allow_site_vgpr_scratch_spills =
        allow_vgpr_scratch_spills && kind == "branch";
    // Fixed-counter fallback does not preserve a previous-BB value, so it can
    // form scratch addresses directly from EXEC and spill only VGPR temporaries.
    const bool direct_exec_fixed_scratch_guarded_by_opaque = false;
    const bool allow_direct_exec_fixed_counter_scratch_spills =
        !direct_exec_fixed_scratch_guarded_by_opaque;
    const bool sgpr_scratch_spill_guarded_by_opaque =
        site_previous_bb_probe_registers && allow_site_vgpr_scratch_spills &&
        summary.unmodeled_opaque_instruction_count != 0;
    const bool sgpr_scratch_spill_guarded_by_exec_condition =
        site_previous_bb_probe_registers && allow_site_vgpr_scratch_spills &&
        branch_condition_depends_on_exec(site);
    // Product edge coverage is adaptive: use previous-BB SGPR scratch spills
    // for modeled scalar-controlled sites, but degrade divergent EXEC-
    // conditioned branch probes to fixed counters until that restore sequence is
    // proven under changing EXEC masks.
    const bool product_sgpr_scratch_spills_enabled = true;
    const bool allow_site_sgpr_scratch_spills =
        allow_site_vgpr_scratch_spills && !sgpr_scratch_spill_guarded_by_opaque &&
        !sgpr_scratch_spill_guarded_by_exec_condition &&
        product_sgpr_scratch_spills_enabled;
    const char *failure_reason = nullptr;
    std::optional<EdgeProbeRegisterSelection> regs =
        select_edge_probe_registers_from_liveness(
            kernel, arch, liveness, single_point, site_previous_bb_probe_registers,
            stable_state_sgpr,
            base_registers, allow_fresh_registers,
            allow_site_vgpr_scratch_spills, &failure_reason,
            allow_direct_exec_fixed_counter_scratch_spills,
            allow_site_sgpr_scratch_spills, force_fresh_sgprs,
            force_lane0_fixed_counter, force_lane0_fixed_counter || force_fresh_vgprs,
            /*force_fresh_scratch_address_vgpr=*/false);
    bool force_lane0_probe_selected = force_lane0_fixed_counter && regs.has_value();
    if (!regs && force_lane0_fixed_counter) {
      regs = select_edge_probe_registers_from_liveness(
          kernel, arch, liveness, single_point, site_previous_bb_probe_registers,
          stable_state_sgpr,
          base_registers, allow_fresh_registers,
          allow_site_vgpr_scratch_spills, &failure_reason,
          allow_direct_exec_fixed_counter_scratch_spills,
          allow_site_sgpr_scratch_spills, force_fresh_sgprs,
          /*force_saved_exec_sgpr_pair=*/false, force_fresh_vgprs);
      force_lane0_probe_selected = false;
    }
    if (regs) {
      sites[site_index].probe_registers = regs->probe_registers;
      sites[site_index].scratch_spill_plan = regs->scratch_spill_plan;
      sites[site_index].force_lane0_exec_for_fixed_counter =
          force_lane0_probe_selected;
      ++result.kept_probe_points;
      if (regs->uses_fresh_registers)
        ++result.fresh_register_probe_points;
      if (regs->scratch_spill_plan)
        record_scratch_spill_result(result, *regs->scratch_spill_plan);
      continue;
    }
    const bool saved_exec_pressure = failure_mentions_saved_exec_pressure(failure_reason);
    const char *guarded_failure_reason = nullptr;
    if (sgpr_scratch_spill_guarded_by_opaque && saved_exec_pressure) {
      ++result.sgpr_scratch_spill_guarded_probe_points;
      guarded_failure_reason = "SGPR scratch spill disabled by unmodeled opaque instructions";
    }
    if (sgpr_scratch_spill_guarded_by_exec_condition && saved_exec_pressure) {
      ++result.sgpr_scratch_spill_exec_guarded_probe_points;
      if (guarded_failure_reason == nullptr)
        guarded_failure_reason = "SGPR scratch spill disabled for EXEC-conditioned branch";
    }

    if (allow_fixed_counter_branch_fallback && edge_slots != nullptr &&
        kind == "branch" && site.slot_policy == EdgeSlotPolicyKind::PreviousBbHash) {
      const char *fallback_failure_reason = nullptr;
      const bool force_lane0_fixed_counter_fallback =
          branch_condition_depends_on_exec(site);
      std::optional<EdgeProbeRegisterSelection> fallback_regs =
          select_edge_probe_registers_from_liveness(
              kernel, arch, liveness, single_point,
              /*previous_bb_probe_registers=*/false, stable_state_sgpr, base_registers,
              allow_fresh_registers, allow_site_vgpr_scratch_spills,
              &fallback_failure_reason,
              allow_direct_exec_fixed_counter_scratch_spills,
              allow_site_sgpr_scratch_spills, force_fresh_sgprs,
              force_lane0_fixed_counter_fallback,
              force_lane0_fixed_counter_fallback || force_fresh_vgprs,
              /*force_fresh_scratch_address_vgpr=*/false);
      bool force_lane0_fallback_selected =
          force_lane0_fixed_counter_fallback && fallback_regs.has_value();
      if (!fallback_regs && force_lane0_fixed_counter_fallback) {
        fallback_regs = select_edge_probe_registers_from_liveness(
            kernel, arch, liveness, single_point,
            /*previous_bb_probe_registers=*/false, stable_state_sgpr, base_registers,
            allow_fresh_registers, allow_site_vgpr_scratch_spills,
            &fallback_failure_reason,
            allow_direct_exec_fixed_counter_scratch_spills,
            allow_site_sgpr_scratch_spills, force_fresh_sgprs,
            /*force_saved_exec_sgpr_pair=*/false, force_fresh_vgprs);
        force_lane0_fallback_selected = false;
      }
      if (fallback_regs) {
        const uint32_t edge_count = edge_count_for_site(site);
        const bool previous_bb_branch_site =
            kind == "branch" && site.slot_policy == EdgeSlotPolicyKind::PreviousBbHash;
        std::optional<EdgeSlotAssignment> fallback_slot =
            edge_slots->assign_branch_terminator_fallback(
                EdgeSlotPolicyKind::FixedCounter, edge_count, site.bb_id,
                site.fallthrough_bb_id);
        if (fallback_slot) {
          add_site_slot_summary(result.dropped_slot_summary, site);
          site.slot_policy = fallback_slot->policy;
          site.fixed_slot = fallback_slot->primary_slot;
          site.fallthrough_slot = fallback_slot->secondary_slot;
          site.fixed_slot_collisions = fallback_slot->fixed_slot_collisions;
          site.probe_registers = fallback_regs->probe_registers;
          site.scratch_spill_plan = fallback_regs->scratch_spill_plan;
          site.force_lane0_exec_for_fixed_counter =
              force_lane0_fallback_selected;
          ++result.kept_probe_points;
          if (fallback_regs->uses_fresh_registers)
            ++result.fresh_register_probe_points;
          if (fallback_regs->scratch_spill_plan)
            record_scratch_spill_result(result, *fallback_regs->scratch_spill_plan);
          result.degraded_edges += edge_count;
          if (previous_bb_branch_site) {
            result.degraded_previous_bb_branch_edges += edge_count;
            ++result.degraded_previous_bb_branch_sites;
            result.fixed_counter_branch_edge_liveness_fallback_used += edge_count;
          }
          const std::string reason = contextual_liveness_failure_reason(
              guarded_failure_reason != nullptr
                  ? guarded_failure_reason
                  : (failure_reason != nullptr
                         ? failure_reason
                         : "previous-BB liveness rejected probe registers"),
              fresh_register_growth_disabled_reason);
          record_degradation_count(summary, kind, reason, edge_count);
          continue;
        }
        fallback_failure_reason = "fixed-counter fallback slot assignment failed";
      }
      if (fallback_failure_reason != nullptr &&
          direct_exec_fixed_scratch_guarded_by_opaque) {
        ++result.direct_exec_fixed_scratch_guarded_probe_points;
        failure_reason = "direct-EXEC fixed-counter scratch disabled by opaque instructions";
      } else if (fallback_failure_reason != nullptr) {
        failure_reason = fallback_failure_reason;
      }
    }

    if (block_entry_scratch_spills_blocked) {
      failure_reason =
          "scratch-backed block-entry probes require loader-visible private segment growth";
    }
    const std::string reason =
        contextual_liveness_failure_reason(failure_reason,
                                           fresh_register_growth_disabled_reason);
    record_skip_sample(summary, kind, sites[site_index].patch_text_offset, reason,
                       probe_points[i], SkipSamplePolicy::DistinctSite);
    if (kind == "branch" &&
        sites[site_index].slot_policy == EdgeSlotPolicyKind::PreviousBbHash) {
      result.dropped_previous_bb_branch_edges += edge_count_for_site(sites[site_index]);
      ++result.dropped_previous_bb_branch_sites;
    }
    result.dropped_edges += edge_count_for_site(sites[site_index]);
    add_site_slot_summary(result.dropped_slot_summary, sites[site_index]);
    drop_indices.push_back(site_index);
  }

  std::sort(drop_indices.rbegin(), drop_indices.rend());
  for (size_t site_index : drop_indices) {
    if (site_index < sites.size())
      sites.erase(sites.begin() + site_index);
  }

  result.dropped_sites = static_cast<uint32_t>(
      std::min<size_t>(drop_indices.size(), std::numeric_limits<uint32_t>::max()));
  return result;
}

PerSiteLivenessResult drop_liveness_sites(std::span<const size_t> site_indices,
                                          std::vector<EdgeSite> &sites,
                                          KernelEdgeSelectionSummary &summary,
                                          std::string_view kind,
                                          std::string_view reason) {
  PerSiteLivenessResult result;
  std::vector<size_t> drop_indices;

  for (size_t site_index : site_indices) {
    if (site_index >= sites.size())
      continue;
    record_skip_sample(summary, kind, sites[site_index].patch_text_offset, reason,
                       nullptr, SkipSamplePolicy::DistinctSite);
    if (kind == "branch" &&
        sites[site_index].slot_policy == EdgeSlotPolicyKind::PreviousBbHash) {
      result.dropped_previous_bb_branch_edges += edge_count_for_site(sites[site_index]);
      ++result.dropped_previous_bb_branch_sites;
    }
    result.dropped_edges += edge_count_for_site(sites[site_index]);
    add_site_slot_summary(result.dropped_slot_summary, sites[site_index]);
    drop_indices.push_back(site_index);
  }

  std::sort(drop_indices.rbegin(), drop_indices.rend());
  for (size_t site_index : drop_indices) {
    if (site_index < sites.size())
      sites.erase(sites.begin() + site_index);
  }

  result.dropped_sites = static_cast<uint32_t>(
      std::min<size_t>(drop_indices.size(), std::numeric_limits<uint32_t>::max()));
  return result;
}

const rocjitsu::BasicBlock *
find_block_at_offset(const std::vector<std::unique_ptr<rocjitsu::BasicBlock>> &blocks,
                     uint64_t offset) {
  for (const auto &block : blocks) {
    if (block != nullptr && block->start_offset() == offset)
      return block.get();
  }
  return nullptr;
}

std::vector<const rocjitsu::BasicBlock *>
collect_reachable_ordered(const rocjitsu::BasicBlock *entry) {
  std::vector<const rocjitsu::BasicBlock *> stack = {entry};
  std::unordered_set<const rocjitsu::BasicBlock *> reachable;
  while (!stack.empty()) {
    const rocjitsu::BasicBlock *block = stack.back();
    stack.pop_back();
    if (block == nullptr || !reachable.insert(block).second)
      continue;
    for (const rocjitsu::BasicBlock *succ : block->successors())
      stack.push_back(succ);
  }

  std::vector<const rocjitsu::BasicBlock *> ordered(reachable.begin(), reachable.end());
  std::sort(ordered.begin(), ordered.end(), [](const auto *lhs, const auto *rhs) {
    return lhs->start_offset() < rhs->start_offset();
  });
  return ordered;
}

std::vector<rocjitsu::BasicBlock *>
mutable_liveness_scope(const std::vector<const rocjitsu::BasicBlock *> &ordered) {
  std::vector<rocjitsu::BasicBlock *> scope;
  scope.reserve(ordered.size());
  for (const rocjitsu::BasicBlock *block : ordered)
    scope.push_back(const_cast<rocjitsu::BasicBlock *>(block));
  return scope;
}

} // namespace

const char *arch_name(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA2:
    return "CDNA2/gfx90a";
  case ROCJITSU_CODE_ARCH_CDNA3:
    return "CDNA3/gfx94x";
  case ROCJITSU_CODE_ARCH_CDNA4:
    return "CDNA4/gfx950";
  case ROCJITSU_CODE_ARCH_RDNA1:
    return "RDNA1/gfx101x";
  case ROCJITSU_CODE_ARCH_RDNA2:
    return "RDNA2/gfx103x";
  case ROCJITSU_CODE_ARCH_RDNA3:
    return "RDNA3/gfx110x";
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return "RDNA3.5/gfx115x";
  case ROCJITSU_CODE_ARCH_RDNA4:
    return "RDNA4/gfx120x";
  default:
    return "unsupported";
  }
}

uint32_t stable_bb_id(std::string_view kernel_name, uint64_t block_text_offset) {
  uint32_t hash = 2166136261u;
  auto mix = [&](uint8_t byte) {
    hash ^= byte;
    hash *= 16777619u;
  };
  for (char c : kernel_name)
    mix(static_cast<uint8_t>(c));
  for (uint32_t i = 0; i < 8; ++i)
    mix(static_cast<uint8_t>((block_text_offset >> (i * 8)) & 0xffu));
  return hash == 0 ? 1u : hash;
}

uint32_t stable_edge_id(std::string_view kernel_name, uint64_t pred_text_offset,
                        uint64_t target_text_offset) {
  uint32_t hash = 2166136261u;
  auto mix = [&](uint8_t byte) {
    hash ^= byte;
    hash *= 16777619u;
  };
  for (char c : kernel_name)
    mix(static_cast<uint8_t>(c));
  for (uint32_t i = 0; i < 8; ++i)
    mix(static_cast<uint8_t>((pred_text_offset >> (i * 8)) & 0xffu));
  for (uint32_t i = 0; i < 8; ++i)
    mix(static_cast<uint8_t>((target_text_offset >> (i * 8)) & 0xffu));
  return hash == 0 ? 1u : hash;
}

std::vector<KernelSite> find_kernel_sites(std::span<const uint8_t> image) {
  std::vector<KernelSite> sites;
  auto sections = read_sections(image);
  if (!sections)
    return sites;
  const uint32_t elf_mach = read_elf_mach(image).value_or(0);
  const bool sgpr_descriptor_effective =
      descriptor_sgpr_count_is_effective(elf_mach);
  const std::unordered_map<std::string, KernelMetadataResourceInfo> metadata_resources =
      read_kernel_metadata_resources(image, *sections);

  for (const auto &symtab : sections->sections) {
    if (symtab.sh_type != rocjitsu::SHT_SYMTAB && symtab.sh_type != rocjitsu::SHT_DYNSYM)
      continue;
    if (symtab.sh_entsize != sizeof(rocjitsu::Elf64_Sym) ||
        symtab.sh_link >= sections->sections.size())
      continue;
    const auto &strtab = sections->sections[symtab.sh_link];
    if (symtab.sh_offset > image.size() || symtab.sh_size > image.size() - symtab.sh_offset ||
        strtab.sh_offset > image.size() || strtab.sh_size > image.size() - strtab.sh_offset)
      continue;
    const auto *strings = reinterpret_cast<const char *>(image.data() + strtab.sh_offset);
    const size_t string_size = static_cast<size_t>(strtab.sh_size);
    const size_t count = symtab.sh_size / sizeof(rocjitsu::Elf64_Sym);
    for (size_t i = 0; i < count; ++i) {
      auto sym = read_struct<rocjitsu::Elf64_Sym>(image, symtab.sh_offset +
                                                             i * sizeof(rocjitsu::Elf64_Sym));
      if (!sym || sym->st_name >= string_size)
        continue;
      const char *sym_name = strings + sym->st_name;
      const size_t max_len = string_size - sym->st_name;
      std::string name(sym_name, strnlen(sym_name, max_len));
      if (name.size() <= 3 || name.substr(name.size() - 3) != ".kd")
        continue;

      auto desc_file_offset = va_to_file_offset(*sections, sym->st_value);
      if (!desc_file_offset)
        continue;
      auto desc = read_struct<KD>(image, *desc_file_offset);
      if (!desc)
        continue;
      const int64_t entry_va =
          static_cast<int64_t>(sym->st_value) + desc->kernel_code_entry_byte_offset;
      if (entry_va < 0 || static_cast<uint64_t>(entry_va) < sections->text->sh_addr ||
          static_cast<uint64_t>(entry_va) >= sections->text->sh_addr + sections->text->sh_size)
        continue;

      KernelSite site;
      site.name = name.substr(0, name.size() - 3);
      site.descriptor_file_offset = *desc_file_offset;
      site.entry_text_offset = static_cast<uint64_t>(entry_va) - sections->text->sh_addr;
      site.elf_mach = elf_mach;
      site.wave32 = AMDHSA_BITS_GET(desc->kernel_code_properties,
                                    kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) != 0;
      const uint32_t vgpr_granularity = site.wave32 ? 8 : 4;
      site.allocated_vgpr_count = granulated_to_register_count(
          AMDHSA_BITS_GET(desc->compute_pgm_rsrc1,
                          kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
          vgpr_granularity);
      site.descriptor_sgpr_count = granulated_to_register_count(
          AMDHSA_BITS_GET(desc->compute_pgm_rsrc1,
                          kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT),
          /*granularity=*/8);
      const auto metadata = metadata_resources.find(site.name);
      const KernelMetadataResourceInfo *metadata_info =
          metadata == metadata_resources.end() ? nullptr : &metadata->second;
      if (metadata_info != nullptr && metadata_info->sgpr_count) {
        site.has_metadata_sgpr_count = true;
        site.metadata_sgpr_count = *metadata_info->sgpr_count;
      }
      site.descriptor_sgpr_count_effective = sgpr_descriptor_effective;
      site.allocated_sgpr_count =
          planning_sgpr_count(site.descriptor_sgpr_count, metadata_info,
                              site.descriptor_sgpr_count_effective);
      // LLVM intentionally encodes zero SGPR granules on gfx10+; until DBI
      // grows and patches the loader-visible metadata contract, SGPR growth
      // must be backed by the metadata count rather than the ignored
      // descriptor granules.
      site.fresh_sgpr_growth_supported =
          site.descriptor_sgpr_count_effective || site.has_metadata_sgpr_count;
      site.private_segment_fixed_size = desc->private_segment_fixed_size;
      sites.push_back(std::move(site));
    }
  }

  std::sort(sites.begin(), sites.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.descriptor_file_offset < rhs.descriptor_file_offset;
  });
  sites.erase(std::unique(sites.begin(), sites.end(),
                          [](const auto &lhs, const auto &rhs) {
                            return lhs.descriptor_file_offset == rhs.descriptor_file_offset;
                          }),
              sites.end());
  return sites;
}

std::vector<EntryProbeRegisterSelection>
select_entry_probe_registers(const rocjitsu::AmdGpuCodeObject &co, rj_code_arch_t arch,
                             std::span<const KernelSite> kernels,
                             bool masked_entry_probe_registers,
                             bool stable_state_sgpr,
                             const Rdna4ProbeRegisters &base_registers) {
  std::vector<EntryProbeRegisterSelection> selections;
  if (kernels.empty())
    return selections;

  std::vector<uint64_t> entry_offsets;
  entry_offsets.reserve(kernels.size());
  for (const KernelSite &kernel : kernels)
    entry_offsets.push_back(kernel.entry_text_offset);

  auto decoder = create_cfg_planning_decoder(arch);
  if (decoder == nullptr)
    return selections;

  std::vector<std::unique_ptr<rocjitsu::BasicBlock>> blocks;
  try {
    blocks = rocjitsu::BasicBlock::build(co, *decoder, entry_offsets);
  } catch (const std::exception &) {
    return selections;
  }

  for (const KernelSite &kernel : kernels) {
    const rocjitsu::BasicBlock *entry = find_block_at_offset(blocks, kernel.entry_text_offset);
    if (entry == nullptr)
      continue;
    const rocjitsu::Instruction *first = first_instruction(*entry);
    if (first == nullptr)
      continue;

    std::vector<const rocjitsu::BasicBlock *> ordered = collect_reachable_ordered(entry);
    std::vector<rocjitsu::BasicBlock *> liveness_scope = mutable_liveness_scope(ordered);
    std::optional<RegisterSet> live_in = entry_probe_live_in(entry, liveness_scope);
    if (!live_in)
      continue;
    std::optional<Rdna4ProbeRegisters> regs =
        select_probe_registers_from_live_set(kernel, *live_in, masked_entry_probe_registers,
                                             stable_state_sgpr, base_registers);
    if (!regs)
      continue;

    EntryProbeRegisterSelection selection;
    selection.kernel_name = kernel.name;
    selection.probe_registers = *regs;
    selection.liveness_probe_points = 1;
    selections.push_back(selection);
  }

  return selections;
}

InstrumentationPlan select_edge_sites(const rocjitsu::AmdGpuCodeObject &co, rj_code_arch_t arch,
                                      std::span<const KernelSite> kernels,
                                      const InstrumentationPlanOptions &options) {
  InstrumentationPlan result;
  if (kernels.empty())
    return result;

  std::vector<uint64_t> entry_offsets;
  entry_offsets.reserve(kernels.size());
  for (const KernelSite &kernel : kernels)
    entry_offsets.push_back(kernel.entry_text_offset);

  auto decoder = create_cfg_planning_decoder(arch);
  if (decoder == nullptr) {
    result.failure_reason = "no CFG decoder";
    if (options.verbose)
      fprintf(stderr, "rocjitsu-afl: no CFG decoder for %s\n", arch_name(arch));
    return result;
  }
  std::vector<std::unique_ptr<rocjitsu::BasicBlock>> blocks;
  try {
    blocks = rocjitsu::BasicBlock::build(co, *decoder, entry_offsets);
  } catch (const std::exception &e) {
    result.failure_reason = e.what();
    if (options.verbose) {
      fprintf(stderr, "rocjitsu-afl: CFG decode failed for %s: %s\n", arch_name(arch), e.what());
    }
    return result;
  }

  std::unordered_set<uint64_t> patched_blocks;
  std::unordered_set<uint64_t> patched_block_patch_offsets;
  std::unordered_set<uint64_t> patched_branch_offsets;
  FixedEdgeSlotTracker fixed_slot_tracker;

  for (const KernelSite &kernel : kernels) {
    EdgeSlotPolicy edge_slots(options.block_entry_slot_policy,
                              options.branch_terminator_slot_policy,
                              &fixed_slot_tracker);
    const rocjitsu::BasicBlock *entry = nullptr;
    for (const auto &block : blocks) {
      if (block->start_offset() == kernel.entry_text_offset) {
        entry = block.get();
        break;
      }
    }
    if (entry == nullptr)
      continue;

    std::vector<const rocjitsu::BasicBlock *> stack = {entry};
    std::unordered_set<const rocjitsu::BasicBlock *> reachable;
    while (!stack.empty()) {
      const rocjitsu::BasicBlock *block = stack.back();
      stack.pop_back();
      if (block == nullptr || !reachable.insert(block).second)
        continue;
      for (const rocjitsu::BasicBlock *succ : block->successors())
        stack.push_back(succ);
    }

    std::vector<const rocjitsu::BasicBlock *> ordered(reachable.begin(), reachable.end());
    std::sort(ordered.begin(), ordered.end(), [](const auto *lhs, const auto *rhs) {
      return lhs->start_offset() < rhs->start_offset();
    });
    std::vector<rocjitsu::BasicBlock *> liveness_scope;
    liveness_scope.reserve(ordered.size());
    for (const rocjitsu::BasicBlock *block : ordered)
      liveness_scope.push_back(const_cast<rocjitsu::BasicBlock *>(block));

    KernelEdgeSelectionSummary summary;
    summary.kernel_name = kernel.name;
    summary.self_contained_probe = options.self_contained_edge_probes;
    summary.reachable_blocks = static_cast<uint32_t>(
        std::min<size_t>(ordered.size(), std::numeric_limits<uint32_t>::max()));
    record_opaque_instruction_diagnostics(summary, liveness_scope);
    EdgeSlotPolicySummary liveness_dropped_slot_summary;
    if (options.block_entry_site_limit == 0) {
      summary.block_candidates =
          ordered.size() > 1 ? static_cast<uint32_t>(ordered.size() - 1) : 0;
      summary.skipped_limit = summary.block_candidates;
    } else {
      std::vector<const rocjitsu::Instruction *> liveness_probe_points;
      std::vector<size_t> liveness_site_indices;
      bool liveness_requires_previous_bb_registers = false;
      for (const rocjitsu::BasicBlock *block : ordered) {
        if (block == nullptr || block == entry)
          continue;
        ++summary.block_candidates;
        const uint32_t bb_id = stable_bb_id(kernel.name, block->start_offset());
        if (result.sites.size() >= options.block_entry_site_limit) {
          ++summary.skipped_limit;
          record_skip_sample(summary, "block", block->start_offset(), "edge site limit");
          if (options.verbose) {
            fprintf(stderr,
                    "rocjitsu-afl: skipping bb bb_id=0x%x kernel=%s block=%llu "
                    "reason=edge site limit\n",
                    bb_id, kernel.name.c_str(),
                    static_cast<unsigned long long>(block->start_offset()));
          }
          continue;
        }
        const rocjitsu::Instruction *first = first_instruction(*block);
        if (first == nullptr) {
          ++summary.skipped_unsafe;
          record_skip_sample(summary, "block", block->start_offset(),
                             "no block-entry instruction");
          if (options.verbose) {
            fprintf(stderr,
                    "rocjitsu-afl: skipping bb bb_id=0x%x kernel=%s block=%llu "
                    "reason=no block-entry instruction\n",
                    bb_id, kernel.name.c_str(),
                    static_cast<unsigned long long>(block->start_offset()));
          }
          continue;
        }

        std::string_view skip_reason;
        BlockEntryPatchSkip skip;
        std::optional<BlockEntryPatchPoint> patch_point =
            select_block_entry_patch_point(*block, *first, &skip_reason, &skip);
        if (!patch_point) {
          ++summary.skipped_unsafe;
          record_skip_sample(summary, "block", skip.text_offset, skip.reason,
                             skip.instruction);
          if (options.verbose) {
            fprintf(stderr,
                    "rocjitsu-afl: skipping bb bb_id=0x%x kernel=%s block=%llu "
                    "patch=%llu reason=%.*s\n",
                    bb_id, kernel.name.c_str(),
                    static_cast<unsigned long long>(block->start_offset()),
                    static_cast<unsigned long long>(skip.text_offset),
                    static_cast<int>(skip_reason.size()), skip_reason.data());
          }
          continue;
        }
        const rocjitsu::Instruction *patch_inst = patch_point->instruction;
        if (patch_inst == nullptr || patch_inst->raw_encoding() == nullptr ||
            patch_inst->size() < 4 || patch_inst->size() % sizeof(uint32_t) != 0) {
          ++summary.skipped_unsafe;
          record_skip_sample(summary, "block", patch_point->text_offset,
                             "no relocatable block-entry instruction", patch_inst);
          if (options.verbose) {
            fprintf(stderr,
                    "rocjitsu-afl: skipping bb bb_id=0x%x kernel=%s block=%llu "
                    "patch=%llu reason=no relocatable block-entry instruction\n",
                    bb_id, kernel.name.c_str(),
                    static_cast<unsigned long long>(block->start_offset()),
                    static_cast<unsigned long long>(patch_point->text_offset));
          }
          continue;
        }
        const bool conditional_entry_branch = is_conditional_direct_branch(*patch_inst);
        uint64_t conditional_entry_target = 0;
        if (conditional_entry_branch) {
          const int64_t target =
              static_cast<int64_t>(patch_point->text_offset) +
              static_cast<int64_t>(patch_inst->size()) +
              static_cast<int64_t>(*patch_inst->branch_offset_bytes());
          if (target < 0) {
            ++summary.skipped_unsafe;
            record_skip_sample(summary, "block", patch_point->text_offset,
                               "entry conditional branch target is negative", patch_inst);
            continue;
          }
          conditional_entry_target = static_cast<uint64_t>(target);
        }
        if (!patched_blocks.insert(block->start_offset()).second)
          continue;
        if (!patched_block_patch_offsets.insert(patch_point->text_offset).second)
          continue;

        std::optional<EdgeSlotAssignment> slot_assignment =
            edge_slots.assign_block_entry(bb_id);
        if (!slot_assignment) {
          ++summary.skipped_fixed_slot;
          record_skip_sample(summary, "block", block->start_offset(),
                             "edge slot policy could not assign a slot", first);
          if (options.verbose) {
            fprintf(stderr,
                    "rocjitsu-afl: skipping bb bb_id=0x%x kernel=%s block=%llu "
                    "reason=edge slot policy could not assign a slot\n",
                    bb_id, kernel.name.c_str(),
                    static_cast<unsigned long long>(block->start_offset()));
          }
          continue;
        }

        EdgeSite site;
        site.kind = conditional_entry_branch ? EdgePatchKind::ConditionalBlockEntry
                                             : EdgePatchKind::BlockEntry;
        site.kernel_name = kernel.name;
        site.pred_text_offset =
            block->predecessors().empty() ? 0 : block->predecessors().front()->start_offset();
        site.block_text_offset = block->start_offset();
        site.patch_text_offset = patch_point->text_offset;
        site.return_text_offset = conditional_entry_branch
                                      ? conditional_entry_target
                                      : patch_point->text_offset + patch_inst->size();
        site.first_inst_size = static_cast<uint32_t>(patch_inst->size());
        site.predecessor_count = static_cast<uint32_t>(block->predecessors().size());
        site.bb_id = bb_id;
        site.fallthrough_bb_id = bb_id;
        site.slot_policy = slot_assignment->policy;
        site.probe_registers = options.probe_registers;
        site.fixed_slot = slot_assignment->primary_slot;
        site.fallthrough_slot = slot_assignment->primary_slot;
        site.fixed_slot_collisions = slot_assignment->fixed_slot_collisions;
        site.branch_opcode = conditional_entry_branch ? patch_inst->opcode() : 0;
        site.self_contained_probe = options.self_contained_edge_probes;
        liveness_probe_points.push_back(patch_inst);
        liveness_site_indices.push_back(result.sites.size());
        // Entry-backed fixed block counters still mask to the first active lane.
        // They therefore need the saved-EXEC and scalar-temp registers reserved
        // just like previous-BB probes; treating them as flagless counters can
        // alias the stable state pointer SGPR and turn the atomic address bogus.
        liveness_requires_previous_bb_registers |=
            site.slot_policy == EdgeSlotPolicyKind::PreviousBbHash ||
            (site.slot_policy == EdgeSlotPolicyKind::FixedCounter &&
             edge_patch_kind_is_block_entry(site.kind) && !site.self_contained_probe);
        result.sites.push_back(std::move(site));
        ++summary.block_selected;
      }
      if (options.liveness_registers && !liveness_site_indices.empty()) {
        std::span<const rocjitsu::Instruction *const> probe_points(liveness_probe_points.data(),
                                                                   liveness_probe_points.size());
        std::optional<Rdna4ProbeRegisters> regs;
        bool liveness_accounted = false;
        const char *strict_liveness_failure = nullptr;
        if (!options.require_liveness_registers) {
          regs = select_liveness_probe_registers(kernel, liveness_scope, probe_points,
                                                 liveness_requires_previous_bb_registers,
                                                 !options.self_contained_edge_probes,
                                                 options.probe_registers);
        } else {
          try {
            rocjitsu::LivenessAnalysis liveness(liveness_scope);
            // Opaque instructions are modeled well enough for CFG sizing and
            // allocated-register liveness, but not yet well enough to prove
            // descriptor growth. Keep fresh registers off until the VOPD model is
            // opcode-complete or the DBI side can validate fresh ranges another way.
            const bool allow_fresh_registers =
                summary.opaque_instruction_count == 0 ||
                options.allow_opaque_fresh_registers;
            if (!allow_fresh_registers) {
              add_capped(summary.fresh_register_growth_disabled_by_opaque_probe_points,
                         probe_points.size());
              record_opaque_fresh_register_candidate(
                  summary, kernel, arch, liveness, probe_points,
                  std::span<const size_t>(liveness_site_indices.data(),
                                          liveness_site_indices.size()),
                  std::span<const EdgeSite>(result.sites.data(), result.sites.size()),
                  liveness_requires_previous_bb_registers,
                  !options.self_contained_edge_probes, options.probe_registers,
                  "block");
            }
            std::optional<EdgeProbeRegisterSelection> safe_regs =
                select_safe_probe_registers_from_analysis(
                    kernel, arch, liveness, probe_points, liveness_requires_previous_bb_registers,
                    !options.self_contained_edge_probes, options.probe_registers,
                    allow_fresh_registers, /*failure_reason=*/nullptr,
                    options.force_fresh_sgprs,
                    /*force_saved_exec_sgpr_pair=*/false,
                    options.force_fresh_vgprs);
            if (safe_regs) {
              regs = safe_regs->probe_registers;
              if (safe_regs->uses_fresh_registers) {
                summary.fresh_registers = true;
                add_capped(summary.fresh_register_probe_points, probe_points.size());
              }
            }
            if (!regs) {
              const PerSiteLivenessResult per_site = assign_per_site_liveness_or_drop(
                  kernel, arch, liveness, probe_points, liveness_site_indices,
                  !options.self_contained_edge_probes, options.probe_registers,
                  allow_fresh_registers,
                  /*allow_fixed_counter_branch_fallback=*/false,
                  options.allow_vgpr_scratch_spills, options.force_fresh_sgprs,
                  options.force_fresh_vgprs,
                  allow_fresh_registers ? nullptr
                                        : "fresh register growth disabled by opaque instructions",
                  /*edge_slots=*/nullptr, result.sites, summary, "block");
              subtract_capped(summary.block_selected, per_site.dropped_sites);
              summary.skipped_liveness += per_site.dropped_sites;
              add_slot_summary(liveness_dropped_slot_summary,
                               per_site.dropped_slot_summary);
              record_liveness_result(summary, per_site);
              liveness_accounted = true;
            }
          } catch (const std::exception &) {
            strict_liveness_failure = "liveness analysis failed";
          }
        }
        if (regs) {
          for (size_t site_index : liveness_site_indices)
            result.sites[site_index].probe_registers = *regs;
          summary.liveness_registers = true;
          summary.liveness_probe_points = static_cast<uint32_t>(std::min<size_t>(
              liveness_probe_points.size(), std::numeric_limits<uint32_t>::max()));
          liveness_accounted = true;
        } else if (options.require_liveness_registers && !liveness_accounted) {
          const PerSiteLivenessResult dropped = drop_liveness_sites(
              liveness_site_indices, result.sites, summary, "block",
              strict_liveness_failure != nullptr ? strict_liveness_failure
                                                 : "no liveness-safe probe registers");
          subtract_capped(summary.block_selected, dropped.dropped_sites);
          summary.skipped_liveness += dropped.dropped_sites;
          add_slot_summary(liveness_dropped_slot_summary,
                           dropped.dropped_slot_summary);
        }
      }
    }

    if (options.branch_edge_slots) {
      const bool previous_bb_branch_policy =
          options.branch_terminator_slot_policy == EdgeSlotPolicyKind::PreviousBbHash;
      const BranchEdgeBudgetSelection branch_budget_selection =
          select_branch_edge_budget(
              std::span<const rocjitsu::BasicBlock *const>(ordered.data(),
                                                           ordered.size()),
              options, previous_bb_branch_policy);
      const PreviousBbBranchAggregateBudget &branch_budget =
          branch_budget_selection.budget;
      summary.branch_edge_candidate_edges =
          branch_budget_selection.branch_candidate_edges;
      summary.previous_bb_branch_edge_candidate_edges =
          branch_budget.candidate_edges;
      summary.previous_bb_branch_site_candidate_sites = branch_budget.candidate_sites;
      summary.branch_edge_budget = branch_budget.edge_budget;
      summary.previous_bb_branch_site_budget = branch_budget.site_budget;
      summary.branch_edge_budget_reason =
          std::string(branch_budget_selection.edge_reason);
      summary.previous_bb_branch_site_budget_reason =
          std::string(branch_budget_selection.site_reason);
      uint32_t fixed_counter_budget_fallback_edges = 0;

      std::vector<const rocjitsu::Instruction *> liveness_probe_points;
      std::vector<size_t> liveness_site_indices;
      bool liveness_requires_previous_bb_registers = false;
      if (previous_bb_branch_policy) {
        summary.previous_bb_branch_edge_over_budget =
            branch_budget.edge_over_budget;
        summary.previous_bb_branch_site_over_budget =
            branch_budget.site_over_budget;
      }
      summary.fixed_counter_branch_edge_fallback_budget =
          branch_budget.fixed_counter_fallback_budget;
      summary.fixed_counter_branch_edge_fallback_budget_reason =
          std::string(branch_budget_selection.fixed_counter_fallback_reason);
      summary.previous_bb_branch_aggregate_limit_kind =
          previous_bb_branch_aggregate_limit_kind_name(branch_budget.limit_kind());
      summary.previous_bb_branch_aggregate_safety =
          previous_bb_aggregate_safety_name(branch_budget);
      summary.previous_bb_branch_aggregate_safety_reason =
          previous_bb_aggregate_safety_reason(branch_budget);
      for (const rocjitsu::BasicBlock *block : ordered) {
        if (block == nullptr)
          continue;
        const rocjitsu::Instruction *term = block->terminator();
        const bool unconditional = term != nullptr && is_unconditional_direct_branch(*term);
        const bool conditional = term != nullptr && is_conditional_direct_branch(*term);
        if (!unconditional && !conditional) {
          ++summary.skipped_branch_unsafe;
          record_skip_sample(summary, "branch", branch_terminator_skip_offset(*block, term),
                             branch_terminator_skip_reason(*block, term), term);
          continue;
        }
        ++summary.branch_candidates;
        const uint32_t site_edge_count = conditional ? 2 : 1;
        // Fixed-counter fallback preserves branch feedback without writing
        // previous-BB state, so do not charge those logical edges against the
        // previous-BB edge budget.
        const bool over_previous_bb_branch_budget =
            branch_budget.edge_budget_exhausted(
                summary.previous_bb_branch_edges_selected, site_edge_count);
        const bool over_previous_bb_branch_site_budget =
            branch_budget.previous_bb_site_budget_exhausted(
                summary.previous_bb_branch_sites_selected);
        const bool fixed_counter_budget_fallback_available =
            branch_budget.fixed_counter_fallback_available(
                fixed_counter_budget_fallback_edges, site_edge_count);
        const bool exec_conditioned_previous_bb_branch =
            previous_bb_branch_policy && conditional &&
            branch_opcode_depends_on_exec(term->opcode());
        // s_cbranch_execz/execnz sites dispatch on the live EXEC mask while the
        // previous-BB probe also narrows/restores EXEC. A forced rocBLAS
        // MT128x64 run faults in this path, so keep branch feedback but use the
        // smaller fixed-counter probe until the divergent-mask interaction is
        // proven safe.
        const bool use_fixed_counter_exec_condition_fallback =
            exec_conditioned_previous_bb_branch &&
            fixed_counter_budget_fallback_available;
        const bool use_fixed_counter_budget_fallback =
            !exec_conditioned_previous_bb_branch &&
            (over_previous_bb_branch_budget || over_previous_bb_branch_site_budget) &&
            fixed_counter_budget_fallback_available;
        if (!exec_conditioned_previous_bb_branch &&
            (over_previous_bb_branch_budget || over_previous_bb_branch_site_budget) &&
            !use_fixed_counter_budget_fallback) {
          ++summary.skipped_branch_limit;
          record_skip_sample(summary, "branch", block->end_offset() - term->size(),
                             over_previous_bb_branch_site_budget
                                 ? "previous-BB branch site limit"
                                 : "branch edge site limit",
                             term);
          continue;
        }
        if (exec_conditioned_previous_bb_branch &&
            !use_fixed_counter_exec_condition_fallback &&
            !use_fixed_counter_budget_fallback) {
          ++summary.skipped_branch_limit;
          record_skip_sample(
              summary, "branch", block->end_offset() - term->size(),
              "EXEC-conditioned previous-BB branch requires fixed-counter fallback budget",
              term);
          continue;
        }
        if (term->raw_encoding() == nullptr || term->size() != sizeof(uint32_t)) {
          ++summary.skipped_branch_unsafe;
          record_skip_sample(summary, "branch", block->end_offset() - term->size(),
                             "branch terminator encoding is not relocatable", term);
          continue;
        }
        const uint64_t term_offset = block->end_offset() - term->size();
        if (!patched_branch_offsets.insert(term_offset).second)
          continue;

        const int64_t target =
            static_cast<int64_t>(block->end_offset()) + *term->branch_offset_bytes();
        if (target < 0) {
          ++summary.skipped_branch_unsafe;
          record_skip_sample(summary, "branch", term_offset, "branch target is negative",
                             term);
          continue;
        }
        const uint64_t fallthrough_target = block->end_offset();
        const uint32_t taken_edge_id =
            stable_edge_id(kernel.name, block->start_offset(), static_cast<uint64_t>(target));
        const uint32_t fallthrough_edge_id =
            stable_edge_id(kernel.name, block->start_offset(), fallthrough_target);
        std::optional<EdgeSlotAssignment> slot_assignment =
            (use_fixed_counter_budget_fallback ||
             use_fixed_counter_exec_condition_fallback)
                ? edge_slots.assign_branch_terminator_fallback(
                      EdgeSlotPolicyKind::FixedCounter, site_edge_count, taken_edge_id,
                      fallthrough_edge_id)
                : edge_slots.assign_branch_terminator(site_edge_count, taken_edge_id,
                                                      fallthrough_edge_id);
        if (!slot_assignment) {
          ++summary.skipped_fixed_slot;
          record_skip_sample(summary, "branch", term_offset,
                             "edge slot policy could not assign a slot", term);
          continue;
        }

        EdgeSite site;
        site.kind = conditional ? EdgePatchKind::ConditionalBranchTerminator
                                : EdgePatchKind::BranchTerminator;
        site.kernel_name = kernel.name;
        site.pred_text_offset = block->start_offset();
        site.block_text_offset = static_cast<uint64_t>(target);
        site.patch_text_offset = term_offset;
        site.return_text_offset = static_cast<uint64_t>(target);
        site.first_inst_size = static_cast<uint32_t>(term->size());
        site.bb_id = taken_edge_id;
        site.fallthrough_bb_id = fallthrough_edge_id;
        site.slot_policy = slot_assignment->policy;
        site.probe_registers = options.probe_registers;
        site.fixed_slot = slot_assignment->primary_slot;
        site.fixed_slot_collisions = slot_assignment->fixed_slot_collisions;
        site.self_contained_probe = options.self_contained_edge_probes;
        if (conditional) {
          site.fallthrough_slot = slot_assignment->secondary_slot;
          site.branch_opcode = term->opcode();
        }
        liveness_probe_points.push_back(term);
        liveness_site_indices.push_back(result.sites.size());
        liveness_requires_previous_bb_registers |=
            site.slot_policy == EdgeSlotPolicyKind::PreviousBbHash;
        result.sites.push_back(std::move(site));
        summary.branch_edges_selected += site_edge_count;
        if (result.sites.back().slot_policy == EdgeSlotPolicyKind::PreviousBbHash) {
          summary.previous_bb_branch_edges_selected += site_edge_count;
          ++summary.previous_bb_branch_sites_selected;
        }
        if (use_fixed_counter_budget_fallback) {
          fixed_counter_budget_fallback_edges += site_edge_count;
          summary.fixed_counter_branch_edge_fallback_used =
              fixed_counter_budget_fallback_edges;
          summary.fixed_counter_branch_edge_aggregate_fallback_used +=
              site_edge_count;
          summary.branch_edges_degraded_to_fixed += site_edge_count;
          ++summary.previous_bb_branch_sites_degraded_to_fixed;
          record_degradation_count(
              summary, "branch",
              over_previous_bb_branch_site_budget ? "previous-BB branch site cap"
                                                  : "previous-BB branch budget cap",
              site_edge_count);
        } else if (use_fixed_counter_exec_condition_fallback) {
          fixed_counter_budget_fallback_edges += site_edge_count;
          summary.fixed_counter_branch_edge_fallback_used =
              fixed_counter_budget_fallback_edges;
          summary.fixed_counter_branch_edge_safety_fallback_used +=
              site_edge_count;
          summary.branch_edges_degraded_to_fixed += site_edge_count;
          ++summary.previous_bb_branch_sites_degraded_to_fixed;
          record_degradation_count(
              summary, "branch",
              "EXEC-conditioned previous-BB branch probe is not yet proven safe",
              site_edge_count);
        }
      }
      if (options.liveness_registers && !liveness_site_indices.empty()) {
        std::span<const rocjitsu::Instruction *const> probe_points(liveness_probe_points.data(),
                                                                   liveness_probe_points.size());
        std::optional<Rdna4ProbeRegisters> regs;
        bool liveness_accounted = false;
        const char *strict_liveness_failure = nullptr;
        if (!options.require_liveness_registers) {
          regs = select_liveness_probe_registers(kernel, liveness_scope, probe_points,
                                                 liveness_requires_previous_bb_registers,
                                                 !options.self_contained_edge_probes,
                                                 options.probe_registers);
        } else {
          try {
            rocjitsu::LivenessAnalysis liveness(liveness_scope);
            // Opaque instructions are modeled well enough for CFG sizing and
            // allocated-register liveness, but not yet well enough to prove
            // descriptor growth. Keep fresh registers off until the VOPD model is
            // opcode-complete or the DBI side can validate fresh ranges another way.
            const bool allow_fresh_registers =
                summary.opaque_instruction_count == 0 ||
                options.allow_opaque_fresh_registers;
            if (!allow_fresh_registers) {
              add_capped(summary.fresh_register_growth_disabled_by_opaque_probe_points,
                         probe_points.size());
              record_opaque_fresh_register_candidate(
                  summary, kernel, arch, liveness, probe_points,
                  std::span<const size_t>(liveness_site_indices.data(),
                                          liveness_site_indices.size()),
                  std::span<const EdgeSite>(result.sites.data(), result.sites.size()),
                  liveness_requires_previous_bb_registers,
                  !options.self_contained_edge_probes, options.probe_registers,
                  "branch");
            }
            const bool has_exec_empty_fixed_counter_edge =
                std::any_of(liveness_site_indices.begin(), liveness_site_indices.end(),
                            [&](size_t site_index) {
                              return site_index < result.sites.size() &&
                                     fixed_counter_edge_has_exec_empty_outcome(
                                         result.sites[site_index]);
                            });
            std::optional<EdgeProbeRegisterSelection> safe_regs;
            if (!has_exec_empty_fixed_counter_edge) {
              safe_regs = select_safe_probe_registers_from_analysis(
                  kernel, arch, liveness, probe_points, liveness_requires_previous_bb_registers,
                  !options.self_contained_edge_probes, options.probe_registers,
                  allow_fresh_registers, /*failure_reason=*/nullptr,
                  options.force_fresh_sgprs,
                  /*force_saved_exec_sgpr_pair=*/false,
                  options.force_fresh_vgprs);
            }
            if (safe_regs) {
              regs = safe_regs->probe_registers;
              if (safe_regs->uses_fresh_registers) {
                summary.fresh_registers = true;
                add_capped(summary.fresh_register_probe_points, probe_points.size());
              }
            }
            if (!regs) {
              const PerSiteLivenessResult per_site = assign_per_site_liveness_or_drop(
                  kernel, arch, liveness, probe_points, liveness_site_indices,
                  !options.self_contained_edge_probes, options.probe_registers,
                  allow_fresh_registers,
                  options.fixed_counter_fallback_for_branch_liveness,
                  options.allow_vgpr_scratch_spills, options.force_fresh_sgprs,
                  options.force_fresh_vgprs,
                  allow_fresh_registers ? nullptr
                                        : "fresh register growth disabled by opaque instructions",
                  &edge_slots, result.sites, summary, "branch");
              subtract_capped(summary.branch_edges_selected, per_site.dropped_edges);
              subtract_capped(summary.previous_bb_branch_edges_selected,
                              per_site.dropped_previous_bb_branch_edges);
              subtract_capped(summary.previous_bb_branch_edges_selected,
                              per_site.degraded_previous_bb_branch_edges);
              subtract_capped(summary.previous_bb_branch_sites_selected,
                              per_site.dropped_previous_bb_branch_sites);
              subtract_capped(summary.previous_bb_branch_sites_selected,
                              per_site.degraded_previous_bb_branch_sites);
              summary.skipped_branch_liveness += per_site.dropped_sites;
              summary.branch_edges_degraded_to_fixed += per_site.degraded_edges;
              summary.previous_bb_branch_sites_degraded_to_fixed +=
                  per_site.degraded_previous_bb_branch_sites;
              summary.fixed_counter_branch_edge_liveness_fallback_used +=
                  per_site.fixed_counter_branch_edge_liveness_fallback_used;
              add_slot_summary(liveness_dropped_slot_summary,
                               per_site.dropped_slot_summary);
              record_liveness_result(summary, per_site);
              liveness_accounted = true;
            }
          } catch (const std::exception &) {
            strict_liveness_failure = "liveness analysis failed";
          }
        }
        if (regs) {
          for (size_t site_index : liveness_site_indices)
            result.sites[site_index].probe_registers = *regs;
          summary.liveness_registers = true;
          summary.liveness_probe_points = static_cast<uint32_t>(std::min<size_t>(
              summary.liveness_probe_points + liveness_probe_points.size(),
              std::numeric_limits<uint32_t>::max()));
          liveness_accounted = true;
        } else if (options.require_liveness_registers && !liveness_accounted) {
          const PerSiteLivenessResult dropped = drop_liveness_sites(
              liveness_site_indices, result.sites, summary, "branch",
              strict_liveness_failure != nullptr ? strict_liveness_failure
                                                 : "no liveness-safe probe registers");
          subtract_capped(summary.branch_edges_selected, dropped.dropped_edges);
          subtract_capped(summary.previous_bb_branch_edges_selected,
                          dropped.dropped_previous_bb_branch_edges);
          subtract_capped(summary.previous_bb_branch_sites_selected,
                          dropped.dropped_previous_bb_branch_sites);
          summary.skipped_branch_liveness += dropped.dropped_sites;
          add_slot_summary(liveness_dropped_slot_summary,
                           dropped.dropped_slot_summary);
        }
      }
    }
    summary.slot_policy_summary = edge_slots.summary();
    subtract_slot_summary(summary.slot_policy_summary, liveness_dropped_slot_summary);
    summary.inline_slots_reserved = summary.slot_policy_summary.inline_slots_reserved;
    result.slot_policy_summary.hashed_edge_sites +=
        summary.slot_policy_summary.hashed_edge_sites;
    result.slot_policy_summary.fixed_edge_sites += summary.slot_policy_summary.fixed_edge_sites;
    result.slot_policy_summary.fixed_slot_requests +=
        summary.slot_policy_summary.fixed_slot_requests;
    result.slot_policy_summary.fixed_slots_reserved +=
        summary.slot_policy_summary.fixed_slots_reserved;
    result.slot_policy_summary.fixed_slot_exhaustions +=
        summary.slot_policy_summary.fixed_slot_exhaustions;
    result.slot_policy_summary.fixed_slot_collisions +=
        summary.slot_policy_summary.fixed_slot_collisions;
    result.slot_policy_summary.inline_slot_requests +=
        summary.slot_policy_summary.inline_slot_requests;
    result.slot_policy_summary.inline_slots_reserved +=
        summary.slot_policy_summary.inline_slots_reserved;
    result.slot_policy_summary.inline_slot_exhaustions +=
        summary.slot_policy_summary.inline_slot_exhaustions;
    if (options.verbose) {
      fprintf(stderr,
              "rocjitsu-afl: selected %u bb sites for %s "
              "(inline_slots=%u skipped_unsafe=%u skipped_liveness=%u skipped_limit=%u "
              "skipped_fixed_slot=%u branch_edges=%u skipped_branch_unsafe=%u "
              "skipped_branch_liveness=%u skipped_branch_limit=%u "
              "branch_edges_degraded_to_fixed=%u)\n",
              summary.block_selected, kernel.name.c_str(), summary.inline_slots_reserved,
              summary.skipped_unsafe, summary.skipped_liveness, summary.skipped_limit,
              summary.skipped_fixed_slot, summary.branch_edges_selected,
              summary.skipped_branch_unsafe, summary.skipped_branch_liveness,
              summary.skipped_branch_limit, summary.branch_edges_degraded_to_fixed);
    }
    summary.coverage_strategy = kernel_coverage_strategy(summary, options);
    result.kernel_summaries.push_back(std::move(summary));
  }

  return result;
}

namespace {

enum class ConditionalEdgePath {
  Single,
  Fallthrough,
  Taken,
};

bool fixed_counter_edge_requires_lane0_exec(const EdgeSite &site,
                                            ConditionalEdgePath path) {
  if (!site.force_lane0_exec_for_fixed_counter ||
      !fixed_counter_edge_has_exec_empty_outcome(site)) {
    return false;
  }
  if (site.branch_opcode == kGfx11PlusSoppCbranchExecz)
    return path == ConditionalEdgePath::Taken;
  if (site.branch_opcode == kGfx11PlusSoppCbranchExecnz)
    return path == ConditionalEdgePath::Fallthrough;
  return false;
}

std::optional<std::vector<uint32_t>>
edge_probe_words_for_policy(EdgeSlotPolicyKind policy, EdgePatchKind kind, uint32_t bb_id,
                            uint32_t fixed_slot, rj_code_arch_t arch, uint64_t state_pointer,
                            bool self_contained_probe, Rdna4ProbeRegisters regs,
                            bool force_lane0_exec_for_fixed_counter,
                            const char **failure_reason) {
  auto fail = [&](const char *reason) {
    if (failure_reason != nullptr)
      *failure_reason = reason;
    return std::optional<std::vector<uint32_t>>{};
  };

  if (policy == EdgeSlotPolicyKind::FixedCounter) {
    const bool flagless_counter =
        self_contained_probe || !edge_patch_kind_is_block_entry(kind);
    if (force_lane0_exec_for_fixed_counter && !flagless_counter)
      return fail("forced-lane fixed counter is only supported for flagless probes");
    auto fixed_probe =
        flagless_counter
            ? (self_contained_probe
                   ? (force_lane0_exec_for_fixed_counter
                          ? rdna4_flagless_counter_probe_force_lane0_with_state_pointer(
                                fixed_slot, state_pointer, arch, regs)
                          : rdna4_flagless_counter_probe_with_state_pointer(
                                fixed_slot, state_pointer, arch, regs))
                   : (force_lane0_exec_for_fixed_counter
                          ? rdna4_flagless_counter_probe_force_lane0(fixed_slot, arch, regs)
                          : rdna4_flagless_counter_probe(fixed_slot, arch, regs)))
            : rdna4_counter_probe(fixed_slot, arch, regs);
    if (!fixed_probe)
      return fail("coverage probe is not supported for target");
    return std::move(*fixed_probe);
  }

  if (self_contained_probe)
    return rdna4_previous_bb_edge_probe_with_state_pointer(bb_id, state_pointer, arch,
                                                                    regs);

  return rdna4_previous_bb_edge_probe(bb_id,
                                               /*load_state_base=*/false,
                                               /*state_pointer_kernarg_offset=*/0, arch, regs);
}

bool scratch_spill_uses_saved_exec_for_address(const EdgeSite &site) {
  return site.scratch_spill_plan && site.scratch_spill_plan->sgpr_spills.empty() &&
         site.probe_registers.saved_exec_sgpr != site.probe_registers.state_sgpr;
}

std::optional<std::vector<uint32_t>>
edge_probe_words_for_site_edge(const EdgeSite &site, uint32_t bb_id, uint32_t fixed_slot,
                               rj_code_arch_t arch, uint64_t state_pointer,
                               ConditionalEdgePath path, const char **failure_reason) {
  const bool force_lane0_exec =
      fixed_counter_edge_requires_lane0_exec(site, path);
  auto probe = edge_probe_words_for_policy(site.slot_policy, site.kind, bb_id, fixed_slot, arch,
                                           state_pointer, site.self_contained_probe,
                                           site.probe_registers, force_lane0_exec,
                                           failure_reason);
  if (!probe || !site.scratch_spill_plan)
    return probe;

  if (force_lane0_exec) {
    auto wrapped = wrap_forced_lane0_probe_with_vgpr_scratch_spills(
        *probe, *site.scratch_spill_plan, site.probe_registers, arch);
    if (!wrapped && failure_reason != nullptr)
      *failure_reason = "forced-lane scratch spill wrapper is not supported for target";
    return wrapped;
  }

  const bool use_saved_exec_for_scratch_address =
      scratch_spill_uses_saved_exec_for_address(site);
  auto wrapped = wrap_probe_with_vgpr_scratch_spills(
      *probe, *site.scratch_spill_plan, site.probe_registers, arch,
      use_saved_exec_for_scratch_address);
  if (!wrapped && failure_reason != nullptr)
    *failure_reason = "scratch spill wrapper is not supported for target";
  return wrapped;
}

std::optional<std::vector<uint32_t>>
edge_probe_words(const EdgeSite &site, rj_code_arch_t arch, uint64_t state_pointer,
                 const char **failure_reason) {
  return edge_probe_words_for_site_edge(site, site.bb_id, site.fixed_slot, arch, state_pointer,
                                        ConditionalEdgePath::Single, failure_reason);
}

std::optional<uint64_t> edge_trampoline_size_bytes(const EdgeSite &site, rj_code_arch_t arch,
                                                   std::span<const uint8_t> text,
                                                   uint64_t state_pointer,
                                                   const char **failure_reason) {
  auto fail = [&](const char *reason) -> std::optional<uint64_t> {
    if (failure_reason != nullptr)
      *failure_reason = reason;
    return std::nullopt;
  };

  if (edge_patch_kind_is_conditional_dispatch(site.kind)) {
    auto taken_probe = edge_probe_words_for_site_edge(site, site.bb_id, site.fixed_slot, arch,
                                                     state_pointer, ConditionalEdgePath::Taken,
                                                     failure_reason);
    auto fallthrough_probe = edge_probe_words_for_site_edge(
        site, site.fallthrough_bb_id, site.fallthrough_slot, arch, state_pointer,
        ConditionalEdgePath::Fallthrough, failure_reason);
    if (!taken_probe || !fallthrough_probe)
      return fail("coverage probe is not supported for target");
    return (1 + fallthrough_probe->size() + 1 + taken_probe->size() + 1) * sizeof(uint32_t);
  }

  auto probe = edge_probe_words(site, arch, state_pointer, failure_reason);
  if (!probe)
    return std::nullopt;

  uint64_t words = probe->size();
  if (site.kind == EdgePatchKind::BlockEntry) {
    const char *relocation_failure = nullptr;
    auto relocated = relocate_overwritten_instruction(
        text, site.patch_text_offset, site.first_inst_size, arch, &relocation_failure);
    if (!relocated)
      return fail(relocation_failure != nullptr ? relocation_failure
                                                : "overwritten instruction is not relocatable");
    words += relocated->words.size();
  }
  ++words;
  return words * sizeof(uint32_t);
}

std::optional<EdgeTrampoline> build_edge_trampoline(const EdgeSite &site,
                                                    std::span<const uint8_t> text,
                                                    rj_code_arch_t arch, uint64_t state_pointer,
                                                    uint64_t cave_entry,
                                                    const char **failure_reason) {
  auto fail = [&](const char *reason) -> std::optional<EdgeTrampoline> {
    if (failure_reason != nullptr)
      *failure_reason = reason;
    return std::nullopt;
  };

  if (site.patch_text_offset > text.size() ||
      site.first_inst_size > text.size() - site.patch_text_offset)
    return fail("edge site is outside .text");

  auto fwd = s_branch_offset_dwords(site.patch_text_offset, cave_entry);
  if (!fwd)
    return fail("branch to instrumentation cave is out of s_branch range");

  if (edge_patch_kind_is_conditional_dispatch(site.kind)) {
    auto taken_probe = edge_probe_words_for_site_edge(site, site.bb_id, site.fixed_slot, arch,
                                                     state_pointer, ConditionalEdgePath::Taken,
                                                     failure_reason);
    auto fallthrough_probe = edge_probe_words_for_site_edge(
        site, site.fallthrough_bb_id, site.fallthrough_slot, arch, state_pointer,
        ConditionalEdgePath::Fallthrough, failure_reason);
    if (!taken_probe || !fallthrough_probe)
      return fail("coverage probe is not supported for target");

    const uint64_t cond_branch_pc = cave_entry;
    const uint64_t fallthrough_probe_pc = cond_branch_pc + sizeof(uint32_t);
    const uint64_t fallthrough_branch_pc =
        fallthrough_probe_pc + static_cast<uint64_t>(fallthrough_probe->size()) * sizeof(uint32_t);
    const uint64_t taken_probe_pc = fallthrough_branch_pc + sizeof(uint32_t);
    const uint64_t taken_branch_pc =
        taken_probe_pc + static_cast<uint64_t>(taken_probe->size()) * sizeof(uint32_t);

    auto cond = s_branch_offset_dwords(cond_branch_pc, taken_probe_pc);
    auto fallthrough_ret = s_branch_offset_dwords(
        fallthrough_branch_pc, site.patch_text_offset + site.first_inst_size);
    auto taken_ret = s_branch_offset_dwords(taken_branch_pc, site.return_text_offset);
    if (!cond || !fallthrough_ret || !taken_ret)
      return fail("conditional edge dispatcher branch is out of s_branch range");

    EdgeTrampoline trampoline;
    trampoline.patch_branch = rocjitsu::build_s_branch(*fwd, arch);
    trampoline.cave_words.push_back(rocjitsu::pack_sopp(static_cast<uint8_t>(site.branch_opcode),
                                                        static_cast<uint16_t>(*cond)));
    trampoline.cave_words.insert(trampoline.cave_words.end(), fallthrough_probe->begin(),
                                 fallthrough_probe->end());
    trampoline.cave_words.push_back(rocjitsu::build_s_branch(*fallthrough_ret, arch));
    trampoline.cave_words.insert(trampoline.cave_words.end(), taken_probe->begin(),
                                 taken_probe->end());
    trampoline.cave_words.push_back(rocjitsu::build_s_branch(*taken_ret, arch));
    return trampoline;
  }

  auto probe = edge_probe_words(site, arch, state_pointer, failure_reason);
  if (!probe)
    return std::nullopt;

  std::vector<uint32_t> cave_words(std::move(*probe));
  if (site.kind == EdgePatchKind::BlockEntry) {
    const char *relocation_failure = nullptr;
    auto relocated = relocate_overwritten_instruction(
        text, site.patch_text_offset, site.first_inst_size, arch, &relocation_failure);
    if (!relocated)
      return fail(relocation_failure != nullptr ? relocation_failure
                                                : "overwritten instruction is not relocatable");
    cave_words.insert(cave_words.end(), relocated->words.begin(), relocated->words.end());
    if (relocated->direct_branch_target_text_offset) {
      const uint64_t relocated_branch_pc =
          cave_entry + static_cast<uint64_t>(cave_words.size()) * sizeof(uint32_t);
      auto target_branch =
          s_branch_offset_dwords(relocated_branch_pc, *relocated->direct_branch_target_text_offset);
      if (!target_branch)
        return fail("relocated direct branch target is out of s_branch range");

      cave_words.push_back(rocjitsu::build_s_branch(*target_branch, arch));
      EdgeTrampoline trampoline;
      trampoline.patch_branch = rocjitsu::build_s_branch(*fwd, arch);
      trampoline.cave_words = std::move(cave_words);
      return trampoline;
    }
  }

  const uint64_t return_branch_pc =
      cave_entry + static_cast<uint64_t>(cave_words.size()) * sizeof(uint32_t);
  auto ret = s_branch_offset_dwords(return_branch_pc, site.return_text_offset);
  if (!ret)
    return fail("return branch from instrumentation cave is out of s_branch range");

  cave_words.push_back(rocjitsu::build_s_branch(*ret, arch));
  EdgeTrampoline trampoline;
  trampoline.patch_branch = rocjitsu::build_s_branch(*fwd, arch);
  trampoline.cave_words = std::move(cave_words);
  return trampoline;
}

void install_edge_site_redirect(const EdgeSite &site, const EdgeTrampoline &trampoline,
                                std::vector<uint8_t> &text, rj_code_arch_t arch) {
  memcpy(text.data() + site.patch_text_offset, &trampoline.patch_branch,
         sizeof(trampoline.patch_branch));
  for (uint32_t byte = sizeof(trampoline.patch_branch); byte < site.first_inst_size;
       byte += sizeof(uint32_t)) {
    const uint32_t nop = rocjitsu::build_s_nop(0, arch);
    memcpy(text.data() + site.patch_text_offset + byte, &nop, sizeof(nop));
  }
}

void install_local_edge_trampoline(const EdgeSite &site, const EdgeTrampoline &trampoline,
                                   std::vector<uint8_t> &text, uint64_t cave_text_offset,
                                   rj_code_arch_t arch) {
  install_edge_site_redirect(site, trampoline, text, arch);
  memcpy(text.data() + cave_text_offset, trampoline.cave_words.data(),
         trampoline.cave_words.size() * sizeof(uint32_t));
}

const char *edge_trampoline_placement_name(EdgeTrampolinePlacement placement) {
  switch (placement) {
  case EdgeTrampolinePlacement::AppendedCave:
    return "appended-cave";
  case EdgeTrampolinePlacement::LocalTextCave:
    return "local-cave";
  }
  return "unknown";
}

std::string scratch_address_exec_source(const EdgeSite &site) {
  if (!site.scratch_spill_plan)
    return "none";
  if (site.force_lane0_exec_for_fixed_counter)
    return "forced-lane0";
  if (scratch_spill_uses_saved_exec_for_address(site))
    return "saved-exec";
  return "exec";
}

EdgeSiteSelectionSample selected_edge_sample(const PlannedEdgeTrampoline &planned) {
  const EdgeSite &site = planned.site;
  EdgeSiteSelectionSample sample;
  sample.kernel_name = site.kernel_name;
  sample.kind = edge_patch_kind_name(site.kind);
  sample.pred_text_offset = site.pred_text_offset;
  sample.block_text_offset = site.block_text_offset;
  sample.patch_text_offset = site.patch_text_offset;
  sample.return_text_offset = site.return_text_offset;
  sample.cave_text_offset = planned.result.cave_text_offset;
  sample.trampoline_bytes =
      static_cast<uint64_t>(planned.trampoline.cave_words.size()) * sizeof(uint32_t);
  sample.bb_id = site.bb_id;
  sample.fallthrough_bb_id = site.fallthrough_bb_id;
  sample.fixed_slot = site.fixed_slot;
  sample.fallthrough_slot = site.fallthrough_slot;
  sample.fixed_slot_collisions = site.fixed_slot_collisions;
  sample.self_contained_probe = site.self_contained_probe;
  sample.force_lane0_exec_for_fixed_counter =
      site.force_lane0_exec_for_fixed_counter;
  sample.slot_policy = edge_slot_policy_name(site.slot_policy);
  sample.placement = edge_trampoline_placement_name(planned.result.placement);
  sample.scratch_address_exec_source = scratch_address_exec_source(site);
  sample.state_sgpr = site.probe_registers.state_sgpr;
  sample.saved_exec_sgpr = site.probe_registers.saved_exec_sgpr;
  sample.tmp0_sgpr = site.probe_registers.tmp0_sgpr;
  sample.tmp1_sgpr = site.probe_registers.tmp1_sgpr;
  sample.scc_sgpr = site.probe_registers.scc_sgpr;
  sample.workitem_vgpr = site.probe_registers.workitem_vgpr;
  sample.tmp0_vgpr = site.probe_registers.tmp0_vgpr;
  sample.tmp1_vgpr = site.probe_registers.tmp1_vgpr;
  sample.tmp2_vgpr = site.probe_registers.tmp2_vgpr;
  sample.scratch_spill = site.scratch_spill_plan.has_value();
  if (site.scratch_spill_plan) {
    sample.scratch_address_vgpr = site.scratch_spill_plan->address_vgpr;
    sample.vgpr_scratch_spill = !site.scratch_spill_plan->vgpr_spills.empty();
    sample.sgpr_scratch_spill = !site.scratch_spill_plan->sgpr_spills.empty();
    for (const ProbeScratchSpillSlot &slot : site.scratch_spill_plan->vgpr_spills)
      sample.scratch_spilled_vgprs.push_back(slot.vgpr);
    for (const ProbeScratchSgprSpillSlot &slot : site.scratch_spill_plan->sgpr_spills)
      sample.scratch_spilled_sgprs.push_back(slot.sgpr);
  }
  return sample;
}

} // namespace

uint32_t edge_count_for_site(const EdgeSite &site) {
  return site.kind == EdgePatchKind::ConditionalBranchTerminator ? 2u : 1u;
}

bool previous_bb_branch_site(const EdgeSite &site) {
  return site.slot_policy == EdgeSlotPolicyKind::PreviousBbHash &&
         !edge_patch_kind_is_block_entry(site.kind);
}

bool placement_failure_can_degrade_to_fixed(std::string_view reason) {
  return reason.find("range") != std::string_view::npos ||
         reason.find("text cave") != std::string_view::npos ||
         reason.find("cave") != std::string_view::npos;
}

namespace {

uint32_t next_stable_fixed_slot(uint32_t slot) {
  return slot == kMaxFixedCounterSlot ? kFirstEdgeCounterSlot : slot + 1;
}

} // namespace

void prime_fixed_counter_placement_tracker(FixedEdgeSlotTracker &tracker,
                                           std::span<const EdgeSite> sites) {
  for (const EdgeSite &site : sites) {
    if (site.slot_policy != EdgeSlotPolicyKind::FixedCounter)
      continue;
    tracker.record(site.fixed_slot);
    if (edge_count_for_site(site) == 2)
      tracker.record(site.fallthrough_slot);
  }
}

EdgeSite make_stable_fixed_counter_fallback_site(const EdgeSite &site) {
  EdgeSite fallback = site;
  const uint32_t edge_count = edge_count_for_site(site);
  fallback.slot_policy = EdgeSlotPolicyKind::FixedCounter;
  fallback.fixed_slot = FixedEdgeSlotAllocator::fixed_counter_slot_for_id(site.bb_id);
  fallback.fallthrough_slot = 0;
  fallback.fixed_slot_collisions = 0;
  if (edge_count == 2) {
    fallback.fallthrough_slot =
        FixedEdgeSlotAllocator::fixed_counter_slot_for_id(site.fallthrough_bb_id);
    if (fallback.fallthrough_slot == fallback.fixed_slot)
      fallback.fallthrough_slot = next_stable_fixed_slot(fallback.fallthrough_slot);
  }
  return fallback;
}

uint32_t record_fixed_counter_placement_slots(FixedEdgeSlotTracker &tracker,
                                              const EdgeSite &site) {
  uint32_t collisions = tracker.record(site.fixed_slot) ? 1 : 0;
  if (edge_count_for_site(site) == 2)
    collisions += tracker.record(site.fallthrough_slot) ? 1 : 0;
  return collisions;
}

bool placement_fixed_fallback_has_budget(const KernelEdgeSelectionSummary &summary,
                                         uint32_t edge_count) {
  if (summary.fixed_counter_branch_edge_fallback_budget == 0)
    return false;
  if (edge_count > summary.fixed_counter_branch_edge_fallback_budget)
    return false;
  return summary.fixed_counter_branch_edge_fallback_used <=
         summary.fixed_counter_branch_edge_fallback_budget - edge_count;
}

namespace {

void add_degradation_reason_count(KernelEdgeSelectionSummary &summary,
                                  std::string_view reason, uint32_t edge_count) {
  if (edge_count == 0)
    return;
  for (auto &count : summary.degradation_reason_counts) {
    if (count.kind == "branch" && count.reason == reason) {
      count.count += edge_count;
      return;
    }
  }
  summary.degradation_reason_counts.push_back(
      {"branch", std::string(reason), edge_count});
}

void subtract_previous_bb_site(EdgeSlotPolicySummary &summary) {
  if (summary.hashed_edge_sites != 0)
    --summary.hashed_edge_sites;
}

void add_fixed_branch_site(EdgeSlotPolicySummary &summary, uint32_t edge_count,
                           uint32_t collisions) {
  ++summary.fixed_edge_sites;
  summary.fixed_slot_requests += edge_count;
  summary.fixed_slots_reserved += edge_count;
  summary.fixed_slot_collisions += collisions;
  summary.inline_slot_requests += edge_count;
  summary.inline_slots_reserved += edge_count;
}

KernelEdgeSelectionSummary *
find_kernel_summary(InstrumentationPlan &selection, std::string_view kernel_name) {
  for (KernelEdgeSelectionSummary &summary : selection.kernel_summaries) {
    if (summary.kernel_name == kernel_name)
      return &summary;
  }
  return nullptr;
}

const KernelEdgeSelectionSummary *
find_kernel_summary(const InstrumentationPlan &selection,
                    std::string_view kernel_name) {
  for (const KernelEdgeSelectionSummary &summary : selection.kernel_summaries) {
    if (summary.kernel_name == kernel_name)
      return &summary;
  }
  return nullptr;
}

} // namespace

bool placement_fixed_fallback_has_budget(const InstrumentationPlan &selection,
                                         const EdgeSite &site) {
  const KernelEdgeSelectionSummary *summary =
      find_kernel_summary(selection, site.kernel_name);
  return summary != nullptr &&
         placement_fixed_fallback_has_budget(*summary, edge_count_for_site(site));
}

bool record_previous_bb_branch_placement_fallback(
    InstrumentationPlan &selection, const EdgeSite &site,
    uint32_t fixed_slot_collisions, std::string_view failure_reason) {
  const uint32_t edge_count = edge_count_for_site(site);
  KernelEdgeSelectionSummary *summary =
      find_kernel_summary(selection, site.kernel_name);
  if (summary == nullptr ||
      !placement_fixed_fallback_has_budget(*summary, edge_count)) {
    return false;
  }

  if (summary->previous_bb_branch_sites_selected != 0)
    --summary->previous_bb_branch_sites_selected;
  subtract_capped(summary->previous_bb_branch_edges_selected, edge_count);
  summary->branch_edges_degraded_to_fixed += edge_count;
  ++summary->previous_bb_branch_sites_degraded_to_fixed;
  summary->fixed_counter_branch_edge_fallback_used += edge_count;
  summary->fixed_counter_branch_edge_placement_fallback_used += edge_count;
  subtract_previous_bb_site(summary->slot_policy_summary);
  add_fixed_branch_site(summary->slot_policy_summary, edge_count,
                        fixed_slot_collisions);
  subtract_previous_bb_site(selection.slot_policy_summary);
  add_fixed_branch_site(selection.slot_policy_summary, edge_count,
                        fixed_slot_collisions);

  std::string reason =
      "previous-BB trampoline placement failed; fixed counter trampoline fit";
  if (!failure_reason.empty()) {
    reason += " after ";
    reason += failure_reason;
  }
  add_degradation_reason_count(*summary, reason, edge_count);
  return true;
}

std::optional<PlannedEdgeTrampoline>
plan_edge_trampoline(const EdgeSite &site, std::span<const uint8_t> text,
                     uint64_t appended_cave_body_size, uint64_t cave_start,
                     LocalTextCaveAllocator &local_caves, rj_code_arch_t arch,
                     uint64_t state_pointer, const char **failure_reason) {
  auto fail = [&](const char *reason) -> std::optional<PlannedEdgeTrampoline> {
    if (failure_reason != nullptr)
      *failure_reason = reason;
    return std::nullopt;
  };

  const uint64_t appended_cave = cave_start + appended_cave_body_size;
  const char *appended_failure = nullptr;
  if (auto trampoline = build_edge_trampoline(site, text, arch, state_pointer,
                                              appended_cave, &appended_failure)) {
    return PlannedEdgeTrampoline{
        site, std::move(*trampoline),
        EdgePatchResult{EdgeTrampolinePlacement::AppendedCave, appended_cave}};
  }

  const char *size_failure = nullptr;
  auto size_bytes =
      edge_trampoline_size_bytes(site, arch, text, state_pointer, &size_failure);
  if (!size_bytes)
    return fail(size_failure != nullptr ? size_failure : appended_failure);

  auto local_cave = local_caves.allocate(site.patch_text_offset, *size_bytes, [&](uint64_t offset) {
    return build_edge_trampoline(site, text, arch, state_pointer, offset, nullptr)
        .has_value();
  });
  if (!local_cave)
    return fail("no branch-reachable local text cave");

  const char *local_failure = nullptr;
  auto trampoline = build_edge_trampoline(site, text, arch, state_pointer, *local_cave,
                                          &local_failure);
  if (!trampoline)
    return fail(local_failure != nullptr ? local_failure : "local text cave emission failed");

  return PlannedEdgeTrampoline{
      site, std::move(*trampoline),
      EdgePatchResult{EdgeTrampolinePlacement::LocalTextCave, *local_cave}};
}

uint32_t scaled_u32(uint64_t numerator, uint64_t denominator,
                    uint64_t scale) {
  if (denominator == 0 || scale == 0)
    return 0;
  uint64_t scaled = std::numeric_limits<uint64_t>::max();
  if (numerator <= std::numeric_limits<uint64_t>::max() / scale)
    scaled = numerator * scale;
  const uint64_t value = scaled / denominator;
  return static_cast<uint32_t>(
      std::min<uint64_t>(value, std::numeric_limits<uint32_t>::max()));
}

std::pair<std::string, std::string>
previous_bb_branch_overhead_decision(uint32_t selected_edges,
                                     uint32_t aggregate_fallback_edges,
                                     uint32_t safety_fallback_edges,
                                     uint32_t liveness_fallback_edges,
                                     uint32_t placement_fallback_edges,
                                     uint32_t map_pressure_ppm) {
  const uint64_t fallback_edges =
      static_cast<uint64_t>(aggregate_fallback_edges) + safety_fallback_edges +
      liveness_fallback_edges + placement_fallback_edges;
  if (selected_edges == 0 && fallback_edges == 0) {
    return {"no-previous-bb-branch-edges",
            "no previous-BB branch probes were selected for this scope"};
  }
  if (map_pressure_ppm >= 1000000u) {
    return {"afl-map-at-capacity",
            "selected logical previous-BB branch edges meet or exceed the AFL "
            "device half-map capacity; bitmap collisions are expected"};
  }
  if (placement_fallback_edges != 0) {
    return {"placement-degraded",
            "some previous-BB branch probes degraded to fixed counters after "
            "trampoline placement or branch-range checks failed"};
  }
  if (selected_edges == 0) {
    return {"fully-degraded-to-fixed",
            "all candidate previous-BB branch probes degraded to fixed counters"};
  }
  if (fallback_edges != 0) {
    return {"partially-degraded-to-fixed",
            "some branch sites used fixed counters because the stronger "
            "previous-BB probe was not proven safe or budgeted"};
  }
  return {"within-current-resource-guards",
          "selected previous-BB branch probes fit current map, liveness, "
          "descriptor/private-segment, and placement guards"};
}

void finalize_previous_bb_branch_overhead(
    KernelEdgeSelectionSummary &summary) {
  summary.previous_bb_branch_afl_map_budget = kHashEdgeSlots;
  summary.previous_bb_branch_afl_map_pressure_ppm =
      scaled_u32(summary.previous_bb_branch_edges_selected, kHashEdgeSlots,
                 1000000u);
  summary.previous_bb_branch_trampoline_avg_bytes_x100 =
      scaled_u32(summary.previous_bb_branch_edge_trampoline_bytes,
                 summary.previous_bb_branch_edge_trampolines_planned, 100u);
  summary.previous_bb_branch_appended_trampoline_ratio_ppm =
      scaled_u32(summary.previous_bb_branch_planned_appended_edge_trampoline_bytes,
                 summary.previous_bb_branch_edge_trampoline_bytes, 1000000u);
  summary.previous_bb_branch_local_trampoline_ratio_ppm =
      scaled_u32(summary.previous_bb_branch_planned_local_edge_trampoline_bytes,
                 summary.previous_bb_branch_edge_trampoline_bytes, 1000000u);
  auto decision = previous_bb_branch_overhead_decision(
      summary.previous_bb_branch_edges_selected,
      summary.fixed_counter_branch_edge_aggregate_fallback_used,
      summary.fixed_counter_branch_edge_safety_fallback_used,
      summary.fixed_counter_branch_edge_liveness_fallback_used,
      summary.fixed_counter_branch_edge_placement_fallback_used,
      summary.previous_bb_branch_afl_map_pressure_ppm);
  summary.previous_bb_branch_overhead_status = std::move(decision.first);
  summary.previous_bb_branch_overhead_reason = std::move(decision.second);
}

void finalize_previous_bb_branch_overhead(PatchDeviceElfReport &report) {
  report.previous_bb_branch_afl_map_budget = kHashEdgeSlots;
  report.previous_bb_branch_afl_map_pressure_ppm =
      scaled_u32(report.previous_bb_branch_edges_selected, kHashEdgeSlots,
                 1000000u);
  report.previous_bb_branch_trampoline_avg_bytes_x100 =
      scaled_u32(report.previous_bb_branch_edge_trampoline_bytes,
                 report.previous_bb_branch_edge_trampolines_planned, 100u);
  report.previous_bb_branch_appended_trampoline_ratio_ppm =
      scaled_u32(report.previous_bb_branch_planned_appended_edge_trampoline_bytes,
                 report.previous_bb_branch_edge_trampoline_bytes, 1000000u);
  report.previous_bb_branch_local_trampoline_ratio_ppm =
      scaled_u32(report.previous_bb_branch_planned_local_edge_trampoline_bytes,
                 report.previous_bb_branch_edge_trampoline_bytes, 1000000u);
  report.previous_bb_branch_code_growth_pressure_ppm =
      scaled_u32(report.previous_bb_branch_planned_appended_edge_trampoline_bytes,
                 report.input_bytes, 1000000u);
  auto decision = previous_bb_branch_overhead_decision(
      report.previous_bb_branch_edges_selected,
      report.fixed_counter_branch_edge_aggregate_fallback_used,
      report.fixed_counter_branch_edge_safety_fallback_used,
      report.fixed_counter_branch_edge_liveness_fallback_used,
      report.fixed_counter_branch_edge_placement_fallback_used,
      report.previous_bb_branch_afl_map_pressure_ppm);
  report.previous_bb_branch_overhead_status = std::move(decision.first);
  report.previous_bb_branch_overhead_reason = std::move(decision.second);
}

void record_patch_plan_summary(PatchDeviceElfReport &report,
                               const DeviceElfPatchPlan &plan) {
  report.entry_candidate_count = plan.entry_candidate_count;
  report.entry_backed_edge_kernels = static_cast<uint32_t>(
      std::min<size_t>(plan.entry_backed_edge_sites.size(),
                       std::numeric_limits<uint32_t>::max()));
  report.self_contained_edge_kernels =
      static_cast<uint32_t>(std::min<size_t>(plan.self_contained_edge_sites.size(),
                                             std::numeric_limits<uint32_t>::max()));
  report.hybrid_edge_probes = plan.hybrid_edge_probes;
  report.cfg_failure_reason = plan.edge_selection.failure_reason;
  report.kernel_summaries = plan.edge_selection.kernel_summaries;
  report.hashed_edge_sites = plan.edge_selection.slot_policy_summary.hashed_edge_sites;
  report.fixed_edge_sites = plan.edge_selection.slot_policy_summary.fixed_edge_sites;
  report.fixed_slot_requests = plan.edge_selection.slot_policy_summary.fixed_slot_requests;
  report.fixed_slots_reserved = plan.edge_selection.slot_policy_summary.fixed_slots_reserved;
  report.fixed_slot_exhaustions =
      plan.edge_selection.slot_policy_summary.fixed_slot_exhaustions;
  report.fixed_slot_collisions =
      plan.edge_selection.slot_policy_summary.fixed_slot_collisions;
  report.inline_slot_requests = plan.edge_selection.slot_policy_summary.inline_slot_requests;
  report.inline_slot_exhaustions = plan.edge_selection.slot_policy_summary.inline_slot_exhaustions;
  report.branch_edges_degraded_to_fixed = 0;
  report.fixed_counter_branch_edge_aggregate_fallback_used = 0;
  report.fixed_counter_branch_edge_safety_fallback_used = 0;
  report.fixed_counter_branch_edge_liveness_fallback_used = 0;
  report.fixed_counter_branch_edge_placement_fallback_used = 0;
  report.exec_empty_fixed_counter_edges = 0;
  report.previous_bb_branch_edges_selected = 0;
  report.previous_bb_branch_sites_selected = 0;
  report.previous_bb_branch_sites_degraded_to_fixed = 0;
  for (const KernelEdgeSelectionSummary &summary : plan.edge_selection.kernel_summaries) {
    report.branch_edges_degraded_to_fixed += summary.branch_edges_degraded_to_fixed;
    report.fixed_counter_branch_edge_aggregate_fallback_used +=
        summary.fixed_counter_branch_edge_aggregate_fallback_used;
    report.fixed_counter_branch_edge_safety_fallback_used +=
        summary.fixed_counter_branch_edge_safety_fallback_used;
    report.fixed_counter_branch_edge_liveness_fallback_used +=
        summary.fixed_counter_branch_edge_liveness_fallback_used;
    report.fixed_counter_branch_edge_placement_fallback_used +=
        summary.fixed_counter_branch_edge_placement_fallback_used;
    report.previous_bb_branch_edges_selected +=
        summary.previous_bb_branch_edges_selected;
    report.previous_bb_branch_sites_selected +=
        summary.previous_bb_branch_sites_selected;
    report.previous_bb_branch_sites_degraded_to_fixed +=
        summary.previous_bb_branch_sites_degraded_to_fixed;
  }
  report.edge_sites_selected = static_cast<uint32_t>(
      std::min<size_t>(plan.edge_sites.size(), std::numeric_limits<uint32_t>::max()));
  report.local_text_cave_ranges = plan.local_text_cave_summary.range_count;
  report.local_text_cave_bytes = plan.local_text_cave_summary.total_bytes;
  report.largest_local_text_cave_bytes = plan.local_text_cave_summary.largest_range_bytes;
  report.edge_patch_failures = plan.edge_patch_failures;
  report.branch_range_failures = plan.branch_range_failures;
  report.sampled_failures = plan.sampled_edge_failures;
  report.edge_trampolines_planned = 0;
  report.previous_bb_branch_edge_trampolines_planned = 0;
  report.planned_appended_edge_trampolines = 0;
  report.planned_local_edge_trampolines = 0;
  report.previous_bb_branch_planned_appended_edge_trampolines = 0;
  report.previous_bb_branch_planned_local_edge_trampolines = 0;
  report.planned_edge_trampoline_bytes = 0;
  report.previous_bb_branch_edge_trampoline_bytes = 0;
  report.planned_appended_edge_trampoline_bytes = 0;
  report.planned_local_edge_trampoline_bytes = 0;
  report.previous_bb_branch_planned_appended_edge_trampoline_bytes = 0;
  report.previous_bb_branch_planned_local_edge_trampoline_bytes = 0;
  report.largest_edge_trampoline_bytes = 0;
  report.largest_previous_bb_branch_edge_trampoline_bytes = 0;
  report.previous_bb_branch_afl_map_budget = kHashEdgeSlots;
  report.previous_bb_branch_afl_map_pressure_ppm = 0;
  report.previous_bb_branch_trampoline_avg_bytes_x100 = 0;
  report.previous_bb_branch_appended_trampoline_ratio_ppm = 0;
  report.previous_bb_branch_local_trampoline_ratio_ppm = 0;
  report.previous_bb_branch_code_growth_pressure_ppm = 0;
  std::unordered_map<std::string, size_t> summary_indices;
  for (size_t index = 0; index < report.kernel_summaries.size(); ++index)
    summary_indices.emplace(report.kernel_summaries[index].kernel_name, index);
  for (const PlannedEdgeTrampoline &planned : plan.edge_trampolines) {
    const uint64_t trampoline_bytes =
        static_cast<uint64_t>(planned.trampoline.cave_words.size()) * sizeof(uint32_t);
    ++report.edge_trampolines_planned;
    report.planned_edge_trampoline_bytes += trampoline_bytes;
    report.largest_edge_trampoline_bytes =
        std::max(report.largest_edge_trampoline_bytes, trampoline_bytes);
    const bool previous_bb_branch = previous_bb_branch_site(planned.site);
    if (previous_bb_branch) {
      ++report.previous_bb_branch_edge_trampolines_planned;
      report.previous_bb_branch_edge_trampoline_bytes += trampoline_bytes;
      report.largest_previous_bb_branch_edge_trampoline_bytes =
          std::max(report.largest_previous_bb_branch_edge_trampoline_bytes,
                   trampoline_bytes);
    }
    if (planned.site.force_lane0_exec_for_fixed_counter)
      ++report.exec_empty_fixed_counter_edges;

    const bool local = planned.result.placement == EdgeTrampolinePlacement::LocalTextCave;
    if (local) {
      ++report.planned_local_edge_trampolines;
      report.planned_local_edge_trampoline_bytes += trampoline_bytes;
      if (previous_bb_branch) {
        ++report.previous_bb_branch_planned_local_edge_trampolines;
        report.previous_bb_branch_planned_local_edge_trampoline_bytes +=
            trampoline_bytes;
      }
    } else {
      ++report.planned_appended_edge_trampolines;
      report.planned_appended_edge_trampoline_bytes += trampoline_bytes;
      if (previous_bb_branch) {
        ++report.previous_bb_branch_planned_appended_edge_trampolines;
        report.previous_bb_branch_planned_appended_edge_trampoline_bytes +=
            trampoline_bytes;
      }
    }

    auto summary_index = summary_indices.find(planned.site.kernel_name);
    if (summary_index == summary_indices.end())
      continue;
    KernelEdgeSelectionSummary &summary = report.kernel_summaries[summary_index->second];
    ++summary.edge_trampolines_planned;
    summary.planned_edge_trampoline_bytes += trampoline_bytes;
    summary.largest_edge_trampoline_bytes =
        std::max(summary.largest_edge_trampoline_bytes, trampoline_bytes);
    if (previous_bb_branch) {
      ++summary.previous_bb_branch_edge_trampolines_planned;
      summary.previous_bb_branch_edge_trampoline_bytes += trampoline_bytes;
      summary.largest_previous_bb_branch_edge_trampoline_bytes =
          std::max(summary.largest_previous_bb_branch_edge_trampoline_bytes,
                   trampoline_bytes);
    }
    if (planned.site.force_lane0_exec_for_fixed_counter)
      ++summary.exec_empty_fixed_counter_edges;
    if (local) {
      ++summary.planned_local_edge_trampolines;
      summary.planned_local_edge_trampoline_bytes += trampoline_bytes;
      if (previous_bb_branch) {
        ++summary.previous_bb_branch_planned_local_edge_trampolines;
        summary.previous_bb_branch_planned_local_edge_trampoline_bytes +=
            trampoline_bytes;
      }
    } else {
      ++summary.planned_appended_edge_trampolines;
      summary.planned_appended_edge_trampoline_bytes += trampoline_bytes;
      if (previous_bb_branch) {
        ++summary.previous_bb_branch_planned_appended_edge_trampolines;
        summary.previous_bb_branch_planned_appended_edge_trampoline_bytes +=
            trampoline_bytes;
      }
    }
  }
  finalize_previous_bb_branch_overhead(report);
  for (KernelEdgeSelectionSummary &summary : report.kernel_summaries)
    finalize_previous_bb_branch_overhead(summary);
  report.sampled_selected_edges.clear();
  const bool previous_bb_aggregate_cap_active =
      std::any_of(report.kernel_summaries.begin(), report.kernel_summaries.end(),
                  [](const KernelEdgeSelectionSummary &summary) {
                    return summary.previous_bb_branch_edge_over_budget != 0 ||
                           summary.previous_bb_branch_site_over_budget != 0;
                  });
  const size_t selected_edge_sample_limit =
      (report.allow_opaque_fresh_registers || previous_bb_aggregate_cap_active)
          ? std::numeric_limits<size_t>::max()
          : 32;
  for (const PlannedEdgeTrampoline &planned : plan.edge_trampolines) {
    if (report.sampled_selected_edges.size() >= selected_edge_sample_limit)
      break;
    report.sampled_selected_edges.push_back(selected_edge_sample(planned));
  }
}

void install_planned_edge_trampoline(const PlannedEdgeTrampoline &planned,
                                     std::vector<uint8_t> &text,
                                     rocjitsu::CodeObjectPatcher &patcher,
                                     rj_code_arch_t arch) {
  if (planned.result.placement == EdgeTrampolinePlacement::LocalTextCave) {
    install_local_edge_trampoline(planned.site, planned.trampoline, text,
                                  planned.result.cave_text_offset, arch);
    return;
  }

  install_edge_site_redirect(planned.site, planned.trampoline, text, arch);
  patcher.append_cave_body(planned.trampoline.cave_words);
}

std::optional<AmdgpuMetadataPrivateSegmentPatch>
plan_amdgpu_metadata_integer_patch(std::span<const uint8_t> image,
                                   std::string_view kernel_name,
                                   std::string_view field_name,
                                   uint32_t new_value,
                                   const char **failure_reason) {
  auto fail = [&](const char *reason) -> std::optional<AmdgpuMetadataPrivateSegmentPatch> {
    if (failure_reason != nullptr)
      *failure_reason = reason;
    return std::nullopt;
  };

  auto ehdr = read_struct<rocjitsu::Elf64_Ehdr>(image, 0);
  if (!ehdr ||
      std::memcmp(ehdr->e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE) != 0 ||
      ehdr->e_ident[rocjitsu::EI_CLASS] != rocjitsu::ELFCLASS64)
    return fail("AMDGPU metadata patch requires a 64-bit ELF image");
  if (ehdr->e_shoff > image.size() ||
      ehdr->e_shnum > (image.size() - ehdr->e_shoff) / sizeof(rocjitsu::Elf64_Shdr))
    return fail("AMDGPU metadata patch requires valid ELF section headers");

  const uint64_t shdr_table = ehdr->e_shoff;
  const char *last_failure = nullptr;
  for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
    auto shdr = read_struct<rocjitsu::Elf64_Shdr>(
        image, shdr_table + static_cast<uint64_t>(i) * sizeof(rocjitsu::Elf64_Shdr));
    if (!shdr || shdr->sh_type != rocjitsu::SHT_NOTE)
      continue;
    if (shdr->sh_offset > image.size() || shdr->sh_size > image.size() - shdr->sh_offset)
      return fail("AMDGPU metadata note section is out of range");

    uint64_t note_offset = shdr->sh_offset;
    const uint64_t note_end = shdr->sh_offset + shdr->sh_size;
    while (note_offset + sizeof(ElfNoteHeader) <= note_end) {
      auto note = read_struct<ElfNoteHeader>(image, note_offset);
      if (!note)
        return fail("AMDGPU metadata note header is malformed");
      const uint64_t name_offset = note_offset + sizeof(ElfNoteHeader);
      const uint64_t desc_offset = align_up_u64(name_offset + note->namesz, 4);
      const uint64_t next_note = align_up_u64(desc_offset + note->descsz, 4);
      if (name_offset > note_end || note->namesz > note_end - name_offset ||
          desc_offset > note_end || note->descsz > note_end - desc_offset ||
          next_note > note_end) {
        return fail("AMDGPU metadata note payload is out of range");
      }

      const bool amdgpu_note =
          note->type == rocjitsu::NT_AMDGPU_METADATA && note->namesz >= 6 &&
          std::memcmp(image.data() + name_offset, "AMDGPU", 6) == 0;
      if (amdgpu_note) {
        MsgpackIntegerPatch integer_patch;
        const char *metadata_failure = nullptr;
        std::span<const uint8_t> metadata(image.data() + desc_offset, note->descsz);
        if (plan_metadata_integer_patch_from_metadata(metadata, kernel_name, field_name,
                                                     new_value, &integer_patch,
                                                     &metadata_failure)) {
          AmdgpuMetadataPrivateSegmentPatch patch;
          if (integer_patch.kind == MsgpackIntegerPatch::Kind::InPlaceBytes) {
            patch.kind = AmdgpuMetadataPrivateSegmentPatch::Kind::InPlaceBytes;
            patch.file_offset = desc_offset + integer_patch.offset;
            patch.bytes = integer_patch.bytes;
            patch.size = integer_patch.size;
            return patch;
          }

          std::optional<std::vector<uint8_t>> rebuilt_note_section =
              rebuild_note_section_with_metadata_patch(
                  image, *shdr, note_offset, *note, name_offset, desc_offset, next_note,
                  metadata, integer_patch, &metadata_failure);
          if (!rebuilt_note_section) {
            last_failure = metadata_failure;
            note_offset = next_note;
            continue;
          }
          patch.kind = AmdgpuMetadataPrivateSegmentPatch::Kind::RebuiltNoteSection;
          patch.note_section_file_offset = shdr->sh_offset;
          patch.note_section_bytes = std::move(*rebuilt_note_section);
          return patch;
        }
        last_failure = metadata_failure;
      }
      note_offset = next_note;
    }
  }

  return fail(last_failure != nullptr ? last_failure : "AMDGPU metadata note is missing");
}

std::optional<AmdgpuMetadataPrivateSegmentPatch>
plan_amdgpu_metadata_private_segment_patch(std::span<const uint8_t> image,
                                           std::string_view kernel_name,
                                           uint32_t private_segment_fixed_size,
                                           const char **failure_reason) {
  return plan_amdgpu_metadata_integer_patch(
      image, kernel_name, ".private_segment_fixed_size", private_segment_fixed_size,
      failure_reason);
}

std::optional<AmdgpuMetadataPrivateSegmentPatch>
plan_amdgpu_metadata_sgpr_count_patch(std::span<const uint8_t> image,
                                      std::string_view kernel_name,
                                      uint32_t sgpr_count,
                                      const char **failure_reason) {
  return plan_amdgpu_metadata_integer_patch(image, kernel_name, ".sgpr_count",
                                           sgpr_count, failure_reason);
}

const char *amdgpu_metadata_private_segment_patch_kind_name(
    AmdgpuMetadataPrivateSegmentPatch::Kind kind) {
  switch (kind) {
  case AmdgpuMetadataPrivateSegmentPatch::Kind::InPlaceBytes:
    return "in-place-bytes";
  case AmdgpuMetadataPrivateSegmentPatch::Kind::RebuiltNoteSection:
    return "rebuilt-note-section";
  }
  return "unknown";
}

std::optional<KernelDescriptorResourceSummary>
plan_kernel_descriptor_resources(std::span<const uint8_t> image, const KernelSite &site,
                                 const ProbeRegisterRequirements &requirements,
                                 const char **failure_reason) {
  auto fail = [&](const char *reason) -> std::optional<KernelDescriptorResourceSummary> {
    if (failure_reason != nullptr)
      *failure_reason = reason;
    return std::nullopt;
  };

  auto desc = read_struct<KD>(image, site.descriptor_file_offset);
  if (!desc)
    return fail("kernel descriptor is out of range");

  const bool wave32 = AMDHSA_BITS_GET(desc->kernel_code_properties,
                                      kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) != 0;
  const uint32_t vgpr_granularity = wave32 ? 8 : 4;
  const uint32_t min_vgpr_granulated =
      register_count_to_granulated(requirements.vgprs, vgpr_granularity);
  const uint32_t sgpr_granularity = 8;

  const uint32_t old_vgpr_granulated = AMDHSA_BITS_GET(
      desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  const uint32_t old_sgpr_granulated = AMDHSA_BITS_GET(
      desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  const uint32_t descriptor_sgpr_count =
      granulated_to_register_count(old_sgpr_granulated, sgpr_granularity);
  const bool descriptor_sgpr_effective = site.descriptor_sgpr_count_effective;
  const uint32_t metadata_sgpr_count =
      site.has_metadata_sgpr_count ? site.metadata_sgpr_count : 0;
  uint32_t old_sgpr_count = site.allocated_sgpr_count;
  if (old_sgpr_count == 0) {
    if (!descriptor_sgpr_effective)
      old_sgpr_count = metadata_sgpr_count;
    else if (site.has_metadata_sgpr_count)
      old_sgpr_count = std::max(descriptor_sgpr_count, metadata_sgpr_count);
    else
      old_sgpr_count = descriptor_sgpr_count;
  }
  if (!descriptor_sgpr_effective && requirements.sgprs > old_sgpr_count &&
      !site.has_metadata_sgpr_count)
    return fail("SGPR growth unsupported because AMDGPU metadata SGPR count is missing");
  const uint32_t patched_sgpr_count = std::max(requirements.sgprs, old_sgpr_count);
  const uint32_t min_sgpr_granulated =
      register_count_to_granulated(patched_sgpr_count, sgpr_granularity);
  const uint32_t patched_vgpr_granulated =
      std::max(old_vgpr_granulated, min_vgpr_granulated);
  const uint32_t patched_sgpr_granulated =
      descriptor_sgpr_effective ? std::max(old_sgpr_granulated, min_sgpr_granulated)
                                : old_sgpr_granulated;
  const uint32_t old_private_segment_fixed_size = desc->private_segment_fixed_size;
  const uint32_t patched_private_segment_fixed_size =
      std::max(old_private_segment_fixed_size, requirements.private_segment_bytes);
  const bool old_private_segment_enabled =
      AMDHSA_BITS_GET(desc->compute_pgm_rsrc2,
                      kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT) != 0;
  const bool patched_private_segment_enabled =
      old_private_segment_enabled || patched_private_segment_fixed_size != 0;

  KernelDescriptorResourceSummary summary;
  summary.kernel_name = site.name;
  summary.descriptor_file_offset = site.descriptor_file_offset;
  summary.wave32 = wave32;
  summary.has_metadata_sgpr_count = site.has_metadata_sgpr_count;
  summary.descriptor_sgpr_count_effective = descriptor_sgpr_effective;
  summary.fresh_sgpr_growth_supported = site.fresh_sgpr_growth_supported;
  summary.old_private_segment_enabled = old_private_segment_enabled;
  summary.patched_private_segment_enabled = patched_private_segment_enabled;
  summary.vgpr_granularity = vgpr_granularity;
  summary.sgpr_granularity = sgpr_granularity;
  summary.old_vgpr_granulated = old_vgpr_granulated;
  summary.old_sgpr_granulated = old_sgpr_granulated;
  summary.patched_vgpr_granulated = patched_vgpr_granulated;
  summary.patched_sgpr_granulated = patched_sgpr_granulated;
  summary.descriptor_sgpr_count = descriptor_sgpr_count;
  summary.metadata_sgpr_count = metadata_sgpr_count;
  summary.old_vgpr_count = granulated_to_register_count(old_vgpr_granulated,
                                                        vgpr_granularity);
  summary.old_sgpr_count = old_sgpr_count;
  summary.patched_vgpr_count = granulated_to_register_count(patched_vgpr_granulated,
                                                            vgpr_granularity);
  summary.patched_sgpr_count =
      descriptor_sgpr_effective
          ? granulated_to_register_count(patched_sgpr_granulated, sgpr_granularity)
          : patched_sgpr_count;
  summary.old_private_segment_fixed_size = old_private_segment_fixed_size;
  summary.patched_private_segment_fixed_size = patched_private_segment_fixed_size;
  summary.spill_bytes =
      patched_private_segment_fixed_size - old_private_segment_fixed_size;
  summary.resource_fields_changed =
      old_vgpr_granulated != patched_vgpr_granulated ||
      (descriptor_sgpr_effective &&
       old_sgpr_granulated != patched_sgpr_granulated) ||
      (!descriptor_sgpr_effective && old_sgpr_count != patched_sgpr_count) ||
      old_private_segment_fixed_size != patched_private_segment_fixed_size ||
      old_private_segment_enabled != patched_private_segment_enabled;

  return summary;
}

bool patch_kernel_descriptor_resources(rocjitsu::CodeObjectPatcher &patcher,
                                       const KernelDescriptorResourceSummary &summary,
                                       const char **failure_reason,
                                       std::optional<AmdgpuMetadataPrivateSegmentPatch::Kind>
                                           *applied_private_segment_metadata_patch,
                                       std::optional<AmdgpuMetadataPrivateSegmentPatch::Kind>
                                           *applied_sgpr_count_metadata_patch) {
  auto fail = [&](const char *reason) -> bool {
    if (failure_reason != nullptr)
      *failure_reason = reason;
    return false;
  };

  if (applied_private_segment_metadata_patch != nullptr)
    applied_private_segment_metadata_patch->reset();
  if (applied_sgpr_count_metadata_patch != nullptr)
    applied_sgpr_count_metadata_patch->reset();

  auto desc = read_struct<KD>(patcher.image_bytes(), summary.descriptor_file_offset);
  if (!desc)
    return fail("descriptor resource patch descriptor is out of range");

  auto apply_metadata_patch = [&](const AmdgpuMetadataPrivateSegmentPatch &patch,
                                  const char *write_failure,
                                  const char *replace_failure) -> bool {
    if (patch.kind == AmdgpuMetadataPrivateSegmentPatch::Kind::InPlaceBytes) {
      if (!patcher.patch_bytes(patch.file_offset, {patch.bytes.data(), patch.size}))
        return fail(write_failure);
      return true;
    }
    if (!patcher.replace_note_section(patch.note_section_file_offset,
                                      patch.note_section_bytes))
      return fail(replace_failure);
    return true;
  };

  if (!summary.descriptor_sgpr_count_effective &&
      summary.patched_sgpr_count != summary.old_sgpr_count) {
    const char *metadata_failure = nullptr;
    std::optional<AmdgpuMetadataPrivateSegmentPatch> sgpr_count_metadata_patch =
        plan_amdgpu_metadata_sgpr_count_patch(
            patcher.image_bytes(), summary.kernel_name, summary.patched_sgpr_count,
            &metadata_failure);
    if (!sgpr_count_metadata_patch)
      return fail(metadata_failure != nullptr ? metadata_failure
                                              : "AMDGPU metadata SGPR count patch failed");
    if (!apply_metadata_patch(*sgpr_count_metadata_patch,
                              "AMDGPU metadata SGPR count write failed",
                              "AMDGPU metadata note section replacement failed"))
      return false;
    if (applied_sgpr_count_metadata_patch != nullptr)
      *applied_sgpr_count_metadata_patch = sgpr_count_metadata_patch->kind;
  }

  if (summary.patched_private_segment_fixed_size != summary.old_private_segment_fixed_size) {
    const char *metadata_failure = nullptr;
    std::optional<AmdgpuMetadataPrivateSegmentPatch> private_segment_metadata_patch =
        plan_amdgpu_metadata_private_segment_patch(
        patcher.image_bytes(), summary.kernel_name,
        summary.patched_private_segment_fixed_size, &metadata_failure);
    if (!private_segment_metadata_patch)
      return fail(metadata_failure != nullptr ? metadata_failure
                                              : "AMDGPU metadata private segment patch failed");
    if (!apply_metadata_patch(*private_segment_metadata_patch,
                              "AMDGPU metadata private segment write failed",
                              "AMDGPU metadata note section replacement failed"))
      return false;
    if (applied_private_segment_metadata_patch != nullptr)
      *applied_private_segment_metadata_patch = private_segment_metadata_patch->kind;
  }

  AMDHSA_BITS_SET(desc->compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  summary.patched_vgpr_granulated);
  if (summary.descriptor_sgpr_count_effective) {
    AMDHSA_BITS_SET(desc->compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                    summary.patched_sgpr_granulated);
  }
  desc->private_segment_fixed_size = summary.patched_private_segment_fixed_size;
  AMDHSA_BITS_SET(desc->compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT,
                  summary.patched_private_segment_enabled ? 1 : 0);

  const auto *bytes = reinterpret_cast<const uint8_t *>(&*desc);
  if (!patcher.patch_kernel_descriptor(summary.descriptor_file_offset, {bytes, sizeof(KD)}))
    return fail("descriptor resource patch descriptor write failed");
  return true;
}

std::optional<KernelDescriptorResourceSummary>
patch_kernel_descriptor_for_requirements(rocjitsu::CodeObjectPatcher &patcher,
                                         const KernelSite &site,
                                         const ProbeRegisterRequirements &requirements) {
  std::optional<KernelDescriptorResourceSummary> summary =
      plan_kernel_descriptor_resources(patcher.image_bytes(), site, requirements);
  if (!summary)
    return std::nullopt;
  if (!patch_kernel_descriptor_resources(patcher, *summary))
    return std::nullopt;
  return summary;
}

std::optional<KernelDescriptorResourceSummary>
patch_kernel_descriptor_for_probe(rocjitsu::CodeObjectPatcher &patcher, const KernelSite &site,
                                  const Rdna4ProbeRegisters &probe_registers) {
  return patch_kernel_descriptor_for_requirements(patcher, site,
                                                 probe_register_requirements(probe_registers));
}

std::optional<ProbeScratchSpillPlan>
plan_probe_scratch_spills(rj_code_arch_t arch, uint8_t address_vgpr,
                          std::span<const uint8_t> spilled_vgprs,
                          std::span<const uint8_t> spilled_sgprs,
                          uint32_t original_private_segment_bytes, bool wave32) {
  uint32_t private_segment_limit = 0;
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    private_segment_limit = 0x1000u;
    break;
  case ROCJITSU_CODE_ARCH_RDNA4:
    private_segment_limit = 0x800000u;
    break;
  default:
    return std::nullopt;
  }

  SpillManager spill_manager(original_private_segment_bytes, private_segment_limit);
  ProbeScratchSpillPlan plan;
  plan.address_vgpr = address_vgpr;
  plan.wave32 = wave32;
  plan.vgpr_spills.reserve(spilled_vgprs.size());
  plan.sgpr_spills.reserve(spilled_sgprs.size());
  for (uint8_t vgpr : spilled_vgprs) {
    if (vgpr == address_vgpr)
      return std::nullopt;
    std::optional<uint32_t> offset =
        spill_manager.allocate_slot({RegClass::VGPR, vgpr, 1});
    if (!offset)
      return std::nullopt;
    plan.vgpr_spills.push_back({vgpr, *offset});
  }
  for (uint8_t sgpr : spilled_sgprs) {
    std::optional<uint32_t> offset =
        spill_manager.allocate_slot({RegClass::SGPR, sgpr, 1});
    if (!offset)
      return std::nullopt;
    plan.sgpr_spills.push_back({sgpr, *offset});
  }
  std::optional<uint32_t> private_segment_bytes =
      round_up_power_of_two(spill_manager.total_private_bytes(), private_segment_limit);
  if (!private_segment_bytes)
    return std::nullopt;
  plan.private_segment_bytes = *private_segment_bytes;
  return plan;
}

std::optional<ProbeScratchSpillPlan>
plan_vgpr_scratch_spills(rj_code_arch_t arch, uint8_t address_vgpr,
                         std::span<const uint8_t> spilled_vgprs,
                         uint32_t original_private_segment_bytes, bool wave32) {
  return plan_probe_scratch_spills(arch, address_vgpr, spilled_vgprs,
                                   std::span<const uint8_t>(),
                                   original_private_segment_bytes, wave32);
}

} // namespace rocjitsu::fuzzer::afl_dbi
