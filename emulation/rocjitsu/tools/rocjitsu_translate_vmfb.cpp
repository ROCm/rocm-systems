// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu_translate_common.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

constexpr uint32_t kZipLocalHeaderSig = 0x04034B50u;
constexpr uint32_t kZipCentralHeaderSig = 0x02014B50u;
constexpr uint32_t kZipEndOfCentralDirectorySig = 0x06054B50u;
constexpr uint32_t kZip64EndOfCentralDirectorySig = 0x06064B50u;
constexpr uint32_t kZip64EndOfCentralDirectoryLocatorSig = 0x07064B50u;
constexpr uint16_t kZip64ExtraId = 0x0001u;
constexpr uint16_t kZipStored = 0u;
constexpr uint32_t kU32Max = 0xFFFFFFFFu;
constexpr uint16_t kU16Max = 0xFFFFu;
constexpr size_t kIreeFlatbufferHeaderSize = 64;
constexpr std::array<uint8_t, 4> kElfMagic = {0x7Fu, 'E', 'L', 'F'};
constexpr std::array<uint8_t, 4> kHipExecutableMagic = {'H', 'I', 'P', '1'};

struct ZipEntry {
  std::string name;
  uint64_t central_header_offset = 0;
  uint64_t local_header_offset = 0;
  uint64_t data_offset = 0;
  uint64_t compressed_size = 0;
  uint64_t uncompressed_size = 0;
  uint16_t flags = 0;
  uint16_t compression = 0;
};

struct ZipCentralDirectory {
  uint64_t offset = 0;
  uint64_t size = 0;
  uint64_t entry_count = 0;
};

struct Zip64Values {
  std::optional<uint64_t> uncompressed_size;
  std::optional<uint64_t> compressed_size;
  std::optional<uint64_t> local_header_offset;
};

struct HipHsacoString {
  size_t field_offset = 0;
  size_t string_offset = 0;
  size_t data_offset = 0;
  size_t size = 0;
};

struct MemberReplacement {
  std::vector<uint8_t> bytes;
  uint64_t old_size = 0;
};

[[nodiscard]] uint16_t read_u16(std::span<const uint8_t> bytes, size_t offset) {
  return static_cast<uint16_t>(bytes[offset]) | (static_cast<uint16_t>(bytes[offset + 1]) << 8);
}

[[nodiscard]] uint32_t read_u32(std::span<const uint8_t> bytes, size_t offset) {
  return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
         (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
         (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

[[nodiscard]] int32_t read_i32(std::span<const uint8_t> bytes, size_t offset) {
  return static_cast<int32_t>(read_u32(bytes, offset));
}

[[nodiscard]] uint64_t read_u64(std::span<const uint8_t> bytes, size_t offset) {
  uint64_t value = 0;
  for (size_t i = 0; i < sizeof(uint64_t); ++i)
    value |= static_cast<uint64_t>(bytes[offset + i]) << (8 * i);
  return value;
}

void write_u32(std::span<uint8_t> bytes, size_t offset, uint32_t value) {
  bytes[offset + 0] = static_cast<uint8_t>(value & 0xFFu);
  bytes[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFFu);
  bytes[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xFFu);
  bytes[offset + 3] = static_cast<uint8_t>((value >> 24) & 0xFFu);
}

void write_u64(std::span<uint8_t> bytes, size_t offset, uint64_t value) {
  for (size_t i = 0; i < sizeof(uint64_t); ++i)
    bytes[offset + i] = static_cast<uint8_t>((value >> (8 * i)) & 0xFFu);
}

[[nodiscard]] uint32_t crc32(std::span<const uint8_t> bytes) {
  uint32_t crc = 0xFFFFFFFFu;
  for (const uint8_t byte : bytes) {
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit)
      crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}

[[nodiscard]] bool has_bytes(std::span<const uint8_t> bytes, size_t offset,
                             std::span<const uint8_t> expected) {
  return offset <= bytes.size() && expected.size() <= bytes.size() - offset &&
         std::equal(expected.begin(), expected.end(), bytes.begin() + offset);
}

[[nodiscard]] std::optional<size_t>
find_signature_from_end(std::span<const uint8_t> bytes, uint32_t signature, size_t max_back_scan) {
  if (bytes.size() < sizeof(uint32_t))
    return std::nullopt;
  const size_t min_pos = bytes.size() > max_back_scan ? bytes.size() - max_back_scan : 0;
  for (size_t pos = bytes.size() - sizeof(uint32_t); pos >= min_pos; --pos) {
    if (read_u32(bytes, pos) == signature)
      return pos;
    if (pos == 0)
      break;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<size_t> find_elf_magic(std::span<const uint8_t> bytes) {
  const auto it = std::search(bytes.begin(), bytes.end(), kElfMagic.begin(), kElfMagic.end());
  if (it == bytes.end())
    return std::nullopt;
  return static_cast<size_t>(std::distance(bytes.begin(), it));
}

[[nodiscard]] std::optional<size_t>
flatbuffer_table_field(std::span<const uint8_t> bytes, size_t table_offset, uint16_t field_index) {
  if (table_offset + sizeof(uint32_t) > bytes.size())
    return std::nullopt;

  const int32_t vtable_distance = read_i32(bytes, table_offset);
  const int64_t vtable_offset_signed =
      static_cast<int64_t>(table_offset) - static_cast<int64_t>(vtable_distance);
  if (vtable_offset_signed < 0 || static_cast<uint64_t>(vtable_offset_signed) > bytes.size())
    return std::nullopt;
  const size_t vtable_offset = static_cast<size_t>(vtable_offset_signed);
  if (vtable_offset + 4 > bytes.size())
    return std::nullopt;

  const uint16_t vtable_size = read_u16(bytes, vtable_offset);
  const size_t slot_offset = 4u + static_cast<size_t>(field_index) * sizeof(uint16_t);
  if (slot_offset + sizeof(uint16_t) > vtable_size ||
      vtable_offset + slot_offset + 2 > bytes.size())
    return std::nullopt;

  const uint16_t object_field_offset = read_u16(bytes, vtable_offset + slot_offset);
  if (object_field_offset == 0)
    return std::nullopt;
  const size_t field_offset = table_offset + object_field_offset;
  if (field_offset + sizeof(uint32_t) > bytes.size())
    return std::nullopt;
  return field_offset;
}

[[nodiscard]] std::optional<HipHsacoString> find_hip_hsaco_string(std::span<const uint8_t> member,
                                                                  std::string &error) {
  if (!has_bytes(member, 0, kHipExecutableMagic))
    return std::nullopt;
  if (member.size() < kIreeFlatbufferHeaderSize + 8) {
    error = "HIP executable FlatBuffer member is smaller than the IREE executable header";
    return std::nullopt;
  }
  const uint64_t content_size = read_u64(member, 8);
  if (content_size == 0 || content_size > member.size() - kIreeFlatbufferHeaderSize) {
    error = "HIP executable FlatBuffer content_size is out of bounds";
    return std::nullopt;
  }

  const size_t fb_base = kIreeFlatbufferHeaderSize;
  if (!has_bytes(member, fb_base + 4, kHipExecutableMagic)) {
    error = "HIP executable FlatBuffer payload is missing HIP1 file identifier";
    return std::nullopt;
  }

  const size_t root_table = fb_base + read_u32(member, fb_base);
  const auto modules_field = flatbuffer_table_field(member, root_table, 1);
  if (!modules_field) {
    error = "HIP executable FlatBuffer is missing ExecutableDef.modules";
    return std::nullopt;
  }
  const size_t modules_vec = *modules_field + read_u32(member, *modules_field);
  if (modules_vec + sizeof(uint32_t) > member.size()) {
    error = "HIP executable modules vector is out of bounds";
    return std::nullopt;
  }

  const uint32_t module_count = read_u32(member, modules_vec);
  std::optional<HipHsacoString> found;
  for (uint32_t i = 0; i < module_count; ++i) {
    const size_t element_offset = modules_vec + sizeof(uint32_t) + static_cast<size_t>(i) * 4u;
    if (element_offset + sizeof(uint32_t) > member.size()) {
      error = "HIP executable modules vector element is out of bounds";
      return std::nullopt;
    }
    const size_t module_table = element_offset + read_u32(member, element_offset);
    const auto hsaco_field = flatbuffer_table_field(member, module_table, 0);
    if (!hsaco_field)
      continue;
    const size_t string_offset = *hsaco_field + read_u32(member, *hsaco_field);
    if (string_offset + sizeof(uint32_t) > member.size()) {
      error = "HIP executable hsaco_image string is out of bounds";
      return std::nullopt;
    }
    const uint32_t string_size = read_u32(member, string_offset);
    const size_t data_offset = string_offset + sizeof(uint32_t);
    if (data_offset > member.size() || string_size > member.size() - data_offset ||
        data_offset + string_size >= member.size()) {
      error = "HIP executable hsaco_image string contents are out of bounds";
      return std::nullopt;
    }
    if (!has_bytes(member, data_offset, kElfMagic))
      continue;
    if (found) {
      error = "HIP executable contains multiple embedded HSACO images; multi-module VMFB rewrite "
              "is not implemented";
      return std::nullopt;
    }
    found = HipHsacoString{*hsaco_field, string_offset, data_offset, string_size};
  }
  if (!found)
    error = "HIP executable FlatBuffer does not contain an embedded HSACO string";
  return found;
}

void append_flatbuffer_string(std::vector<uint8_t> &member, size_t fb_base,
                              size_t string_field_offset, std::span<const uint8_t> value) {
  while ((member.size() - fb_base) % 4 != 0)
    member.push_back(0);

  const size_t string_offset = member.size();
  const uint32_t string_size = static_cast<uint32_t>(value.size());
  member.resize(member.size() + sizeof(uint32_t) + value.size() + 1);
  write_u32(member, string_offset, string_size);
  std::memcpy(member.data() + string_offset + sizeof(uint32_t), value.data(), value.size());
  member[string_offset + sizeof(uint32_t) + value.size()] = 0;
  while ((member.size() - fb_base) % 4 != 0)
    member.push_back(0);

  write_u32(member, string_field_offset,
            static_cast<uint32_t>(string_offset - string_field_offset));
}

[[nodiscard]] bool patch_unique_u64(std::span<uint8_t> bytes, uint64_t old_value,
                                    uint64_t new_value) {
  std::optional<size_t> found;
  for (size_t offset = 0; offset + sizeof(uint64_t) <= bytes.size(); ++offset) {
    if (read_u64(bytes, offset) != old_value)
      continue;
    if (found)
      return false;
    found = offset;
  }
  if (!found)
    return false;
  write_u64(bytes, *found, new_value);
  return true;
}

bool patch_zip64_sizes(std::span<uint8_t> extra, uint64_t new_size) {
  size_t pos = 0;
  while (pos + 4 <= extra.size()) {
    const uint16_t id = read_u16(extra, pos);
    const uint16_t size = read_u16(extra, pos + 2);
    pos += 4;
    if (pos + size > extra.size())
      return false;
    if (id == kZip64ExtraId) {
      if (size >= 8)
        write_u64(extra, pos, new_size);
      if (size >= 16)
        write_u64(extra, pos + 8, new_size);
      return true;
    }
    pos += size;
  }
  return false;
}

void patch_zip_entry_metadata(std::span<uint8_t> bytes, uint64_t local_header_offset,
                              uint64_t central_header_offset, uint64_t new_size, uint32_t crc) {
  const size_t local = static_cast<size_t>(local_header_offset);
  write_u32(bytes, local + 14, crc);
  if (read_u32(bytes, local + 18) != kU32Max)
    write_u32(bytes, local + 18, static_cast<uint32_t>(new_size));
  if (read_u32(bytes, local + 22) != kU32Max)
    write_u32(bytes, local + 22, static_cast<uint32_t>(new_size));
  const uint16_t local_name_len = read_u16(bytes, local + 26);
  const uint16_t local_extra_len = read_u16(bytes, local + 28);
  if (local_extra_len != 0) {
    patch_zip64_sizes(
        std::span<uint8_t>(bytes.data() + local + 30u + local_name_len, local_extra_len), new_size);
  }

  const size_t central = static_cast<size_t>(central_header_offset);
  write_u32(bytes, central + 16, crc);
  if (read_u32(bytes, central + 20) != kU32Max)
    write_u32(bytes, central + 20, static_cast<uint32_t>(new_size));
  if (read_u32(bytes, central + 24) != kU32Max)
    write_u32(bytes, central + 24, static_cast<uint32_t>(new_size));
  const uint16_t central_name_len = read_u16(bytes, central + 28);
  const uint16_t central_extra_len = read_u16(bytes, central + 30);
  if (central_extra_len != 0) {
    patch_zip64_sizes(
        std::span<uint8_t>(bytes.data() + central + 46u + central_name_len, central_extra_len),
        new_size);
  }
}

bool shift_zip_central_directory_offsets(std::span<uint8_t> bytes, uint64_t delta) {
  if (delta == 0)
    return true;

  const auto eocd_pos = find_signature_from_end(bytes, kZipEndOfCentralDirectorySig, 65557);
  if (!eocd_pos || *eocd_pos + 22 > bytes.size())
    return false;

  if (read_u32(bytes, *eocd_pos + 16) != kU32Max)
    write_u32(bytes, *eocd_pos + 16,
              static_cast<uint32_t>(read_u32(bytes, *eocd_pos + 16) + delta));

  if (*eocd_pos >= 20 && read_u32(bytes, *eocd_pos - 20) == kZip64EndOfCentralDirectoryLocatorSig) {
    const size_t locator = *eocd_pos - 20;
    const uint64_t old_zip64_eocd_offset = read_u64(bytes, locator + 8);
    write_u64(bytes, locator + 8, old_zip64_eocd_offset + delta);

    const size_t zip64_eocd = static_cast<size_t>(old_zip64_eocd_offset + delta);
    if (zip64_eocd + 56 > bytes.size() ||
        read_u32(bytes, zip64_eocd) != kZip64EndOfCentralDirectorySig)
      return false;
    write_u64(bytes, zip64_eocd + 48, read_u64(bytes, zip64_eocd + 48) + delta);
  }
  return true;
}

[[nodiscard]] std::optional<std::vector<uint8_t>>
translate_member(const ZipEntry &entry, std::span<const uint8_t> member,
                 rocjitsu::BinaryTranslator &translator, bool fail_on_warnings,
                 std::string &error) {
  std::vector<uint8_t> translated_member(member.begin(), member.end());

  std::string hip_error;
  const auto hip_hsaco = find_hip_hsaco_string(translated_member, hip_error);
  if (!hip_hsaco && has_bytes(translated_member, 0, kHipExecutableMagic)) {
    error = entry.name + ": " + hip_error;
    return std::nullopt;
  }

  std::span<const uint8_t> code_object_bytes;
  if (hip_hsaco) {
    code_object_bytes = std::span<const uint8_t>(translated_member.data() + hip_hsaco->data_offset,
                                                 hip_hsaco->size);
  } else {
    const auto elf_offset = find_elf_magic(translated_member);
    if (!elf_offset)
      return std::vector<uint8_t>{};
    code_object_bytes = std::span<const uint8_t>(translated_member.data() + *elf_offset,
                                                 translated_member.size() - *elf_offset);
  }

  rocjitsu::AmdGpuCodeObject code_object(code_object_bytes.data(), code_object_bytes.size());
  if (!code_object.is_valid()) {
    error = "embedded ELF is not a valid AMDGPU code object in member: " + entry.name;
    return std::nullopt;
  }

  auto result = translator.translate(code_object);
  if (result.elf_bytes.empty()) {
    error = "translation produced an empty code object for member: " + entry.name;
    return std::nullopt;
  }
  for (const auto &warning : result.warnings)
    std::cerr << "warning: " << entry.name << ": " << warning << '\n';
  if (fail_on_warnings && !result.warnings.empty()) {
    error = "translation warnings are fatal for member: " + entry.name;
    return std::nullopt;
  }

  if (hip_hsaco) {
    if (result.elf_bytes.size() <= hip_hsaco->size) {
      write_u32(translated_member, hip_hsaco->string_offset,
                static_cast<uint32_t>(result.elf_bytes.size()));
      std::memcpy(translated_member.data() + hip_hsaco->data_offset, result.elf_bytes.data(),
                  result.elf_bytes.size());
      translated_member[hip_hsaco->data_offset + result.elf_bytes.size()] = 0;
    } else {
      append_flatbuffer_string(translated_member, kIreeFlatbufferHeaderSize,
                               hip_hsaco->field_offset, result.elf_bytes);
      write_u64(translated_member, 8, translated_member.size() - kIreeFlatbufferHeaderSize);
    }
    return translated_member;
  }

  const auto elf_offset = find_elf_magic(translated_member);
  if (!elf_offset || result.elf_bytes.size() != translated_member.size() - *elf_offset) {
    error = "translated HSACO size changed for non-HIP member " + entry.name +
            "; only IREE HIP executable FlatBuffer members can be resized";
    return std::nullopt;
  }
  std::memcpy(translated_member.data() + *elf_offset, result.elf_bytes.data(),
              result.elf_bytes.size());
  return translated_member;
}

[[nodiscard]] Zip64Values parse_zip64_extra(std::span<const uint8_t> extra,
                                            bool needs_uncompressed_size,
                                            bool needs_compressed_size,
                                            bool needs_local_header_offset) {
  Zip64Values values;
  size_t pos = 0;
  while (pos + 4 <= extra.size()) {
    const uint16_t id = read_u16(extra, pos);
    const uint16_t size = read_u16(extra, pos + 2);
    pos += 4;
    if (pos + size > extra.size())
      break;

    if (id == kZip64ExtraId) {
      size_t zip64_pos = pos;
      if (needs_uncompressed_size && zip64_pos + 8 <= pos + size) {
        values.uncompressed_size = read_u64(extra, zip64_pos);
        zip64_pos += 8;
      }
      if (needs_compressed_size && zip64_pos + 8 <= pos + size) {
        values.compressed_size = read_u64(extra, zip64_pos);
        zip64_pos += 8;
      }
      if (needs_local_header_offset && zip64_pos + 8 <= pos + size)
        values.local_header_offset = read_u64(extra, zip64_pos);
      return values;
    }
    pos += size;
  }
  return values;
}

[[nodiscard]] std::optional<ZipCentralDirectory>
find_central_directory(std::span<const uint8_t> bytes) {
  const auto eocd_pos = find_signature_from_end(bytes, kZipEndOfCentralDirectorySig, 65557);
  if (!eocd_pos || *eocd_pos + 22 > bytes.size())
    return std::nullopt;

  ZipCentralDirectory cd;
  cd.entry_count = read_u16(bytes, *eocd_pos + 10);
  cd.size = read_u32(bytes, *eocd_pos + 12);
  cd.offset = read_u32(bytes, *eocd_pos + 16);
  if (cd.entry_count != kU16Max && cd.size != kU32Max && cd.offset != kU32Max)
    return cd;

  if (*eocd_pos < 20 || read_u32(bytes, *eocd_pos - 20) != kZip64EndOfCentralDirectoryLocatorSig)
    return std::nullopt;
  const uint64_t zip64_eocd_offset = read_u64(bytes, *eocd_pos - 12);
  if (zip64_eocd_offset + 56 > bytes.size() ||
      read_u32(bytes, static_cast<size_t>(zip64_eocd_offset)) != kZip64EndOfCentralDirectorySig)
    return std::nullopt;

  cd.entry_count = read_u64(bytes, static_cast<size_t>(zip64_eocd_offset) + 32);
  cd.size = read_u64(bytes, static_cast<size_t>(zip64_eocd_offset) + 40);
  cd.offset = read_u64(bytes, static_cast<size_t>(zip64_eocd_offset) + 48);
  return cd;
}

[[nodiscard]] std::optional<std::vector<ZipEntry>> parse_zip_entries(std::span<const uint8_t> bytes,
                                                                     std::string &error) {
  const auto cd = find_central_directory(bytes);
  if (!cd) {
    error = "failed to locate ZIP central directory";
    return std::nullopt;
  }
  if (cd->offset > bytes.size() || cd->size > bytes.size() - cd->offset) {
    error = "ZIP central directory is outside the file";
    return std::nullopt;
  }

  std::vector<ZipEntry> entries;
  entries.reserve(static_cast<size_t>(std::min<uint64_t>(cd->entry_count, 1024)));
  size_t pos = static_cast<size_t>(cd->offset);
  const size_t end = static_cast<size_t>(cd->offset + cd->size);
  for (uint64_t i = 0; i < cd->entry_count; ++i) {
    if (pos + 46 > end || read_u32(bytes, pos) != kZipCentralHeaderSig) {
      error = "malformed ZIP central directory entry";
      return std::nullopt;
    }

    ZipEntry entry;
    entry.central_header_offset = pos;
    entry.flags = read_u16(bytes, pos + 8);
    entry.compression = read_u16(bytes, pos + 10);
    const uint32_t compressed_size_32 = read_u32(bytes, pos + 20);
    const uint32_t uncompressed_size_32 = read_u32(bytes, pos + 24);
    const uint16_t name_len = read_u16(bytes, pos + 28);
    const uint16_t extra_len = read_u16(bytes, pos + 30);
    const uint16_t comment_len = read_u16(bytes, pos + 32);
    const uint32_t local_header_offset_32 = read_u32(bytes, pos + 42);
    if (pos + 46u + name_len + extra_len + comment_len > end) {
      error = "ZIP central directory entry exceeds central directory bounds";
      return std::nullopt;
    }

    const auto name_begin = bytes.begin() + static_cast<std::ptrdiff_t>(pos + 46);
    entry.name.assign(name_begin, name_begin + name_len);
    const std::span<const uint8_t> extra(bytes.data() + pos + 46u + name_len, extra_len);
    const auto zip64 =
        parse_zip64_extra(extra, uncompressed_size_32 == kU32Max, compressed_size_32 == kU32Max,
                          local_header_offset_32 == kU32Max);
    entry.uncompressed_size = uncompressed_size_32 == kU32Max ? zip64.uncompressed_size.value_or(0)
                                                              : uncompressed_size_32;
    entry.compressed_size =
        compressed_size_32 == kU32Max ? zip64.compressed_size.value_or(0) : compressed_size_32;
    entry.local_header_offset = local_header_offset_32 == kU32Max
                                    ? zip64.local_header_offset.value_or(0)
                                    : local_header_offset_32;

    if (entry.local_header_offset > bytes.size() || entry.local_header_offset + 30 > bytes.size() ||
        read_u32(bytes, static_cast<size_t>(entry.local_header_offset)) != kZipLocalHeaderSig) {
      error = "ZIP local header is invalid for " + entry.name;
      return std::nullopt;
    }
    const size_t local = static_cast<size_t>(entry.local_header_offset);
    const uint16_t local_name_len = read_u16(bytes, local + 26);
    const uint16_t local_extra_len = read_u16(bytes, local + 28);
    entry.data_offset = entry.local_header_offset + 30u + local_name_len + local_extra_len;
    if (entry.data_offset > bytes.size() ||
        entry.compressed_size > bytes.size() - entry.data_offset) {
      error = "ZIP member data exceeds file bounds for " + entry.name;
      return std::nullopt;
    }

    entries.push_back(entry);
    pos += 46u + name_len + extra_len + comment_len;
  }
  return entries;
}

void print_usage(const char *argv0) {
  std::cerr << "usage: " << argv0
            << " <input.vmfb> <output.vmfb> <guest-arch> <host-arch> [host-mach] "
               "[--fail-on-warnings]\n"
            << "example: " << argv0 << " in.vmfb out.vmfb gfx1250 rdna4 gfx1201 "
            << "--fail-on-warnings\n";
}

struct Options {
  std::filesystem::path input;
  std::filesystem::path output;
  rj_code_arch_t guest_arch = ROCJITSU_CODE_ARCH_INVALID;
  rj_code_arch_t host_arch = ROCJITSU_CODE_ARCH_INVALID;
  uint32_t host_mach = 0;
  bool fail_on_warnings = false;
};

[[nodiscard]] std::optional<Options> parse_options(int argc, char **argv) {
  if (argc < 5 || argc > 7)
    return std::nullopt;

  Options opts;
  opts.input = argv[1];
  opts.output = argv[2];
  const auto guest_arch = rocjitsu::tools::parse_arch(argv[3]);
  const auto host_arch = rocjitsu::tools::parse_arch(argv[4]);
  if (!guest_arch || !host_arch)
    return std::nullopt;
  opts.guest_arch = *guest_arch;
  opts.host_arch = *host_arch;

  for (int i = 5; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--fail-on-warnings") {
      opts.fail_on_warnings = true;
      continue;
    }
    const auto host_mach = rocjitsu::tools::parse_mach(arg);
    if (!host_mach)
      return std::nullopt;
    opts.host_mach = *host_mach;
  }
  return opts;
}

} // namespace

int main(int argc, char **argv) {
  const auto opts = parse_options(argc, argv);
  if (!opts) {
    print_usage(argv[0]);
    return 1;
  }

  auto bytes = rocjitsu::tools::read_file(opts->input);
  if (!bytes) {
    std::cerr << "failed to read VMFB: " << opts->input << '\n';
    return 1;
  }

  std::string zip_error;
  const auto central_directory = find_central_directory(*bytes);
  if (!central_directory) {
    std::cerr << "failed to locate ZIP central directory\n";
    return 1;
  }
  auto entries = parse_zip_entries(*bytes, zip_error);
  if (!entries) {
    std::cerr << zip_error << '\n';
    return 1;
  }

  rocjitsu::BinaryTranslator translator(opts->guest_arch, opts->host_arch, opts->host_mach);
  size_t translated_count = 0;
  std::unordered_map<std::string, MemberReplacement> replacements;
  for (const ZipEntry &entry : *entries) {
    if (entry.compression != kZipStored)
      continue;

    std::string error;
    auto translated_member =
        translate_member(entry,
                         std::span<const uint8_t>(bytes->data() + entry.data_offset,
                                                  static_cast<size_t>(entry.uncompressed_size)),
                         translator, opts->fail_on_warnings, error);
    if (!translated_member) {
      std::cerr << error << '\n';
      return 1;
    }
    if (translated_member->empty())
      continue;

    replacements.emplace(entry.name,
                         MemberReplacement{std::move(*translated_member), entry.uncompressed_size});
    ++translated_count;
    std::cout << "translated member " << entry.name << " size=" << entry.uncompressed_size << " -> "
              << replacements.at(entry.name).bytes.size() << '\n';
  }

  if (translated_count == 0) {
    std::cerr << "no embedded AMDGPU code objects found in VMFB: " << opts->input << '\n';
    return 1;
  }

  uint64_t total_delta = 0;
  const ZipEntry *size_changed_entry = nullptr;
  for (const ZipEntry &entry : *entries) {
    const auto it = replacements.find(entry.name);
    if (it == replacements.end())
      continue;
    const int64_t delta = static_cast<int64_t>(it->second.bytes.size()) -
                          static_cast<int64_t>(entry.uncompressed_size);
    if (delta == 0)
      continue;
    if (delta < 0) {
      std::cerr << "shrinking VMFB members is not implemented for member: " << entry.name << '\n';
      return 1;
    }
    if (size_changed_entry != nullptr) {
      std::cerr << "multiple size-changing VMFB executable members are not implemented\n";
      return 1;
    }
    if (entry.data_offset + entry.uncompressed_size != central_directory->offset) {
      std::cerr << "size-changing VMFB member is not the final local ZIP member before the central "
                   "directory: "
                << entry.name << '\n';
      return 1;
    }
    total_delta = static_cast<uint64_t>(delta);
    size_changed_entry = &entry;
  }

  if (size_changed_entry != nullptr) {
    const auto module_it = std::ranges::find_if(
        *entries, [](const ZipEntry &entry) { return entry.name == "module.fb"; });
    if (module_it == entries->end()) {
      std::cerr << "size-changing VMFB rewrite requires a module.fb member\n";
      return 1;
    }
    std::vector<uint8_t> module(
        bytes->begin() + static_cast<std::ptrdiff_t>(module_it->data_offset),
        bytes->begin() +
            static_cast<std::ptrdiff_t>(module_it->data_offset + module_it->uncompressed_size));
    if (!patch_unique_u64(module, size_changed_entry->uncompressed_size,
                          replacements.at(size_changed_entry->name).bytes.size())) {
      std::cerr << "failed to patch unique executable byte length in module.fb for member: "
                << size_changed_entry->name << '\n';
      return 1;
    }
    replacements[module_it->name] =
        MemberReplacement{std::move(module), module_it->uncompressed_size};
  }

  std::vector<const ZipEntry *> entries_by_data_offset;
  entries_by_data_offset.reserve(entries->size());
  for (const ZipEntry &entry : *entries)
    entries_by_data_offset.push_back(&entry);
  std::ranges::sort(entries_by_data_offset, [](const ZipEntry *lhs, const ZipEntry *rhs) {
    return lhs->data_offset < rhs->data_offset;
  });

  std::vector<uint8_t> output;
  output.reserve(bytes->size() + total_delta);
  size_t cursor = 0;
  for (const ZipEntry *entry : entries_by_data_offset) {
    const auto it = replacements.find(entry->name);
    if (it == replacements.end())
      continue;
    const size_t data_offset = static_cast<size_t>(entry->data_offset);
    const size_t old_end = static_cast<size_t>(entry->data_offset + entry->uncompressed_size);
    if (cursor > data_offset) {
      std::cerr << "overlapping VMFB member replacements are not supported\n";
      return 1;
    }
    output.insert(output.end(), bytes->begin() + static_cast<std::ptrdiff_t>(cursor),
                  bytes->begin() + static_cast<std::ptrdiff_t>(data_offset));
    output.insert(output.end(), it->second.bytes.begin(), it->second.bytes.end());
    cursor = old_end;
  }
  output.insert(output.end(), bytes->begin() + static_cast<std::ptrdiff_t>(cursor), bytes->end());

  auto shifted_offset = [&](uint64_t original_offset) {
    if (size_changed_entry != nullptr &&
        original_offset >= size_changed_entry->data_offset + size_changed_entry->uncompressed_size)
      return original_offset + total_delta;
    return original_offset;
  };

  for (const ZipEntry &entry : *entries) {
    const auto it = replacements.find(entry.name);
    if (it == replacements.end())
      continue;
    const uint32_t member_crc = crc32(it->second.bytes);
    patch_zip_entry_metadata(output, shifted_offset(entry.local_header_offset),
                             shifted_offset(entry.central_header_offset), it->second.bytes.size(),
                             member_crc);
  }

  if (!shift_zip_central_directory_offsets(output, total_delta)) {
    std::cerr << "failed to update ZIP central directory offsets\n";
    return 1;
  }

  if (!rocjitsu::tools::write_file(opts->output, output)) {
    std::cerr << "failed to write VMFB: " << opts->output << '\n';
    return 1;
  }
  std::cout << "translated " << translated_count << " embedded AMDGPU code object(s)\n";
  return 0;
}
