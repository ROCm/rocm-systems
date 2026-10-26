// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdhsa_metadata.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "util/bit.h"

#include <array>
#include <cstring>
#include <limits>
#include <utility>

namespace rocjitsu::amdhsa {
namespace {

class Cursor {
public:
  explicit Cursor(std::span<const uint8_t> bytes) : bytes_(bytes) {}

  bool read(MetadataValue &value, unsigned depth = 0) {
    if (depth > 32 || ++nodes_ > 65536 || offset_ >= bytes_.size())
      return false;
    value.offset = offset_;
    const uint8_t tag = bytes_[offset_++];
    uint64_t count = 0;
    if (tag < 128 || (tag >= 0xcc && tag <= 0xcf)) {
      value.kind = MetadataValue::Kind::Integer;
      value.integer_bytes = tag < 128 ? 0 : 1u << (tag - 0xcc);
      if (tag < 128) {
        value.integer = tag;
        return true;
      }
      return big_endian(value.integer_bytes, value.integer);
    }
    if ((tag & 0xe0) == 0xa0 || tag == 0xd9 || tag == 0xda || tag == 0xdb) {
      value.kind = MetadataValue::Kind::String;
      if ((tag & 0xe0) == 0xa0)
        count = tag & 31;
      else if (!big_endian(1u << (tag - 0xd9), count))
        return false;
      if (count > bytes_.size() - offset_)
        return false;
      value.string = {reinterpret_cast<const char *>(bytes_.data() + offset_),
                      static_cast<size_t>(count)};
      offset_ += count;
      return true;
    }
    const bool map = (tag & 0xf0) == 0x80 || tag == 0xde || tag == 0xdf;
    const bool array = (tag & 0xf0) == 0x90 || tag == 0xdc || tag == 0xdd;
    if (map || array) {
      value.kind = map ? MetadataValue::Kind::Map : MetadataValue::Kind::Array;
      if ((tag & 0xe0) == 0x80)
        count = tag & 15;
      else if (!big_endian((tag == 0xdc || tag == 0xde) ? 2 : 4, count))
        return false;
      if (count > 65536 || count > bytes_.size() - offset_)
        return false;
      for (uint64_t index = 0; index < count; ++index) {
        MetadataValue key;
        MetadataValue child;
        if (map && (!read(key, depth + 1) || key.kind != MetadataValue::Kind::String))
          return false;
        if (!read(child, depth + 1))
          return false;
        if (map) {
          if (!value.map.emplace(key.string, std::move(child)).second)
            return false;
        } else {
          value.array.push_back(std::move(child));
        }
      }
      return true;
    }
    if (tag == 0xc2 || tag == 0xc3) {
      value.kind = MetadataValue::Kind::Boolean;
      value.integer = tag == 0xc3;
      return true;
    }
    return tag == 0xc0;
  }

  bool at_end() const { return offset_ == bytes_.size(); }

private:
  bool big_endian(unsigned count, uint64_t &value) {
    if (count > 8 || count > bytes_.size() - offset_)
      return false;
    value = 0;
    for (unsigned index = 0; index < count; ++index)
      value = (value << 8) | bytes_[offset_++];
    return true;
  }

  std::span<const uint8_t> bytes_;
  size_t offset_ = 0;
  size_t nodes_ = 0;
};

std::optional<std::span<const uint8_t>> find_metadata_note(std::span<const uint8_t> image) {
  Elf64_Ehdr header{};
  if (image.size() < sizeof(header))
    return std::nullopt;
  std::memcpy(&header, image.data(), sizeof(header));
  if (header.e_shentsize != sizeof(Elf64_Shdr))
    return std::nullopt;
  const std::optional<std::vector<Elf64_Shdr>> sections =
      read_elf_table<Elf64_Shdr>(image, header.e_shoff, header.e_shnum, header.e_shentsize);
  if (!sections)
    return std::nullopt;
  std::optional<std::span<const uint8_t>> result;
  for (const Elf64_Shdr &section : *sections) {
    if (section.sh_type != SHT_NOTE)
      continue;
    if (!util::is_range_in_bounds(section.sh_offset, section.sh_size, image.size()))
      return std::nullopt;
    const std::span<const uint8_t> bytes = image.subspan(section.sh_offset, section.sh_size);
    for (size_t offset = 0; offset < bytes.size();) {
      if (bytes.size() - offset < 12)
        return std::nullopt;
      std::array<uint32_t, 3> framing{};
      std::memcpy(framing.data(), bytes.data() + offset, 12);
      const uint64_t name_bytes = (uint64_t{framing[0]} + 3) & ~uint64_t{3};
      const uint64_t data_bytes = (uint64_t{framing[1]} + 3) & ~uint64_t{3};
      if (name_bytes + data_bytes > bytes.size() - offset - 12)
        return std::nullopt;
      if (framing[2] == NT_AMDGPU_METADATA) {
        if (result || framing[0] != 7 || std::memcmp(bytes.data() + offset + 12, "AMDGPU", 7) != 0)
          return std::nullopt;
        result = bytes.subspan(offset + 12 + name_bytes, framing[1]);
      }
      offset += 12 + name_bytes + data_bytes;
    }
  }
  return result;
}

Result fail(std::string *error, const char *message) {
  if (error != nullptr)
    *error = message;
  return Result::failure();
}

} // namespace

const MetadataValue *MetadataValue::field(std::string_view name) const {
  const std::map<std::string_view, MetadataValue>::const_iterator found = map.find(name);
  return found == map.end() ? nullptr : &found->second;
}

std::optional<uint32_t> MetadataValue::integer_field(std::string_view name) const {
  const MetadataValue *child = field(name);
  if (child == nullptr || child->kind != Kind::Integer ||
      child->integer > std::numeric_limits<uint32_t>::max())
    return std::nullopt;
  return static_cast<uint32_t>(child->integer);
}

bool MetadataValue::text_field_is(std::string_view name, std::string_view expected) const {
  const MetadataValue *child = field(name);
  return child != nullptr && child->kind == Kind::String && child->string == expected;
}

FailureOr<MetadataDocument> MetadataDocument::parse(std::span<const uint8_t> image,
                                                    std::string *error) {
  const std::optional<std::span<const uint8_t>> payload = find_metadata_note(image);
  if (!payload)
    return fail(error, "missing or invalid AMDHSA metadata note");
  MetadataDocument document;
  document.payload_ = *payload;
  Cursor cursor(*payload);
  if (!cursor.read(document.root_) || !cursor.at_end() ||
      document.root_.kind != MetadataValue::Kind::Map)
    return fail(error, "unsupported AMDHSA metadata encoding");
  return document;
}

Result update_resource_metadata(std::span<uint8_t> image, const KernelDescriptorInfo &kernel,
                                uint32_t kernarg_bytes, std::optional<uint32_t> sgprs,
                                std::optional<uint32_t> vgprs, std::string *error) {
  if (!util::is_range_in_bounds(kernel.descriptor_file_offset, sizeof(kernel.descriptor),
                                image.size()))
    return fail(error, "AMDHSA descriptor lies outside the image");
  const FailureOr<MetadataDocument> document = MetadataDocument::parse(image, error);
  if (document.failed())
    return Result::failure();
  const MetadataValue &root = document.value().root();
  const MetadataValue *version = root.field("amdhsa.version");
  if (version == nullptr || version->kind != MetadataValue::Kind::Array ||
      version->array.size() != 2 || version->array[0].kind != MetadataValue::Kind::Integer ||
      version->array[0].integer != 1 || version->array[1].kind != MetadataValue::Kind::Integer ||
      version->array[1].integer != 2)
    return fail(error, "unsupported AMDHSA metadata version");
  const MetadataValue *kernels = root.field("amdhsa.kernels");
  if (kernels == nullptr || kernels->kind != MetadataValue::Kind::Array)
    return fail(error, "missing AMDHSA kernel metadata");
  const MetadataValue *entry = nullptr;
  for (const MetadataValue &candidate : kernels->array) {
    if (candidate.kind != MetadataValue::Kind::Map)
      return fail(error, "invalid AMDHSA kernel metadata entry");
    if (!candidate.text_field_is(".name", kernel.kernel_name))
      continue;
    if (entry != nullptr || !candidate.text_field_is(".symbol", kernel.kernel_name + ".kd"))
      return fail(error, "ambiguous AMDHSA kernel metadata");
    entry = &candidate;
  }
  if (entry == nullptr)
    return fail(error, "missing AMDHSA metadata for the selected kernel");
  const std::span<const uint8_t> payload = document.value().payload();
  const size_t payload_offset = payload.data() - image.data();
  if (payload_offset < kernel.descriptor_file_offset + sizeof(kernel.descriptor) &&
      kernel.descriptor_file_offset < payload_offset + payload.size())
    return fail(error, "AMDHSA metadata overlaps the kernel descriptor");
  const std::array<std::pair<std::string_view, std::optional<uint32_t>>, 3> updates = {{
      {".kernarg_segment_size", kernarg_bytes},
      {".sgpr_count", sgprs},
      {".vgpr_count", vgprs},
  }};
  for (const auto &[name, number] : updates) {
    if (!number)
      continue;
    const MetadataValue *value = entry->field(name);
    if (value == nullptr || value->kind != MetadataValue::Kind::Integer)
      return fail(error, "missing or non-integer AMDHSA resource field");
    if ((value->integer_bytes == 0 && *number > 127) ||
        (value->integer_bytes != 0 && value->integer_bytes < 4 &&
         *number >= (uint64_t{1} << (8 * value->integer_bytes))))
      return fail(error, "resource growth needs a wider metadata integer encoding");
  }
  for (const auto &[name, number] : updates) {
    if (!number)
      continue;
    const MetadataValue &value = *entry->field(name);
    if (value.integer_bytes == 0) {
      image[payload_offset + value.offset] = static_cast<uint8_t>(*number);
    } else {
      for (unsigned index = 0; index < value.integer_bytes; ++index)
        image[payload_offset + value.offset + 1 + index] =
            static_cast<uint8_t>(uint64_t{*number} >> (8 * (value.integer_bytes - index - 1)));
    }
  }
  return Result::success();
}

} // namespace rocjitsu::amdhsa
