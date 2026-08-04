// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/sidecar_metadata.h"

#include <array>
#include <cstring>
#include <limits>
#include <utility>

namespace rocjitsu {
namespace {

constexpr std::array<uint8_t, 8> kMagic = {'R', 'J', 'S', 'I', 'D', 'E', '1', '\0'};
constexpr uint32_t kVersion = 1;

struct MetadataHeader {
  std::array<uint8_t, 8> magic{};
  uint32_t version = 0;
  uint32_t record_count = 0;
  uint32_t string_bytes = 0;
  uint32_t reserved = 0;
};

struct MetadataRecord {
  uint32_t kernel_name_offset = 0;
  uint32_t kernel_name_size = 0;
  uint32_t variant_name_offset = 0;
  uint32_t variant_name_size = 0;
  uint64_t normal_descriptor_vaddr = 0;
  uint64_t variant_descriptor_vaddr = 0;
};

static_assert(sizeof(MetadataHeader) == 24);
static_assert(sizeof(MetadataRecord) == 32);

template <typename T> void append_pod(std::vector<uint8_t> &out, const T &value) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
  out.insert(out.end(), bytes, bytes + sizeof(T));
}

template <typename T>
[[nodiscard]] bool read_pod(std::span<const uint8_t> bytes, size_t offset, T &value) {
  if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
    return false;
  std::memcpy(&value, bytes.data() + offset, sizeof(T));
  return true;
}

[[nodiscard]] bool append_string(std::vector<uint8_t> &strings, std::string_view value,
                                 uint32_t &offset, uint32_t &size) {
  if (value.size() > std::numeric_limits<uint32_t>::max() ||
      strings.size() > std::numeric_limits<uint32_t>::max() - value.size()) {
    return false;
  }
  offset = static_cast<uint32_t>(strings.size());
  size = static_cast<uint32_t>(value.size());
  strings.insert(strings.end(), value.begin(), value.end());
  return true;
}

[[nodiscard]] bool read_string(std::span<const uint8_t> strings, uint32_t offset, uint32_t size,
                               std::string &value) {
  if (offset > strings.size() || size > strings.size() - offset)
    return false;
  value.assign(reinterpret_cast<const char *>(strings.data() + offset), size);
  return true;
}

} // namespace

std::vector<uint8_t> serialize_sidecar_metadata(std::span<const SidecarVariantMetadata> variants) {
  std::vector<uint8_t> strings;
  std::vector<MetadataRecord> records;
  records.reserve(variants.size());

  for (const SidecarVariantMetadata &variant : variants) {
    MetadataRecord record{};
    if (!append_string(strings, variant.kernel_name, record.kernel_name_offset,
                       record.kernel_name_size) ||
        !append_string(strings, variant.variant_name, record.variant_name_offset,
                       record.variant_name_size)) {
      return {};
    }
    record.normal_descriptor_vaddr = variant.normal_descriptor_vaddr;
    record.variant_descriptor_vaddr = variant.variant_descriptor_vaddr;
    records.push_back(record);
  }

  MetadataHeader header{};
  header.magic = kMagic;
  header.version = kVersion;
  header.record_count = static_cast<uint32_t>(records.size());
  header.string_bytes = static_cast<uint32_t>(strings.size());

  std::vector<uint8_t> out;
  out.reserve(sizeof(header) + records.size() * sizeof(MetadataRecord) + strings.size());
  append_pod(out, header);
  for (const MetadataRecord &record : records)
    append_pod(out, record);
  out.insert(out.end(), strings.begin(), strings.end());
  return out;
}

std::optional<std::vector<SidecarVariantMetadata>>
parse_sidecar_metadata(std::span<const uint8_t> bytes) {
  MetadataHeader header{};
  if (!read_pod(bytes, 0, header))
    return std::nullopt;
  if (header.magic != kMagic || header.version != kVersion || header.reserved != 0)
    return std::nullopt;

  const uint64_t records_bytes =
      static_cast<uint64_t>(header.record_count) * sizeof(MetadataRecord);
  const uint64_t strings_offset = sizeof(MetadataHeader) + records_bytes;
  if (strings_offset > bytes.size() || header.string_bytes > bytes.size() - strings_offset)
    return std::nullopt;
  const auto strings = bytes.subspan(static_cast<size_t>(strings_offset), header.string_bytes);

  std::vector<SidecarVariantMetadata> variants;
  variants.reserve(header.record_count);
  for (uint32_t i = 0; i < header.record_count; ++i) {
    MetadataRecord record{};
    const size_t offset = sizeof(MetadataHeader) + static_cast<size_t>(i) * sizeof(MetadataRecord);
    if (!read_pod(bytes, offset, record))
      return std::nullopt;

    SidecarVariantMetadata variant{};
    if (!read_string(strings, record.kernel_name_offset, record.kernel_name_size,
                     variant.kernel_name) ||
        !read_string(strings, record.variant_name_offset, record.variant_name_size,
                     variant.variant_name)) {
      return std::nullopt;
    }
    variant.normal_descriptor_vaddr = record.normal_descriptor_vaddr;
    variant.variant_descriptor_vaddr = record.variant_descriptor_vaddr;
    variants.push_back(std::move(variant));
  }
  return variants;
}

} // namespace rocjitsu
