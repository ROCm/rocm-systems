// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/sidecar_metadata.h"

#include <array>
#include <cstring>
#include <limits>
#include <utility>

namespace rocjitsu {
namespace {

constexpr std::array<uint8_t, 8> kMagic = {'R', 'J', 'V', 'L', 'D', 'S', '1', '\0'};
constexpr uint32_t kVersion = 5;

struct MetadataHeader {
  std::array<uint8_t, 8> magic{};
  uint32_t version = 0;
  uint32_t record_count = 0;
  uint32_t string_bytes = 0;
  uint32_t reserved = 0;
};

struct MetadataRecord {
  uint32_t name_offset = 0;
  uint32_t name_size = 0;
  uint64_t normal_descriptor_vaddr = 0;
  uint64_t virtual_descriptor_vaddr = 0;
  uint32_t static_lds_bytes = 0;
  uint32_t kernarg_size = 0;
  uint32_t backing_pointer_kernarg_offset = 0;
  uint16_t virtual_lds_base_sgpr = 0;
  uint16_t flags = 0;
};

static_assert(sizeof(MetadataHeader) == 24);
static_assert(sizeof(MetadataRecord) == 40);

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

} // namespace

std::vector<uint8_t>
serialize_virtual_lds_metadata(std::span<const VirtualLdsKernelMetadata> kernels) {
  std::vector<uint8_t> strings;
  std::vector<MetadataRecord> records;
  records.reserve(kernels.size());

  for (const VirtualLdsKernelMetadata &kernel : kernels) {
    if (kernel.kernel_name.size() > std::numeric_limits<uint32_t>::max())
      return {};
    if (strings.size() > std::numeric_limits<uint32_t>::max() - kernel.kernel_name.size())
      return {};

    MetadataRecord record{};
    record.name_offset = static_cast<uint32_t>(strings.size());
    record.name_size = static_cast<uint32_t>(kernel.kernel_name.size());
    record.normal_descriptor_vaddr = kernel.normal_descriptor_vaddr;
    record.virtual_descriptor_vaddr = kernel.virtual_descriptor_vaddr;
    record.static_lds_bytes = kernel.static_lds_bytes;
    record.kernarg_size = kernel.kernarg_size;
    record.backing_pointer_kernarg_offset = kernel.backing_pointer_kernarg_offset;
    record.virtual_lds_base_sgpr = kernel.virtual_lds_base_sgpr;
    record.flags = kernel.flags;
    records.push_back(record);
    strings.insert(strings.end(), kernel.kernel_name.begin(), kernel.kernel_name.end());
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

std::optional<std::vector<VirtualLdsKernelMetadata>>
parse_virtual_lds_metadata(std::span<const uint8_t> bytes) {
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

  std::vector<VirtualLdsKernelMetadata> kernels;
  kernels.reserve(header.record_count);
  for (uint32_t i = 0; i < header.record_count; ++i) {
    MetadataRecord record{};
    const size_t offset = sizeof(MetadataHeader) + static_cast<size_t>(i) * sizeof(MetadataRecord);
    if (!read_pod(bytes, offset, record))
      return std::nullopt;
    if (record.name_offset > strings.size() ||
        record.name_size > strings.size() - record.name_offset)
      return std::nullopt;

    VirtualLdsKernelMetadata kernel{};
    kernel.kernel_name.assign(reinterpret_cast<const char *>(strings.data() + record.name_offset),
                              record.name_size);
    kernel.normal_descriptor_vaddr = record.normal_descriptor_vaddr;
    kernel.virtual_descriptor_vaddr = record.virtual_descriptor_vaddr;
    kernel.static_lds_bytes = record.static_lds_bytes;
    kernel.kernarg_size = record.kernarg_size;
    kernel.backing_pointer_kernarg_offset = record.backing_pointer_kernarg_offset;
    kernel.virtual_lds_base_sgpr = record.virtual_lds_base_sgpr;
    kernel.flags = record.flags;
    kernels.push_back(std::move(kernel));
  }
  return kernels;
}

} // namespace rocjitsu
