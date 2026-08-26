// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/kernarg_extension.h"

#include <array>
#include <cstring>
#include <limits>
#include <string_view>
#include <utility>

namespace rocjitsu {
namespace {

constexpr std::array<uint8_t, 8> kMetadataMagic = {'R', 'J', 'K', 'A', 'R', 'G', '1', '\0'};
constexpr uint32_t kMetadataVersion = 1;

struct MetadataHeader {
  std::array<uint8_t, 8> magic{};
  uint32_t version = 0;
  uint32_t extension_count = 0;
  uint32_t payload_count = 0;
  uint32_t string_bytes = 0;
};

struct ExtensionRecord {
  uint32_t kernel_name_offset = 0;
  uint32_t kernel_name_size = 0;
  uint32_t variant_name_offset = 0;
  uint32_t variant_name_size = 0;
  uint32_t original_kernarg_size = 0;
  uint32_t first_payload = 0;
  uint32_t payload_count = 0;
  uint32_t reserved = 0;
};

struct PayloadRecord {
  uint32_t name_offset = 0;
  uint32_t name_size = 0;
  uint32_t size = 0;
  uint32_t alignment = 0;
};

static_assert(sizeof(MetadataHeader) == 24);
static_assert(sizeof(ExtensionRecord) == 32);
static_assert(sizeof(PayloadRecord) == 16);

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

[[nodiscard]] bool is_power_of_two(uint32_t value) {
  return value != 0 && (value & (value - 1u)) == 0;
}

[[nodiscard]] std::optional<uint32_t> align_up_u32(uint32_t value, uint32_t alignment) {
  if (!is_power_of_two(alignment))
    return std::nullopt;
  const uint32_t mask = alignment - 1u;
  if (value > std::numeric_limits<uint32_t>::max() - mask)
    return std::nullopt;
  return (value + mask) & ~mask;
}

} // namespace

std::optional<KernargExtensionLayout>
make_kernarg_extension_layout(uint32_t original_kernarg_size,
                              std::span<const KernargExtensionPayloadLayout> payloads) {
  KernargExtensionLayout layout;
  layout.original_kernarg_size = original_kernarg_size;

  // The saved original pointer is part of the extension area. Keep it naturally
  // aligned so the entry prologue can load it with one scalar dwordx2 SMEM op.
  auto cursor = align_up_u32(original_kernarg_size, alignof(uint64_t));
  if (!cursor)
    return std::nullopt;
  layout.original_kernarg_pointer_offset = *cursor;
  if (*cursor > std::numeric_limits<uint32_t>::max() - sizeof(uint64_t))
    return std::nullopt;
  *cursor += static_cast<uint32_t>(sizeof(uint64_t));

  layout.payload_offsets.reserve(payloads.size());
  for (const KernargExtensionPayloadLayout &payload : payloads) {
    cursor = align_up_u32(*cursor, payload.alignment);
    if (!cursor || *cursor > std::numeric_limits<uint32_t>::max() - payload.size)
      return std::nullopt;
    layout.payload_offsets.push_back(*cursor);
    *cursor += payload.size;
  }

  layout.wrapper_size = *cursor;
  return layout;
}

bool write_kernarg_extension_wrapper(std::span<uint8_t> wrapper,
                                     const KernargExtensionLayout &layout,
                                     const void *original_kernarg,
                                     uint64_t original_kernarg_pointer,
                                     std::span<const KernargExtensionPayloadWrite> payloads) {
  if (wrapper.size() != layout.wrapper_size)
    return false;
  if (payloads.size() != layout.payload_offsets.size())
    return false;
  if (layout.wrapper_size < sizeof(uint64_t))
    return false;
  if (layout.original_kernarg_size != 0 && original_kernarg == nullptr)
    return false;
  // KernargExtensionLayout is public plain data, so a forged/inconsistent layout
  // can reach here without going through make_kernarg_extension_layout. Reject a
  // copy that would overrun the wrapper or overlap the saved original pointer
  // before memcpy'ing original_kernarg_size bytes.
  if (layout.original_kernarg_size > layout.wrapper_size ||
      layout.original_kernarg_size > layout.original_kernarg_pointer_offset) {
    return false;
  }
  if (layout.original_kernarg_pointer_offset >
      layout.wrapper_size - static_cast<uint32_t>(sizeof(uint64_t))) {
    return false;
  }

  std::memset(wrapper.data(), 0, wrapper.size());
  if (layout.original_kernarg_size != 0)
    std::memcpy(wrapper.data(), original_kernarg, layout.original_kernarg_size);
  std::memcpy(wrapper.data() + layout.original_kernarg_pointer_offset, &original_kernarg_pointer,
              sizeof(original_kernarg_pointer));

  for (size_t i = 0; i < payloads.size(); ++i) {
    const KernargExtensionPayloadWrite &payload = payloads[i];
    if (payload.size == 0)
      continue;
    if (payload.data == nullptr)
      return false;
    const uint32_t offset = layout.payload_offsets[i];
    if (offset > layout.wrapper_size || payload.size > layout.wrapper_size - offset)
      return false;
    std::memcpy(wrapper.data() + offset, payload.data, payload.size);
  }
  return true;
}

std::vector<uint8_t>
serialize_kernarg_extension_metadata(std::span<const KernargExtensionMetadata> extensions) {
  std::vector<uint8_t> strings;
  std::vector<ExtensionRecord> extension_records;
  std::vector<PayloadRecord> payload_records;
  extension_records.reserve(extensions.size());

  for (const KernargExtensionMetadata &extension : extensions) {
    if (payload_records.size() > std::numeric_limits<uint32_t>::max() ||
        extension.payloads.size() > std::numeric_limits<uint32_t>::max() - payload_records.size()) {
      return {};
    }

    ExtensionRecord record{};
    if (!append_string(strings, extension.kernel_name, record.kernel_name_offset,
                       record.kernel_name_size) ||
        !append_string(strings, extension.variant_name, record.variant_name_offset,
                       record.variant_name_size)) {
      return {};
    }
    record.original_kernarg_size = extension.original_kernarg_size;
    record.first_payload = static_cast<uint32_t>(payload_records.size());
    record.payload_count = static_cast<uint32_t>(extension.payloads.size());

    for (const KernargExtensionPayloadMetadata &payload : extension.payloads) {
      PayloadRecord payload_record{};
      if (!append_string(strings, payload.name, payload_record.name_offset,
                         payload_record.name_size)) {
        return {};
      }
      payload_record.size = payload.size;
      payload_record.alignment = payload.alignment;
      payload_records.push_back(payload_record);
    }
    extension_records.push_back(record);
  }

  MetadataHeader header{};
  header.magic = kMetadataMagic;
  header.version = kMetadataVersion;
  header.extension_count = static_cast<uint32_t>(extension_records.size());
  header.payload_count = static_cast<uint32_t>(payload_records.size());
  header.string_bytes = static_cast<uint32_t>(strings.size());

  std::vector<uint8_t> out;
  out.reserve(sizeof(header) + extension_records.size() * sizeof(ExtensionRecord) +
              payload_records.size() * sizeof(PayloadRecord) + strings.size());
  append_pod(out, header);
  for (const ExtensionRecord &record : extension_records)
    append_pod(out, record);
  for (const PayloadRecord &record : payload_records)
    append_pod(out, record);
  out.insert(out.end(), strings.begin(), strings.end());
  return out;
}

std::optional<std::vector<KernargExtensionMetadata>>
parse_kernarg_extension_metadata(std::span<const uint8_t> bytes) {
  MetadataHeader header{};
  if (!read_pod(bytes, 0, header))
    return std::nullopt;
  if (header.magic != kMetadataMagic || header.version != kMetadataVersion)
    return std::nullopt;

  const uint64_t extension_bytes =
      static_cast<uint64_t>(header.extension_count) * sizeof(ExtensionRecord);
  const uint64_t payload_offset = sizeof(MetadataHeader) + extension_bytes;
  const uint64_t payload_bytes =
      static_cast<uint64_t>(header.payload_count) * sizeof(PayloadRecord);
  const uint64_t strings_offset = payload_offset + payload_bytes;
  if (payload_offset > bytes.size() || strings_offset > bytes.size() ||
      header.string_bytes > bytes.size() - strings_offset) {
    return std::nullopt;
  }
  const auto strings = bytes.subspan(static_cast<size_t>(strings_offset), header.string_bytes);

  std::vector<KernargExtensionMetadata> extensions;
  extensions.reserve(header.extension_count);
  for (uint32_t i = 0; i < header.extension_count; ++i) {
    ExtensionRecord record{};
    const size_t record_offset =
        sizeof(MetadataHeader) + static_cast<size_t>(i) * sizeof(ExtensionRecord);
    if (!read_pod(bytes, record_offset, record) || record.reserved != 0 ||
        record.first_payload > header.payload_count ||
        record.payload_count > header.payload_count - record.first_payload) {
      return std::nullopt;
    }

    KernargExtensionMetadata extension{};
    if (!read_string(strings, record.kernel_name_offset, record.kernel_name_size,
                     extension.kernel_name) ||
        !read_string(strings, record.variant_name_offset, record.variant_name_size,
                     extension.variant_name)) {
      return std::nullopt;
    }
    extension.original_kernarg_size = record.original_kernarg_size;
    extension.payloads.reserve(record.payload_count);
    for (uint32_t j = 0; j < record.payload_count; ++j) {
      PayloadRecord payload_record{};
      const size_t offset = static_cast<size_t>(payload_offset) +
                            static_cast<size_t>(record.first_payload + j) * sizeof(PayloadRecord);
      if (!read_pod(bytes, offset, payload_record))
        return std::nullopt;

      // A non-power-of-two alignment would make the wrapper layout math (which
      // masks with alignment-1) produce garbage offsets, so reject it at parse
      // instead of trusting the on-disk value downstream.
      if (!is_power_of_two(payload_record.alignment))
        return std::nullopt;

      KernargExtensionPayloadMetadata payload{};
      if (!read_string(strings, payload_record.name_offset, payload_record.name_size,
                       payload.name)) {
        return std::nullopt;
      }
      payload.size = payload_record.size;
      payload.alignment = payload_record.alignment;
      extension.payloads.push_back(std::move(payload));
    }
    extensions.push_back(std::move(extension));
  }
  return extensions;
}

} // namespace rocjitsu
