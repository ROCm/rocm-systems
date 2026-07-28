// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_kernel_metadata.h"

#include <limits>

namespace rocjitsu::amdgpu_code_object_detail {
namespace {

struct MetadataCursor {
  std::span<const uint8_t> bytes;
  size_t offset = 0;
};

[[nodiscard]] bool skip_metadata_bytes(MetadataCursor &cursor, uint64_t count) {
  if (cursor.offset > cursor.bytes.size() || count > cursor.bytes.size() - cursor.offset)
    return false;
  cursor.offset += static_cast<size_t>(count);
  return true;
}

[[nodiscard]] bool read_metadata_be(MetadataCursor &cursor, unsigned byte_count, uint64_t &value) {
  if (byte_count > sizeof(value) || cursor.offset > cursor.bytes.size() ||
      byte_count > cursor.bytes.size() - cursor.offset) {
    return false;
  }
  value = 0;
  for (unsigned i = 0; i < byte_count; ++i)
    value = (value << 8u) | cursor.bytes[cursor.offset++];
  return true;
}

[[nodiscard]] bool read_metadata_collection_count(MetadataCursor &cursor, bool map,
                                                  uint32_t &count) {
  if (cursor.offset >= cursor.bytes.size())
    return false;
  const uint8_t tag = cursor.bytes[cursor.offset++];
  if (map && (tag & 0xf0u) == 0x80u) {
    count = tag & 0x0fu;
    return true;
  }
  if (!map && (tag & 0xf0u) == 0x90u) {
    count = tag & 0x0fu;
    return true;
  }
  uint64_t wide_count = 0;
  if (tag == (map ? 0xdeu : 0xdcu)) {
    if (!read_metadata_be(cursor, 2u, wide_count))
      return false;
  } else if (tag == (map ? 0xdfu : 0xddu)) {
    if (!read_metadata_be(cursor, 4u, wide_count))
      return false;
  } else {
    return false;
  }
  if (wide_count > std::numeric_limits<uint32_t>::max())
    return false;
  count = static_cast<uint32_t>(wide_count);
  return true;
}

[[nodiscard]] bool read_metadata_string(MetadataCursor &cursor, std::string_view &value) {
  if (cursor.offset >= cursor.bytes.size())
    return false;
  const uint8_t tag = cursor.bytes[cursor.offset++];
  uint64_t length = 0;
  if ((tag & 0xe0u) == 0xa0u) {
    length = tag & 0x1fu;
  } else if (tag == 0xd9u) {
    if (!read_metadata_be(cursor, 1u, length))
      return false;
  } else if (tag == 0xdau) {
    if (!read_metadata_be(cursor, 2u, length))
      return false;
  } else if (tag == 0xdbu) {
    if (!read_metadata_be(cursor, 4u, length))
      return false;
  } else {
    return false;
  }
  if (cursor.offset > cursor.bytes.size() || length > cursor.bytes.size() - cursor.offset)
    return false;
  value = std::string_view(reinterpret_cast<const char *>(cursor.bytes.data() + cursor.offset),
                           static_cast<size_t>(length));
  cursor.offset += static_cast<size_t>(length);
  return true;
}

[[nodiscard]] bool read_metadata_unsigned(MetadataCursor &cursor, uint64_t &value) {
  if (cursor.offset >= cursor.bytes.size())
    return false;
  const uint8_t tag = cursor.bytes[cursor.offset++];
  if (tag <= 0x7fu) {
    value = tag;
    return true;
  }
  switch (tag) {
  case 0xccu:
    return read_metadata_be(cursor, 1u, value);
  case 0xcdu:
    return read_metadata_be(cursor, 2u, value);
  case 0xceu:
    return read_metadata_be(cursor, 4u, value);
  case 0xcfu:
    return read_metadata_be(cursor, 8u, value);
  default:
    return false;
  }
}

[[nodiscard]] bool skip_metadata_value(MetadataCursor &cursor, unsigned depth = 0) {
  if (depth > kMaximumKernelMetadataNestingDepth || cursor.offset >= cursor.bytes.size())
    return false;
  const uint8_t tag = cursor.bytes[cursor.offset];
  if (tag <= 0x7fu || tag >= 0xe0u || tag == 0xc0u || tag == 0xc2u || tag == 0xc3u)
    return skip_metadata_bytes(cursor, 1u);
  if ((tag & 0xe0u) == 0xa0u)
    return skip_metadata_bytes(cursor, 1u + (tag & 0x1fu));
  if ((tag & 0xf0u) == 0x90u || (tag & 0xf0u) == 0x80u) {
    ++cursor.offset;
    const uint32_t elements =
        (tag & 0xf0u) == 0x80u ? static_cast<uint32_t>(tag & 0x0fu) * 2u : tag & 0x0fu;
    for (uint32_t i = 0; i < elements; ++i) {
      if (!skip_metadata_value(cursor, depth + 1u))
        return false;
    }
    return true;
  }

  ++cursor.offset;
  uint64_t length = 0;
  const auto skip_length_prefixed = [&](unsigned length_bytes, uint64_t extra) {
    return read_metadata_be(cursor, length_bytes, length) &&
           skip_metadata_bytes(cursor, length + extra);
  };
  switch (tag) {
  case 0xc4u:
  case 0xd9u:
    return skip_length_prefixed(1u, 0u);
  case 0xc5u:
  case 0xdau:
    return skip_length_prefixed(2u, 0u);
  case 0xc6u:
  case 0xdbu:
    return skip_length_prefixed(4u, 0u);
  case 0xc7u:
    return skip_length_prefixed(1u, 1u);
  case 0xc8u:
    return skip_length_prefixed(2u, 1u);
  case 0xc9u:
    return skip_length_prefixed(4u, 1u);
  case 0xcau:
  case 0xceu:
  case 0xd2u:
    return skip_metadata_bytes(cursor, 4u);
  case 0xcbu:
  case 0xcfu:
  case 0xd3u:
    return skip_metadata_bytes(cursor, 8u);
  case 0xccu:
  case 0xd0u:
    return skip_metadata_bytes(cursor, 1u);
  case 0xcdu:
  case 0xd1u:
    return skip_metadata_bytes(cursor, 2u);
  case 0xd4u:
    return skip_metadata_bytes(cursor, 2u);
  case 0xd5u:
    return skip_metadata_bytes(cursor, 3u);
  case 0xd6u:
    return skip_metadata_bytes(cursor, 5u);
  case 0xd7u:
    return skip_metadata_bytes(cursor, 9u);
  case 0xd8u:
    return skip_metadata_bytes(cursor, 17u);
  case 0xdcu:
  case 0xddu:
  case 0xdeu:
  case 0xdfu: {
    const bool map = tag == 0xdeu || tag == 0xdfu;
    const unsigned count_bytes = tag == 0xdcu || tag == 0xdeu ? 2u : 4u;
    if (!read_metadata_be(cursor, count_bytes, length) ||
        (map && length > std::numeric_limits<uint64_t>::max() / 2u)) {
      return false;
    }
    const uint64_t elements = map ? length * 2u : length;
    for (uint64_t i = 0; i < elements; ++i) {
      if (!skip_metadata_value(cursor, depth + 1u))
        return false;
    }
    return true;
  }
  default:
    return false;
  }
}

[[nodiscard]] bool read_kernel_args_metadata(MetadataCursor &cursor, bool &has_dynamic_lds) {
  uint32_t argument_count = 0;
  if (!read_metadata_collection_count(cursor, /*map=*/false, argument_count))
    return false;
  for (uint32_t argument_index = 0; argument_index < argument_count; ++argument_index) {
    uint32_t argument_entries = 0;
    if (!read_metadata_collection_count(cursor, /*map=*/true, argument_entries))
      return false;
    for (uint32_t argument_entry = 0; argument_entry < argument_entries; ++argument_entry) {
      std::string_view argument_key;
      if (!read_metadata_string(cursor, argument_key))
        return false;
      if (argument_key == ".value_kind") {
        std::string_view value_kind;
        if (!read_metadata_string(cursor, value_kind))
          return false;
        has_dynamic_lds |= value_kind == "hidden_dynamic_lds_size";
      } else if (!skip_metadata_value(cursor)) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] KernelMetadataVisitStatus
parse_kernel_metadata_payload(std::span<const uint8_t> payload, const void *context,
                              KernelMetadataVisitorCallback visitor) {
  MetadataCursor root{payload};
  uint32_t root_entries = 0;
  if (!read_metadata_collection_count(root, /*map=*/true, root_entries))
    return KernelMetadataVisitStatus::Malformed;
  for (uint32_t entry = 0; entry < root_entries; ++entry) {
    std::string_view key;
    if (!read_metadata_string(root, key))
      return KernelMetadataVisitStatus::Malformed;
    if (key != "amdhsa.kernels") {
      if (!skip_metadata_value(root))
        return KernelMetadataVisitStatus::Malformed;
      continue;
    }
    uint32_t kernel_count = 0;
    if (!read_metadata_collection_count(root, /*map=*/false, kernel_count))
      return KernelMetadataVisitStatus::Malformed;
    bool visitor_rejected = false;
    for (uint32_t kernel_index = 0; kernel_index < kernel_count; ++kernel_index) {
      uint32_t kernel_entries = 0;
      if (!read_metadata_collection_count(root, /*map=*/true, kernel_entries))
        return KernelMetadataVisitStatus::Malformed;
      std::optional<std::string_view> name;
      KernelMetadata metadata;
      for (uint32_t kernel_entry = 0; kernel_entry < kernel_entries; ++kernel_entry) {
        std::string_view kernel_key;
        if (!read_metadata_string(root, kernel_key))
          return KernelMetadataVisitStatus::Malformed;
        if (kernel_key == ".name") {
          std::string_view parsed_name;
          if (!read_metadata_string(root, parsed_name))
            return KernelMetadataVisitStatus::Malformed;
          name = parsed_name;
        } else if (kernel_key == ".args") {
          if (!read_kernel_args_metadata(root, metadata.has_dynamic_lds))
            return KernelMetadataVisitStatus::Malformed;
        } else if (kernel_key == ".uses_dynamic_stack") {
          if (root.offset >= root.bytes.size())
            return KernelMetadataVisitStatus::Malformed;
          const uint8_t tag = root.bytes[root.offset++];
          if (tag != 0xc2u && tag != 0xc3u)
            return KernelMetadataVisitStatus::Malformed;
          metadata.uses_dynamic_stack = tag == 0xc3u;
        } else if (kernel_key == ".sgpr_count") {
          uint64_t count = 0;
          if (!read_metadata_unsigned(root, count) ||
              count > std::numeric_limits<uint16_t>::max()) {
            return KernelMetadataVisitStatus::Malformed;
          }
          metadata.sgpr_count = static_cast<uint16_t>(count);
        } else if (kernel_key == ".reqd_workgroup_size") {
          uint32_t dimension_count = 0;
          if (!read_metadata_collection_count(root, /*map=*/false, dimension_count) ||
              dimension_count != 3u) {
            return KernelMetadataVisitStatus::Malformed;
          }
          std::array<uint32_t, 3> dimensions{};
          for (uint32_t dimension = 0; dimension < dimension_count; ++dimension) {
            uint64_t value = 0;
            if (!read_metadata_unsigned(root, value) || value == 0u ||
                value > std::numeric_limits<uint32_t>::max()) {
              return KernelMetadataVisitStatus::Malformed;
            }
            dimensions[dimension] = static_cast<uint32_t>(value);
          }
          metadata.required_workgroup_size = dimensions;
        } else if (!skip_metadata_value(root)) {
          return KernelMetadataVisitStatus::Malformed;
        }
      }
      if (!visitor_rejected && name &&
          (metadata.has_dynamic_lds || metadata.uses_dynamic_stack || metadata.sgpr_count ||
           metadata.required_workgroup_size) &&
          !visitor(context, *name, metadata)) {
        visitor_rejected = true;
      }
    }
    return visitor_rejected ? KernelMetadataVisitStatus::VisitorRejected
                            : KernelMetadataVisitStatus::Complete;
  }
  return KernelMetadataVisitStatus::Complete;
}

} // namespace

KernelMetadataVisitStatus
visit_kernel_metadata_payload_erased(std::span<const uint8_t> payload, const void *context,
                                     KernelMetadataVisitorCallback visitor) {
  return parse_kernel_metadata_payload(payload, context, visitor);
}

} // namespace rocjitsu::amdgpu_code_object_detail
