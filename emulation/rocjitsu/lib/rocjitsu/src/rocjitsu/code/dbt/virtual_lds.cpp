// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/virtual_lds.h"
#include "util/bit.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>

namespace rocjitsu {
namespace {

constexpr std::array<uint8_t, 8> kMagic = {'R', 'J', 'V', 'L', 'D', 'S', '1', '\0'};
constexpr uint32_t kVersion = 7;

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
  uint32_t sidecar_name_offset = 0;
  uint32_t sidecar_name_size = 0;
  uint32_t static_lds_bytes = 0;
  uint32_t normal_private_segment_size = 0;
  uint32_t virtual_private_segment_size = 0;
  uint16_t virtual_lds_base_sgpr = 0;
  uint16_t flags = 0;
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

std::vector<uint8_t>
serialize_virtual_lds_metadata(std::span<const VirtualLdsKernelMetadata> kernels) {
  std::vector<uint8_t> strings;
  std::vector<MetadataRecord> records;
  records.reserve(kernels.size());

  for (const VirtualLdsKernelMetadata &kernel : kernels) {
    MetadataRecord record{};
    if (!append_string(strings, kernel.kernel_name, record.kernel_name_offset,
                       record.kernel_name_size) ||
        !append_string(strings, kernel.sidecar_variant_name, record.sidecar_name_offset,
                       record.sidecar_name_size)) {
      return {};
    }
    record.static_lds_bytes = kernel.static_lds_bytes;
    record.normal_private_segment_size = kernel.normal_private_segment_size;
    record.virtual_private_segment_size = kernel.virtual_private_segment_size;
    record.virtual_lds_base_sgpr = kernel.virtual_lds_base_sgpr;
    record.flags = kernel.flags;
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

    // A flag bit outside the known mask means version skew or corruption; the
    // record cannot be interpreted safely, so reject the whole section.
    if ((record.flags & ~kVirtualLdsKnownFlagsMask) != 0)
      return std::nullopt;

    VirtualLdsKernelMetadata kernel{};
    if (!read_string(strings, record.kernel_name_offset, record.kernel_name_size,
                     kernel.kernel_name) ||
        !read_string(strings, record.sidecar_name_offset, record.sidecar_name_size,
                     kernel.sidecar_variant_name)) {
      return std::nullopt;
    }
    kernel.static_lds_bytes = record.static_lds_bytes;
    kernel.normal_private_segment_size = record.normal_private_segment_size;
    kernel.virtual_private_segment_size = record.virtual_private_segment_size;
    kernel.virtual_lds_base_sgpr = record.virtual_lds_base_sgpr;
    kernel.flags = record.flags;
    kernels.push_back(std::move(kernel));
  }
  return kernels;
}

namespace {

[[nodiscard]] std::optional<uint32_t> ceil_div_u32(uint32_t value, uint16_t divisor) {
  if (divisor == 0)
    return std::nullopt;
  return util::ceil_div(value, static_cast<uint32_t>(divisor));
}

[[nodiscard]] VirtualLdsDispatchRewrite reject_dispatch(std::string diagnostic,
                                                        uint64_t requested_lds) {
  VirtualLdsDispatchRewrite result;
  result.decision = VirtualLdsDispatchDecision::Reject;
  result.diagnostic = std::move(diagnostic);
  result.requested_lds = requested_lds;
  return result;
}

} // namespace

VirtualLdsDispatchRewrite plan_virtual_lds_dispatch(const VirtualLdsDispatchPacketFields &packet,
                                                    const VirtualLdsDispatchKernelFacts &kernel,
                                                    uint32_t host_lds_bytes) {
  const uint64_t requested_lds =
      std::max<uint64_t>(kernel.static_lds_bytes, packet.group_segment_size);
  if (host_lds_bytes == 0 || requested_lds <= host_lds_bytes) {
    VirtualLdsDispatchRewrite result;
    result.decision = VirtualLdsDispatchDecision::KeepNormal;
    result.requested_lds = requested_lds;
    result.private_segment_size = packet.private_segment_size;
    return result;
  }

  if (requested_lds == 0 || requested_lds > std::numeric_limits<uint32_t>::max())
    return reject_dispatch("virtual-LDS request exceeds the runtime-state ABI", requested_lds);

  const auto groups_x = ceil_div_u32(packet.grid_size_x, packet.workgroup_size_x);
  const auto groups_y = ceil_div_u32(packet.grid_size_y, packet.workgroup_size_y);
  const auto groups_z = ceil_div_u32(packet.grid_size_z, packet.workgroup_size_z);
  if (!groups_x || !groups_y || !groups_z)
    return reject_dispatch("virtual-LDS dispatch has a zero workgroup dimension", requested_lds);

  if ((*groups_x > 1 && !kernel.has_workgroup_id_x) ||
      (*groups_y > 1 && !kernel.has_workgroup_id_y) ||
      (*groups_z > 1 && !kernel.has_workgroup_id_z)) {
    return reject_dispatch("virtual-LDS dispatch needs an unavailable workgroup-id SGPR",
                           requested_lds);
  }

  const uint32_t allocation_stride_x = static_cast<uint32_t>(requested_lds);
  const auto allocation_stride_y = util::checked_mul(*groups_x, allocation_stride_x);
  if (!allocation_stride_y)
    return reject_dispatch("virtual-LDS X stride exceeds the runtime-state ABI", requested_lds);
  const auto allocation_stride_z = util::checked_mul(*groups_y, *allocation_stride_y);
  if (!allocation_stride_z)
    return reject_dispatch("virtual-LDS Y stride exceeds the runtime-state ABI", requested_lds);
  const auto backing = util::checked_mul(*groups_z, *allocation_stride_z);
  if (!backing)
    return reject_dispatch("virtual-LDS backing allocation exceeds the runtime-state ABI",
                           requested_lds);

  const uint32_t normalized_packet_private =
      std::max(packet.private_segment_size, kernel.normal_private_segment_size);
  const uint32_t dynamic_private = normalized_packet_private - kernel.normal_private_segment_size;
  const auto virtual_private =
      util::checked_add(kernel.virtual_private_segment_size, dynamic_private);
  if (!virtual_private)
    return reject_dispatch("virtual-LDS private-segment request overflows the AQL field",
                           requested_lds);

  VirtualLdsDispatchRewrite result;
  result.decision = VirtualLdsDispatchDecision::UseSidecar;
  result.requested_lds = requested_lds;
  result.private_segment_size = *virtual_private;
  result.geometry = {
      .groups_x = *groups_x,
      .groups_y = *groups_y,
      .groups_z = *groups_z,
      .stride_x = *groups_x > 1 ? allocation_stride_x : 0,
      .stride_y = *groups_y > 1 ? *allocation_stride_y : 0,
      .stride_z = *groups_z > 1 ? *allocation_stride_z : 0,
      .backing_bytes = static_cast<size_t>(std::max(*backing, allocation_stride_x)),
  };
  return result;
}

} // namespace rocjitsu
