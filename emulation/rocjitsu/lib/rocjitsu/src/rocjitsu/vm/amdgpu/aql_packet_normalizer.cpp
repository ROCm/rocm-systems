// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file aql_packet_normalizer.cpp
/// @brief Backend policy for guest AQL packets crossing a DBT execution boundary.

#include "rocjitsu/vm/amdgpu/aql_packet_normalizer.h"

#include "rocjitsu/vm/amdgpu/amd_ext_aql_packet.h"

#include <cstring>
#include <limits>

namespace rocjitsu::amdgpu {
namespace {

constexpr uint16_t field_mask(uint32_t offset, uint32_t width) {
  return static_cast<uint16_t>(((uint32_t{1} << width) - 1u) << offset);
}

constexpr uint16_t kPacketTypeMask =
    field_mask(HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE);
constexpr uint16_t kReleaseFenceMask = field_mask(HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                                                  HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);

constexpr uint16_t replace_packet_type(uint16_t header, hsa_packet_type_t type) {
  return static_cast<uint16_t>((header & ~kPacketTypeMask) |
                               (static_cast<uint16_t>(type) << HSA_PACKET_HEADER_TYPE));
}

constexpr hsa_packet_type_t packet_type(uint16_t header) {
  return static_cast<hsa_packet_type_t>((header & kPacketTypeMask) >> HSA_PACKET_HEADER_TYPE);
}

void copy_slot(AqlPacketSlot &destination, const void *source) {
  std::memcpy(destination.bytes.data(), source, destination.bytes.size());
}

template <typename Packet> void store_packet(AqlPacketSlot &destination, const Packet &packet) {
  static_assert(sizeof(Packet) == sizeof(AqlPacketSlot));
  std::memcpy(destination.bytes.data(), &packet, sizeof(packet));
}

bool checked_grid_size(uint32_t clusters, uint8_t cluster_size, uint16_t workgroup_size,
                       uint32_t &result) {
  if (clusters == 0 || cluster_size == 0 || workgroup_size == 0)
    return false;
  const uint64_t value = static_cast<uint64_t>(clusters) * cluster_size * workgroup_size;
  if (value > std::numeric_limits<uint32_t>::max())
    return false;
  result = static_cast<uint32_t>(value);
  return true;
}

} // namespace

AqlPacketNormalization normalize_aql_packet(const void *packet, AqlExecutionBackend backend,
                                            AqlKernelResourceRequirements resources) {
  AqlPacketNormalization result;
  if (packet == nullptr) {
    result.disposition = AqlPacketDisposition::MalformedExtendedDispatch;
    return result;
  }

  copy_slot(result.packets[0], packet);
  result.packet_count = 1;

  uint16_t header = 0;
  std::memcpy(&header, packet, sizeof(header));
  if (packet_type(header) == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
    if (backend == AqlExecutionBackend::Simulator)
      return result;

    hsa_kernel_dispatch_packet_t dispatch{};
    std::memcpy(&dispatch, packet, sizeof(dispatch));
    const uint32_t private_segment_size =
        std::max(dispatch.private_segment_size, resources.private_segment_size);
    if (private_segment_size == dispatch.private_segment_size)
      return result;

    dispatch.private_segment_size = private_segment_size;
    store_packet(result.packets[0], dispatch);
    result.disposition = AqlPacketDisposition::NormalizedKernelResources;
    return result;
  }

  if (packet_type(header) != HSA_PACKET_TYPE_VENDOR_SPECIFIC)
    return result;

  AmdExtKernelDispatchPacket extended{};
  std::memcpy(&extended, packet, sizeof(extended));
  if (extended.amd_format != kHsaAmdPacketTypeExtKernelDispatch)
    return result;

  // The simulator implements the AMD extended packet and cluster scheduling,
  // so preserving all 64 bytes is the only lossless policy for that backend.
  if (backend == AqlExecutionBackend::Simulator)
    return result;

  uint32_t grid_x = 0;
  uint32_t grid_y = 0;
  uint32_t grid_z = 0;
  if (!checked_grid_size(extended.cluster_count_x, extended.cluster_size_x,
                         extended.workgroup_size_x, grid_x) ||
      !checked_grid_size(extended.cluster_count_y, extended.cluster_size_y,
                         extended.workgroup_size_y, grid_y) ||
      !checked_grid_size(extended.cluster_count_z, extended.cluster_size_z,
                         extended.workgroup_size_z, grid_z)) {
    result.disposition = AqlPacketDisposition::MalformedExtendedDispatch;
    return result;
  }

  if (extended.cluster_size_x != 1 || extended.cluster_size_y != 1 ||
      extended.cluster_size_z != 1) {
    result.disposition = AqlPacketDisposition::UnsupportedClusterDispatch;
    return result;
  }

  hsa_kernel_dispatch_packet_t dispatch{};
  dispatch.header = replace_packet_type(extended.header, HSA_PACKET_TYPE_KERNEL_DISPATCH);
  dispatch.setup = extended.setup;
  dispatch.workgroup_size_x = extended.workgroup_size_x;
  dispatch.workgroup_size_y = extended.workgroup_size_y;
  dispatch.workgroup_size_z = extended.workgroup_size_z;
  dispatch.grid_size_x = grid_x;
  dispatch.grid_size_y = grid_y;
  dispatch.grid_size_z = grid_z;
  dispatch.private_segment_size =
      std::max(extended.private_segment_size, resources.private_segment_size);
  dispatch.group_segment_size = extended.group_segment_size;
  dispatch.kernel_object = extended.kernel_object;
  dispatch.kernarg_address = extended.kernarg_address;
  dispatch.completion_signal = extended.completion_signal;

  if (extended.dep_signal.handle == 0) {
    store_packet(result.packets[0], dispatch);
    result.packet_count = 1;
  } else {
    hsa_barrier_and_packet_t barrier{};
    barrier.header = replace_packet_type(extended.header, HSA_PACKET_TYPE_BARRIER_AND);
    barrier.header = static_cast<uint16_t>(barrier.header & ~kReleaseFenceMask);
    barrier.dep_signal[0] = extended.dep_signal;

    // The ordinary dispatch has no dependent-signal field. Force its barrier
    // bit so it cannot enter launch before the inserted barrier observes the
    // dependency, independently of the original packet's queue-ordering bit.
    dispatch.header = static_cast<uint16_t>(dispatch.header | (1u << HSA_PACKET_HEADER_BARRIER));
    store_packet(result.packets[0], barrier);
    store_packet(result.packets[1], dispatch);
    result.packet_count = 2;
  }

  result.disposition = AqlPacketDisposition::NormalizedExtendedDispatch;
  return result;
}

uint64_t aql_packet_kernel_object(const void *packet) {
  if (packet == nullptr)
    return 0;

  uint16_t header = 0;
  std::memcpy(&header, packet, sizeof(header));
  if (packet_type(header) == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
    hsa_kernel_dispatch_packet_t dispatch{};
    std::memcpy(&dispatch, packet, sizeof(dispatch));
    return dispatch.kernel_object;
  }
  if (packet_type(header) != HSA_PACKET_TYPE_VENDOR_SPECIFIC)
    return 0;

  AmdExtKernelDispatchPacket extended{};
  std::memcpy(&extended, packet, sizeof(extended));
  if (extended.amd_format != kHsaAmdPacketTypeExtKernelDispatch)
    return 0;
  return extended.kernel_object;
}

const char *aql_packet_disposition_name(AqlPacketDisposition disposition) {
  switch (disposition) {
  case AqlPacketDisposition::Forwarded:
    return "forwarded";
  case AqlPacketDisposition::NormalizedKernelResources:
    return "normalized-kernel-resources";
  case AqlPacketDisposition::NormalizedExtendedDispatch:
    return "normalized-extended-dispatch";
  case AqlPacketDisposition::UnsupportedClusterDispatch:
    return "unsupported-cluster-dispatch";
  case AqlPacketDisposition::MalformedExtendedDispatch:
    return "malformed-extended-dispatch";
  }
  return "unknown";
}

} // namespace rocjitsu::amdgpu
