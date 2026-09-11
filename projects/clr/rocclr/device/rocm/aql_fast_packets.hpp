// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hsa/hsa.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace amd::roc::aql_fast {

// Indirect-buffer-only packet layouts. Each packet occupies one 64-byte slot.
// DW0[15:0] is the standard AQL header: vendor type, barrier and fence scopes.
// DW0[23:16] selects FAST_BARRIER(8), FAST_DISPATCH(9), or SET_USER_ARGS(10).
// SET_USER_ARGS uses DW0[27:24] for the first user register and DW0[31:28]
// for count-1; 1..14 payload dwords occupy the trailing slots. Larger ranges
// require a second packet. Barrier/acquire precede register writes; dispatch
// release follows execution. No packet here carries a completion signal.
// FAST_DISPATCH DW1..2 encode workgroup dimensions, DW3..5 grid dimensions,
// DW6..7 the 256-byte-aligned entry address shifted by 8, DW8..10 resource
// registers, DW11 launch initiator, and DW12 private bytes per work-item.
// Other words are reserved zero. Descriptor validation is deliberately stricter
// than ordinary AQL; unsupported inputs fall back before queue publication.
using Packet = std::array<uint32_t, 16>;
struct Descriptor {
  uint32_t group_size, private_size, kernarg_size, reserved0;
  int64_t entry_offset;
  uint8_t reserved1[20];
  uint32_t rsrc3, rsrc1, rsrc2;
  uint16_t properties, preload;
  uint32_t reserved2;
};
static_assert(sizeof(Descriptor) == 64);
static_assert(offsetof(Descriptor, rsrc3) == 44);
static_assert(offsetof(Descriptor, properties) == 56);
static_assert(sizeof(Packet) == 64);

enum class Result { Encoded, Header, Scratch, Descriptor, Geometry, Address, Arguments, Signal };

// Drain shader execution before returning to root completion. Keep this
// boundary even when the last kernel has no release scope. Cache release
// and completion signaling remain owned by the system-release PQ root.
inline Packet completionBarrier() {
  Packet packet{};
  packet[0] = HSA_PACKET_TYPE_VENDOR_SPECIFIC | (8u << 16) |
              (1u << HSA_PACKET_HEADER_BARRIER);
  return packet;
}

// gfx1201 agent release has no cache operation. A following execution wait
// can own that wait for an unsignaled kernel. Passing last=true requires an
// explicit completion-boundary execution wait before the terminal jump.
inline uint16_t coalesceAgentRelease(uint16_t header, uint16_t nextHeader, bool last) {
  constexpr uint16_t releaseMask = 3u << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
  const bool nextWaits = (nextHeader & 0xff) == HSA_PACKET_TYPE_KERNEL_DISPATCH &&
                        (nextHeader & (1u << HSA_PACKET_HEADER_BARRIER));
  if ((header & releaseMask) == (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE) &&
      (last || nextWaits)) return header & ~releaseMask;
  return header;
}

// The caller owns descriptor and kernarg snapshots and guarantees their size.
// No GPU addresses are dereferenced here. Failure leaves output unchanged.
inline Result appendDispatch(const hsa_kernel_dispatch_packet_t& dispatch,
                             const Descriptor& descriptor, const void* kernargs,
                             size_t kernargBytes, std::vector<Packet>& output) {
  constexpr uint16_t typeMask = 0xff;
  constexpr uint16_t barrierMask = 1u << HSA_PACKET_HEADER_BARRIER;
  constexpr uint16_t acquireMask = 3u << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  constexpr uint16_t releaseMask = 3u << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
  const uint16_t header = dispatch.header;
  if ((header & typeMask) != HSA_PACKET_TYPE_KERNEL_DISPATCH ||
      (header & ~(typeMask | barrierMask | acquireMask | releaseMask)) ||
      (header & acquireMask) == acquireMask || (header & releaseMask) == releaseMask ||
      dispatch.setup == 0 || dispatch.setup > 3 || dispatch.reserved0 || dispatch.reserved2) {
    return Result::Header;
  }
  if (dispatch.completion_signal.handle) return Result::Signal;
  const bool scratchEnabled = (descriptor.rsrc2 & 1u) != 0;
  const bool needsScratch = dispatch.private_segment_size || descriptor.private_size || scratchEnabled;
  if ((descriptor.properties & (1u << 11)) ||
      (needsScratch && (!scratchEnabled || !dispatch.private_segment_size ||
                       dispatch.private_segment_size < descriptor.private_size))) return Result::Scratch;
  constexpr uint16_t kernargProperty = 1u << 3;
  constexpr uint16_t wave32Property = 1u << 10;
  if (descriptor.properties & ~(kernargProperty | wave32Property)) return Result::Descriptor;
  const uint32_t userCount = (descriptor.rsrc2 >> 1) & 31u;
  const uint32_t prefix = (descriptor.properties & kernargProperty) ? 2 : 0;
  const uint32_t preloadCount = descriptor.preload & 127u;
  const uint32_t preloadOffset = descriptor.preload >> 7;
  if (userCount > 16 || userCount < prefix || (!prefix && userCount) ||
      preloadCount > userCount - prefix || (!preloadCount && preloadOffset) ||
      uint64_t(preloadOffset + preloadCount) * 4 > descriptor.kernarg_size) {
    return Result::Descriptor;
  }
  if (preloadCount && (!kernargs ||
      uint64_t(preloadOffset + preloadCount) * 4 > kernargBytes)) return Result::Arguments;
  if (prefix && !dispatch.kernarg_address) return Result::Arguments;
  const uint64_t threads = uint64_t(dispatch.workgroup_size_x) *
                           dispatch.workgroup_size_y * dispatch.workgroup_size_z;
  if (!threads || threads > 1024 || !dispatch.grid_size_x || !dispatch.grid_size_y ||
      !dispatch.grid_size_z || dispatch.grid_size_x % dispatch.workgroup_size_x ||
      dispatch.grid_size_y % dispatch.workgroup_size_y ||
      dispatch.grid_size_z % dispatch.workgroup_size_z) return Result::Geometry;
  const uint64_t lds = (uint64_t(dispatch.group_segment_size) + 511) / 512;
  if (dispatch.group_segment_size < descriptor.group_size || lds > 511) return Result::Geometry;
  uint64_t entry;
  if (descriptor.entry_offset >= 0) {
    const uint64_t offset = uint64_t(descriptor.entry_offset);
    if (dispatch.kernel_object > std::numeric_limits<uint64_t>::max() - offset) return Result::Address;
    entry = dispatch.kernel_object + offset;
  } else {
    const uint64_t offset = uint64_t(-(descriptor.entry_offset + 1)) + 1;
    if (dispatch.kernel_object < offset) return Result::Address;
    entry = dispatch.kernel_object - offset;
  }
  if (!entry || (entry & 255u) || (entry >> 48)) return Result::Address;

  uint32_t users[16]{};
  if (prefix) {
    const uint64_t address = reinterpret_cast<uintptr_t>(dispatch.kernarg_address);
    users[0] = uint32_t(address);
    users[1] = uint32_t(address >> 32);
  }
  if (preloadCount) {
    std::memcpy(users + prefix, static_cast<const uint8_t*>(kernargs) + preloadOffset * 4,
                preloadCount * 4);
  }
  // Place prefix dependency before changing registers; release belongs to launch.
  const uint16_t dependency = header & (barrierMask | acquireMask);
  for (uint32_t base = 0; base < userCount;) {
    const uint32_t count = userCount - base > 14 ? 14 : userCount - base;
    Packet packet{};
    packet[0] = HSA_PACKET_TYPE_VENDOR_SPECIFIC | (base == 0 ? dependency : 0) |
                (10u << 16) | ((base | ((count - 1) << 4)) << 24);
    std::memcpy(packet.data() + 16 - count, users + base, count * 4);
    output.push_back(packet);
    base += count;
  }
  Packet packet{};
  packet[0] = HSA_PACKET_TYPE_VENDOR_SPECIFIC | (9u << 16) |
              (header & releaseMask) | (userCount ? 0 : dependency);
  packet[1] = dispatch.workgroup_size_x | (uint32_t(dispatch.workgroup_size_y) << 16);
  packet[2] = dispatch.workgroup_size_z;
  packet[3] = dispatch.grid_size_x;
  packet[4] = dispatch.grid_size_y;
  packet[5] = dispatch.grid_size_z;
  packet[6] = uint32_t(entry >> 8);
  packet[7] = uint32_t(entry >> 40);
  packet[8] = descriptor.rsrc1;
  packet[9] = (descriptor.rsrc2 & ~(511u << 15)) | (uint32_t(lds) << 15);
  packet[10] = descriptor.rsrc3;
  packet[11] = 1u | (1u << 2) | (1u << 5) |
               ((descriptor.properties & wave32Property) ? (1u << 15) : 0);
  // DW12 is private segment bytes per work-item, matching ordinary AQL.
  packet[12] = dispatch.private_segment_size;
  output.push_back(packet);
  return Result::Encoded;
}

}  // namespace amd::roc::aql_fast
