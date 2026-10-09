// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace rocjitsu::amdgpu::aql_metadata {

// ROCr hsa_amd_metadata_kernel_dispatch_packet_t / metadata_barrier_packet_t,
// version 0.0: one 256-byte companion per 64-byte AQL slot. CLR publishes all
// four blocks for kernels, but only the first block for barriers. It publishes
// companions before the AQL header; an INVALID companion means memory fallback.
inline constexpr uint32_t kBlockBytes = 64;
inline constexpr uint32_t kBlockWords = kBlockBytes / sizeof(uint32_t);
inline constexpr uint32_t kBlocks = 4;
inline constexpr uint32_t kPacketBytes = kBlocks * kBlockBytes;
inline constexpr uint32_t kPacketWords = kPacketBytes / sizeof(uint32_t);
inline constexpr uint32_t kEventIdWord = 1;
inline constexpr uint32_t kDescriptorWord = 2;
inline constexpr uint32_t kDescriptorBytes = 48;
inline constexpr uint32_t kKernelDescriptorOffset = 16;
inline constexpr uint32_t kPreloadWords = 32;
using Packet = std::array<uint32_t, kPacketWords>;

constexpr uint32_t header_word(uint32_t block) { return block * kBlockWords; }
constexpr uint32_t kernarg_word(uint32_t index) {
  return kBlockWords + 1 + index + index / (kBlockWords - 1);
}
inline uint32_t event_id(std::span<const uint32_t> packet) { return packet[kEventIdWord]; }
inline std::span<const std::byte> kernel_descriptor(std::span<const uint32_t> packet) {
  return std::as_bytes(packet.subspan(kDescriptorWord, kDescriptorBytes / sizeof(uint32_t)));
}
inline std::array<uint32_t, kPreloadWords> kernargs(std::span<const uint32_t> packet) {
  std::array<uint32_t, kPreloadWords> result{};
  for (uint32_t index = 0; index < result.size(); ++index)
    result[index] = packet[kernarg_word(index)];
  return result;
}

// Validate before mapping or reserving queue state, and again on ring updates.
inline bool valid_ring_layout(uint64_t base, uint32_t ring_bytes, uint32_t metadata_bytes) {
  return metadata_bytes == 0 ||
         (uint64_t{metadata_bytes} == uint64_t{ring_bytes} * kBlocks &&
          base <= std::numeric_limits<uint64_t>::max() - ring_bytes - metadata_bytes);
}

} // namespace rocjitsu::amdgpu::aql_metadata
