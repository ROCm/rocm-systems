// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file aql_packet_normalizer.h
/// @brief Backend policy for guest AQL packets crossing a DBT execution boundary.

#ifndef ROCJITSU_VM_AMDGPU_AQL_PACKET_NORMALIZER_H_
#define ROCJITSU_VM_AMDGPU_AQL_PACKET_NORMALIZER_H_

#include <array>
#include <cstddef>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Execution backend receiving a guest-produced AQL packet.
enum class AqlExecutionBackend {
  Hardware,
  Simulator,
};

/// @brief Result of applying backend AQL policy to one 64-byte packet.
enum class AqlPacketDisposition {
  Forwarded,
  NormalizedKernelResources,
  NormalizedExtendedDispatch,
  UnsupportedClusterDispatch,
  MalformedExtendedDispatch,
};

/// @brief Kernel resources known from the loaded executable symbol.
struct AqlKernelResourceRequirements {
  uint32_t private_segment_size = 0;
  uint32_t group_segment_fixed_size = 0;
};

/// @brief One naturally aligned AQL packet slot.
struct alignas(8) AqlPacketSlot {
  std::array<std::byte, 64> bytes{};
};

static_assert(sizeof(AqlPacketSlot) == 64);

/// @brief At most two target packets produced from one guest packet.
///
/// @details An extended dispatch with a dependent signal becomes a barrier-and
/// packet followed by an ordinary dispatch on hardware. Unsupported or
/// malformed packets remain in `packets[0]` so the caller can forward them to
/// the runtime's normal error path after emitting a backend-specific diagnostic.
struct AqlPacketNormalization {
  AqlPacketDisposition disposition = AqlPacketDisposition::Forwarded;
  std::array<AqlPacketSlot, 2> packets{};
  uint32_t packet_count = 0;
};

/// @brief Normalize one guest AQL packet for the selected execution backend.
[[nodiscard]] AqlPacketNormalization
normalize_aql_packet(const void *packet, AqlExecutionBackend backend,
                     AqlKernelResourceRequirements resources = {});

/// @brief Return the kernel-object handle carried by a dispatch packet, or zero.
[[nodiscard]] uint64_t aql_packet_kernel_object(const void *packet);

/// @brief Stable diagnostic spelling for an AQL normalization result.
[[nodiscard]] const char *aql_packet_disposition_name(AqlPacketDisposition disposition);

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_VM_AMDGPU_AQL_PACKET_NORMALIZER_H_
