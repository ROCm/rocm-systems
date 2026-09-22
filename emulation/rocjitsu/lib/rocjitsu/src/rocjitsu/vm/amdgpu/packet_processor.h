// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file packet_processor.h
/// @brief Common contract for processing one protocol packet.

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace rocjitsu::amdgpu {

/// @brief Protocol-independent disposition of one packet-processing attempt.
enum class PacketProcessStatus : uint8_t {
  Complete,
  NeedInput,
  Blocked,
  Faulted,
  Malformed,
  Unsupported,
};

/// @brief Whether the owning ring may advance beyond the head packet.
enum class PacketRetirement : uint8_t {
  Hold,
  Retire,
};

/// @brief Information shared between a protocol processor and its ring owner.
/// @details Required input and retirement extents are normalized to bytes only
/// at this internal boundary. The retirement extent may include protocol-defined
/// skipped commands beyond the decoded packet, so the ring owner must validate it
/// against both ring capacity and the producer-visible extent before advancing.
/// Guest-visible cursors remain protocol-specific and are advanced exclusively
/// by the owning queue or ring component.
struct PacketProcessResult {
  PacketProcessStatus status = PacketProcessStatus::Malformed;
  PacketRetirement retirement = PacketRetirement::Hold;
  std::size_t retirement_bytes = 0;
  std::size_t required_bytes = 0;
};

/// @brief Validate the protocol-independent part of one processor result.
/// @details Protocol owners remain responsible for packet-specific bounds and
/// diagnostics. This helper only rejects contradictory common states so every
/// ring consumer enforces the same retirement and input-growth contract.
[[nodiscard]] constexpr bool valid_packet_process_result(const PacketProcessResult &result,
                                                         std::size_t available_bytes,
                                                         std::size_t byte_alignment = 1) noexcept {
  if (byte_alignment == 0 || (result.retirement_bytes % byte_alignment) != 0 ||
      (result.required_bytes % byte_alignment) != 0) {
    return false;
  }

  if (result.status == PacketProcessStatus::NeedInput) {
    return result.retirement == PacketRetirement::Hold && result.required_bytes > available_bytes;
  }
  if (result.required_bytes != 0)
    return false;

  if (result.status == PacketProcessStatus::Complete) {
    return result.retirement == PacketRetirement::Retire && result.retirement_bytes != 0;
  }
  if (result.status == PacketProcessStatus::Blocked)
    return result.retirement == PacketRetirement::Hold;

  return result.retirement == PacketRetirement::Hold || result.retirement_bytes != 0;
}

/// @brief Result type exposing the common packet-processing envelope.
template <typename Result>
concept PacketProcessorResult = requires(const Result &result) {
  { result.packet_result() } -> std::same_as<const PacketProcessResult &>;
};

/// @brief Compile-time interface implemented independently by each packet protocol.
/// @details The request may carry protocol-specific, caller-owned continuation
/// state. The ring owner retains the relevant address-space snapshot across a
/// blocked attempt and publishes its cursor only after retirement.
template <typename Processor>
concept PacketProcessor = requires(Processor &processor, typename Processor::Request request) {
  requires PacketProcessorResult<typename Processor::Result>;
  { processor.process(std::move(request)) } -> std::same_as<typename Processor::Result>;
};

} // namespace rocjitsu::amdgpu
