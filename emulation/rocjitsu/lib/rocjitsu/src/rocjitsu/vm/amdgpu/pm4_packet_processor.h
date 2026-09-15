// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file pm4_packet_processor.h
/// @brief Processing for the supported PM4 compute-queue packet subset.

#pragma once

#include "rocjitsu/vm/amdgpu/packet_processor.h"

#include <cstdint>
#include <functional>
#include <span>
#include <utility>

namespace rocjitsu::amdgpu {

enum class Pm4PacketDiagnostic : uint8_t {
  None,
  UnsupportedPacket,
  RegisterWriteRejected,
};

enum class Pm4RegisterWriteStatus : uint8_t {
  Complete,
  Blocked,
  Faulted,
  Rejected,
};

class Pm4PacketCallbacks {
public:
  /// Apply one SET_UCONFIG_REG write through the owning register model.
  /// Non-complete outcomes must mean that no register effect occurred.
  std::function<Pm4RegisterWriteStatus(uint64_t register_dword, uint32_t value)>
      write_uconfig_register;
};

struct Pm4PacketProcessRequest {
  std::span<const uint32_t> available_dwords;
};

struct Pm4PacketProcessResult {
  [[nodiscard]] const PacketProcessResult &packet_result() const noexcept { return packet; }

  PacketProcessResult packet;
  Pm4PacketDiagnostic diagnostic = Pm4PacketDiagnostic::None;
  uint32_t packet_header = 0;
  uint64_t register_dword = 0;
};

/// @brief Processes one PM4 packet without owning its ring or queue lifetime.
class Pm4PacketProcessor {
public:
  using Request = Pm4PacketProcessRequest;
  using Result = Pm4PacketProcessResult;

  explicit Pm4PacketProcessor(Pm4PacketCallbacks callbacks = {})
      : callbacks_(std::move(callbacks)) {}

  [[nodiscard]] Result process(Request request) const;

private:
  Pm4PacketCallbacks callbacks_;
};

static_assert(PacketProcessor<Pm4PacketProcessor>);

} // namespace rocjitsu::amdgpu
