// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <functional>

namespace rocjitsu::amdgpu {

/// @brief Outcome of a write through the owning user-config register model.
enum class Pm4RegisterWriteStatus : uint8_t {
  Complete, ///< The register effect occurred; packet processing may continue.
  Blocked,  ///< No effect occurred; retry the same packet later.
  Faulted,  ///< No effect occurred; cancel the queue after a register-access failure.
  Rejected, ///< No effect occurred; cancel the queue for an unsupported write.
};

/// @brief Optional register-model services copied into each CP-owned PM4 queue.
/// @details Captured objects must outlive every queue that holds these callbacks.
class Pm4PacketCallbacks {
public:
  /// @brief Apply one SET_UCONFIG_REG write through the owning register model.
  /// @details Non-complete outcomes must mean that no register effect occurred.
  /// register_dword is an absolute register dword index (0xc000 plus the packet
  /// offset); value is the 32-bit register value. A native ring requires this
  /// callback for SET_UCONFIG_REG. A submission queue may omit it and use its
  /// local user-config register state.
  std::function<Pm4RegisterWriteStatus(uint64_t register_dword, uint32_t value)>
      write_uconfig_register;
};

} // namespace rocjitsu::amdgpu
