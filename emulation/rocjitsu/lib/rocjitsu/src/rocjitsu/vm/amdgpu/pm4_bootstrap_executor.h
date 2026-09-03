// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file pm4_bootstrap_executor.h
/// @brief Transport-neutral execution of the narrow compute startup PM4 subset.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include <cstdint>
#include <functional>
#include <utility>

namespace rocjitsu::amdgpu {

/// @brief Result of consuming a firmware-free compute startup ring.
enum class Pm4BootstrapOutcome : uint8_t {
  Complete,
  Unavailable,
  Faulted,
  Malformed,
  UnsupportedPacket,
  RegisterWriteRejected,
};

struct Pm4BootstrapResult {
  Pm4BootstrapOutcome outcome = Pm4BootstrapOutcome::Malformed;
  uint64_t read_pointer = 0;
  uint32_t packet_header = 0;
  uint64_t register_dword = 0;
};

struct Pm4BootstrapCallbacks {
  /// @brief Apply one SET_UCONFIG_REG write to the owning MMIO model.
  std::function<bool(uint64_t register_dword, uint32_t value)> write_uconfig_register;
};

/// @brief Execute only the PM4 packets required before normal queue bring-up.
///
/// @details This is deliberately not a general graphics command processor. It
/// keeps packet decoding and ring semantics in the core GPU model while the
/// PCI frontend supplies only a register-write adapter. Normal AQL and SDMA
/// queues continue to use QueueService and their existing shared backends.
class Pm4BootstrapExecutor {
public:
  explicit Pm4BootstrapExecutor(Pm4BootstrapCallbacks callbacks)
      : callbacks_(std::move(callbacks)) {}

  [[nodiscard]] Pm4BootstrapResult execute(const GpuVmAccess &access, uint64_t ring_base,
                                           uint64_t ring_dwords, uint64_t read_pointer,
                                           uint64_t write_pointer) const;

private:
  Pm4BootstrapCallbacks callbacks_;
};

} // namespace rocjitsu::amdgpu
