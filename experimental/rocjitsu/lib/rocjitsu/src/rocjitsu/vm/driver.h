// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file driver.h
/// @brief Hardware driver interface for a VirtualMachine.

#ifndef ROCJITSU_VM_DRIVER_H_
#define ROCJITSU_VM_DRIVER_H_

#include "rocjitsu/vm/amdgpu/command_processor.h"

#include <cstdint>

namespace rocjitsu {

class VirtualMachine;

/// @brief Hardware driver interface for a VirtualMachine.
///
/// Models the kernel-mode driver interface presented to a user-mode driver
/// (e.g. rocr). Owned as a value member of VirtualMachine - the driver is
/// part of the hardware being modeled.
///
/// The driver is a thin bridge between the host (app) thread and the
/// event-driven simulation. It translates API calls into simulation events
/// via schedule_event_async().
class Driver {
public:
  /// @brief Construct a driver for the given VM.
  /// @param vm The virtual machine that owns this driver.
  explicit Driver(VirtualMachine &vm);
  ~Driver() = default;

  Driver(const Driver &) = delete;
  Driver &operator=(const Driver &) = delete;

  /// @brief Submit a dispatch packet to an XCD's command processor.
  ///
  /// Thread-safe. Injects a doorbell event into the simulation via
  /// schedule_event_async().
  /// @param packet The dispatch parameters.
  /// @param xcd_idx Which XCD to submit to (default: 0).
  void submit(amdgpu::DispatchPacket packet, uint32_t xcd_idx = 0);

  /// @brief Request the simulation to stop.
  ///
  /// Thread-safe. Signals the engine to exit at the next epoch boundary.
  void close();

private:
  VirtualMachine &vm_;
};

} // namespace rocjitsu

#endif // ROCJITSU_VM_DRIVER_H_
