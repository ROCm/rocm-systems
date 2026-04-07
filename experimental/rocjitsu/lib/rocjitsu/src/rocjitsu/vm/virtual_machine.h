// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file virtual_machine.h
/// @brief AMDGPU virtual machine modeling a complete SoC hierarchy.

#ifndef ROCJITSU_VM_VIRTUAL_MACHINE_H_
#define ROCJITSU_VM_VIRTUAL_MACHINE_H_

#include "rocjitsu/vm/driver.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/component.h"

#include <cstdint>

namespace rocjitsu {

/// @brief AMDGPU virtual machine modeling a complete SoC hierarchy.
///
/// The VirtualMachine is a CompositeComponent representing the hardware
/// being modeled. It owns the SoC (which owns shader engines, compute
/// units, etc.) as a child component. The simulation infrastructure
/// (engine, topology, partitioning) is managed by SimulationEngine, which
/// owns the topology containing this VM as the root.
///
/// Fully event-driven: the VM is a structural container. Its descendants
/// (command processors) schedule and handle events through the engine.
/// The Driver injects external work as async events.
class VirtualMachine : public simdojo::CompositeComponent {
public:
  /// @brief Configuration for a VirtualMachine (the "system" being modeled).
  struct Config {
    SoC::Config soc; ///< SoC hierarchy configuration.
  };

  /// @brief Construct a virtual machine from configuration.
  /// @param config Complete VM configuration.
  explicit VirtualMachine(const Config &config);

  ~VirtualMachine() override = default;

  VirtualMachine(const VirtualMachine &) = delete;
  VirtualMachine &operator=(const VirtualMachine &) = delete;
  VirtualMachine(VirtualMachine &&) = delete;
  VirtualMachine &operator=(VirtualMachine &&) = delete;

  /// @brief Return the SoC root of the component hierarchy.
  /// @returns Pointer to the SoC.
  SoC *soc() { return soc_; }
  /// @returns Const pointer to the SoC.
  const SoC *soc() const { return soc_; }

  /// @brief Convenience: return GPU memory from the SoC.
  /// @returns Pointer to the GPU memory.
  amdgpu::GpuMemory *memory() { return soc_->memory(); }
  /// @returns Const pointer to the GPU memory.
  const amdgpu::GpuMemory *memory() const { return soc_->memory(); }

  /// @brief Return the hardware driver interface.
  /// @returns Reference to the driver.
  Driver &driver() { return driver_; }
  /// @returns Const reference to the driver.
  const Driver &driver() const { return driver_; }

  /// @brief Return the configuration used to construct this VM.
  /// @returns Const reference to the VM configuration.
  const Config &config() const { return config_; }

private:
  Config config_;
  SoC *soc_ = nullptr;
  Driver driver_{*this};
};

} // namespace rocjitsu

#endif // ROCJITSU_VM_VIRTUAL_MACHINE_H_
