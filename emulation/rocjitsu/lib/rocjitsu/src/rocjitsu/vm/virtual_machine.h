// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file virtual_machine.h
/// @brief AMDGPU virtual machine modeling a complete SoC hierarchy.

#ifndef ROCJITSU_VM_VIRTUAL_MACHINE_H_
#define ROCJITSU_VM_VIRTUAL_MACHINE_H_

#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/component.h"

#include <cstdint>
#include <memory>
#include <vector>

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
class SimulatedDriver;

class VirtualMachine : public simdojo::CompositeComponent {
public:
  struct Config {
    SoC::Config soc;
  };

  explicit VirtualMachine(const Config &config);
  VirtualMachine(std::unique_ptr<SoC> soc, bool daemon_mode = false);
  VirtualMachine(std::vector<std::unique_ptr<SoC>> socs, std::vector<uint32_t> gpu_ids,
                 bool daemon_mode = false);
  ~VirtualMachine() override;

  VirtualMachine(const VirtualMachine &) = delete;
  VirtualMachine &operator=(const VirtualMachine &) = delete;
  VirtualMachine(VirtualMachine &&) = delete;
  VirtualMachine &operator=(VirtualMachine &&) = delete;

  SoC *soc() { return soc_; }
  const SoC *soc() const { return soc_; }
  amdgpu::GpuMemory *memory() { return soc_->memory(); }
  const amdgpu::GpuMemory *memory() const { return soc_->memory(); }
  const Config &config() const { return config_; }

  SimulatedDriver *driver() { return driver_.get(); }

private:
  Config config_;
  SoC *soc_ = nullptr;
  std::unique_ptr<SimulatedDriver> driver_;
};

} // namespace rocjitsu

#endif // ROCJITSU_VM_VIRTUAL_MACHINE_H_
