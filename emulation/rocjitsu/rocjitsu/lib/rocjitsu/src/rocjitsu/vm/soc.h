// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file soc.h
/// @brief System-on-Chip container with XCDs, I/O Dies, and shared GPU memory.

#ifndef ROCJITSU_VM_SOC_H_
#define ROCJITSU_VM_SOC_H_

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/hbm_controller.h"
#include "rocjitsu/vm/amdgpu/iod.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "rocjitsu/vm/execution_plugin.h"

#include "simdojo/sim/component.h"
#include "simdojo/sim/exec_mode.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rocjitsu {

/// @brief System-on-Chip container with XCDs, I/O Dies, and shared GPU memory.
///
/// Constructs the full GPU hierarchy (memory + IODs + XCDs + shader engines + CUs)
/// from a Config. When num_iods > 0, each IOD owns a memory-side cache and HBM
/// controller; XCDs are assigned to IODs in order. When num_iods == 0, XCDs
/// connect directly to a standalone HBM controller (simplified path).
///
/// Child components are owned via the CompositeComponent tree.
class SoC : public simdojo::CompositeComponent {
public:
  /// @brief Configuration for an AMDGPU SoC.
  struct Config {
    rj_code_arch_t arch;     ///< ISA architecture.
    uint32_t num_xcds;       ///< Number of Accelerator Complex Dies.
    uint32_t num_iods = 0;   ///< Number of I/O Dies (0 = no IOD modeling).
    amdgpu::Xcd::Config xcd; ///< Config applied to each XCD.
    simdojo::ExecMode exec_mode = simdojo::ExecMode::FUNCTIONAL; ///< Execution mode.
  };

  /// @brief Construct a named SoC from configuration.
  /// @param name Human-readable name (e.g., "mi300x").
  /// @param config SoC configuration parameters.
  SoC(std::string name, const Config &config);

  /// @brief Construct an empty SoC (children added externally by the config loader).
  explicit SoC(std::string name, amdgpu::GpuMemory *memory = nullptr)
      : simdojo::CompositeComponent(std::move(name)), memory_(memory) {
    set_weight(0);
  }

  void add_xcd(amdgpu::Xcd *xcd) { xcds_.push_back(xcd); }
  void add_iod(amdgpu::Iod *iod) { iods_.push_back(iod); }
  void set_memory(amdgpu::GpuMemory *m); // Defined in soc.cpp.

  /// @brief Wire L2 → HBM backing store links (call after engine build).
  void wire_backing(simdojo::Topology &topo);
  void set_arch(rj_code_arch_t a) { arch_ = a; }
  rj_code_arch_t arch() const { return arch_; }

  /// @brief Wire topology links between XCDs, IODs, and memory.
  void initialize() override;

  /// @brief Return the number of XCDs.
  /// @returns Number of Accelerator Complex Dies.
  uint32_t num_xcds() const { return static_cast<uint32_t>(xcds_.size()); }

  /// @brief Access an XCD by index.
  /// @param idx Zero-based XCD index.
  /// @returns Pointer to the XCD.
  amdgpu::Xcd *xcd(uint32_t idx) {
    assert(idx < xcds_.size());
    return xcds_[idx];
  }
  /// @returns Const pointer to the XCD.
  const amdgpu::Xcd *xcd(uint32_t idx) const {
    assert(idx < xcds_.size());
    return xcds_[idx];
  }

  /// @brief Return all XCDs.
  /// @returns Const reference to the vector of XCD pointers.
  const std::vector<amdgpu::Xcd *> &xcds() const { return xcds_; }

  /// @brief Return the command processor for this device.
  ///
  /// @details In amdkfd, a gpu_id identifies the whole device, not an individual XCD.
  /// The SoC owns the topology and is the right place to decide which CP serves
  /// the device. For single-XCD SoCs this is xcd(0)'s CP; a future multi-XCD
  /// implementation would return a MES dispatcher instead.
  ///
  /// @returns Pointer to the primary CommandProcessor, or nullptr if no XCDs.
  amdgpu::CommandProcessor *command_processor() {
    return xcds_.empty() ? nullptr : xcds_[0]->command_processor();
  }

  /// @brief Return the number of I/O Dies.
  /// @returns Number of I/O Dies.
  uint32_t num_iods() const { return static_cast<uint32_t>(iods_.size()); }

  /// @brief Access an IOD by index.
  /// @param idx Zero-based IOD index.
  /// @returns Pointer to the IOD.
  amdgpu::Iod *iod(uint32_t idx) {
    assert(idx < iods_.size());
    return iods_[idx];
  }

  /// @brief Flush the entire cache hierarchy to GPU memory.
  ///
  /// Flushes all CU caches (L1 → L2), then flushes all IOD memory-side
  /// caches (MSC → HBM). After this call, all dirty data is visible in
  /// GpuMemory.
  void flush_all();

  /// @brief Return the GPU memory.
  /// @returns Pointer to the GPU memory.
  amdgpu::GpuMemory *memory() { return memory_; }
  /// @returns Const pointer to the GPU memory.
  const amdgpu::GpuMemory *memory() const { return memory_; }

  /// @brief Set the execution plugin group and distribute to CPs/CUs.
  void set_plugin_group(std::shared_ptr<ExecutionPluginGroup> plugin_group);

private:
  rj_code_arch_t arch_ = ROCJITSU_CODE_ARCH_INVALID;
  simdojo::ExecMode exec_mode_ = simdojo::ExecMode::FUNCTIONAL;
  std::vector<amdgpu::Xcd *> xcds_;
  std::vector<amdgpu::Iod *> iods_;
  amdgpu::GpuMemory *memory_ = nullptr;
  std::unique_ptr<amdgpu::HbmController> hbm_standalone_; ///< Used when num_iods == 0.
  std::shared_ptr<ExecutionPluginGroup> plugin_group_;
};

} // namespace rocjitsu

#endif // ROCJITSU_VM_SOC_H_
