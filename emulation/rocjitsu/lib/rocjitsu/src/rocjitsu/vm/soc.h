// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file soc.h
/// @brief System-on-Chip container with XCDs, I/O Dies, and shared GPU memory.

#ifndef ROCJITSU_VM_SOC_H_
#define ROCJITSU_VM_SOC_H_

#include "rocjitsu/vm/amdgpu/cpu_dispatch_pool.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/hbm_controller.h"
#include "rocjitsu/vm/amdgpu/iod.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include "simdojo/sim/component.h"
#include "simdojo/sim/exec_mode.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rocjitsu {

namespace test {
class SoCTestAccess;
}

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

  uint32_t gpu_id() const { return gpu_id_; }

  void add_xcd(amdgpu::Xcd *xcd) { xcds_.push_back(xcd); }
  void add_iod(amdgpu::Iod *iod) { iods_.push_back(iod); }

  /// @brief Set flat-address-space aperture boundaries on all CUs via the SPI hierarchy.
  void set_apertures(uint64_t shared_base, uint64_t shared_limit, uint64_t private_base,
                     uint64_t private_limit) {
    for (auto *xcd : xcds_)
      xcd->set_apertures(shared_base, shared_limit, private_base, private_limit);
  }
  void set_memory(amdgpu::GpuMemory *m); // Defined in soc.cpp.

  /// @brief Wire L2 → HBM backing store links (call after engine build).
  void wire_backing(simdojo::Topology &topo);
  void set_arch(rj_code_arch_t a) { arch_ = a; }
  rj_code_arch_t arch() const { return arch_; }
  void set_exec_mode(simdojo::ExecMode mode) { exec_mode_ = mode; }
  simdojo::ExecMode exec_mode() const { return exec_mode_; }

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

  /// @brief Pick the XCD command processor that will own a HW queue.
  ///
  /// @details On real MI300X hardware, the MES firmware distributes HW queues
  /// across XCDs. Use the process-local queue ordinal so equivalent queues from
  /// independent processes compete for the same XCD resources. The owner reads
  /// the queue's ring and holds each dispatch's completion signal. It is not the
  /// only XCD that runs the work: a queue marked HwQueue::xcd_fanout spreads each
  /// dispatch over every XCD, and which XCD owns the queue does not change the
  /// workgroup-to-XCD mapping.
  ///
  /// @returns Pointer to the selected CommandProcessor, or nullptr if no XCDs.
  amdgpu::CommandProcessor *assign_queue_owner_cp(uint32_t queue_ordinal) {
    if (xcds_.empty())
      return nullptr;
    return xcds_[queue_xcd_id(queue_ordinal)]->command_processor();
  }

  /// @brief Return the XCD selected for a process-local queue ordinal.
  uint32_t queue_xcd_id(uint32_t queue_ordinal) const {
    assert(!xcds_.empty());
    return queue_ordinal % static_cast<uint32_t>(xcds_.size());
  }

  /// @brief Per-XCD histogram of workgroups placed by each XCD's command processor.
  ///
  /// @details Index i is the count for xcd(i). Each entry is that CP's lifetime
  /// running total, so this describes one grid only when it is the sole dispatch,
  /// as in the tests; otherwise diff a snapshot taken before the dispatch against
  /// one taken after. A single non-zero bucket means the work ran on one XCD.
  /// @returns One entry per XCD, in XCD index order.
  std::vector<uint64_t> dispatched_workgroups_per_xcd() const {
    std::vector<uint64_t> counts;
    counts.reserve(xcds_.size());
    for (const auto *xcd_ptr : xcds_) {
      const auto *cp = xcd_ptr->command_processor();
      counts.push_back(cp ? cp->dispatched_workgroups() : 0);
    }
    return counts;
  }

  /// @brief Apply a function to all XCD command processors.
  ///
  /// @details Used for broadcast operations (setting callbacks, flushing, etc.)
  /// that must reach every CP on the device.
  template <typename Fn> void for_each_cp(Fn &&fn) {
    for (auto *xcd_ptr : xcds_)
      if (auto *cp = xcd_ptr->command_processor())
        fn(cp);
  }

  bool has_active_wfs_for_process(uint32_t process_id) const {
    for (auto *xcd_ptr : xcds_) {
      if (auto *cp = xcd_ptr->command_processor()) {
        for (auto *cu : cp->compute_units())
          if (cu->has_active_wfs_for_process(process_id))
            return true;
      }
    }
    return false;
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

  /// @brief Set the shared host-thread budget for CU dispatch across this SoC.
  void set_dispatch_threads(uint32_t threads);
  uint32_t dispatch_threads() const { return dispatch_threads_; }

  const std::vector<amdgpu::ComputeUnitCore *> &all_cus();

  ExecutionPluginGroup &plugin_group() { return *plugin_group_; }

private:
  friend class test::SoCTestAccess;

  void apply_dispatch_threads();

  static inline std::atomic<uint32_t> next_gpu_id_{0};
  uint32_t gpu_id_ = next_gpu_id_++;
  rj_code_arch_t arch_ = ROCJITSU_CODE_ARCH_INVALID;
  simdojo::ExecMode exec_mode_ = simdojo::ExecMode::FUNCTIONAL;
  std::vector<amdgpu::Xcd *> xcds_;
  std::vector<amdgpu::Iod *> iods_;
  amdgpu::GpuMemory *memory_ = nullptr;
  std::unique_ptr<amdgpu::HbmController> hbm_standalone_; ///< Used when num_iods == 0.
  std::unique_ptr<amdgpu::CpuDispatchPool> dispatch_pool_;
  uint32_t requested_dispatch_threads_ = 1;
  uint32_t dispatch_threads_ = 1;
  std::shared_ptr<ExecutionPluginGroup> plugin_group_;
  std::vector<amdgpu::ComputeUnitCore *> all_cus_cache_;
};

} // namespace rocjitsu

#endif // ROCJITSU_VM_SOC_H_
