// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin.h
/// @brief Architecture-neutral plugin interface for execution hooks.
///
/// A single plugin class serves both AMDGPU and RISC-V execution paths.
/// Hooks are prefixed with their architecture name (onAmdgpu*, onRiscv*).

#pragma once

#include "plugins/kernel_dispatch_info.h"
#include "plugins/wavefront_state.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

/// @brief Abstract plugin interface for execution hooks.
///
/// Plugins receive callbacks at key points during simulation execution.
/// All hooks have empty default implementations so plugins only override
/// what they need. The no-plugin path has near-zero overhead (the
/// default empty group's delegation loops iterate an empty vector).
///
/// Ownership: plugins are owned by an ExecutionPluginGroup via
/// unique_ptr. The group itself is shared (via shared_ptr) between
/// the SoC and all components that fire hooks (CommandProcessor,
/// ComputeUnit, Hart). A static empty group is used as the default
/// so the plugin group pointer is never null.
class ExecutionPlugin {
public:
  explicit ExecutionPlugin(std::string name) : name_(std::move(name)) {}
  virtual ~ExecutionPlugin() = default;

  const std::string &name() const { return name_; }

  /// Index into Wavefront::plugin_states_, assigned by the group on add().
  uint32_t slot_index() const { return slot_index_; }

  // -- Lifecycle hooks ------------------------------------------------------

  /// Called when the emulated driver opens (simulation is ready to accept work).
  virtual void onInit() {}

  /// Called when the emulated driver closes (simulation is shutting down).
  /// All simulation state is still valid during this callback.
  virtual void onShutdown() {}

  // -- AMDGPU hooks --------------------------------------------------------

  /// Called before every AMDGPU instruction is executed.
  /// Wavefront state reflects the state prior to the instruction's effects.
  virtual void beforeAmdgpuExecuteInstruction(uint64_t /*pc*/, const Instruction & /*inst*/,
                                              amdgpu::Wavefront & /*wf*/) {}

  /// Called after every AMDGPU instruction is executed.
  /// Wavefront state (wait targets, PC, etc.) reflects the instruction's effects.
  virtual void afterAmdgpuExecuteInstruction(uint64_t /*pc*/, const Instruction & /*inst*/,
                                             amdgpu::Wavefront & /*wf*/) {}

  /// Called when an AMDGPU memory instruction is routed to a pipeline.
  virtual void onAmdgpuRouteMemoryInstruction(const Instruction & /*inst*/,
                                              amdgpu::Wavefront & /*wf*/) {}

  /// Called when the command processor has parsed an AQL kernel dispatch packet
  /// and created a DispatchEntry. Fires during packet fetching, before any
  /// workgroups are placed. Multiple packets may be parsed in a single fetch.
  virtual void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo & /*info*/) {}

  /// Called when the command processor begins executing a dispatch — barriers
  /// are satisfied and workgroup placement is about to start.
  virtual void onAmdgpuDispatchExecutionBegin(uint32_t /*dispatch_id*/) {}

  /// Called when all workgroups of a dispatch have completed execution.
  virtual void onAmdgpuDispatchExecutionEnd(uint32_t /*dispatch_id*/) {}

  /// Called after a workgroup's wavefronts have been dispatched to a CU.
  virtual void onAmdgpuWorkgroupDispatched(uint32_t /*wg_id*/, uint32_t /*dispatch_id*/,
                                           uint32_t /*vgpr_count*/, uint32_t /*sgpr_count*/,
                                           std::span<amdgpu::Wavefront *> /*wavefronts*/) {}

  /// Called when the last wavefront of a workgroup has halted.
  virtual void onAmdgpuWorkgroupCompleted(uint32_t /*dispatch_id*/, uint32_t /*wg_id*/) {}

  /// Called after a wavefront is initialized and before its first instruction.
  virtual void onAmdgpuWavefrontDispatched(amdgpu::Wavefront & /*wf*/) {}

  /// Called when a wavefront halts, before its resources are freed.
  virtual void onAmdgpuWavefrontHalted(amdgpu::Wavefront & /*wf*/) {}

  /// Called when a VGPR is read during instruction execution.
  /// @param wf Owning wavefront, or nullptr if the register is unallocated.
  /// @param physical_reg Physical register index in the VGPR file.
  /// @param lane Lane index within the wavefront.
  virtual void onAmdgpuReadVgpr(amdgpu::Wavefront * /*wf*/, uint32_t /*physical_reg*/,
                                uint32_t /*lane*/, uint8_t /*byteMask*/ = 0xF) {}

  /// Called when an SGPR is read during instruction execution.
  /// @param wf Owning wavefront, or nullptr if the register is unallocated.
  /// @param physical_reg Physical register index in the SGPR file.
  virtual void onAmdgpuReadSgpr(amdgpu::Wavefront * /*wf*/, uint32_t /*physical_reg*/) {}

  /// Called when all waves in a workgroup have reached s_barrier.
  virtual void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> /*wavefronts*/) {}

  // -- RISC-V hooks --------------------------------------------------------

  /// Called before every RISC-V instruction is executed.
  virtual void onRiscvExecuteInstruction(uint64_t /*pc*/, const Instruction & /*inst*/) {}

private:
  friend class ExecutionPluginGroup;
  std::string name_;
  uint32_t slot_index_ = 0;
};

/// @brief Collection of plugins that delegates to each member.
class ExecutionPluginGroup {
  using Clock = std::chrono::steady_clock;

  struct HookProfile {
    uint64_t count = 0;
    uint64_t timed_count = 0;
    double total_ns = 0;

    double estimated_total_ms() const {
      if (timed_count == 0)
        return 0;
      return (total_ns * static_cast<double>(count) / static_cast<double>(timed_count)) / 1e6;
    }
  };

public:
  ExecutionPluginGroup() : start_time_(Clock::now()) {}

  bool add(std::unique_ptr<ExecutionPlugin> p) {
    if (!p)
      return false;
    for (const auto &existing : plugins_)
      if (existing->name() == p->name())
        return false;
    if (plugins_.empty()) {
      active_start_ = &start_time_;
      std::atexit([]() {
        if (active_start_) {
          double total = std::chrono::duration<double>(Clock::now() - *active_start_).count();
          std::cerr << "[rocjitsu] total emulation time: " << std::fixed << std::setprecision(3)
                    << total << " s" << std::endl;
        }
      });
    }
    p->slot_index_ = static_cast<uint32_t>(plugins_.size());
    plugins_.push_back(std::move(p));
    return true;
  }

  uint32_t num_plugins() const { return static_cast<uint32_t>(plugins_.size()); }

  // -- Lifecycle --
  void onInit() {
    for (auto &p : plugins_)
      p->onInit();
  }

  void onShutdown() {
    for (auto &p : plugins_)
      p->onShutdown();
  }

  // -- AMDGPU --
  void beforeAmdgpuExecuteInstruction(uint64_t pc, const Instruction &inst, amdgpu::Wavefront &wf) {
    timed_dispatch(prof_before_exec_, [&]() {
      for (auto &p : plugins_)
        p->beforeAmdgpuExecuteInstruction(pc, inst, wf);
    });
  }

  void afterAmdgpuExecuteInstruction(uint64_t pc, const Instruction &inst, amdgpu::Wavefront &wf) {
    timed_dispatch(prof_after_exec_, [&]() {
      for (auto &p : plugins_)
        p->afterAmdgpuExecuteInstruction(pc, inst, wf);
    });
  }

  void onAmdgpuRouteMemoryInstruction(const Instruction &inst, amdgpu::Wavefront &wf) {
    timed_dispatch(prof_route_mem_, [&]() {
      for (auto &p : plugins_)
        p->onAmdgpuRouteMemoryInstruction(inst, wf);
    });
  }

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
    if (prof_before_exec_.count > 0)
      print_profile_summary();
    for (auto &p : plugins_)
      p->onAmdgpuDispatchPacketProcessed(info);
    reset_profiles();
  }

  void onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) {
    for (auto &p : plugins_)
      p->onAmdgpuDispatchExecutionBegin(dispatch_id);
  }

  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) {
    if (prof_before_exec_.count > 0)
      print_profile_summary();
    for (auto &p : plugins_)
      p->onAmdgpuDispatchExecutionEnd(dispatch_id);
    reset_profiles();
  }

  void onAmdgpuWorkgroupDispatched(uint32_t wg_id, uint32_t dispatch_id, uint32_t vgpr_count,
                                   uint32_t sgpr_count, std::span<amdgpu::Wavefront *> wavefronts) {
    timed_dispatch(prof_wg_dispatched_, [&]() {
      for (auto &p : plugins_)
        p->onAmdgpuWorkgroupDispatched(wg_id, dispatch_id, vgpr_count, sgpr_count, wavefronts);
    });
  }

  void onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t wg_id) {
    timed_dispatch(prof_wg_completed_, [&]() {
      for (auto &p : plugins_)
        p->onAmdgpuWorkgroupCompleted(dispatch_id, wg_id);
    });
  }

  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) {
    timed_dispatch(prof_wf_dispatched_, [&]() {
      for (auto &p : plugins_)
        p->onAmdgpuWavefrontDispatched(wf);
    });
  }

  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) {
    timed_dispatch(prof_wf_halted_, [&]() {
      for (auto &p : plugins_)
        p->onAmdgpuWavefrontHalted(wf);
    });
  }

  void onAmdgpuReadVgpr(amdgpu::Wavefront *wf, uint32_t physical_reg, uint32_t lane,
                        uint8_t byteMask = 0xF) {
    sampled_dispatch(prof_read_vgpr_, [&]() {
      for (auto &p : plugins_)
        p->onAmdgpuReadVgpr(wf, physical_reg, lane, byteMask);
    });
  }

  void onAmdgpuReadSgpr(amdgpu::Wavefront *wf, uint32_t physical_reg) {
    sampled_dispatch(prof_read_sgpr_, [&]() {
      for (auto &p : plugins_)
        p->onAmdgpuReadSgpr(wf, physical_reg);
    });
  }

  void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) {
    timed_dispatch(prof_barrier_, [&]() {
      for (auto &p : plugins_)
        p->onAmdgpuBarrierResolved(wavefronts);
    });
  }

  // -- RISC-V --
  void onRiscvExecuteInstruction(uint64_t pc, const Instruction &inst) {
    for (auto &p : plugins_)
      p->onRiscvExecuteInstruction(pc, inst);
  }

  bool empty() const { return plugins_.empty(); }

  /// A shared empty group used as the default when no plugins are attached.
  static std::shared_ptr<ExecutionPluginGroup> empty_group() {
    static auto instance = std::make_shared<ExecutionPluginGroup>();
    return instance;
  }

private:
  template <typename F> void timed_dispatch(HookProfile &prof, F &&fn) {
    auto t0 = Clock::now();
    fn();
    prof.total_ns += std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
    prof.count++;
    prof.timed_count++;
  }

  template <typename F> void sampled_dispatch(HookProfile &prof, F &&fn) {
    prof.count++;
    if ((prof.count % sample_rate_) == 0) {
      auto t0 = Clock::now();
      fn();
      prof.total_ns += std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
      prof.timed_count++;
    } else {
      fn();
    }
  }

  void reset_profiles() {
    prof_before_exec_ = {};
    prof_after_exec_ = {};
    prof_read_vgpr_ = {};
    prof_read_sgpr_ = {};
    prof_route_mem_ = {};
    prof_barrier_ = {};
    prof_wg_dispatched_ = {};
    prof_wg_completed_ = {};
    prof_wf_dispatched_ = {};
    prof_wf_halted_ = {};
  }

  void print_profile_summary() const {
    auto print_hook = [](const char *name, const HookProfile &p) {
      if (p.count == 0)
        return;
      std::fprintf(stderr, "HOOK_PROFILE %-30s  calls=%-12lu  est_total=%.1f ms\n", name,
                   static_cast<unsigned long>(p.count), p.estimated_total_ms());
    };
    std::fprintf(stderr, "HOOK_PROFILE --- (sample_rate=%lu) ---\n",
                 static_cast<unsigned long>(sample_rate_));
    print_hook("beforeExecuteInstruction", prof_before_exec_);
    print_hook("afterExecuteInstruction", prof_after_exec_);
    print_hook("readVgpr", prof_read_vgpr_);
    print_hook("readSgpr", prof_read_sgpr_);
    print_hook("routeMemoryInstruction", prof_route_mem_);
    print_hook("barrierResolved", prof_barrier_);
    print_hook("workgroupDispatched", prof_wg_dispatched_);
    print_hook("workgroupCompleted", prof_wg_completed_);
    print_hook("wavefrontDispatched", prof_wf_dispatched_);
    print_hook("wavefrontHalted", prof_wf_halted_);
    std::fprintf(stderr, "HOOK_PROFILE ---\n");
  }

  std::vector<std::unique_ptr<ExecutionPlugin>> plugins_;
  Clock::time_point start_time_;
  static inline const Clock::time_point *active_start_ = nullptr;

  uint64_t sample_rate_ = 113;
  HookProfile prof_before_exec_;
  HookProfile prof_after_exec_;
  HookProfile prof_read_vgpr_;
  HookProfile prof_read_sgpr_;
  HookProfile prof_route_mem_;
  HookProfile prof_barrier_;
  HookProfile prof_wg_dispatched_;
  HookProfile prof_wg_completed_;
  HookProfile prof_wf_dispatched_;
  HookProfile prof_wf_halted_;
};

} // namespace rocjitsu
