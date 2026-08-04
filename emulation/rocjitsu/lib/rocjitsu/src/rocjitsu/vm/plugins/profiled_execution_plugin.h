// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file profiled_execution_plugin.h
/// @brief ExecutionPlugin decorator with adaptive-sampling hook profiling.
///
/// Wraps each hook with timing instrumentation. Sampling frequency decreases
/// as the hook fires more often (interval scales as sqrt(count/1000)), so
/// hot hooks are profiled without dominating execution time. Enable by
/// wrapping a plugin group in ProfiledExecutionPlugin. On shutdown, prints
/// per-hook timing summaries.

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>

namespace rocjitsu {

class ProfiledExecutionPlugin final : public ExecutionPlugin {
  using Clock = std::chrono::steady_clock;

  struct HookProfile {
    uint64_t count = 0;
    uint64_t last_sampled = 0;
    uint64_t interval = 1;
    double total_ns = 0;

    double estimated_total_ms() const { return total_ns / 1e6; }
  };

public:
  /// Add a profiling decorator as the sole plugin in @p outer. The outer and
  /// nested groups share a slot allocator, keeping wavefront-state indices
  /// unique across arbitrarily nested decorator trees.
  static ProfiledExecutionPlugin *add_to(ExecutionPluginGroup &outer,
                                         std::unique_ptr<ExecutionPluginGroup> plugins) {
    if (!outer.empty())
      return nullptr;
    if (!plugins)
      plugins = std::make_unique<ExecutionPluginGroup>(PluginSinkConfig{});

    outer.slot_allocator_ = plugins->slot_allocator_;
    auto profile =
        std::unique_ptr<ProfiledExecutionPlugin>(new ProfiledExecutionPlugin(std::move(plugins)));
    auto *result = profile.get();
    return outer.add(std::move(profile)) ? result : nullptr;
  }

  /// Profiling counters and the dispatch-name map are unsynchronized, so the
  /// containing group must serialize high-frequency callbacks. If they become
  /// concurrency-safe, return plugins_->requires_serial_execution().
  bool requires_serial_execution() const override { return true; }

  void onInit() override { plugins_->onInit(); }

  void onShutdown() override {
    if (prof_before_exec_.count > 0)
      print_profile_summary();
    plugins_->onShutdown();
    double total = std::chrono::duration<double>(Clock::now() - start_time_).count();
    sink().write(std::format("[rocjitsu] total emulation time: {:.3f} s\n", total));
  }

  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                        amdgpu::Wavefront &wf) override {
    profiled_dispatch(prof_before_exec_,
                      [&]() { plugins_->onAmdgpuBeforeExecuteInstruction(pc, inst, wf); });
  }

  void onAmdgpuAfterExecuteInstruction(uint64_t pc, const Instruction &inst,
                                       amdgpu::Wavefront &wf) override {
    profiled_dispatch(prof_after_exec_,
                      [&]() { plugins_->onAmdgpuAfterExecuteInstruction(pc, inst, wf); });
  }

  void onAmdgpuRouteMemoryInstruction(const Instruction &inst, amdgpu::Wavefront &wf) override {
    profiled_dispatch(prof_route_mem_,
                      [&]() { plugins_->onAmdgpuRouteMemoryInstruction(inst, wf); });
  }

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override {
    if (prof_before_exec_.count > 0)
      print_profile_summary();
    current_kernel_name_ = info.kernel_name;
    dispatch_kernel_names_[info.dispatch_id] = info.kernel_name;
    plugins_->onAmdgpuDispatchPacketProcessed(info);
    reset_profiles();
  }

  void onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) override {
    plugins_->onAmdgpuDispatchExecutionBegin(dispatch_id);
  }

  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) override {
    if (prof_before_exec_.count > 0) {
      auto it = dispatch_kernel_names_.find(dispatch_id);
      if (it != dispatch_kernel_names_.end())
        current_kernel_name_ = it->second;
      print_profile_summary();
    }
    plugins_->onAmdgpuDispatchExecutionEnd(dispatch_id);
    reset_profiles();
  }

  void onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id,
                                   uint32_t physical_vgpr_count, uint32_t sgpr_count,
                                   std::span<amdgpu::Wavefront *> wavefronts) override {
    profiled_dispatch(prof_wg_dispatched_, [&]() {
      plugins_->onAmdgpuWorkgroupDispatched(dispatch_id, wg_id, physical_vgpr_count, sgpr_count,
                                            wavefronts);
    });
  }

  void onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t wg_id) override {
    profiled_dispatch(prof_wg_completed_,
                      [&]() { plugins_->onAmdgpuWorkgroupCompleted(dispatch_id, wg_id); });
  }

  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) override {
    profiled_dispatch(prof_wf_dispatched_, [&]() { plugins_->onAmdgpuWavefrontDispatched(wf); });
  }

  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) override {
    profiled_dispatch(prof_wf_halted_, [&]() { plugins_->onAmdgpuWavefrontHalted(wf); });
  }

  void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg, uint64_t lane_mask,
                             uint8_t byte_mask = ExecutionPlugin::kFullByteMask) override {
    profiled_dispatch(prof_read_vgpr_, [&]() {
      plugins_->onAmdgpuReadVgprLanes(wf, physical_reg, lane_mask, byte_mask);
    });
  }

  void onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                              uint64_t lane_mask,
                              uint8_t byte_mask = ExecutionPlugin::kFullByteMask) override {
    profiled_dispatch(prof_write_vgpr_, [&]() {
      plugins_->onAmdgpuWriteVgprLanes(wf, physical_reg, lane_mask, byte_mask);
    });
  }

  void onAmdgpuReadSgpr(const amdgpu::Wavefront *wf, uint32_t physical_reg) override {
    profiled_dispatch(prof_read_sgpr_, [&]() { plugins_->onAmdgpuReadSgpr(wf, physical_reg); });
  }

  void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) override {
    profiled_dispatch(prof_barrier_, [&]() { plugins_->onAmdgpuBarrierResolved(wavefronts); });
  }

private:
  explicit ProfiledExecutionPlugin(std::unique_ptr<ExecutionPluginGroup> plugins)
      : ExecutionPlugin("profile"), plugins_(std::move(plugins)), start_time_(Clock::now()) {}

  template <typename F> void profiled_dispatch(HookProfile &prof, F &&fn) {
    prof.count++;
    if ((prof.count % prof.interval) == 0) {
      auto t0 = Clock::now();
      fn();
      double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
      uint64_t gap = prof.count - prof.last_sampled;
      prof.total_ns += ns * static_cast<double>(gap);
      prof.last_sampled = prof.count;
      prof.interval = static_cast<uint64_t>(std::sqrt(prof.count / 1000.0)) + 1;
    } else {
      fn();
    }
  }

  void reset_profiles() {
    prof_before_exec_ = {};
    prof_after_exec_ = {};
    prof_read_vgpr_ = {};
    prof_write_vgpr_ = {};
    prof_read_sgpr_ = {};
    prof_route_mem_ = {};
    prof_barrier_ = {};
    prof_wg_dispatched_ = {};
    prof_wg_completed_ = {};
    prof_wf_dispatched_ = {};
    prof_wf_halted_ = {};
  }

  void print_profile_summary() {
    const char *kname =
        current_kernel_name_.empty() ? kUnknownKernelIdentity : current_kernel_name_.c_str();
    sink().write(std::format("HOOK_PROFILE --- {} ---\n", kname));
    auto print_hook = [this](const char *name, const HookProfile &p) {
      if (p.count == 0)
        return;
      sink().write(std::format("HOOK_PROFILE {:<30s}  calls={:<12}  est_total={:.1f} ms\n", name,
                               p.count, p.estimated_total_ms()));
    };
    print_hook("beforeExecuteInstruction", prof_before_exec_);
    print_hook("afterExecuteInstruction", prof_after_exec_);
    print_hook("readVgpr", prof_read_vgpr_);
    print_hook("writeVgpr", prof_write_vgpr_);
    print_hook("readSgpr", prof_read_sgpr_);
    print_hook("routeMemoryInstruction", prof_route_mem_);
    print_hook("barrierResolved", prof_barrier_);
    print_hook("workgroupDispatched", prof_wg_dispatched_);
    print_hook("workgroupCompleted", prof_wg_completed_);
    print_hook("wavefrontDispatched", prof_wf_dispatched_);
    print_hook("wavefrontHalted", prof_wf_halted_);
    sink().write("HOOK_PROFILE ---\n");
  }

  std::unique_ptr<ExecutionPluginGroup> plugins_;
  Clock::time_point start_time_;
  std::string current_kernel_name_;
  std::unordered_map<uint32_t, std::string> dispatch_kernel_names_;

  HookProfile prof_before_exec_;
  HookProfile prof_after_exec_;
  HookProfile prof_read_vgpr_;
  HookProfile prof_write_vgpr_;
  HookProfile prof_read_sgpr_;
  HookProfile prof_route_mem_;
  HookProfile prof_barrier_;
  HookProfile prof_wg_dispatched_;
  HookProfile prof_wg_completed_;
  HookProfile prof_wf_dispatched_;
  HookProfile prof_wf_halted_;
};

} // namespace rocjitsu
