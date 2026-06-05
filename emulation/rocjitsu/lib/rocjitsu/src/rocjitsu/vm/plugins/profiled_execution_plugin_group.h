// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file profiled_execution_plugin_group.h
/// @brief ExecutionPluginGroup subclass with adaptive-sampling hook profiling.
///
/// Wraps each hook with timing instrumentation. Sampling frequency decreases
/// as the hook fires more often (interval scales as sqrt(count/1000)), so
/// hot hooks are profiled without dominating execution time. Enable by
/// constructing a ProfiledExecutionPluginGroup instead of a plain
/// ExecutionPluginGroup. On shutdown, prints per-hook timing summaries.

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <unordered_map>

namespace rocjitsu {

// NOT thread-safe: HookProfile counters are shared across all CUs without
// synchronization. Use only with num_threads=1.

class ProfiledExecutionPluginGroup : public ExecutionPluginGroup {
  using Clock = std::chrono::steady_clock;

  struct HookProfile {
    uint64_t count = 0;
    uint64_t last_sampled = 0;
    uint64_t interval = 1;
    double total_ns = 0;

    double estimated_total_ms() const { return total_ns / 1e6; }
  };

public:
  ProfiledExecutionPluginGroup() : start_time_(Clock::now()) {}

  void onInit() {
    if (auto *s = build_composite_sink("profile.log"))
      sink_ = s;
    ExecutionPluginGroup::onInit();
  }

  void onShutdown() {
    if (prof_before_exec_.count > 0)
      print_profile_summary();
    ExecutionPluginGroup::onShutdown();
    double total = std::chrono::duration<double>(Clock::now() - start_time_).count();
    sink().write(std::format("[rocjitsu] total emulation time: {:.3f} s\n", total));
  }

  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                        amdgpu::Wavefront &wf) override {
    profiled_dispatch(prof_before_exec_, [&]() {
      ExecutionPluginGroup::onAmdgpuBeforeExecuteInstruction(pc, inst, wf);
    });
  }

  void onAmdgpuAfterExecuteInstruction(uint64_t pc, const Instruction &inst,
                                       amdgpu::Wavefront &wf) override {
    profiled_dispatch(prof_after_exec_, [&]() {
      ExecutionPluginGroup::onAmdgpuAfterExecuteInstruction(pc, inst, wf);
    });
  }

  void onAmdgpuRouteMemoryInstruction(const Instruction &inst, amdgpu::Wavefront &wf) override {
    profiled_dispatch(prof_route_mem_,
                      [&]() { ExecutionPluginGroup::onAmdgpuRouteMemoryInstruction(inst, wf); });
  }

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override {
    if (prof_before_exec_.count > 0)
      print_profile_summary();
    current_kernel_name_ = info.kernel_name;
    dispatch_kernel_names_[info.dispatch_id] = info.kernel_name;
    ExecutionPluginGroup::onAmdgpuDispatchPacketProcessed(info);
    reset_profiles();
  }

  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) override {
    if (prof_before_exec_.count > 0) {
      auto it = dispatch_kernel_names_.find(dispatch_id);
      if (it != dispatch_kernel_names_.end())
        current_kernel_name_ = it->second;
      print_profile_summary();
    }
    ExecutionPluginGroup::onAmdgpuDispatchExecutionEnd(dispatch_id);
    reset_profiles();
  }

  void onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id, uint32_t vgpr_count,
                                   uint32_t sgpr_count,
                                   std::span<amdgpu::Wavefront *> wavefronts) override {
    profiled_dispatch(prof_wg_dispatched_, [&]() {
      ExecutionPluginGroup::onAmdgpuWorkgroupDispatched(dispatch_id, wg_id, vgpr_count, sgpr_count,
                                                        wavefronts);
    });
  }

  void onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t wg_id) override {
    profiled_dispatch(prof_wg_completed_, [&]() {
      ExecutionPluginGroup::onAmdgpuWorkgroupCompleted(dispatch_id, wg_id);
    });
  }

  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) override {
    profiled_dispatch(prof_wf_dispatched_,
                      [&]() { ExecutionPluginGroup::onAmdgpuWavefrontDispatched(wf); });
  }

  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) override {
    profiled_dispatch(prof_wf_halted_,
                      [&]() { ExecutionPluginGroup::onAmdgpuWavefrontHalted(wf); });
  }

  void onAmdgpuReadVgprs(const amdgpu::Wavefront *wf, uint32_t physical_reg, uint32_t lane_begin,
                         uint32_t lane_end,
                         uint8_t byte_mask = ExecutionPlugin::kFullByteMask) override {
    profiled_dispatch(prof_read_vgpr_, [&]() {
      ExecutionPluginGroup::onAmdgpuReadVgprs(wf, physical_reg, lane_begin, lane_end, byte_mask);
    });
  }

  void onAmdgpuReadSgpr(const amdgpu::Wavefront *wf, uint32_t physical_reg) override {
    profiled_dispatch(prof_read_sgpr_,
                      [&]() { ExecutionPluginGroup::onAmdgpuReadSgpr(wf, physical_reg); });
  }

  void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) override {
    profiled_dispatch(prof_barrier_,
                      [&]() { ExecutionPluginGroup::onAmdgpuBarrierResolved(wavefronts); });
  }

private:
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
    prof_read_sgpr_ = {};
    prof_route_mem_ = {};
    prof_barrier_ = {};
    prof_wg_dispatched_ = {};
    prof_wg_completed_ = {};
    prof_wf_dispatched_ = {};
    prof_wf_halted_ = {};
  }

  void print_profile_summary() {
    const char *kname = current_kernel_name_.empty() ? "?" : current_kernel_name_.c_str();
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
    print_hook("readSgpr", prof_read_sgpr_);
    print_hook("routeMemoryInstruction", prof_route_mem_);
    print_hook("barrierResolved", prof_barrier_);
    print_hook("workgroupDispatched", prof_wg_dispatched_);
    print_hook("workgroupCompleted", prof_wg_completed_);
    print_hook("wavefrontDispatched", prof_wf_dispatched_);
    print_hook("wavefrontHalted", prof_wf_halted_);
    sink().write("HOOK_PROFILE ---\n");
  }

  PluginSink &sink() { return *sink_; }

  Clock::time_point start_time_;
  PluginSink *sink_ = &StderrSink::instance();
  std::string current_kernel_name_;
  std::unordered_map<uint32_t, std::string> dispatch_kernel_names_;

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
