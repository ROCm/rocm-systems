// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin_group.h
/// @brief Collection of plugins that delegates to each member.
///
/// ## Sink configuration
///
/// The group controls where plugin output goes. Call add_sink() to add
/// external sinks (applied to all plugins). Call set_sink_dir() before
/// adding plugins to enable per-plugin file sinks at <dir>/<name>.log.
///
/// When a plugin is added via add(), the group constructs a CompositeSink
/// combining all external sinks + an optional per-plugin FileSink, and
/// assigns it to the plugin.

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

class ExecutionPluginGroup {
public:
  ExecutionPluginGroup() = default;
  virtual ~ExecutionPluginGroup() = default;

  /// Add an external sink that receives output from all plugins.
  /// The caller retains ownership (e.g., a static singleton or test fixture).
  void add_sink(PluginSink *s) {
    if (s)
      external_sinks_.push_back(s);
  }

  /// Set directory for per-plugin file sinks. Each plugin added after
  /// this call gets a FileSink at <dir>/<plugin_name>.log.
  void set_sink_dir(const std::string &dir) { sink_dir_ = dir; }

  bool add(std::unique_ptr<ExecutionPlugin> p) {
    if (!p)
      return false;
    for (const auto &existing : plugins_)
      if (existing->name() == p->name())
        return false;
    p->slot_index_ = static_cast<uint32_t>(plugins_.size());
    build_sink_for(*p);
    plugins_.push_back(std::move(p));
    return true;
  }

  uint32_t num_plugins() const { return static_cast<uint32_t>(plugins_.size()); }
  bool empty() const { return plugins_.empty(); }

  // -- Lifecycle (non-virtual) --
  void onInit() {
    for (auto &p : plugins_)
      p->onInit();
  }

  void onShutdown() {
    for (auto &p : plugins_)
      p->onShutdown();
  }

  // -- AMDGPU (virtual) --
  virtual void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                                amdgpu::Wavefront &wf) {
    for (auto &p : plugins_)
      p->onAmdgpuBeforeExecuteInstruction(pc, inst, wf);
  }

  virtual void onAmdgpuAfterExecuteInstruction(uint64_t pc, const Instruction &inst,
                                               amdgpu::Wavefront &wf) {
    for (auto &p : plugins_)
      p->onAmdgpuAfterExecuteInstruction(pc, inst, wf);
  }

  virtual void onAmdgpuRouteMemoryInstruction(const Instruction &inst, amdgpu::Wavefront &wf) {
    for (auto &p : plugins_)
      p->onAmdgpuRouteMemoryInstruction(inst, wf);
  }

  virtual void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
    for (auto &p : plugins_)
      p->onAmdgpuDispatchPacketProcessed(info);
  }

  virtual void onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) {
    for (auto &p : plugins_)
      p->onAmdgpuDispatchExecutionBegin(dispatch_id);
  }

  virtual void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) {
    for (auto &p : plugins_)
      p->onAmdgpuDispatchExecutionEnd(dispatch_id);
  }

  virtual void onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id,
                                           uint32_t vgpr_count, uint32_t sgpr_count,
                                           std::span<amdgpu::Wavefront *> wavefronts) {
    for (auto &p : plugins_)
      p->onAmdgpuWorkgroupDispatched(dispatch_id, wg_id, vgpr_count, sgpr_count, wavefronts);
  }

  virtual void onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t wg_id) {
    for (auto &p : plugins_)
      p->onAmdgpuWorkgroupCompleted(dispatch_id, wg_id);
  }

  virtual void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) {
    for (auto &p : plugins_)
      p->onAmdgpuWavefrontDispatched(wf);
  }

  virtual void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) {
    for (auto &p : plugins_)
      p->onAmdgpuWavefrontHalted(wf);
  }

  virtual void onAmdgpuReadVgprs(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                                 uint32_t lane_begin, uint32_t lane_end,
                                 uint8_t byte_mask = ExecutionPlugin::kFullByteMask) {
    for (auto &p : plugins_)
      p->onAmdgpuReadVgprs(wf, physical_reg, lane_begin, lane_end, byte_mask);
  }

  virtual void onAmdgpuReadSgpr(const amdgpu::Wavefront *wf, uint32_t physical_reg) {
    for (auto &p : plugins_)
      p->onAmdgpuReadSgpr(wf, physical_reg);
  }

  virtual void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) {
    for (auto &p : plugins_)
      p->onAmdgpuBarrierResolved(wavefronts);
  }

  static std::shared_ptr<ExecutionPluginGroup> empty_group() {
    static auto instance = std::make_shared<ExecutionPluginGroup>();
    return instance;
  }

protected:
  /// Build a sink combining external sinks + optional file sink.
  /// Returns nullptr if no sinks are configured (caller should keep default).
  PluginSink *build_composite_sink(const std::string &file_name = {}) {
    bool has_file = !sink_dir_.empty() && !file_name.empty();
    if (external_sinks_.empty() && !has_file)
      return nullptr;
    if (external_sinks_.size() == 1 && !has_file)
      return external_sinks_[0];

    auto composite = std::make_unique<CompositeSink>();
    for (auto *s : external_sinks_)
      composite->add(s);
    if (has_file) {
      auto fs = std::make_unique<FileSink>(sink_dir_ + "/" + file_name);
      composite->add(fs.get());
      owned_sinks_.push_back(std::move(fs));
    }
    auto *result = composite.get();
    owned_sinks_.push_back(std::move(composite));
    return result;
  }

  std::vector<PluginSink *> external_sinks_;
  std::string sink_dir_;
  std::vector<std::unique_ptr<PluginSink>> owned_sinks_;

private:
  void build_sink_for(ExecutionPlugin &p) {
    if (auto *s = build_composite_sink(p.name() + ".log"))
      p.sink_ = s;
  }

  std::vector<std::unique_ptr<ExecutionPlugin>> plugins_;
};

} // namespace rocjitsu
