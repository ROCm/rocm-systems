// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin_group.h
/// @brief Collection of plugins that delegates to each member.
///
/// ## Sink configuration
///
/// The group receives its complete, owned sink configuration at construction.
/// Sink destinations cannot be added after plugins have received their output
/// sink, so construction order and lifetime are explicit in the type.
///
/// When a plugin is added via add(), the group constructs an internal fanout sink
/// combining all configured sinks + an optional per-plugin FileSink, and
/// assigns it to the plugin.

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace test {
class ExecutionPluginGroupTestAccess;
}

/// @brief Deleter for owned plugin instances.
///
/// Every plugin instance is freed through a destroy function, so allocation and
/// deallocation always stay on the same side of the plugin library boundary. For
/// dynamically loaded plugins, @c destroyFn is the plugin library's
/// `rocjitsu_plugin_destroy` export. For in-tree plugins created with `new`, it
/// is a small host trampoline that `delete`s the instance (see
/// delete_execution_plugin()). The pointer is only ever handed back to the
/// deleter that created it, and `std::unique_ptr` never invokes the deleter on
/// a null instance, so no null check is needed here.
struct PluginDeleter {
  void (*destroyFn)(void *) = nullptr;
  void operator()(ExecutionPlugin *p) const { destroyFn(static_cast<void *>(p)); }
};

/// @brief Owning pointer to a plugin instance with a boundary-aware deleter.
using OwnedPlugin = std::unique_ptr<ExecutionPlugin, PluginDeleter>;

/// @brief Host destroy trampoline for in-tree plugins created with `new`.
inline void delete_execution_plugin(void *p) { delete static_cast<ExecutionPlugin *>(p); }

/// @brief Move-only sink configuration transferred into an ExecutionPluginGroup.
struct PluginSinkConfig {
  /// Add an owned sink and return a reference to it.
  ///
  /// The reference remains valid while this configuration, or the
  /// ExecutionPluginGroup that receives it, owns the sink. Moving the
  /// configuration transfers ownership without moving the sink itself.
  template <typename Sink, typename... Args> Sink &emplace(Args &&...args) {
    static_assert(std::is_base_of_v<PluginSink, Sink>, "Sink must derive from PluginSink");
    auto sink = std::make_unique<Sink>(std::forward<Args>(args)...);
    Sink &result = *sink;
    sinks_.push_back(std::move(sink));
    return result;
  }

  /// Enable one per-plugin FileSink rooted at @p directory.
  void set_file_directory(std::string directory) { file_directory_ = std::move(directory); }

private:
  friend class ExecutionPluginGroup;
  std::vector<std::unique_ptr<PluginSink>> sinks_;
  std::string file_directory_;
};

/// @brief Immutable-after-publication collection of execution plugins.
///
/// Configure sinks at construction and call add() before handing the group to
/// simulation components. add() is not thread-safe; after publication, the
/// plugin collection and sampled hook policy must remain immutable while
/// callbacks may dispatch concurrently.
class ExecutionPluginGroup final {
public:
  explicit ExecutionPluginGroup(PluginSinkConfig config)
      : configured_sinks_(std::move(config.sinks_)), sink_dir_(std::move(config.file_directory_)) {}

  ~ExecutionPluginGroup() = default;

  /// Add a plugin owned with a boundary-aware deleter (see OwnedPlugin).
  bool add(OwnedPlugin p) {
    if (!p)
      return false;
    for (const auto &existing : plugins_)
      if (existing.plugin->name() == p->name())
        return false;
    p->slot_index_ = static_cast<uint32_t>(plugins_.size());
    serialize_hot_hooks_ |= p->requires_serial_hot_hooks();
    observes_sgpr_reads_ |= p->observes_sgpr_reads();
    SinkBundle sink = build_sink_bundle(p->name() + ".log");
    if (auto *configured_sink = sink.get())
      p->sink_ = configured_sink;
    plugins_.push_back(PluginEntry{std::move(sink), std::move(p)});
    return true;
  }

  /// Add an in-tree plugin freed with `delete` (host/test convenience).
  bool add(std::unique_ptr<ExecutionPlugin> p) {
    return add(OwnedPlugin(p.release(), PluginDeleter{&delete_execution_plugin}));
  }

  uint32_t num_plugins() const { return static_cast<uint32_t>(plugins_.size()); }
  bool empty() const { return plugins_.empty(); }

  /// Whether high-frequency callbacks are serialized for this group. Plugin
  /// policy is sampled when each plugin is added so hot dispatch stays O(1).
  bool requires_serial_hot_hooks() const { return serialize_hot_hooks_; }

  /// Whether any contained plugin observes SGPR reads. Plugin policy is
  /// sampled when each plugin is added so SGPR access stays O(1).
  bool observes_sgpr_reads() const { return observes_sgpr_reads_; }

  // -- Lifecycle (non-virtual) --
  // The host calls these outside the simulation-callback interval: onInit()
  // completes before callbacks start, and onShutdown() starts after they stop.
  // callback_mutex_ orders simulation callbacks, not lifecycle teardown.
  void onInit() {
    dispatch_with_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onInit();
    });
  }

  void onShutdown() {
    dispatch_with_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onShutdown();
    });
  }

  // -- AMDGPU (non-virtual) --
  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                        amdgpu::Wavefront &wf) {
    dispatch_with_optional_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuBeforeExecuteInstruction(pc, inst, wf);
    });
  }

  void onAmdgpuAfterExecuteInstruction(uint64_t pc, const Instruction &inst,
                                       amdgpu::Wavefront &wf) {
    dispatch_with_optional_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuAfterExecuteInstruction(pc, inst, wf);
    });
  }

  void onAmdgpuRouteMemoryInstruction(const Instruction &inst, amdgpu::Wavefront &wf) {
    dispatch_with_optional_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuRouteMemoryInstruction(inst, wf);
    });
  }

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
    dispatch_with_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuDispatchPacketProcessed(info);
    });
  }

  void onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) {
    dispatch_with_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuDispatchExecutionBegin(dispatch_id);
    });
  }

  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) {
    dispatch_with_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuDispatchExecutionEnd(dispatch_id);
    });
  }

  void onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id,
                                   uint32_t physical_vgpr_count, uint32_t physical_sgpr_count,
                                   std::span<amdgpu::Wavefront *> wavefronts) {
    dispatch_with_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuWorkgroupDispatched(dispatch_id, wg_id, physical_vgpr_count,
                                                  physical_sgpr_count, wavefronts);
    });
  }

  void onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t wg_id) {
    dispatch_with_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuWorkgroupCompleted(dispatch_id, wg_id);
    });
  }

  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) {
    dispatch_with_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuWavefrontDispatched(wf);
    });
  }

  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) {
    dispatch_with_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuWavefrontHalted(wf);
    });
  }

  void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg, uint64_t lane_mask,
                             uint8_t byte_mask = ExecutionPlugin::kFullByteMask) {
    dispatch_with_optional_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuReadVgprLanes(wf, physical_reg, lane_mask, byte_mask);
    });
  }

  void onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                              uint64_t lane_mask,
                              uint8_t byte_mask = ExecutionPlugin::kFullByteMask) {
    dispatch_with_optional_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuWriteVgprLanes(wf, physical_reg, lane_mask, byte_mask);
    });
  }

  void onAmdgpuReadSgpr(const amdgpu::Wavefront *wf, uint32_t physical_reg) {
    dispatch_with_optional_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuReadSgpr(wf, physical_reg);
    });
  }

  void onAmdgpuReadScalarRegister(const amdgpu::Wavefront *wf, RegisterRef reg) {
    dispatch_with_optional_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuReadScalarRegister(wf, reg);
    });
  }

  void onAmdgpuWriteScalarRegister(const amdgpu::Wavefront *wf, RegisterRef reg) {
    dispatch_with_optional_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuWriteScalarRegister(wf, reg);
    });
  }

  void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) {
    dispatch_with_plugin_lock([&]() {
      for (auto &entry : plugins_)
        entry.plugin->onAmdgpuBarrierResolved(wavefronts);
    });
  }

  static std::shared_ptr<ExecutionPluginGroup> empty_group() {
    // Immortal singleton: the shared_ptr is heap-allocated and deliberately never
    // deleted, so its control block outlives process teardown. In local-mode
    // (LD_PRELOAD interposer) the simulation engine runs on a detached thread that
    // is still executing when exit() drives static/atexit destructors on the main
    // thread. A plain function-local `static shared_ptr` would have its control
    // block destroyed by a __cxa_atexit handler during __run_exit_handlers while the
    // engine thread is mid-startup() copying this default plugin group into a
    // CompletionTracker/ComputeUnit — a data race on the refcount that surfaced as a
    // use-after-free SIGSEGV in _Sp_counted_base::_M_release under `ctest -jN`.
    // Leaking the control block removes that teardown race; the OS reclaims the
    // memory at process death. Matches the interposer singleton's never-destructed
    // design for the same reason.
    static std::shared_ptr<ExecutionPluginGroup> *instance =
        new std::shared_ptr<ExecutionPluginGroup>(
            std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{}));
    return *instance;
  }

private:
  friend class test::ExecutionPluginGroupTestAccess;

  template <typename Callback> void dispatch_with_plugin_lock(Callback &&callback) {
    if (plugins_.empty())
      return;
    std::lock_guard<std::recursive_mutex> lock(callback_mutex_);
    ++callback_lock_acquisitions_;
    std::forward<Callback>(callback)();
  }

  template <typename Callback> void dispatch_with_optional_plugin_lock(Callback &&callback) {
    if (plugins_.empty())
      return;
    if (serialize_hot_hooks_)
      dispatch_with_plugin_lock(std::forward<Callback>(callback));
    else
      std::forward<Callback>(callback)();
  }

  // Infrequent hooks may synchronously fire hot register hooks. Recursive
  // acquisition preserves one cross-hook serialization domain without
  // deadlocking that same-thread re-entry.
  std::recursive_mutex callback_mutex_;
  // Read only through the friend test seam. The mutex protects increments;
  // empty groups return before touching either the mutex or this counter.
  uint64_t callback_lock_acquisitions_ = 0;
  bool serialize_hot_hooks_ = false;
  bool observes_sgpr_reads_ = false;

  /// Internal fanout over sinks whose lifetime is guaranteed by the owning
  /// group or SinkBundle. It is deliberately not part of the public sink API.
  class FanoutSink final : public PluginSink {
  public:
    void add(PluginSink &sink) { children_.push_back(&sink); }
    void write(std::string_view msg) override {
      for (auto *sink : children_)
        sink->write(msg);
    }
    bool empty() const { return children_.empty(); }

  private:
    std::vector<PluginSink *> children_;
  };

private:
  /// Owns one fanout sink and any per-fanout child sinks. Member order ensures
  /// the fanout is destroyed before the children it references.
  class SinkBundle {
  public:
    SinkBundle() = default;

    [[nodiscard]] PluginSink *get() const { return fanout_.get(); }

  private:
    friend class ExecutionPluginGroup;
    std::vector<std::unique_ptr<PluginSink>> children_;
    std::unique_ptr<FanoutSink> fanout_;
  };

  /// Build a sink combining configured sinks + optional file sink.
  /// Returns an empty bundle if no sinks are configured.
  [[nodiscard]] SinkBundle build_sink_bundle(const std::string &file_name) const {
    bool has_file = !sink_dir_.empty() && !file_name.empty();
    if (configured_sinks_.empty() && !has_file)
      return {};

    SinkBundle result;
    auto fanout = std::make_unique<FanoutSink>();
    for (const auto &s : configured_sinks_)
      fanout->add(*s);
    if (has_file) {
      auto fs = std::make_unique<FileSink>(sink_dir_ + "/" + file_name);
      if (fs->is_open()) {
        fanout->add(*fs);
        result.children_.push_back(std::move(fs));
      } else if (fanout->empty()) {
        // Preserve output when the file was the only requested destination.
        // Do not add stderr when another configured sink is already usable,
        // because doing so would unexpectedly duplicate output.
        auto fallback = std::make_unique<StderrSink>();
        fanout->add(*fallback);
        result.children_.push_back(std::move(fallback));
      }
    }
    result.fanout_ = std::move(fanout);
    return result;
  }

  // These sinks are declared before plugin entries so plugins and their local
  // fanouts are destroyed before the configured sinks they reference.
  std::vector<std::unique_ptr<PluginSink>> configured_sinks_;
  std::string sink_dir_;
  /// Owns the sink assigned to one plugin. The plugin is declared last and is
  /// therefore destroyed before its sink bundle.
  struct PluginEntry {
    SinkBundle sink;
    OwnedPlugin plugin;
  };

  std::vector<PluginEntry> plugins_;
};

} // namespace rocjitsu
