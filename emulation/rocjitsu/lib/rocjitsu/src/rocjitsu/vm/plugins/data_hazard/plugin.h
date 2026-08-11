// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin.h
/// @brief Data hazard detection plugin.
///
/// Detects RAW, WAR and WAW hazards caused by missing or insufficient wait
/// instructions, plus LDS and global memory races, by observing register
/// accesses and memory routing during execution.
///
/// The plugin is a thin translation layer: it converts rocJitsu callback
/// payloads into the neutral view structs in adapter.h, hands them to
/// DataHazardAdapter, and reports whatever the engine finds through report.h.
///
/// Configuration (see kConfigSchema in plugin_export.cpp):
///   - `report_path`: file to write the JSON hazard report to. Empty means the
///     summary is only written to the plugin sink.
///   - `verbose`: report every hazard occurrence instead of folding duplicates.

#pragma once

#include "rocjitsu/vm/plugins/data_hazard/adapter.h"
#include "rocjitsu/vm/plugins/data_hazard/report.h"
#include "rocjitsu/vm/plugins/disasm_cache.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/wavefront_state.h"

#include "data_hazard_engine.h"
#include "simulator_api.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace rocjitsu::plugins::data_hazard {

/// @brief Renders instruction identity for hazard messages.
///
/// The engine only carries an ::hazard_core::InstructionDescriptor, so what an
/// instruction was is remembered here as instructions are observed and looked
/// up again when a hazard is reported. The text comes from the shared
/// DisasmCache, keyed by PC so each static instruction is disassembled once;
/// the per-execution metadata is appended at format time.
class InstructionFormatter final : public hazard_core::SimulatorInstructionFormatter {
public:
  void remember(const InstructionView &view, const Instruction &inst);

  std::string
  format_instruction(const hazard_core::InstructionDescriptor &instruction) const override;

  void clear();

private:
  /// Bounds retention so long-running dispatches do not grow without limit.
  static constexpr size_t kMaxCachedInstructions = 4096;

  static std::string format(const InstructionView &view, const std::string &disassembly);
  static std::string format_fallback(const hazard_core::InstructionDescriptor &instruction);

  DisasmCache disassembly_;
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, InstructionView> instructions_;
  std::deque<uint64_t> insertion_order_;
};

/// @brief Tracks the instruction currently executing on one wavefront.
///
/// Register-read and memory-route callbacks carry no instruction identity, so
/// they are attributed to whatever this wave most recently began executing.
struct DataHazardWavefrontState : WavefrontState {
  InstructionView current;
  bool is_memory_op = false;

  void reset() {
    current = InstructionView{};
    is_memory_op = false;
  }
};

/// @brief Resolved plugin configuration.
struct PluginConfig {
  /// Destination of the JSON hazard report; empty disables file output.
  std::string report_path;
  /// Retain every hazard occurrence rather than folding duplicates.
  bool verbose = false;
};

class DataHazardPlugin : public ExecutionPlugin {
public:
  /// @param config_json Resolved plugin configuration object. May be null.
  explicit DataHazardPlugin(const char *config_json = nullptr);
  ~DataHazardPlugin() override;

  void onInit() override;
  void onShutdown() override;

  void onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) override;
  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) override;

  void onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id,
                                   uint32_t physical_vgpr_count, uint32_t sgpr_count,
                                   std::span<amdgpu::Wavefront *> wavefronts) override;
  void onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t wg_id) override;

  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) override;
  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) override;

  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                        amdgpu::Wavefront &wf) override;
  void onAmdgpuRouteMemoryInstruction(const Instruction &inst, amdgpu::Wavefront &wf) override;

  void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg, uint64_t lane_mask,
                             uint8_t byte_mask = ExecutionPlugin::kFullByteMask) override;
  void onAmdgpuReadSgpr(const amdgpu::Wavefront *wf, uint32_t physical_reg) override;

  void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) override;

  /// Exposed for tests: the hazards collected so far.
  const WarningCollector &warnings() const { return collector_; }

  /// @brief Stops accepting callbacks, so a report is not written while new
  /// hazards are still arriving. Called before every report write.
  void begin_shutdown();

  /// @brief Emits the summary and, when configured, the JSON report file.
  /// @details Idempotent: only the first call of a plugin's lifetime writes.
  /// Called from shutdown, from the destructor, and from a process-exit handler,
  /// because a launched application need not tear the VM down before exiting.
  void write_report();

private:
  /// Installs the state slot. Wavefront::plugin_state reads the slot without a
  /// bounds check, so this must run for every wave in onAmdgpuWorkgroupDispatched
  /// before any other callback touches it.
  void seed_wave_state(amdgpu::Wavefront &wf);
  DataHazardWavefrontState *wave_state(amdgpu::Wavefront &wf) const;
  const DataHazardWavefrontState *wave_state(const amdgpu::Wavefront *wf) const;

  uint64_t next_instruction_id() { return next_instruction_id_.fetch_add(1) + 1; }

  /// Writes one line to the plugin sink. Detection runs on several partition
  /// threads and the file sink does no locking of its own, so every write of
  /// this plugin's goes through here.
  void write_to_sink(const std::string &text);

  /// True once teardown has begun. Callbacks return early from that point so a
  /// report written at process exit is not racing fresh detections.
  bool shutting_down() const { return shutting_down_.load(std::memory_order_acquire); }

  void emit_source_reads(const InstructionView &view, const Instruction &inst, bool is_memory_op);
  void emit_destination_writes(const InstructionView &view, const Instruction &inst);
  void emit_tensor_lds_write(const InstructionView &view, const amdgpu::Wavefront &wf);
  void route_scalar_memory(const Instruction &inst, amdgpu::Wavefront &wf,
                           const InstructionView &current);
  void route_vector_memory(const Instruction &inst, amdgpu::Wavefront &wf,
                           const InstructionView &current);

  PluginConfig config_;
  hazard_core::DataHazardEngine engine_;
  InstructionFormatter formatter_;
  WarningCollector collector_;
  CollectingWarningSink warning_sink_;
  DataHazardAdapter adapter_;

  std::atomic<uint64_t> next_instruction_id_{0};
  std::atomic<bool> shutting_down_{false};
  mutable std::mutex sink_mutex_;
  std::once_flag report_once_;
};

} // namespace rocjitsu::plugins::data_hazard
