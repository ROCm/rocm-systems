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

#include "rocjitsu/code/rj_code.h"
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
#include <vector>

namespace rocjitsu::amdgpu {
struct VectorMemState;
}

namespace rocjitsu::amdgpu::tensor_dma_detail {
struct TensorDmaDescriptor;
}

namespace rocjitsu::plugins::data_hazard {

/// A stretch of LDS one access covers, relative to the wave's LDS allocation.
struct LocalMemoryRange {
  uint64_t address = 0;
  uint32_t size = 0;
};

/// The LDS a tensor DMA transfer touches — written by `tensor_load_to_lds`,
/// read by `tensor_store_from_lds` — derived from the descriptor it transfers
/// under: a variable base, an element size, tile dimensions, an iteration
/// stride, and padding that skews the rows apart. A descriptor that transfers
/// nothing, or one the executor rejects, touches no LDS and yields no ranges.
///
/// The direction is part of where the transfer lands, not only of what it does
/// to the LDS it names: the ISA applies descriptor padding to memory-to-LDS
/// transfers alone, so a @p store_from_lds transfer reads the dense element
/// stream that the same descriptor would have skewed on the way in.
///
/// A padded run is reported as the whole span it skews across, gaps included.
/// The gaps hold no transferred data, but they sit between rows of the same
/// tile, and reporting them apart would leave a pending write per row for every
/// later LDS access to walk.
std::vector<LocalMemoryRange>
tensor_lds_ranges(const amdgpu::tensor_dma_detail::TensorDmaDescriptor &desc, bool store_from_lds);

/// @brief Renders instruction identity for hazard messages.
///
/// The engine only carries an ::hazard_core::InstructionDescriptor, so what an
/// instruction was is remembered here as instructions are observed and looked
/// up again when a hazard is reported. The text comes from a DisasmCache keyed
/// by PC, so each static instruction is disassembled once; the per-execution
/// metadata is appended at format time.
class InstructionFormatter final : public hazard_core::SimulatorInstructionFormatter {
public:
  void remember(const InstructionView &view, const Instruction &inst);

  /// Names the architecture the suggestions must be assemblable on. Until this
  /// is called the wording defaults to the split-counter waits.
  void set_architecture(rj_code_arch_t arch);

  std::string
  format_instruction(const hazard_core::InstructionDescriptor &instruction) const override;

  /// The split waits (`s_wait_loadcnt` and the rest) exist only on gfx12-era
  /// targets. Everywhere else the same counters are drained by the legacy
  /// `s_waitcnt` forms, so a report on those targets has to name those instead
  /// of instructions the assembler would reject.
  std::string format_wait_suggestion(hazard_core::WaitCntType kind,
                                     hazard_core::HazardAccessKind access,
                                     hazard_core::HazardResourceLabel resource) const override;

  /// A flat load holds two counters at once, which gfx12-era targets drain with
  /// the combined `s_wait_loadcnt_dscnt`. The legacy targets have no such
  /// instruction: a flat load counts on vmcnt and lgkmcnt, and both are named.
  std::string format_flat_load_wait_suggestion() const override;

  /// Drops the disassembly recorded for @p dispatch_id, once nothing more can
  /// be reported against it.
  void forget_dispatch(hazard_core::EntityId dispatch_id);

  void clear();

private:
  /// Bounds retention so long-running dispatches do not grow without limit.
  static constexpr size_t kMaxCachedInstructions = 4096;

  static std::string format(const InstructionView &view, const std::string &disassembly);
  static std::string format_fallback(const hazard_core::InstructionDescriptor &instruction);

  std::shared_ptr<DisasmCache> cache_for(hazard_core::EntityId dispatch_id);
  std::shared_ptr<const DisasmCache> find_cache(hazard_core::EntityId dispatch_id) const;

  std::atomic<rj_code_arch_t> arch_{ROCJITSU_CODE_ARCH_INVALID};
  mutable std::mutex mutex_;
  /// One cache per dispatch rather than one for the plugin: a code object can
  /// be unloaded and a later dispatch can run different code at the same
  /// address, and a PC-keyed entry outliving its code object would quote that
  /// later dispatch the instruction it replaced.
  std::unordered_map<hazard_core::EntityId, std::shared_ptr<DisasmCache>> disassembly_;
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

  void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts,
                               AmdgpuBarrierScope scope) override;

  /// Exposed for tests: the hazards collected so far.
  const WarningCollector &warnings() const { return collector_; }

  /// @brief Stops accepting callbacks, so a report is not written while new
  /// hazards are still arriving. Called before every report write.
  void begin_shutdown();

  /// @brief Checks the outstanding local memory epochs, then emits the summary
  /// and, when configured, the JSON report file.
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

  void emit_source_reads(const InstructionView &view, const Instruction &inst);
  void emit_destination_writes(const InstructionView &view, const Instruction &inst);
  /// Routes the LDS a tensor DMA transfer touches, derived from the descriptor
  /// the transfer will execute from. @p reads_lds distinguishes a store
  /// streaming LDS out from a load streaming it in.
  void emit_tensor_lds_access(const InstructionView &view, const Instruction &inst,
                              const amdgpu::Wavefront &wf, bool reads_lds);
  void route_scalar_memory(const Instruction &inst, amdgpu::Wavefront &wf,
                           const InstructionView &current);
  void route_vector_memory(const Instruction &inst, amdgpu::Wavefront &wf,
                           const InstructionView &current);
  /// Routes the second access of a dual-address DS instruction, which @p first
  /// describes the first half of.
  void route_second_local_access(const amdgpu::VectorMemState &vmem, const amdgpu::Wavefront &wf,
                                 const MemoryRouteView &first, size_t lanes);

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
