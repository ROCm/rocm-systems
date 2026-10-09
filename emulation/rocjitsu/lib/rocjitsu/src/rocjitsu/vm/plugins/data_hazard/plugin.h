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
///   - `global_memory_races`: check global memory for cross-workgroup races.
///     Off narrows detection to registers and LDS within a workgroup.

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
#include <unordered_set>
#include <vector>

namespace rocjitsu::amdgpu {
class VectorMemState;
}

namespace rocjitsu::amdgpu::tensor_dma_detail {
class TensorDmaDescriptor;
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
  /// The disassembly cache of @p dispatch_id, created on first use. Callers on
  /// the execution path fetch it once per wave and record into it directly, so
  /// the formatter's own lock is not taken per instruction.
  std::shared_ptr<DisasmCache> cache_for(hazard_core::EntityId dispatch_id);

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
  static std::string format(const hazard_core::InstructionDescriptor &instruction,
                            const std::string &disassembly);

  std::shared_ptr<const DisasmCache> find_cache(hazard_core::EntityId dispatch_id) const;

  std::atomic<rj_code_arch_t> arch_{ROCJITSU_CODE_ARCH_INVALID};
  mutable std::mutex mutex_;
  /// One cache per dispatch rather than one for the plugin: a code object can
  /// be unloaded and a later dispatch can run different code at the same
  /// address, and a PC-keyed entry outliving its code object would quote that
  /// later dispatch the instruction it replaced.
  std::unordered_map<hazard_core::EntityId, std::shared_ptr<DisasmCache>> disassembly_;
};

/// @brief Tracks the instruction currently executing on one wavefront.
///
/// Register-read and memory-route callbacks carry no instruction identity, so
/// they are attributed to whatever this wave most recently began executing.
/// The lanes of each VGPR an instruction has not yet reported reading.
///
/// An instruction can read one register many times: a scalar-path MFMA reads
/// each operand element through its own one-lane read, tens of thousands per
/// instruction. A repeat read of a lane checks the same pending state as the
/// first and can find nothing it did not, so only the first read of each lane
/// is passed on. Entries are stamped with the instruction that set them, so a
/// new instruction starts clean without clearing anything.
class VgprReadFilter {
public:
  /// The lanes of @p lane_mask that @p instruction_id has not yet read from
  /// VGPR @p reg, which are recorded as read.
  uint64_t take_new_lanes(uint32_t reg, hazard_core::EntityId instruction_id, uint64_t lane_mask) {
    if (reg >= lanes_.size()) {
      lanes_.resize(reg + 1, 0);
      instructions_.resize(reg + 1, 0);
    }
    if (instructions_[reg] != instruction_id) {
      instructions_[reg] = instruction_id;
      lanes_[reg] = 0;
    }
    const uint64_t new_lanes = lane_mask & ~lanes_[reg];
    lanes_[reg] |= new_lanes;
    return new_lanes;
  }

  void clear() {
    lanes_.clear();
    instructions_.clear();
  }

private:
  std::vector<uint64_t> lanes_;
  std::vector<hazard_core::EntityId> instructions_;
};

struct DataHazardWavefrontState : WavefrontState {
  InstructionView current;
  bool is_memory_op = false;

  /// Instruction ids are this wave's serial in the high 32 bits and a per-wave
  /// count in the low 32: unique across waves and in program order within one,
  /// with no counter shared between threads.
  hazard_core::EntityId id_base = 0;
  uint32_t id_count = 0;

  /// The dispatch's disassembly cache and the PCs this wave has already put in
  /// it, so the cache's lock is taken once per wave per PC rather than once per
  /// executed instruction.
  std::shared_ptr<DisasmCache> disassembly;
  hazard_core::EntityId disassembly_dispatch = 0;
  std::unordered_set<uint64_t> recorded_pcs;

  /// The wave's engine state, fetched once at wave dispatch, so its events are
  /// delivered by pointer rather than looked up by key.
  hazard_core::DataHazardEngine::WaveHandle engine_wave;

  /// Read hooks are const, so the filter of VGPR reads already reported is
  /// mutable.
  mutable VgprReadFilter vgpr_reads;

  void reset() {
    current = InstructionView{};
    is_memory_op = false;
    id_base = 0;
    id_count = 0;
    disassembly.reset();
    disassembly_dispatch = 0;
    recorded_pcs.clear();
    engine_wave.reset();
    vgpr_reads.clear();
  }
};

/// @brief Resolved plugin configuration.
struct PluginConfig {
  /// Destination of the JSON hazard report; empty disables file output.
  std::string report_path;
  /// Retain every hazard occurrence rather than folding duplicates.
  bool verbose = false;
  /// Check global memory accesses for cross-workgroup races.
  bool global_memory_races = true;
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

  /// The next instruction id for @p state's wave; see DataHazardWavefrontState.
  /// Waves without plugin state draw from a plugin-wide counter instead.
  uint64_t next_instruction_id(DataHazardWavefrontState *state);
  /// Records @p inst's disassembly for its dispatch the first time its wave
  /// executes @p view.pc.
  void record_disassembly(DataHazardWavefrontState *state, const InstructionView &view,
                          const Instruction &inst);

  /// Writes one line to the plugin sink. Detection runs on several partition
  /// threads and the file sink does no locking of its own, so every write of
  /// this plugin's goes through here.
  void write_to_sink(const std::string &text);

  /// True once teardown has begun. Callbacks return early from that point so a
  /// report written at process exit is not racing fresh detections.
  bool shutting_down() const { return shutting_down_.load(std::memory_order_acquire); }

  void emit_source_reads(const InstructionView &view, const Instruction &inst,
                         const amdgpu::Wavefront &wf);
  void emit_destination_writes(const InstructionView &view, const Instruction &inst,
                               const amdgpu::Wavefront &wf);
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
  /// Source of each wave's id_base. Taken once per wave.
  std::atomic<uint32_t> next_wave_serial_{0};
  std::atomic<bool> shutting_down_{false};
  mutable std::mutex sink_mutex_;
  std::once_flag report_once_;
};

} // namespace rocjitsu::plugins::data_hazard
