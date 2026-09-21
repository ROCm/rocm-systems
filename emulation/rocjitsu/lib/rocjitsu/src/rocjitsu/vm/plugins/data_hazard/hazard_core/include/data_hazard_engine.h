// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Simulator-neutral data hazard engine. Adapters translate native simulator
// callbacks into generic events and delegate to this engine.

#pragma once

#include "data_hazard_state.h"
#include "simulator_api.h"

#include <atomic>
#include <functional>
#include <optional>
#include <string>

namespace hazard_core {

/// Notified when the engine drops a malformed event. The engine runs on the
/// simulator's callback path, where a throw would abort wave execution, so
/// frontends install this to log instead.
using EngineDiagnosticHandler = std::function<void(const std::string &message)>;

class DataHazardEngine final : public DataHazardSimulatorApi {
public:
  void reset();
  void set_instruction_formatter(const SimulatorInstructionFormatter *formatter);
  void set_warning_sink(const HazardWarningSink *sink);
  void set_diagnostic_handler(EngineDiagnosticHandler handler);

  /// Number of events dropped as malformed since the last reset().
  uint64_t rejected_event_count() const;

  std::vector<EngineWarning> warning_snapshot() const;
  std::optional<EngineWaveSnapshot> wave_snapshot(const ExecutionKey &wave) const;
  std::vector<EngineWaveSnapshot> wave_snapshots() const;
  size_t wave_count() const;

  void on_dispatch_begin(EntityId dispatch_id) override;
  void on_dispatch_end(EntityId dispatch_id) override;
  void on_workgroup_begin(EntityId dispatch_id, EntityId cluster_id,
                          EntityId workgroup_id) override;
  void on_workgroup_end(EntityId dispatch_id, EntityId cluster_id, EntityId workgroup_id) override;
  void on_wave_begin(const ExecutionKey &wave) override;
  void on_wave_end(const ExecutionKey &wave) override;
  void on_instruction(const InstructionEvent &instruction) override;
  void on_resource_access(const ResourceAccessEvent &event) override;
  void on_barrier(const BarrierEvent &barrier) override;
  void on_shutdown() override;

private:
  std::shared_ptr<EngineWaveState> get_or_create_wave(const ExecutionKey &key,
                                                      const EngineWorkgroupKey &workgroup_key,
                                                      const EngineWaveKey &partial);
  std::shared_ptr<EngineWaveState> get_wave(const ExecutionKey &key);
  std::shared_ptr<EngineWaveState> get_wave(const EngineWaveKey &partial);
  std::vector<std::shared_ptr<EngineWaveState>>
  waves_for_barrier(const BarrierEvent &barrier) const;
  std::vector<EngineWorkgroupKey> workgroup_keys_for_barrier(const BarrierEvent &barrier) const;
  std::shared_ptr<EngineWorkgroupState> get_or_create_workgroup(const EngineWorkgroupKey &key);
  void reset_dispatch(EntityId dispatch_id, bool flush_epochs);
  void erase_wave(const ExecutionKey &key, const EngineWaveKey &partial);
  void record_warning(const EngineInstructionContext &ctx, const ResourceAccessEvent &event,
                      HazardKind kind, ResourceKind resource_kind, HazardAccessKind access_kind,
                      uint32_t resource_index, uint64_t address, uint32_t size_bytes,
                      WaitCntType required_wait, const PendingAsyncOp &pending,
                      const std::array<uint32_t, 4> &source_raw_isa, uint64_t source_pc,
                      const std::string &message, const std::string &suggestion);
  void handle_vector_access(EngineWaveState &wave, const EngineInstructionContext &ctx,
                            const ResourceAccessEvent &event, WaitCntType vector_write_wait,
                            bool vector_write_also_waits_lds, WaitCntType vector_read_wait);
  void check_vector_raw_hazards(EngineWaveState &wave, const EngineInstructionContext &ctx,
                                const ResourceAccessEvent &event, ResourceKind resource_kind,
                                uint32_t reg, uint32_t count);
  void check_vector_war_waw_hazards(EngineWaveState &wave, const EngineInstructionContext &ctx,
                                    const ResourceAccessEvent &event, ResourceKind resource_kind,
                                    uint32_t reg, uint32_t count);
  void track_vector_pending(EngineWaveState &wave, const EngineInstructionContext &ctx,
                            const ResourceAccessEvent &event, ResourceKind resource_kind,
                            uint32_t reg, uint32_t count, WaitCntType vector_write_wait,
                            bool vector_write_also_waits_lds, WaitCntType vector_read_wait);
  void handle_scalar_access(EngineWaveState &wave, const EngineInstructionContext &ctx,
                            const ResourceAccessEvent &event, WaitCntType scalar_write_wait);
  void handle_lds_access(EngineWaveState &wave, const EngineInstructionContext &ctx,
                         const ResourceAccessEvent &event, WaitCntType local_write_wait);
  void check_global_access_for_races(const EngineInstructionContext &ctx,
                                     const ResourceAccessEvent &event);
  void flush_workgroup_epoch(const EngineWorkgroupKey &key);
  void check_lds_epoch_for_races(const EngineWorkgroupKey &key,
                                 const std::vector<EngineLdsAccessRecord> &epoch);
  void emit_warning(const EngineWarning &warning);
  const SimulatorInstructionFormatter *instruction_formatter() const;

  /// Record and report a dropped event. Never throws.
  void reject_event(const std::string &message) const;
  bool check_access_size(const char *handler, uint32_t size_bytes) const;

  const SimulatorInstructionFormatter *instruction_formatter_ = nullptr;
  const HazardWarningSink *warning_sink_ = nullptr;
  EngineDiagnosticHandler diagnostic_handler_;
  mutable std::atomic<uint64_t> rejected_events_{0};
  EngineState state_;
};

DataHazardEngine &data_hazard_engine();

} // namespace hazard_core
