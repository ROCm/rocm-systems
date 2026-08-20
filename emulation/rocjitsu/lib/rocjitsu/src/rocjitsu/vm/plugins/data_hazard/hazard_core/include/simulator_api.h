// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Abstract simulator-facing API for the data hazard engine. Concrete adapters
// translate native simulator callbacks into these events and may provide
// simulator-specific instruction formatting.

#pragma once

#include "hazard_events.h"
#include "wait_suggestion.h"

#include <string>

namespace hazard_core {

class SimulatorInstructionFormatter {
public:
  virtual ~SimulatorInstructionFormatter() = default;

  virtual std::string format_instruction(const InstructionDescriptor &instruction) const = 0;
  virtual std::string format_location(const InstructionDescriptor &instruction) const;

  /// Wording for the wait a hazard needs. The default builds it from
  /// hazard_core::make_wait_suggestion; override only for simulator-specific text.
  virtual std::string format_wait_suggestion(WaitCntType kind, HazardAccessKind access,
                                             HazardResourceLabel resource) const;
};

class DataHazardSimulatorApi {
public:
  virtual ~DataHazardSimulatorApi() = default;

  virtual void on_dispatch_begin(EntityId dispatch_id) = 0;
  virtual void on_dispatch_end(EntityId dispatch_id) = 0;
  virtual void on_workgroup_begin(EntityId dispatch_id, EntityId cluster_id,
                                  EntityId workgroup_id) = 0;
  virtual void on_workgroup_end(EntityId dispatch_id, EntityId cluster_id,
                                EntityId workgroup_id) = 0;
  /// Wavegroup lifecycle. Both default to no-ops, and epochs are created
  /// lazily, so a simulator without a wavegroup hierarchy need not call them.
  /// on_wavegroup_end flushes the wavegroup's LDS epoch, which is what reports
  /// a race when the kernel never issues the s_sema_wait that would close it.
  virtual void on_wavegroup_begin(EntityId /*dispatch_id*/, EntityId /*cluster_id*/,
                                  EntityId /*workgroup_id*/, EntityId /*wavegroup_id*/) {}
  virtual void on_wavegroup_end(EntityId /*dispatch_id*/, EntityId /*cluster_id*/,
                                EntityId /*workgroup_id*/, EntityId /*wavegroup_id*/) {}
  virtual void on_wave_begin(const ExecutionKey &wave) = 0;
  virtual void on_wave_end(const ExecutionKey &wave) = 0;
  virtual void on_instruction(const InstructionEvent &instruction) = 0;
  virtual void on_resource_access(const ResourceAccessEvent &access) = 0;
  virtual void on_barrier(const BarrierEvent &barrier) = 0;
  /// s_sema_signal. Kept separate from on_barrier because a semaphore is a
  /// wavegroup-scope directed signal rather than a workgroup barrier. Defaults
  /// to a no-op so adapters without semaphore support need not override it.
  virtual void on_semaphore(const SemaphoreEvent & /*semaphore*/) {}
  virtual void on_shutdown() = 0;
};

} // namespace hazard_core
