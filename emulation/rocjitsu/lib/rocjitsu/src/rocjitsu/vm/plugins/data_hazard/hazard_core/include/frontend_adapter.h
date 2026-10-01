// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Shared helpers for simulator frontend adapters.
//
// Frontends remain responsible for simulator-specific translation. These helpers
// centralize the common "translate native callback payload, then forward to the
// generic engine API" pattern used by simulator adapters.

#pragma once

#include "hazard_events.h"
#include "simulator_api.h"

namespace data_hazard_frontend {

inline void on_dispatch_begin(hazard_core::DataHazardSimulatorApi &api,
                              hazard_core::EntityId dispatch_id) {
  api.on_dispatch_begin(dispatch_id);
}

inline void on_dispatch_end(hazard_core::DataHazardSimulatorApi &api,
                            hazard_core::EntityId dispatch_id) {
  api.on_dispatch_end(dispatch_id);
}

inline void on_workgroup_begin(hazard_core::DataHazardSimulatorApi &api,
                               const hazard_core::ExecutionKey &workgroup) {
  api.on_workgroup_begin(workgroup.dispatch_id, workgroup.cluster_id, workgroup.workgroup_id);
}

inline void on_workgroup_end(hazard_core::DataHazardSimulatorApi &api,
                             const hazard_core::ExecutionKey &workgroup) {
  api.on_workgroup_end(workgroup.dispatch_id, workgroup.cluster_id, workgroup.workgroup_id);
}

inline void on_wave_begin(hazard_core::DataHazardSimulatorApi &api,
                          const hazard_core::ExecutionKey &wave) {
  api.on_wave_begin(wave);
}

inline void on_wave_end(hazard_core::DataHazardSimulatorApi &api,
                        const hazard_core::ExecutionKey &wave) {
  api.on_wave_end(wave);
}

template <typename NativeInstruction, typename MakeEvent>
hazard_core::InstructionEvent on_instruction(hazard_core::DataHazardSimulatorApi &api,
                                             const NativeInstruction &native_instruction,
                                             MakeEvent make_event) {
  hazard_core::InstructionEvent event = make_event(native_instruction);
  api.on_instruction(event);
  return event;
}

template <typename NativeAccess, typename MakeEvent>
void on_resource_access(hazard_core::DataHazardSimulatorApi &api, const NativeAccess &native_access,
                        MakeEvent make_event) {
  api.on_resource_access(make_event(native_access));
}

template <typename NativeBarrier, typename MakeEvent>
void on_barrier(hazard_core::DataHazardSimulatorApi &api, const NativeBarrier &native_barrier,
                MakeEvent make_event) {
  api.on_barrier(make_event(native_barrier));
}

inline void on_shutdown(hazard_core::DataHazardSimulatorApi &api) { api.on_shutdown(); }

} // namespace data_hazard_frontend
