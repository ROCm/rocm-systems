// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cu_cycle_model.h
/// @brief Per-CU clocked shell around cycle_model::ArchModel. One per compute
/// unit, registered in the topology under its CU (pre-build) by the RJ_CYCLE
/// driver path. Its advance() ticks the ArchModel one CU cycle per clock edge;
/// the ArchModel is built lazily on configure() once the arch is known.
///
/// Each edge, the model's pending L1-miss/bypass requests are sent as MemReqMsg
/// over the OUT port to the shared MemSysCycleModel; completions arrive
/// asynchronously on the IN port and are routed via on_mem_completion().

#pragma once

#include "simdojo/sim/clocked.h"
#include "simdojo/sim/component.h"

#include "cycle_model/ArchModel.h"
#include "cycle_model/UarchConfig.h"

#include <memory>
#include <string>

namespace rocjitsu {
namespace amdgpu {

class CuCycleModel : public simdojo::Clocked<simdojo::Component> {
public:
  CuCycleModel(std::string name, const simdojo::ClockDomain &domain); // out-of-line: adds ports

  // Build the per-CU ArchModel from a parsed config (idempotent: first call wins).
  void configure(const cycle_model::UarchConfig &cfg);

  bool configured() const { return model_ != nullptr; }
  cycle_model::ArchModel &model() { return *model_; }

  // Per-dispatch reset of the per-CU memory state: the ArchModel's L1/MSHR submodel
  // plus its rid_owner_ map. The shared L2/HBM reset is on MemSysCycleModel.
  void reset_memory() {
    if (model_) model_->reset_memory();
  }

  simdojo::Port *req_out() { return req_out_; } // wired by the driver to the MemSys req_in
  simdojo::Port *cpl_in() { return cpl_in_; }   // wired by the driver from the MemSys cpl_out

  // One CU clock edge: tick the model one cycle. Returns true to keep clocking,
  // false to idle (resume_clock() re-arms it when new work arrives).
  bool advance(simdojo::Tick now) override;

private:
  std::unique_ptr<cycle_model::ArchModel> model_; // built on configure()
  simdojo::Port *req_out_ = nullptr;              // OUT: MemReqMsg to the shared model
  simdojo::Port *cpl_in_ = nullptr;               // IN: MemCompletionMsg from the shared model
};

} // namespace amdgpu
} // namespace rocjitsu
