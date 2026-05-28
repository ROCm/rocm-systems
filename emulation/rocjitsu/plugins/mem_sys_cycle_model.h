// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mem_sys_cycle_model.h
/// @brief Shared (one per Xcd/L2) event-driven timing component owning the L2/HBM
/// SharedMemModel. Each connected CU gets a dedicated inbound request port + paired
/// outbound completion port; the per-port handler services MemReqMsg synchronously
/// (computing cross-CU contention via the shared BwQueues) and replies on the paired
/// port. Not Clocked — purely message/event-driven.

#pragma once

#include "simdojo/sim/component.h"

#include "cycle_model/SharedMemModel.h"
#include "cycle_model/UarchConfig.h"

#include <memory>
#include <string>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

class MemSysCycleModel : public simdojo::Component {
public:
  explicit MemSysCycleModel(std::string name) : simdojo::Component(std::move(name)) {}

  // Build the shared L2/HBM model (idempotent: first config wins). Lazy, like
  // CuCycleModel::configure — the plugin calls it on first dispatch.
  void configure(const cycle_model::UarchConfig &cfg);
  bool configured() const { return shared_ != nullptr; }

  // Cross-dispatch reset of L2/HBM. The plugin calls this only when every CU feeding
  // this MemSys is idle (no overlapping in-flight dispatch is using its warmth), so a
  // cross-CU dispatch cannot wipe another's shared state mid-stream.
  void reset() { if (shared_) shared_->reset(); }

  // Driver wiring: create this CU's dedicated inbound (req) + outbound (cpl) port
  // pair and bind the inbound handler to reply on the paired outbound. Returns the
  // {req_in, cpl_out} pointers for the driver to Link to the CU. Call once per CU,
  // pre-build, BEFORE engine.build().
  struct CuPorts { simdojo::Port *req_in; simdojo::Port *cpl_out; };
  CuPorts add_cu_ports();

private:
  void handle_request(simdojo::Port *cpl_out, simdojo::Message *msg);

  std::unique_ptr<cycle_model::SharedMemModel> shared_;
  uint32_t next_port_id_ = 0;     // local PortIDs, two per CU
};

}  // namespace amdgpu
}  // namespace rocjitsu
