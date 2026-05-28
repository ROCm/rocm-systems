// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "plugins/mem_sys_cycle_model.h"

#include "plugins/mem_messages.h"

#include <cassert>
#include <cstdio>

namespace rocjitsu {
namespace amdgpu {

void MemSysCycleModel::configure(const cycle_model::UarchConfig &cfg) {
  if (shared_)
    return;
  shared_ = std::make_unique<cycle_model::SharedMemModel>(cfg);
}

MemSysCycleModel::CuPorts MemSysCycleModel::add_cu_ports() {
  uint32_t req_id = next_port_id_++;
  auto *req_in = add_port(
      std::make_unique<simdojo::Port>(name() + ".req_in" + std::to_string(req_id), req_id, this,
                                      simdojo::PortDirection::IN, simdojo::PortProtocol::MEMORY));
  uint32_t cpl_id = next_port_id_++;
  auto *cpl_out = add_port(
      std::make_unique<simdojo::Port>(name() + ".cpl_out" + std::to_string(cpl_id), cpl_id, this,
                                      simdojo::PortDirection::OUT, simdojo::PortProtocol::MEMORY));
  // The handler closure captures this CU's paired outbound port → request identity is
  // the receiving (MemSys-local) port; the CU-local rid never collides across CUs.
  req_in->set_handler(
      [this, cpl_out](simdojo::Tick, simdojo::Message *msg) { handle_request(cpl_out, msg); });
  return {req_in, cpl_out};
}

void MemSysCycleModel::handle_request(simdojo::Port *cpl_out, simdojo::Message *msg) {
  auto *req = static_cast<MemReqMsg *>(msg); // only MemReqMsg arrives on a req_in port
  if (!shared_) {
    // Wired but never configured (the plugin's CU->Xcd walk failed). Dropping the request
    // leaves the sender's in_flight entry unresolved -> hang, so make it loud rather than
    // silent. The plugin also warns at resolution time; this is the last-line guard.
    static bool warned = false;
    if (!warned) {
      warned = true;
      std::fprintf(stderr,
                   "[rocjitsu] cycle: MemSysCycleModel serviced a request before configure() "
                   "— dropped (sender will stall). Topology wiring is inconsistent.\n");
    }
    assert(shared_ && "MemSysCycleModel serviced a request before configure()");
    return;
  }
  uint64_t completion_cyc = shared_->service(req->shared); // cross-CU contention serializes here
  cpl_out->send(std::make_unique<MemCompletionMsg>(req->rid, completion_cyc));
}

} // namespace amdgpu
} // namespace rocjitsu
