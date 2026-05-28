// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "plugins/cu_cycle_model.h"

#include "plugins/mem_messages.h"

namespace rocjitsu {
namespace amdgpu {

CuCycleModel::CuCycleModel(std::string name, const simdojo::ClockDomain &domain)
    : simdojo::Clocked<simdojo::Component>(std::move(name), domain) {
  req_out_ = add_port(std::make_unique<simdojo::Port>(
      this->name() + ".mem_req", 0, this, simdojo::PortDirection::OUT,
      simdojo::PortProtocol::MEMORY));
  cpl_in_ = add_port(std::make_unique<simdojo::Port>(
      this->name() + ".mem_cpl", 1, this, simdojo::PortDirection::IN,
      simdojo::PortProtocol::MEMORY));
  // Async completion handler: route the shared-hierarchy completion back into the
  // model and re-arm the (possibly idled) clock at the delivery tick (hardening #2).
  cpl_in_->set_handler([this](simdojo::Tick now, simdojo::Message *msg) {
    if (!model_) return;
    auto *c = static_cast<MemCompletionMsg *>(msg);
    model_->on_mem_completion(c->rid, c->completion_cyc);
    resume_clock(now);
  });
}

void CuCycleModel::configure(const cycle_model::UarchConfig &cfg) {
  if (model_) return; // first config wins
  model_ = std::make_unique<cycle_model::ArchModel>(cfg);
}

bool CuCycleModel::advance(simdojo::Tick /*now*/) {
  if (!model_) return false;        // not configured yet => idle
  bool more = model_->tick_cycle(); // one CU cycle per edge
  // Send this edge's L1-miss/bypass traffic to the shared MemSysCycleModel.
  // Completions arrive asynchronously on cpl_in_ via on_mem_completion().
  auto &reqs = model_->mem_requests();
  for (auto &req : reqs)
    req_out_->send(std::make_unique<MemReqMsg>(req.rid, std::move(req.shared), id()));
  reqs.clear();
  // has_work() (-> more) stays true while in_flight is non-empty (the request is
  // outstanding awaiting its async completion), so the clock keeps ticking until the
  // op retires. Do NOT OR in "had requests this edge" — that is subsumed by in_flight.
  return more;
}

} // namespace amdgpu
} // namespace rocjitsu
