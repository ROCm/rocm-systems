// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/iod.h"

#include "simdojo/sim/exec_mode.h"

#include <memory>
#include <string>

namespace rocjitsu {
namespace amdgpu {

Iod::Iod(std::string name, const Config &config, GpuMemory *memory)
    : simdojo::CompositeComponent(std::move(name)) {
  auto hbm = std::make_unique<HbmController>("hbm", memory);
  hbm_ = hbm.get();
  add_child(std::move(hbm));

  auto msc = std::make_unique<MemorySideCache>("msc");
  msc_ = msc.get();
  add_child(std::move(msc));

  // Wire MSC.req -> HBM.cpl with a functional link.
  msc_hbm_link_ = std::make_unique<simdojo::Link>(0, msc_->req_port(), hbm_->cpl_port(), 0);
  msc_hbm_link_->set_exec_mode(simdojo::ExecMode::FUNCTIONAL);
  msc_->req_port()->set_link(msc_hbm_link_.get());
  hbm_->cpl_port()->set_link(msc_hbm_link_.get());

  // Peer interconnect ports (structural, for cross-IOD topology).
  peer_req_ = add_port(std::make_unique<simdojo::Port>(
      "peer_req", 0, this, simdojo::PortDirection::OUT, simdojo::PortProtocol::MEMORY));
  peer_cpl_ = add_port(std::make_unique<simdojo::Port>(
      "peer_cpl", 1, this, simdojo::PortDirection::IN, simdojo::PortProtocol::MEMORY));

  // HBM requester ports — one per HBM stack (structural).
  for (uint32_t i = 0; i < config.num_hbm_stacks; ++i) {
    auto port_id = static_cast<simdojo::PortID>(2 + i);
    auto port =
        std::make_unique<simdojo::Port>("req_" + std::to_string(i), port_id, this,
                                        simdojo::PortDirection::OUT, simdojo::PortProtocol::MEMORY);
    req_ports_.push_back(add_port(std::move(port)));
  }
}

simdojo::Port *Iod::create_cpl_port(const std::string &xcd_name) {
  auto port_id = static_cast<simdojo::PortID>(2 + req_ports_.size() + cpl_ports_.size());
  auto port = std::make_unique<simdojo::Port>(
      "cpl_" + xcd_name, port_id, this, simdojo::PortDirection::IN, simdojo::PortProtocol::MEMORY);
  auto *raw = add_port(std::move(port));
  cpl_ports_.push_back(raw);
  return raw;
}

void Iod::initialize() {}

} // namespace amdgpu
} // namespace rocjitsu
