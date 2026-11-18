// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/virtual_machine.h"

#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/fabric_endpoint.h"

#include "simdojo/sim/topology.h"
#include "util/except.h"

#include <cassert>
#include <memory>
#include <sys/syscall.h>
#include <unistd.h>

namespace rocjitsu {

VirtualMachine::VirtualMachine(const Config &config)
    : simdojo::CompositeComponent("vm"), config_(config) {
  set_weight(0);
  auto soc = std::make_unique<SoC>("gpu_soc", config.soc);
  soc_ = soc.get();
  socs_.push_back(soc_);
  add_child(std::move(soc));
}

VirtualMachine::VirtualMachine(std::unique_ptr<SoC> soc, bool daemon_mode)
    : simdojo::CompositeComponent(soc->name()), soc_(soc.get()) {
  set_weight(0);
  socs_.push_back(soc_);
  soc->move_components_to(*this);
  add_child(std::move(soc));
  driver_ = std::make_unique<SimulatedKfd>(*soc_, daemon_mode);
}

VirtualMachine::VirtualMachine(std::vector<std::unique_ptr<SoC>> socs,
                               std::vector<uint32_t> gpu_ids, bool daemon_mode)
    : simdojo::CompositeComponent("vm") {
  set_weight(0);
  std::vector<SoC *> ptrs;
  ptrs.reserve(socs.size());
  for (size_t i = 0; i < socs.size(); ++i) {
    auto *p = socs[i].get();
    ptrs.push_back(p);
    socs_.push_back(p);
    if (i == 0)
      soc_ = p;
    auto wrapper = std::make_unique<simdojo::CompositeComponent>("gpu" + std::to_string(i));
    wrapper->set_weight(0);
    socs[i]->move_components_to(*wrapper);
    // Keep the now-emptied SoC alive/owned under its own uniquely-named gpuN
    // wrapper rather than directly under the VM: every SoC is named "soc" from
    // the config, so adding them at the VM level would recreate the duplicate-
    // name-in-tree hazard this change fixes (find_child returns the first match).
    wrapper->add_child(std::move(socs[i]));
    add_child(std::move(wrapper));
  }
  driver_ = std::make_unique<SimulatedKfd>(ptrs, std::move(gpu_ids), daemon_mode);
}

VirtualMachine::~VirtualMachine() {
  // Drain frontend-owned processes, queues, and debugger state before the core
  // queue services close their registries and detach command processors.
  driver_.reset();
  for (SoC *soc : socs_)
    soc->release_vm_resources();
  if (statistics_) {
    for (uint32_t index = 0; index < socs_.size(); ++index) {
      for (amdgpu::Xcd *xcd : socs_[index]->xcds()) {
        amdgpu::L2Cache *cache = xcd->l2_cache();
        if (!cache)
          continue;
        auto scope = statistics_->scope(cache->full_path(), index);
        if (!scope)
          continue;
        if (auto counter = scope->counter(
                "cache.backing_read_attempts", "operations",
                "Existing L2 backing-read transaction attempts, including failed attempts"))
          counter->add(cache->backing_read_transactions());
        if (auto counter = scope->counter(
                "cache.backing_write_attempts", "operations",
                "Existing L2 backing-write transaction attempts, including failed attempts"))
          counter->add(cache->backing_write_transactions());
      }
    }
    statistics_->finish();
  }
}

util::Result VirtualMachine::initialize_statistics(const StatisticsConfig &config, std::string node,
                                                   simdojo::ExecMode mode,
                                                   util::DiagnosticEmitter emit_error) {
  auto session = StatisticsSession::create(
      config, std::move(node), mode == simdojo::ExecMode::FUNCTIONAL ? "functional" : "clocked",
      emit_error);
  if (session.failed())
    return util::Result::failure();
  statistics_ = std::move(session).value();
  for (uint32_t index = 0; index < socs_.size(); ++index)
    socs_[index]->set_statistics(statistics_, index);
  return util::Result::success();
}

void VirtualMachine::install_peer_routes(config::SystemPeerRouteProjection routes,
                                         simdojo::Topology &topology, simdojo::ExecMode mode) {
  peer_routes_ = std::move(routes);
  if (peer_routes_.routes().empty() && !driver_)
    return;
  std::vector<config::SystemPeerRoute> route_copy(peer_routes_.routes().begin(),
                                                  peer_routes_.routes().end());
  std::vector<std::shared_ptr<amdgpu::DeviceCacheCoherence>> coherence_domains;
  std::vector<amdgpu::DeviceCacheDomain> prepared_domains;
  coherence_domains.reserve(socs_.size());
  if (driver_ && driver_->gpus_.size() != socs_.size())
    throw util::ConfigError("fabric GPU and coherence domain counts differ");
  prepared_domains.reserve(socs_.size());
  for (size_t i = 0; i < socs_.size(); ++i) {
    auto domain = socs_[i]->cache_coherence();
    coherence_domains.push_back(domain);
    if (driver_) {
      if (driver_->gpus_[i].soc != socs_[i])
        throw util::ConfigError("fabric GPU and coherence domain order differs");
      prepared_domains.push_back({std::move(domain), driver_->gpus_[i].gpu_id});
    }
  }
  auto fabric =
      std::make_unique<FabricEndpoint>(std::move(route_copy), std::move(coherence_domains));
  if (driver_)
    fabric->configure_cache_writeback(std::move(prepared_domains),
                                      driver_->cache_writeback_resolver());
  fabric_endpoint_ = fabric.get();
  add_child(std::move(fabric));
  fabric_endpoint_->wire(topology, mode);
  if (driver_)
    driver_->attach_fabric_endpoint(fabric_endpoint_);
}

} // namespace rocjitsu
