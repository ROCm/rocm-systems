// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <memory>
#include <string>

namespace rocjitsu {

void SoC::set_memory(amdgpu::GpuMemory *m) {
  memory_ = m;
  // Create a standalone HBM controller for the config-loader path where the
  // parameterized constructor (which creates it) was not used.
  if (m && !hbm_standalone_ && iods_.empty())
    hbm_standalone_ = std::make_unique<amdgpu::HbmController>(m);
}

void SoC::wire_backing(simdojo::Topology &topo) {
  if (!hbm_standalone_) {
    return;
  }
  for (auto *x : xcds_) {
    if (!x->l2_cache())
      continue;
    auto *l2_req = x->l2_cache()->req_port();
    if (l2_req->link() == nullptr) {
      auto *hbm_cpl = hbm_standalone_->create_cpl_port(x->name());
      auto *link = topo.add_link(l2_req, hbm_cpl, /*latency=*/1);
      link->set_exec_mode(exec_mode_);
    }
  }
}

SoC::SoC(std::string name, const Config &config)
    : simdojo::CompositeComponent(std::move(name)), arch_(config.arch),
      exec_mode_(config.exec_mode) {
  set_weight(0); // Structural container, not a work-producing component.
  auto soc_name = this->name();

  // GPU memory is shared across all XCDs.
  auto mem = std::make_unique<amdgpu::GpuMemory>("vram");
  memory_ = mem.get();
  add_child(std::move(mem));

  if (config.num_iods > 0) {
    // Create IODs, each with its own memory-side cache and HBM controller.
    for (uint32_t j = 0; j < config.num_iods; ++j) {
      amdgpu::Iod::Config iod_config{};
      iod_config.num_hbm_stacks = 4; // Structural only in functional mode.
      auto iod_ptr =
          std::make_unique<amdgpu::Iod>(soc_name + ".iod" + std::to_string(j), iod_config, memory_);
      iods_.push_back(iod_ptr.get());
      add_child(std::move(iod_ptr));
    }

    // Create XCDs, each assigned to its parent IOD.
    for (uint32_t i = 0; i < config.num_xcds; ++i) {
      auto xcd_ptr =
          std::make_unique<amdgpu::Xcd>(soc_name + ".xcd" + std::to_string(i), config.xcd,
                                        config.arch, memory_, config.exec_mode);
      xcds_.push_back(xcd_ptr.get());
      add_child(std::move(xcd_ptr));
    }
  } else {
    // No IOD modeling: XCDs connect directly to a standalone HBM controller.
    hbm_standalone_ = std::make_unique<amdgpu::HbmController>(memory_);
    for (uint32_t i = 0; i < config.num_xcds; ++i) {
      auto xcd_ptr =
          std::make_unique<amdgpu::Xcd>(soc_name + ".xcd" + std::to_string(i), config.xcd,
                                        config.arch, memory_, config.exec_mode);
      xcds_.push_back(xcd_ptr.get());
      add_child(std::move(xcd_ptr));
    }
  }
}

void SoC::set_plugin_group(std::shared_ptr<ExecutionPluginGroup> plugin_group) {
  plugin_group_ = plugin_group ? plugin_group
                               : ExecutionPluginGroup::empty_group();
  for (auto *xcd : xcds_)
    xcd->set_plugin_group(plugin_group_);
}

void SoC::flush_all() {
  // Flush all per-CU L1 caches (invalidate, since L1 is write-through).
  for (auto *x : xcds_) {
    for (uint32_t si = 0; si < x->num_shader_engines(); ++si) {
      auto *se = x->shader_engine(si);
      for (uint32_t ci = 0; ci < se->num_compute_units(); ++ci)
        se->compute_unit(ci)->flush_l1();
    }
  }

  // Flush each XCD's L2 once (L2 is shared across all CUs in an XCD).
  for (auto *x : xcds_)
    x->l2_cache()->flush_all();

  // Flush all IOD memory-side caches (MSC → HBM).
  for (auto *i : iods_)
    i->msc()->flush_all();
}

void SoC::initialize() {
  if (iods_.empty()) {
    // No IOD modeling: wire each L2's req port directly to the standalone HBM controller.
    // Create the HBM controller lazily if set_memory() was called but the
    // parameterized constructor (which creates it) was not used.
    if (!hbm_standalone_ && memory_)
      hbm_standalone_ = std::make_unique<amdgpu::HbmController>(memory_);
    if (hbm_standalone_) {
      auto &topo = engine()->topology();
      for (auto *x : xcds_) {
        auto *l2_req = x->l2_cache()->req_port();
        if (l2_req->link() == nullptr) {
          auto *hbm_cpl = hbm_standalone_->create_cpl_port(x->name());
          auto *link = topo.add_link(l2_req, hbm_cpl, /*latency=*/1);
          link->set_exec_mode(exec_mode_);
        }
      }
    }
    return;
  }

  auto &topo = engine()->topology();
  uint32_t xcds_per_iod = static_cast<uint32_t>(xcds_.size()) / static_cast<uint32_t>(iods_.size());

  // Wire each XCD's L2 req port → its parent IOD's cpl port.
  // Skip if the port already has a link (set by the declarative topology config).
  for (uint32_t i = 0; i < xcds_.size(); ++i) {
    auto *xcd_out = xcds_[i]->l2_cache()->req_port();
    if (xcd_out->link() != nullptr)
      continue;
    uint32_t iod_idx = i / xcds_per_iod;
    if (iod_idx >= iods_.size())
      iod_idx = static_cast<uint32_t>(iods_.size()) - 1;
    auto *iod_in = iods_[iod_idx]->create_cpl_port(xcds_[i]->name());
    auto *link = topo.add_link(xcd_out, iod_in, /*latency=*/1,
                               /*weight=*/3);
    link->set_exec_mode(exec_mode_);
  }

  // Wire IOD peer interconnect links (bidirectional between all IOD pairs).
  // Skip if already wired by the declarative topology config.
  for (uint32_t a = 0; a < iods_.size(); ++a) {
    if (iods_[a]->peer_req_port()->link() != nullptr)
      continue;
    for (uint32_t b = a + 1; b < iods_.size(); ++b) {
      auto *link_ab =
          topo.add_link(iods_[a]->peer_req_port(), iods_[b]->peer_cpl_port(), /*latency=*/1);
      link_ab->set_exec_mode(exec_mode_);
      auto *link_ba =
          topo.add_link(iods_[b]->peer_req_port(), iods_[a]->peer_cpl_port(), /*latency=*/1);
      link_ba->set_exec_mode(exec_mode_);
    }
  }
}

} // namespace rocjitsu
