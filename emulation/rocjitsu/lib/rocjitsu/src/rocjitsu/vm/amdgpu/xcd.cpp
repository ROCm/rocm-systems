// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/xcd.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <cassert>
#include <memory>
#include <string>

namespace rocjitsu {
namespace amdgpu {

Xcd::Xcd(std::string name, const Config &config, rj_code_arch_t arch, GpuMemory *memory,
         simdojo::ExecMode exec_mode)
    : simdojo::CompositeComponent(std::move(name)), exec_mode_(exec_mode) {
  set_weight(0); // Structural container, not a work-producing component.
  auto xcd_name = this->name();

  // Create shared L2 cache for this XCD. The backing store is connected
  // via the L2's requester port (wired by SoC::initialize()).
  auto l2 = std::make_unique<L2Cache>(xcd_name + ".l2");
  l2_cache_ = l2.get();
  add_child(std::move(l2));

  // Create command processor for this XCD.
  auto cp = std::make_unique<CommandProcessor>(xcd_name + ".cp");
  cp_ = cp.get();

  // Create shader engines, each with its own CU array sharing the XCD's L2.
  for (uint32_t i = 0; i < config.num_shader_engines; ++i) {
    auto se =
        std::make_unique<ShaderEngine>(xcd_name + ".se" + std::to_string(i), config.shader_engine,
                                       arch, memory, l2_cache_, exec_mode);
    // Register all CUs in this SE with the XCD's command processor.
    // This also sets up on_idle callbacks from CUs to CP::check_all_idle().
    for (uint32_t c = 0; c < se->num_compute_units(); ++c)
      cp_->add_compute_unit(se->compute_unit(c));
    // Register the SE's SPI with the CP so ace_dispatch_all() can use the
    // interleaved SPI path for cross-workgroup spin-wait resolution.
    cp_->add_spi(&se->spi());
    shader_engines_.push_back(se.get());
    add_child(std::move(se));
  }

  cp_->add_l2_cache(l2_cache_);
  add_child(std::move(cp));
}

void Xcd::initialize() {
  auto &topo = engine()->topology();

  // Wire CP req ports → CU cpl ports.
  auto &dispatch_ports = cp_->dispatch_ports();
  auto &cus = cp_->compute_units();
  assert(dispatch_ports.size() == cus.size() && "dispatch ports and CU count must match");
  for (size_t i = 0; i < cus.size(); ++i) {
    auto *link = topo.add_link(dispatch_ports[i], cus[i]->cpl_port(), /*latency=*/1,
                               /*weight=*/2); // CP→CU weight = 2
    link->set_exec_mode(exec_mode_);
  }

  // Wire each CU's req port → a dedicated L2 cpl port (one per CU).
  for (size_t i = 0; i < cus.size(); ++i) {
    auto *l2_cpl = l2_cache_->create_cpl_port(cus[i]->name());
    auto *link = topo.add_link(cus[i]->req_port(), l2_cpl, /*latency=*/1,
                               /*weight=*/10); // CU↔L2 weight = 10
    link->set_exec_mode(exec_mode_);
  }

  // Add synthetic intra-SE edges between adjacent CUs within each SE.
  // These lightweight links help the partitioner detect SE grouping.
  for (auto *se : shader_engines_) {
    uint32_t n = se->num_compute_units();
    for (uint32_t c = 0; c + 1 < n; ++c) {
      auto *cu_a = se->compute_unit(c);
      auto *cu_b = se->compute_unit(c + 1);

      // Create UNTYPED ports for synthetic adjacency (not used for data transfer).
      auto port_a_out =
          std::make_unique<simdojo::Port>(cu_a->name() + ".adj_req_" + std::to_string(c),
                                          /*id=*/0, cu_a, simdojo::PortDirection::OUT);
      auto port_b_in =
          std::make_unique<simdojo::Port>(cu_b->name() + ".adj_cpl_" + std::to_string(c),
                                          /*id=*/0, cu_b, simdojo::PortDirection::IN);
      auto port_b_out =
          std::make_unique<simdojo::Port>(cu_b->name() + ".adj_req_r_" + std::to_string(c),
                                          /*id=*/0, cu_b, simdojo::PortDirection::OUT);
      auto port_a_in =
          std::make_unique<simdojo::Port>(cu_a->name() + ".adj_cpl_r_" + std::to_string(c),
                                          /*id=*/0, cu_a, simdojo::PortDirection::IN);

      auto *pa_out = cu_a->add_port(std::move(port_a_out));
      auto *pb_in = cu_b->add_port(std::move(port_b_in));
      auto *pb_out = cu_b->add_port(std::move(port_b_out));
      auto *pa_in = cu_a->add_port(std::move(port_a_in));

      auto *link_ab = topo.add_link(pa_out, pb_in, /*latency=*/1, /*weight=*/2);
      link_ab->set_exec_mode(exec_mode_);
      auto *link_ba = topo.add_link(pb_out, pa_in, /*latency=*/1, /*weight=*/2);
      link_ba->set_exec_mode(exec_mode_);
    }
  }

  // L2→HBM/fabric port wiring is done by SoC::initialize() since the
  // destination depends on the IOD topology.
}

} // namespace amdgpu
} // namespace rocjitsu
