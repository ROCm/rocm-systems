// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/soc.h"

#include "rocjitsu/vm/amdgpu/aql_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/pm4_queue_binding_factory.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

amdgpu::SdmaPacketDialect sdma_dialect(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    return amdgpu::SdmaPacketDialect::Gfx1250;
  if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
      arch == ROCJITSU_CODE_ARCH_RDNA4) {
    return amdgpu::SdmaPacketDialect::Gfx11Plus;
  }
  return amdgpu::SdmaPacketDialect::Legacy;
}

} // namespace

SoC::SoC(std::string name, amdgpu::GpuMemory *memory, rj_code_arch_t arch)
    : simdojo::CompositeComponent(std::move(name)), arch_(arch),
      sdma_queue_scheduler_(gpu_vm_, cache_coherence_, sdma_dialect(arch_)),
      queue_registry_(gpu_vm_), mes_engine_(gpu_vm_, queue_registry_, *this), memory_(memory) {
  set_weight(0);
  if (!install_internal_address_space(memory_))
    throw std::logic_error("failed to install the internal GPU address space");
}

SoC::~SoC() { release_vm_resources(); }

void SoC::release_vm_resources() {
  if (vm_resources_released_)
    return;
  vm_resources_released_ = true;

  // Composite children (including CPs) may be adopted by VirtualMachine.
  // Release every child-held VM lease before either owner's child list starts
  // destroying components, while gpu_vm_ is still alive.
  (void)mes_engine_.teardown_queues();
  queue_registry_.close_all();
  for_each_cp([](amdgpu::CommandProcessor *cp) { cp->shutdown(); });
  sdma_queue_scheduler_.shutdown();
}

void SoC::add_xcd(amdgpu::Xcd *xcd) {
  if (xcd == nullptr)
    throw std::invalid_argument("cannot add a null XCD");
  xcd->set_coherence_domain(cache_coherence_);
  if (xcd->l2_cache()) {
    xcd->l2_cache()->set_legacy_maintenance_memory(memory_);
    xcd->l2_cache()->set_legacy_maintenance_vm(&gpu_vm_);
    xcd->l2_cache()->set_gpu_vm(&gpu_vm_);
  }
  if (amdgpu::CommandProcessor *cp = xcd->command_processor())
    cp->set_gpu_vm(&gpu_vm_, internal_address_space_);
  xcds_.push_back(xcd);
}

void SoC::add_iod(amdgpu::Iod *iod) {
  if (iod == nullptr)
    throw std::invalid_argument("cannot add a null IOD");
  iod->set_coherence_domain(cache_coherence_);
  iod->set_gpu_vm(&gpu_vm_);
  iod->set_legacy_maintenance_memory(memory_);
  iod->msc()->set_legacy_maintenance_vm(&gpu_vm_);
  iods_.push_back(iod);
}

bool SoC::install_internal_address_space(amdgpu::GpuMemory *memory) {
  const bool internal_registered =
      internal_address_space_ && gpu_vm_.lookup(internal_address_space_).has_value();
  if (memory == memory_ && (memory == nullptr || internal_registered))
    return true;
  if (memory_ != nullptr && memory != memory_)
    return false;
  if (!internal_registered) {
    internal_address_space_ = {};
    internal_memory_access_.reset();
  }
  const std::size_t expected_address_spaces = internal_registered ? 1 : 0;
  if (gpu_vm_.active_address_spaces() != expected_address_spaces)
    return false;
  if (queue_registry_.active_queues() != 0 || sdma_queue_scheduler_.active_queues() != 0)
    return false;
  for (const amdgpu::Xcd *xcd : xcds_) {
    const amdgpu::CommandProcessor *cp = xcd->command_processor();
    if (cp != nullptr && cp->has_registered_queues()) {
      return false;
    }
  }
  if (internal_registered && !gpu_vm_.unregister_address_space(internal_address_space_))
    return false;

  memory_ = memory;
  internal_memory_access_.reset();
  internal_address_space_ = {};
  if (memory_ != nullptr) {
    internal_memory_access_ = std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(*memory_);
    internal_address_space_ = gpu_vm_.register_unrouted_address_space(
        0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(), internal_memory_access_, {},
        true);
    if (!internal_address_space_) {
      internal_memory_access_.reset();
      memory_ = nullptr;
      return false;
    }
  }
  return true;
}

bool SoC::set_memory(amdgpu::GpuMemory *memory) {
  if (!install_internal_address_space(memory))
    return false;
  for (amdgpu::Xcd *xcd : xcds_) {
    if (xcd->l2_cache()) {
      xcd->l2_cache()->set_legacy_maintenance_memory(memory_);
      xcd->l2_cache()->set_legacy_maintenance_vm(&gpu_vm_);
      xcd->l2_cache()->set_gpu_vm(&gpu_vm_);
    }
    if (amdgpu::CommandProcessor *cp = xcd->command_processor())
      cp->set_gpu_vm(&gpu_vm_, internal_address_space_);
  }
  for (amdgpu::Iod *iod : iods_) {
    iod->set_legacy_maintenance_memory(memory_);
    iod->msc()->set_legacy_maintenance_vm(&gpu_vm_);
  }
  // Create a standalone HBM controller for the config-loader path where the
  // parameterized constructor (which creates it) was not used.
  if (memory && !hbm_standalone_ && iods_.empty())
    hbm_standalone_ = std::make_unique<amdgpu::HbmController>(memory, &gpu_vm_);
  return true;
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
      exec_mode_(config.exec_mode),
      sdma_queue_scheduler_(gpu_vm_, cache_coherence_, sdma_dialect(arch_)),
      queue_registry_(gpu_vm_), mes_engine_(gpu_vm_, queue_registry_, *this) {
  set_weight(0); // Structural container, not a work-producing component.
  auto soc_name = this->name();

  // GPU memory is shared across all XCDs.
  auto mem = std::make_unique<amdgpu::GpuMemory>("vram");
  memory_ = mem.get();
  add_child(std::move(mem));
  if (!install_internal_address_space(memory_))
    throw std::logic_error("failed to install the internal GPU address space");

  if (config.num_iods > 0) {
    // Create IODs, each with its own memory-side cache and HBM controller.
    for (uint32_t j = 0; j < config.num_iods; ++j) {
      amdgpu::Iod::Config iod_config{};
      iod_config.num_hbm_stacks = 4; // Structural only in functional mode.
      auto iod_ptr = std::make_unique<amdgpu::Iod>(soc_name + ".iod" + std::to_string(j),
                                                   iod_config, memory_, cache_coherence_);
      iod_ptr->set_gpu_vm(&gpu_vm_);
      iod_ptr->msc()->set_legacy_maintenance_vm(&gpu_vm_);
      iods_.push_back(iod_ptr.get());
      add_child(std::move(iod_ptr));
    }

    // Create XCDs, each assigned to its parent IOD.
    for (uint32_t i = 0; i < config.num_xcds; ++i) {
      auto xcd_ptr =
          std::make_unique<amdgpu::Xcd>(soc_name + ".xcd" + std::to_string(i), config.xcd,
                                        config.arch, memory_, config.exec_mode, cache_coherence_);
      xcd_ptr->l2_cache()->set_legacy_maintenance_vm(&gpu_vm_);
      xcd_ptr->l2_cache()->set_gpu_vm(&gpu_vm_);
      xcds_.push_back(xcd_ptr.get());
      add_child(std::move(xcd_ptr));
    }
  } else {
    // No IOD modeling: XCDs connect directly to a standalone HBM controller.
    hbm_standalone_ = std::make_unique<amdgpu::HbmController>(memory_, &gpu_vm_);
    for (uint32_t i = 0; i < config.num_xcds; ++i) {
      auto xcd_ptr =
          std::make_unique<amdgpu::Xcd>(soc_name + ".xcd" + std::to_string(i), config.xcd,
                                        config.arch, memory_, config.exec_mode, cache_coherence_);
      xcd_ptr->l2_cache()->set_legacy_maintenance_vm(&gpu_vm_);
      xcd_ptr->l2_cache()->set_gpu_vm(&gpu_vm_);
      xcds_.push_back(xcd_ptr.get());
      add_child(std::move(xcd_ptr));
    }
  }
}

void SoC::set_arch(rj_code_arch_t arch) {
  if (!sdma_queue_scheduler_.set_packet_dialect(sdma_dialect(arch)))
    throw std::logic_error("SoC architecture cannot change while copy queues are active");
  arch_ = arch;
}

std::optional<amdgpu::ComputeQueueBindingPlan> SoC::make_aql(uint32_t queue_ordinal) {
  amdgpu::CommandProcessor *owner = assign_queue_owner_cp(queue_ordinal);
  if (owner == nullptr)
    return std::nullopt;
  return amdgpu::ComputeQueueBindingPlan{.factory = amdgpu::make_aql_queue_binding_factory(*owner),
                                         .xcd_fanout = true};
}

std::optional<amdgpu::ComputeQueueBindingPlan> SoC::make_pm4(uint32_t queue_ordinal,
                                                             amdgpu::Pm4PacketCallbacks callbacks) {
  amdgpu::CommandProcessor *owner = assign_queue_owner_cp(queue_ordinal);
  if (owner == nullptr)
    return std::nullopt;
  return amdgpu::ComputeQueueBindingPlan{
      .factory = amdgpu::make_pm4_queue_binding_factory(*owner, std::move(callbacks)),
      .xcd_fanout = false};
}

void SoC::set_plugin_group(std::shared_ptr<ExecutionPluginGroup> plugin_group) {
  plugin_group_ = plugin_group ? plugin_group : ExecutionPluginGroup::empty_group();
  for (auto *xcd : xcds_)
    xcd->set_plugin_group(plugin_group_);
}

void SoC::set_dispatch_threads(uint32_t threads) {
  requested_dispatch_threads_ = std::max(threads, 1u);
  apply_dispatch_threads();
}

void SoC::apply_dispatch_threads() {
  size_t max_cp_cus = 1;
  for_each_cp(
      [&max_cp_cus](auto *cp) { max_cp_cus = std::max(max_cp_cus, cp->compute_units().size()); });
  uint32_t effective_threads =
      static_cast<uint32_t>(std::min<size_t>(requested_dispatch_threads_, max_cp_cus));
  if (exec_mode_ != simdojo::ExecMode::FUNCTIONAL)
    effective_threads = 1;

  std::unique_ptr<amdgpu::CpuDispatchPool> new_pool;
  if (effective_threads > 1)
    new_pool = std::make_unique<amdgpu::CpuDispatchPool>(effective_threads);

  auto *pool = new_pool.get();
  for_each_cp([pool, effective_threads](auto *cp) {
    cp->set_shared_dispatch_pool(pool);
    cp->set_dispatch_threads(effective_threads);
  });
  dispatch_pool_ = std::move(new_pool);
  dispatch_threads_ = effective_threads;
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
  if (!cache_coherence_->has_legacy_maintenance_backing())
    throw std::logic_error("SoC cache hierarchy lacks legacy maintenance backing");

  // Let every XCD's command processor see its siblings, so a queue marked for
  // fan-out can split its dispatches across the whole device. Each CP's rank is
  // its own XCD index, which fixes the workgroup-to-XCD mapping independently of
  // which XCD a given queue happened to be assigned to.
  {
    std::vector<amdgpu::CommandProcessor *> cps;
    cps.reserve(xcds_.size());
    for (auto *xcd_ptr : xcds_)
      if (auto *cp = xcd_ptr->command_processor()) {
        cp->set_gpu_vm(&gpu_vm_, internal_address_space_);
        cps.push_back(cp);
      } else {
        cps.push_back(nullptr);
      }
    if (std::find(cps.begin(), cps.end(), nullptr) == cps.end()) {
      for (uint32_t i = 0; i < cps.size(); ++i)
        cps[i]->set_xcd_topology(i, cps);
    }
  }

  if (iods_.empty()) {
    // No IOD modeling: wire each L2's req port directly to the standalone HBM controller.
    // Create the HBM controller lazily if set_memory() was called but the
    // parameterized constructor (which creates it) was not used.
    if (!hbm_standalone_ && memory_)
      hbm_standalone_ = std::make_unique<amdgpu::HbmController>(memory_, &gpu_vm_);
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

const std::vector<amdgpu::ComputeUnitCore *> &SoC::all_cus() {
  if (all_cus_cache_.empty()) {
    for (auto *x : xcds_)
      for (uint32_t s = 0; s < x->num_shader_engines(); ++s)
        for (auto *cu : x->shader_engine(s)->compute_units())
          all_cus_cache_.push_back(cu);
  }
  return all_cus_cache_;
}

} // namespace rocjitsu
