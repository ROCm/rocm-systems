// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/soc.h"

#include "rocjitsu/vm/amdgpu/compute_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/cwsr.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace rocjitsu {
namespace {

amdgpu::SdmaPacketDialect sdma_dialect(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    return amdgpu::SdmaPacketDialect::Gfx1250;
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    return amdgpu::SdmaPacketDialect::Rdna4;
  if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5)
    return amdgpu::SdmaPacketDialect::Gfx11Plus;
  if (arch == ROCJITSU_CODE_ARCH_CDNA2 || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
      arch == ROCJITSU_CODE_ARCH_CDNA4 || arch == ROCJITSU_CODE_ARCH_RDNA2) {
    return amdgpu::SdmaPacketDialect::LegacyExtendedCount;
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
  if (amdgpu::CommandProcessor *cp = xcd->command_processor()) {
    cp->set_gpu_vm(&gpu_vm_, internal_address_space_);
    cp->set_scratch_reservations(scratch_reservations_);
  }
  // GWS is device-global: share one store across every XCD so a dispatch that
  // fans out across XCDs still rendezvouses (see GwsDevice). Supersedes the XCD's
  // standalone default store. The XCD's CUs are already attached by this point.
  xcd->adopt_gws_store(gws_device_);
  xcds_.push_back(xcd);
}

void SoC::wire_gws_to_xcds() {
  for (amdgpu::Xcd *xcd : xcds_)
    xcd->adopt_gws_store(gws_device_);
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
  // GWS is device-global: share one store across every XCD (see GwsDevice). The
  // XCDs above were pushed directly rather than through add_xcd, so wire here.
  wire_gws_to_xcds();
  for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->set_scratch_reservations(scratch_reservations_);
    if (config.scratch_slots_per_cu != 0)
      cp->set_scratch_slots_per_cu(config.scratch_slots_per_cu);
  });
}

void SoC::set_arch(rj_code_arch_t arch) {
  if (!sdma_queue_scheduler_.set_packet_dialect(sdma_dialect(arch)))
    throw std::logic_error("SoC architecture cannot change while copy queues are active");
  arch_ = arch;
}

amdgpu::CommandProcessor *SoC::acquire_mes_queue_owner_cp() {
  if (mes_queue_owner_ != nullptr)
    return mes_queue_owner_;
  amdgpu::CommandProcessor *owner = assign_queue_owner_cp(0);
  if (owner == nullptr)
    return nullptr;
  for (auto *cu : owner->compute_units()) {
    cu->set_trap_handler_resolver(
        [this](const amdgpu::Wavefront &wave) -> std::optional<amdgpu::TrapHandlerConfig> {
          return mes_engine_.resolve_trap_config({.address_space = wave.address_space(),
                                                  .process_id = wave.process_id(),
                                                  .queue_id = wave.queue_id()});
        });
    cu->set_sendmsg_handler([this](amdgpu::Wavefront &wave, uint32_t message) {
      constexpr uint32_t kMessageIdMask = 0xf;
      constexpr uint32_t kMessageInterrupt = 1;
      if (!wave.in_trap_handler() || (message & kMessageIdMask) != kMessageInterrupt)
        return false;
      return mes_engine_.deliver_trap_interrupt(wave.process_id(), wave.m0());
    });
  }
  mes_queue_owner_ = owner;
  return owner;
}

std::optional<amdgpu::ComputeQueueBindingPlan> SoC::make_aql(uint32_t /*queue_ordinal*/) {
  amdgpu::CommandProcessor *owner = acquire_mes_queue_owner_cp();
  if (owner == nullptr)
    return std::nullopt;
  // The PCI discovery profile currently exposes one graphics instance even
  // when the simulation config contains additional XCDs. A MES-created queue
  // must therefore stay on its selected owner: fanning it out would execute on
  // XCDs the guest did not size resources such as scratch for. The simulated
  // KFD path enables fan-out separately when it publishes every XCD.
  return amdgpu::ComputeQueueBindingPlan{
      .factory = amdgpu::make_compute_queue_binding_factory(*owner), .xcd_fanout = false};
}

std::optional<amdgpu::ComputeQueueBindingPlan> SoC::make_pm4(uint32_t /*queue_ordinal*/,
                                                             amdgpu::Pm4PacketCallbacks callbacks) {
  amdgpu::CommandProcessor *owner = acquire_mes_queue_owner_cp();
  if (owner == nullptr)
    return std::nullopt;
  return amdgpu::ComputeQueueBindingPlan{
      .factory = amdgpu::make_compute_queue_binding_factory(*owner, std::move(callbacks)),
      .xcd_fanout = false};
}

amdgpu::QueueContextSaveResult SoC::save_context(const amdgpu::QueueIdentity &identity,
                                                 uint64_t address, uint32_t area_size,
                                                 uint32_t control_stack_size) {
  using amdgpu::VmAccessOutcome;
  const auto access = gpu_vm_.snapshot(identity.address_space);
  // A known CWSR layout does not imply complete MES state support: gfx9.4
  // accumulator registers are not preserved by this save/restore path.
  if (!access || !mes_queue_owner_ ||
      amdgpu::cwsr_layout_kind(arch_) != amdgpu::CwsrLayoutKind::Gfx12_5)
    return {};

  std::vector<amdgpu::CwsrWaveState> states;
  std::unordered_set<amdgpu::L2Cache *> caches;
  const auto belongs_to_queue = [&](const amdgpu::Wavefront *wave) {
    return wave && !wave->is_halted() && wave->process_id() == identity.process_id &&
           wave->queue_id() == identity.queue_id && wave->address_space() == identity.address_space;
  };
  const auto suspended = mes_queue_owner_->suspend_queue_context(identity);
  if (suspended != VmAccessOutcome::Complete && suspended != VmAccessOutcome::Unavailable)
    return {.outcome = suspended};
  bool pending = suspended == VmAccessOutcome::Unavailable;
  for_each_cp([&](amdgpu::CommandProcessor *cp) {
    for (auto *cu : cp->compute_units()) {
      cu->retry_paused_queue_memory(identity.process_id, identity.queue_id, /*replay_loads=*/true);
      cu->with_wave_state_locked([&] {
        for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
          auto *wave = cu->wf(slot);
          if (!belongs_to_queue(wave) || wave->in_trap_handler())
            continue;
          if (!wave->wait_counters().empty()) {
            pending = true;
            continue;
          }
          states.push_back(amdgpu::build_cwsr_wave_state(*wave, arch_));
        }
      });
      if (cu->l2())
        caches.insert(cu->l2());
    }
  });
  if (pending)
    return {.outcome = VmAccessOutcome::Unavailable};
  for_each_cp([&](amdgpu::CommandProcessor *cp) {
    for (auto *cu : cp->compute_units())
      cu->with_wave_state_locked([&] { cu->flush_l1(identity.process_id); });
  });
  for (auto *cache : caches) {
    const auto status = cache->flush_all(identity.process_id);
    if (status != VmAccessOutcome::Complete)
      return {.outcome = status};
  }
  for (auto *iod : iods_) {
    const auto status = iod->msc()->flush_all();
    if (status != VmAccessOutcome::Complete)
      return {.outcome = status};
  }
  std::vector<uint8_t> queue_context;
  const auto saved = mes_queue_owner_->save_queue_context(identity, queue_context);
  if (saved != amdgpu::QueueContextSaveOutcome{VmAccessOutcome::Complete})
    return {.outcome = saved};
  std::array<uint32_t, 2> debug_region{};
  const auto read = access->read(address + amdgpu::cwsr_debug_fields_offset(),
                                 std::as_writable_bytes(std::span(debug_region)));
  if (read != VmAccessOutcome::Complete)
    return {.outcome = read};
  amdgpu::prepare_cwsr_wave_states(states);
  VmAccessOutcome status = VmAccessOutcome::Complete;
  auto write = [&](uint64_t target, std::span<const uint8_t> bytes) {
    if (status == VmAccessOutcome::Complete)
      status = access->write(target, std::as_bytes(bytes));
  };
  const amdgpu::CwsrLayout layout = amdgpu::serialize_queue_cwsr_bulk(
      address, area_size, states, write, arch_, control_stack_size,
      amdgpu::CwsrDebugRegion{debug_region[0], debug_region[1]});
  if (!layout.ok)
    return {};
  if (status != VmAccessOutcome::Complete)
    return {.outcome = status};
  auto lease = gpu_vm_.retain_binding(identity.address_space);
  if (!lease)
    return {};
  saved_contexts_.insert_or_assign(SavedContextKey{identity, address},
                                   SavedContext{std::move(*lease), std::move(queue_context)});
  return {.outcome = status,
          .control_stack_offset = layout.control_stack_offset,
          .wave_state_offset = layout.wave_state_offset};
}

void SoC::cancel_context_save(const amdgpu::QueueIdentity &identity) {
  if (mes_queue_owner_)
    mes_queue_owner_->cancel_queue_context_save(identity);
}

void SoC::release_context(const amdgpu::QueueIdentity &identity) {
  if (mes_queue_owner_)
    mes_queue_owner_->release_queue_context(identity);
}

void SoC::discard_saved_context(const amdgpu::QueueIdentity &identity, uint64_t address) {
  saved_contexts_.erase(SavedContextKey{identity, address});
}

void SoC::clear_saved_contexts() { saved_contexts_.clear(); }

amdgpu::VmAccessOutcome SoC::restore_context(const amdgpu::QueueIdentity &identity,
                                             uint64_t address, uint32_t area_size,
                                             uint32_t control_stack_size) {
  using amdgpu::VmAccessOutcome;
  const auto access = gpu_vm_.snapshot(identity.address_space);
  if (!access || amdgpu::cwsr_layout_kind(arch_) != amdgpu::CwsrLayoutKind::Gfx12_5)
    return VmAccessOutcome::Faulted;
  if (!mes_queue_owner_)
    return VmAccessOutcome::Faulted;
  VmAccessOutcome status = VmAccessOutcome::Complete;
  auto read = [&](uint64_t source, std::span<uint8_t> bytes) {
    if (status == VmAccessOutcome::Complete)
      status = access->read(source, std::as_writable_bytes(bytes));
  };
  const auto saved = saved_contexts_.find(SavedContextKey{identity, address});
  if (saved == saved_contexts_.end())
    return VmAccessOutcome::Malformed;
  std::vector<amdgpu::CwsrWaveState> states;
  const bool decoded = amdgpu::deserialize_queue_cwsr_bulk(address, area_size, states, read, arch_,
                                                           control_stack_size);
  if (status != VmAccessOutcome::Complete)
    return status;
  if (!decoded)
    return VmAccessOutcome::Malformed;
  mes_queue_owner_->set_queue_debug_suspended(identity.queue_id, identity.process_id, true);
  const auto restored = mes_queue_owner_->restore_queue_context(identity, saved->second.bytes);
  if (restored != VmAccessOutcome::Complete)
    return restored;
  std::vector<amdgpu::Wavefront *> waves;
  std::vector<amdgpu::ComputeUnitCore *> owners;
  for (auto *cu : mes_queue_owner_->compute_units())
    cu->with_wave_state_locked([&] {
      for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
        auto *wave = cu->wf(slot);
        if (wave && !wave->is_halted() && wave->process_id() == identity.process_id &&
            wave->queue_id() == identity.queue_id &&
            wave->address_space() == identity.address_space) {
          waves.push_back(wave);
          owners.push_back(cu);
        }
      }
    });
  const auto matches = amdgpu::match_cwsr_wave_states(states, waves);
  if (!matches) {
    release_context(identity);
    return VmAccessOutcome::Malformed;
  }
  // Validate every allocation before applying any debugger edits.
  for (size_t index = 0; index < states.size(); ++index) {
    const auto *wave = waves[(*matches)[index]];
    auto &state = states[index];
    if (state.num_sgprs < wave->num_sgprs() || state.num_vgprs < wave->num_vgprs() ||
        (!state.lds.empty() && state.lds.size() < wave->lds_size())) {
      release_context(identity);
      return VmAccessOutcome::Malformed;
    }
    state.num_sgprs = wave->num_sgprs();
    state.num_vgprs = wave->num_vgprs();
  }
  for (size_t index = 0; index < states.size(); ++index) {
    const size_t wave_index = (*matches)[index];
    owners[wave_index]->with_wave_state_locked([&] {
      auto &wave = *waves[wave_index];
      const auto &state = states[index];
      amdgpu::apply_cwsr_to_wave(wave, state, arch_,
                                 amdgpu::CwsrRestorePolicy::RestoreSavedRegisters);
      // Scratch remains in its original backing even when execution moves.
      wave.set_scratch_scoreboard_id(state.scratch_scoreboard_id);
      wave.set_shader_engine_id(state.shader_engine_id);
      wave.set_trap_id(state.trap_id);
    });
  }
  return VmAccessOutcome::Complete;
}

void SoC::resume_context(const amdgpu::QueueIdentity &identity) {
  if (!mes_queue_owner_)
    return;
  // MES has published the new queue identity and trap configuration before
  // opening admission, so a resumed wave can immediately resolve its handler.
  mes_queue_owner_->set_queue_debug_suspended(identity.queue_id, identity.process_id, false);
  for (auto *cu : mes_queue_owner_->compute_units()) {
    cu->with_wave_state_locked([&] {
      for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
        auto *wave = cu->wf(slot);
        if (wave && !wave->is_halted() && wave->process_id() == identity.process_id &&
            wave->queue_id() == identity.queue_id &&
            wave->address_space() == identity.address_space)
          wave->set_debug_suspended(false);
      }
    });
    cu->schedule_work_async();
  }
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
  // Each XCD caller uses at most CUs-1 workers. Concurrent callers share the
  // SoC pool, so its capacity is not limited to the largest individual XCD.
  // Fewer engine partitions can leave some of this maximum capacity unused.
  size_t capacity = 1;
  for_each_cp([&capacity](auto *cp) {
    if (!cp->compute_units().empty())
      capacity += cp->compute_units().size() - 1;
  });
  uint32_t effective_threads =
      static_cast<uint32_t>(std::min<size_t>(requested_dispatch_threads_, capacity));
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
    if (std::ranges::find(cps, nullptr) == cps.end()) {
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
