// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/compute_unit.h"

#include "rocjitsu/vm/amdgpu/command_processor.h"

#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "util/except.h"
#include "util/log.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <stdexcept>

namespace rocjitsu {
namespace amdgpu {

namespace {
constexpr uint32_t kPrivilegedStatusBit = 1u << 5;

bool is_privileged(const Wavefront &wf) { return (wf.status_raw() & kPrivilegedStatusBit) != 0; }

uint32_t pack_barrier_state(uint32_t member_count, uint32_t signal_count,
                            uint32_t allocation_blocks = 0) {
  return 1u | ((member_count & 0x7fu) << 4) | ((signal_count & 0x7fu) << 16) |
         ((allocation_blocks & 0x7u) << 24);
}
} // namespace

template <GpuIsa Isa> void validate_compute_unit_config(const ComputeUnitCore::Config &config) {
  using Limits = IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, Isa>;

  const int32_t max_ordinary_selector = isa_properties(config.arch).scalar_sgpr_max_selector;
  if (max_ordinary_selector >= 0 &&
      config.sgprs_per_wf <= static_cast<uint32_t>(max_ordinary_selector)) {
    throw util::ConfigError("sgprs_per_wf is " + std::to_string(config.sgprs_per_wf) +
                            ", but the architecture exposes ordinary selectors through s" +
                            std::to_string(max_ordinary_selector) + " and requires at least " +
                            std::to_string(max_ordinary_selector + 1) + " slots");
  }

  if (config.num_wf_slots > Isa::MAX_WF_SLOTS) {
    throw util::ConfigError("num_wf_slots exceeds the ISA maximum of " +
                            std::to_string(Isa::MAX_WF_SLOTS));
  }

  const uint32_t vgprs_per_block =
      std::max(config.vgprs_per_wf, Limits::MAX_ACCVGPR_PHYSICAL_LIMIT);
  if (vgprs_per_block > Limits::MAX_VGPRS_PER_BLOCK) {
    throw util::ConfigError("effective VGPRs per wavefront exceeds the ISA maximum of " +
                            std::to_string(Limits::MAX_VGPRS_PER_BLOCK));
  }

  const uint64_t vgpr_file_registers = static_cast<uint64_t>(config.num_wf_slots) * vgprs_per_block;
  if (vgpr_file_registers > std::numeric_limits<uint32_t>::max() ||
      vgpr_file_registers > Limits::MAX_VGPR_FILE_REGISTERS) {
    throw util::ConfigError("configured VGPR file exceeds the ISA maximum capacity");
  }
}

ComputeUnitCore::ComputeUnitCore(std::string name, const Config &config, GpuMemory *memory,
                                 L2Cache *l2, uint32_t wf_size)
    : simdojo::CompositeComponent(std::move(name)), config_(config), memory_(memory),
      wf_size_(wf_size), decoder_(Decoder::create(config.arch)), l2_(l2), l1_scalar_(l2),
      l1_vector_(l2), lds_(config.lds_size_kb), scalar_mem_pipeline_(&l1_scalar_),
      global_mem_pipeline_(&l1_vector_, l2), local_mem_pipeline_() {
  if (!decoder_)
    throw std::runtime_error("Unsupported architecture for ComputeUnit decoder");

  // Enable pool allocation for the hot decode-execute path.
  // Instructions decoded during step() are always deleted before the CU
  // (and its decoder) are destroyed, so pool allocation is safe here.
  decoder_->enable_pool();

  wfs_.resize(config.num_wf_slots);
  sgpr_file_.init(config.num_wf_slots * config.sgprs_per_wf, config.sgprs_per_wf);
  sgpr_to_wave_.resize(config.num_wf_slots * config.sgprs_per_wf, nullptr);

  // Completer port: CP sends dispatch activation messages here.
  cpl_ = add_port(std::make_unique<simdojo::Port>("cpl", 0, this, simdojo::PortDirection::IN,
                                                  simdojo::PortProtocol::DISPATCH));
  cpl_->set_handler([this](simdojo::Tick, simdojo::Message *) { schedule_work(); });

  // Requester port: structural connection to shared L2 cache.
  req_ = add_port(std::make_unique<simdojo::Port>("req", 1, this, simdojo::PortDirection::OUT,
                                                  simdojo::PortProtocol::MEMORY));
}

std::unique_ptr<ComputeUnitCore> ComputeUnitCore::create(std::string name, const Config &config,
                                                         GpuMemory *memory, L2Cache *l2,
                                                         simdojo::ExecMode exec_mode) {
  // Helper: instantiate the ISA-specific CU for the given execution mode.
#define ROCJITSU_CU_CASE(ARCH_ENUM, ISA_TYPE)                                                      \
  case ARCH_ENUM:                                                                                  \
    validate_compute_unit_config<ISA_TYPE>(config);                                                \
    switch (exec_mode) {                                                                           \
    case simdojo::ExecMode::FUNCTIONAL:                                                            \
      return std::make_unique<IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, ISA_TYPE>>(        \
          std::move(name), config, memory, l2);                                                    \
    case simdojo::ExecMode::CLOCKED:                                                               \
      return std::make_unique<IsaExecComputeUnit<simdojo::ExecMode::CLOCKED, ISA_TYPE>>(           \
          std::move(name), config, memory, l2);                                                    \
    }                                                                                              \
    break

  switch (config.arch) {
    // \NPI new ISA family: add ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_<NAME>, <isa>::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_CDNA1, cdna1::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_CDNA2, cdna2::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_CDNA3, cdna3::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_CDNA4, cdna4::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_RDNA1, rdna1::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_RDNA2, rdna2::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_RDNA3, rdna3::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_RDNA3_5, rdna3_5::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_RDNA4, rdna4::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_GFX1250, cdna5::Isa);
  default:
    break;
  }
#undef ROCJITSU_CU_CASE
  throw std::runtime_error("Unsupported architecture for ComputeUnit");
}

Wavefront *ComputeUnitCore::dispatch_wf(uint32_t wg_id, uint64_t pc, uint32_t num_sgprs,
                                        uint32_t num_vgprs) {
  assert(wfs_.size() == config_.num_wf_slots && "wavefront slots not properly initialized");
  // Halted wavefronts have already freed their SGPR/VGPR blocks at s_endpgm, so a
  // halted slot is immediately available. Find an idle slot.
  size_t slot = config_.num_wf_slots;
  for (size_t i = 0; i < wfs_.size(); ++i) {
    if (wfs_[i]->is_halted()) {
      slot = i;
      break;
    }
  }

  // No free slot: fail the dispatch (like the register-allocation failures below)
  // rather than indexing wfs_ out of bounds. The CP normally gates placement on
  // can_accept_workgroup(), but returning nullptr is part of this API's contract
  // and must hold even when a caller dispatches directly to a full CU.
  if (slot >= config_.num_wf_slots)
    return nullptr;

  return dispatch_wf_at(static_cast<uint32_t>(slot), wg_id, pc, num_sgprs, num_vgprs);
}

Wavefront *ComputeUnitCore::dispatch_wf_at(uint32_t wf_id, uint32_t wg_id, uint64_t pc,
                                           uint32_t num_sgprs, uint32_t num_vgprs) {
  assert(wfs_.size() == config_.num_wf_slots && "wavefront slots not properly initialized");
  if (wf_id >= config_.num_wf_slots || !wfs_[wf_id]->is_halted())
    return nullptr;

  int32_t sgpr_base = sgpr_file_.allocate(num_sgprs);
  if (sgpr_base < 0)
    return nullptr;

  int32_t vgpr_base = allocate_vgprs(num_vgprs);
  if (vgpr_base < 0) {
    sgpr_file_.free(static_cast<uint32_t>(sgpr_base));
    return nullptr;
  }

  // Invalidate the L1 scalar cache so this wavefront reads fresh kernel
  // arguments from L2/memory rather than stale lines from a prior kernel.
  // On real hardware, the driver issues s_dcache_inv at kernel launch.
  l1_scalar_.invalidate_all();

  auto *wf = wfs_[wf_id].get();
  wf->wg_id_ = wg_id;
  wf->pc = pc;
  wf->sgpr_alloc_ = {static_cast<uint32_t>(sgpr_base), num_sgprs};
  wf->vgpr_alloc_ = {static_cast<uint32_t>(vgpr_base), num_vgprs};
  wf->num_sgprs_ = num_sgprs;
  wf->num_vgprs_ = num_vgprs;
  wf->exec_ = wf_size_ == 64 ? ~0ULL : (1ULL << wf_size_) - 1;
  wf->vcc_ = 0;
  wf->m0_ = 0;
  wf->trap_registers_.fill(0);
  wf->set_apertures(shared_aperture_base_, shared_aperture_limit_, private_aperture_base_,
                    private_aperture_limit_);
  wf->state_ = WfState::RUNNING;
  wf->set_ready_cycle(cycle_counter_);
  wf->trace_inst_count_ = 0;

  std::fill(sgpr_to_wave_.begin() + sgpr_base, sgpr_to_wave_.begin() + sgpr_base + num_sgprs, wf);
  fill_vgpr_to_wave(static_cast<uint32_t>(vgpr_base), vgpr_allocation_block_size(), wf);

  util::Logger::cp("DISPATCH_WF cu=", this->full_path(), " wf=", wf->wf_id(), " slot=", wf_id,
                   " pc=0x", std::hex, pc, std::dec, " wg=", wg_id, " pid=", wf->process_id());

  schedule_work();
  return wf;
}

size_t ComputeUnitCore::num_wfs() const {
  size_t count = 0;
  for (const auto &w : wfs_)
    if (!w->is_halted())
      ++count;
  return count;
}

void ComputeUnitCore::free_wavefront_resources(Wavefront &wf) {
  if (wf.sgpr_alloc().count > 0) {
    sgpr_file_.free(wf.sgpr_alloc().base);
    free_vgprs(wf.vgpr_alloc().base);
  }
  wf.trace_inst_count_ = 0;
  wf.reset();
}

void ComputeUnitCore::maybe_reset_lds_alloc() {
  if (!has_active_wfs() && !lds_allocation_pinned())
    reset_lds_alloc();
}

void ComputeUnitCore::begin_workgroup(uint32_t dispatch_id, uint32_t wg_id, uint32_t wf_count,
                                      uint32_t num_named_barriers) {
  const uint64_t key = wg_key(dispatch_id, wg_id);
  active_wgs_[key] = wf_count;
  if (wf_count <= 1) {
    // Single-wave workgroups are not allocated workgroup or named barriers.
    barrier_wgs_.erase(key);
    return;
  }

  auto &group = barrier_wgs_[key];
  group = {};
  group.allocated_count = std::min(num_named_barriers, kMaxNamedBarriers);
  for (auto &barrier : group.workgroup)
    barrier.member_count = wf_count;
}

void ComputeUnitCore::named_barrier_init(Wavefront &wf, int32_t barrier_id, uint32_t member_count) {
  auto group = barrier_wgs_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
  if (group == barrier_wgs_.end() || barrier_id <= 0 ||
      static_cast<uint32_t>(barrier_id) > group->second.allocated_count)
    return;

  auto &barrier = group->second.named[static_cast<uint32_t>(barrier_id)];
  if (member_count != 0)
    barrier.member_count = member_count & 0x7fu;
  barrier.signal_count = 0;
}

void ComputeUnitCore::named_barrier_join(Wavefront &wf, int32_t barrier_id) {
  auto group = barrier_wgs_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
  if (group == barrier_wgs_.end())
    return;
  if (barrier_id == 0) {
    wf.named_barrier_id_ = 0;
    wf.barrier_complete_[kNamedBarrierBit] = false;
    if (wf.waiting_barrier_bit_ == kNamedBarrierBit)
      wf.waiting_barrier_bit_ = Wavefront::kNoBarrierWait;
    return;
  }
  if (barrier_id < 0 || static_cast<uint32_t>(barrier_id) > group->second.allocated_count)
    return;
  wf.named_barrier_id_ = static_cast<uint32_t>(barrier_id);
  wf.barrier_complete_[kNamedBarrierBit] = false;
  if (wf.waiting_barrier_bit_ == kNamedBarrierBit)
    wf.waiting_barrier_bit_ = Wavefront::kNoBarrierWait;
}

bool ComputeUnitCore::barrier_signal(Wavefront &wf, int32_t barrier_id, uint32_t member_count) {
  if (barrier_id == kClusterBarrierId || barrier_id == kClusterTrapBarrierId) {
    if (barrier_id == kClusterTrapBarrierId && !is_privileged(wf))
      return false;
    return cp_ ? cp_->cluster_barrier_signal(wf, barrier_id) : false;
  }

  auto group = barrier_wgs_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
  if (group == barrier_wgs_.end() || barrier_id == 0 || barrier_id < kWorkgroupTrapBarrierId)
    return false;

  BarrierCounter *barrier = nullptr;
  uint8_t completion_bit = kNamedBarrierBit;
  uint32_t named_id = 0;
  if (barrier_id > 0) {
    named_id = static_cast<uint32_t>(barrier_id);
    if (named_id > group->second.allocated_count)
      return false;
    barrier = &group->second.named[named_id];
    if (member_count != 0)
      barrier->member_count = member_count & 0x7fu;
  } else {
    if (barrier_id == kWorkgroupTrapBarrierId && !is_privileged(wf))
      return false;
    completion_bit = static_cast<uint8_t>(-barrier_id);
    barrier = &group->second.workgroup[completion_bit - kWorkgroupBarrierBit];
  }

  if (barrier->member_count == 0)
    return false;
  const bool is_first = barrier->signal_count == 0;
  barrier->signal_count = std::min(barrier->signal_count + 1, 0x7fu);
  if (barrier->signal_count < barrier->member_count)
    return is_first;

  barrier->signal_count = 0;
  auto members = complete_barrier(wf.dispatch_id(), wf.wg_id(), completion_bit, named_id);
  notify_barrier_complete(members);
  return is_first;
}

std::vector<Wavefront *> ComputeUnitCore::complete_barrier(uint32_t dispatch_id, uint32_t wg_id,
                                                           uint8_t completion_bit,
                                                           uint32_t named_barrier_id) {
  std::vector<Wavefront *> members;
  for (const auto &candidate : wfs_) {
    if (candidate->is_halted() || candidate->dispatch_id() != dispatch_id ||
        candidate->wg_id() != wg_id)
      continue;
    if (completion_bit == kNamedBarrierBit && candidate->named_barrier_id_ != named_barrier_id)
      continue;
    candidate->barrier_complete_[completion_bit] = true;
    members.push_back(candidate.get());
  }
  for (auto *member : members) {
    if (member->state() == WfState::BARRIER && member->waiting_barrier_bit_ == completion_bit) {
      member->barrier_complete_[completion_bit] = false;
      member->waiting_barrier_bit_ = Wavefront::kNoBarrierWait;
      member->set_state(WfState::RUNNING);
      member->set_ready_cycle(cycle_counter_);
    }
  }
  return members;
}

void ComputeUnitCore::notify_barrier_complete(std::span<Wavefront *> members) {
  if (!members.empty())
    plugin_group_->onAmdgpuBarrierResolved(members);
}

uint32_t ComputeUnitCore::barrier_state(const Wavefront &wf, int32_t barrier_id) const {
  auto group = barrier_wgs_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
  const uint32_t allocation_blocks =
      group == barrier_wgs_.end() ? 0 : (group->second.allocated_count + 3) / 4;

  if (barrier_id == kClusterBarrierId || barrier_id == kClusterTrapBarrierId) {
    if (barrier_id == kClusterTrapBarrierId && !is_privileged(wf))
      return 0;
    return cp_ ? cp_->cluster_barrier_state(wf, barrier_id, allocation_blocks) : 0;
  }

  if (group == barrier_wgs_.end() || barrier_id == 0 || barrier_id < kWorkgroupTrapBarrierId)
    return 0;

  if (barrier_id < 0) {
    if (barrier_id == kWorkgroupTrapBarrierId && !is_privileged(wf))
      return 0;
    const auto &barrier = group->second.workgroup[static_cast<uint32_t>(-barrier_id - 1)];
    return pack_barrier_state(barrier.member_count, barrier.signal_count, allocation_blocks);
  }

  const uint32_t id = static_cast<uint32_t>(barrier_id);
  if (id > group->second.allocated_count)
    return 0;
  const auto &barrier = group->second.named[id];
  return pack_barrier_state(barrier.member_count, barrier.signal_count, allocation_blocks);
}

void ComputeUnitCore::barrier_wait(Wavefront &wf, int32_t barrier_id) {
  uint8_t completion_bit = kNamedBarrierBit;
  if (barrier_id >= 0) {
    auto group = barrier_wgs_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
    if (group == barrier_wgs_.end() || wf.named_barrier_id_ == 0 ||
        wf.named_barrier_id_ > group->second.allocated_count)
      return;
  } else {
    if (barrier_id < kClusterTrapBarrierId ||
        ((barrier_id == kWorkgroupTrapBarrierId || barrier_id == kClusterTrapBarrierId) &&
         !is_privileged(wf)))
      return;
    completion_bit = static_cast<uint8_t>(-barrier_id);
    if (completion_bit <= kWorkgroupTrapBarrierBit) {
      if (!barrier_wgs_.contains(wg_key(wf.dispatch_id(), wf.wg_id())))
        return;
    } else if (!cp_ || !cp_->cluster_barrier_valid(wf, barrier_id)) {
      return;
    }
  }

  if (wf.barrier_complete_[completion_bit]) {
    wf.barrier_complete_[completion_bit] = false;
    return;
  }
  wf.waiting_barrier_bit_ = completion_bit;
  wf.set_state(WfState::BARRIER);
}

bool ComputeUnitCore::named_barrier_leave(Wavefront &wf) {
  auto group = barrier_wgs_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
  const uint32_t id = wf.named_barrier_id_;
  if (group == barrier_wgs_.end())
    return false;
  if (id == 0)
    return true;
  if (id > group->second.allocated_count)
    return false;

  wf.named_barrier_id_ = 0;
  wf.barrier_complete_[kNamedBarrierBit] = false;
  if (wf.waiting_barrier_bit_ == kNamedBarrierBit)
    wf.waiting_barrier_bit_ = Wavefront::kNoBarrierWait;
  auto &barrier = group->second.named[id];
  if (barrier.member_count != 0)
    --barrier.member_count;
  if (barrier.signal_count >= barrier.member_count) {
    barrier.signal_count = 0;
    auto members = complete_barrier(wf.dispatch_id(), wf.wg_id(), kNamedBarrierBit, id);
    notify_barrier_complete(members);
  }
  return barrier.member_count == 0;
}

void ComputeUnitCore::release_wf(uint32_t dispatch_id, uint32_t wg_id) {
  auto key = wg_key(dispatch_id, wg_id);
  auto group = barrier_wgs_.find(key);
  if (group != barrier_wgs_.end()) {
    const auto retire_member = [&](BarrierCounter &barrier, uint8_t completion_bit,
                                   uint32_t joined_id = 0) {
      if (barrier.member_count == 0)
        return;
      --barrier.member_count;
      if (barrier.member_count == 0 || barrier.signal_count < barrier.member_count)
        return;
      barrier.signal_count = 0;
      auto members = complete_barrier(dispatch_id, wg_id, completion_bit, joined_id);
      notify_barrier_complete(members);
    };

    retire_member(group->second.workgroup[0], kWorkgroupBarrierBit);
    retire_member(group->second.workgroup[1], kWorkgroupTrapBarrierBit);
  }

  auto it = active_wgs_.find(key);
  if (it != active_wgs_.end() && --it->second == 0) {
    plugin_group_->onAmdgpuWorkgroupCompleted(dispatch_id, wg_id);
    active_wgs_.erase(it);
    barrier_wgs_.erase(key);
    if (cp_)
      cp_->notify_wg_complete(dispatch_id, wg_id);
  }
  // The whole workgroup's per-WG LDS region can be reclaimed once the CU has fully
  // drained and no cluster peer can still multicast into it.
  maybe_reset_lds_alloc();
}

void ComputeUnitCore::abort_workgroup(uint32_t dispatch_id, uint32_t wg_id) {
  // Roll back a workgroup that was committed via begin_workgroup() but whose peers
  // in the same clustered dispatch failed to fully dispatch. Unlike release_wf(),
  // this fires no completion hook and no CP notify — the WG never executed. Free any
  // resident (not-yet-halted) waves belonging to this WG, drop the refcount entry,
  // and reclaim LDS if the CU is now idle and unpinned. The caller unpins the cluster
  // LDS separately (the pin is CP-side bookkeeping).
  for (const auto &w : wfs_) {
    if (!w->is_halted() && w->dispatch_id() == dispatch_id && w->wg_id() == wg_id)
      free_wavefront_resources(*w);
  }
  active_wgs_.erase(wg_key(dispatch_id, wg_id));
  barrier_wgs_.erase(wg_key(dispatch_id, wg_id));
  maybe_reset_lds_alloc();
}

bool ComputeUnitCore::can_accept_workgroup(uint32_t num_wfs, uint32_t lds_bytes) const {
  // Count free wavefront slots.
  uint32_t free_slots = 0;
  for (const auto &w : wfs_)
    if (w->is_halted())
      ++free_slots;
  if (free_slots < num_wfs) {
    util::Logger::vm("CU ", this->name(), " can_accept_wg: REJECT free_slots=", free_slots,
                     " < num_wfs=", num_wfs);
    return false;
  }

  // Check SGPR register blocks.
  uint32_t free_sgpr = sgpr_file_.free_block_count();
  if (free_sgpr < num_wfs) {
    util::Logger::vm("CU ", this->name(), " can_accept_wg: REJECT free_sgpr=", free_sgpr,
                     " < num_wfs=", num_wfs);
    return false;
  }

  // Check VGPR register blocks.
  uint32_t free_vgpr = free_vgpr_blocks();
  if (free_vgpr < num_wfs) {
    util::Logger::vm("CU ", this->name(), " can_accept_wg: REJECT free_vgpr=", free_vgpr,
                     " < num_wfs=", num_wfs);
    return false;
  }

  if (lds_bytes > 0) {
    uint32_t aligned = util::align_up(lds_bytes, 256u);
    uint32_t lds_capacity_bytes = config_.lds_size_kb * 1024u;
    if (next_lds_alloc_ + aligned > lds_capacity_bytes) {
      return false;
    }
  }

  return true;
}

void ComputeUnitCore::tick_pipelines() {
  scalar_mem_pipeline_.tick();
  global_mem_pipeline_.tick();
  local_mem_pipeline_.tick();
}

void ComputeUnitCore::route_memory_inst(Instruction *inst, Wavefront &wf) {
  plugin_group_->onAmdgpuRouteMemoryInstruction(*inst, wf);

  if (inst->data()->tag() == GLOBAL_MEM && shared_aperture_base_ != 0) {
    auto &d = *inst->data_as<VectorMemState>();
    uint64_t probe = 0;
    for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
      if (d.lane_mask & (1ULL << lane)) {
        probe = d.per_lane_addr[lane];
        break;
      }
    }
    // FLAT ops targeting the shared aperture are routed to LDS (LGKMCNT,
    // not VMCNT).  Scratch-targeting FLATs stay on the global path.
    if (probe >= shared_aperture_base_ && probe <= shared_aperture_limit_) {
      for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
        if (d.lane_mask & (1ULL << lane))
          d.per_lane_addr[lane] = (d.per_lane_addr[lane] - shared_aperture_base_) + wf.lds_base();
      }
      inst->data()->set_tag(LOCAL_MEM);
      d.wait_counter_type = WaitCounterType::LGKMCNT;
      local_mem_pipeline_.issue(inst, wf);
      return;
    }
  }

  switch (inst->data()->tag()) {
  case SCALAR_MEM:
    scalar_mem_pipeline_.issue(inst, wf);
    break;
  case LOCAL_MEM:
    local_mem_pipeline_.issue(inst, wf);
    break;
  case GLOBAL_MEM:
    global_mem_pipeline_.issue(inst, wf);
    break;
  default:
    break;
  }
}

void ComputeUnitCore::update_wf_states() {
  ++cycle_counter_;

  for (auto &w : wfs_) {
    if (w->state() == WfState::WAITCNT && w->wait_satisfied()) {
      w->set_state(WfState::RUNNING);
      w->set_ready_cycle(cycle_counter_);
    } else if (w->state() == WfState::ENDING && w->wait_counters().empty()) {
      w->halt();
    }
  }

  for (auto &w : wfs_) {
    if (w->state() != WfState::BARRIER || w->waiting_barrier_bit_ != Wavefront::kNoBarrierWait)
      continue;
    uint32_t did = w->dispatch_id();
    uint32_t wg = w->wg_id();
    bool all_at_barrier = true;
    for (auto &w2 : wfs_) {
      if (w2->dispatch_id() == did && w2->wg_id() == wg && w2->state() != WfState::HALTED &&
          (w2->state() != WfState::BARRIER ||
           w2->waiting_barrier_bit_ != Wavefront::kNoBarrierWait)) {
        all_at_barrier = false;
        break;
      }
    }
    if (all_at_barrier) {
      std::vector<Wavefront *> barrier_wfs;
      for (auto &w2 : wfs_)
        if (w2->dispatch_id() == did && w2->wg_id() == wg && w2->state() == WfState::BARRIER &&
            w2->waiting_barrier_bit_ == Wavefront::kNoBarrierWait)
          barrier_wfs.push_back(w2.get());
      plugin_group_->onAmdgpuBarrierResolved(std::span<Wavefront *>(barrier_wfs));
      for (auto *bwf : barrier_wfs) {
        bwf->set_state(WfState::RUNNING);
        bwf->set_ready_cycle(cycle_counter_);
      }
    }
  }
}

void ComputeUnitCore::issue_instruction(Wavefront *active) {
  uint32_t vmid = active->process_id();

  rj_code_binary_inst_t words[4];
  for (int i = 0; i < 4; ++i)
    words[i] = memory_->fetch32(active->pc + i * 4, vmid);

  active->trace_inst_count_++;

  util::StringDiagnostic decode_error;
  DecodeResult decoded = decoder_->decode(words, decode_error.emitter());
  if (decoded.failed()) {
    util::Logger::vm("CU ", this->name(), ": wf", active->wf_id(), " HALT(decode rejection) pc=0x",
                     std::hex, active->pc, " words=[0x", words[0], ",0x", words[1], ",0x", words[2],
                     ",0x", words[3], "]", std::dec, " what=", decode_error.message());
    active->halt();
    return;
  }
  Instruction *inst = decoded.value().release();

  int inst_size_signed = inst->size();
  assert(inst_size_signed > 0 && "instruction size must be positive");
  auto inst_size = static_cast<uint64_t>(inst_size_signed);

  if constexpr (util::Logger::group_enabled(util::Logger::GROUP_VM)) {
    if (active->num_vgprs_ > 0) {
      util::Logger::vm([&](auto &os) {
        uint32_t vb = active->vgpr_alloc().base;
        os << std::format("{} wg[{}] wf[{}] EXECUTE #{} pc={:#x} {} sz={}", this->full_path(),
                          active->wg_id(), active->wf_id(), active->trace_inst_count_, active->pc,
                          inst->mnemonic(), inst_size);
        os << " enc=";
        for (uint64_t w = 0; w < inst_size / 4; ++w)
          os << std::format("{}{:08x}", w ? "," : "", words[w]);
        os << std::format(" scc={} vcc={:x} exec={:x}", active->read_scc(), active->vcc(),
                          active->exec());
        uint32_t nvr = std::min(active->num_vgprs_, 16u);
        for (uint32_t ln = 0; ln < active->wf_size_; ++ln) {
          os << std::format("\n[rj log VM]  PRE L{}: v[0:{}]=", ln, nvr - 1);
          for (uint32_t r = 0; r < nvr; ++r)
            os << std::format("{}{:x}", r ? "," : "", read_vgpr(vb + r, ln));
        }
      });
    }
  }

  plugin_group_->onAmdgpuBeforeExecuteInstruction(active->pc, *inst, *active);

  {
    auto mn = std::string_view(inst->mnemonic());
    if (mn.find("s_setpc") != std::string_view::npos ||
        mn.find("s_swappc") != std::string_view::npos) {
      const Operand *target_operand = inst->src_operand(0);
      assert(target_operand && "indirect PC instruction must have a target operand");
      uint64_t target = RegisterAccess(*active).read_scalar64(*target_operand);
      if (target == 0) {
        active->halt();
        delete inst;
        return;
      }
    }
  }

  execute_instruction(inst, *active);

  // A terminating instruction (s_endpgm with no pending waits) halts the wave
  // inside execute_instruction, which frees and resets its slot. Its registers,
  // pc, and allocations are now zeroed, so the after-execute hook, result logging,
  // and pc-advance below must not run on the dead slot. The dedicated
  // onAmdgpuWavefrontHalted hook already fired (with live state) from halt().
  // s_endpgm is never a memory op, so just reclaim the decoded instruction.
  //
  // Note the intentional asymmetry: an s_endpgm that defers to ENDING (pending
  // memory waits) is NOT halted here, so it DOES fire onAmdgpuAfterExecuteInstruction
  // below; the immediate-halt case does not. onAmdgpuWavefrontHalted is the
  // authoritative terminal hook and fires in both cases — consumers should observe
  // termination there, not via the after-execute hook.
  if (active->is_halted()) {
    delete inst;
    return;
  }

  plugin_group_->onAmdgpuAfterExecuteInstruction(active->pc, *inst, *active);

  if constexpr (util::Logger::group_enabled(util::Logger::GROUP_VM)) {
    if (active->num_vgprs_ > 0) {
      util::Logger::vm([&](auto &os) {
        uint32_t vb = active->vgpr_alloc().base;
        os << std::format("RESULT #{} scc={} vcc={:x} exec={:x}", active->trace_inst_count_,
                          active->read_scc(), active->vcc(), active->exec());
        uint32_t nvr = std::min(active->num_vgprs_, 16u);
        for (uint32_t ln = 0; ln < active->wf_size_; ++ln) {
          os << std::format("\n[rj log VM]  POST L{}: v[0:{}]=", ln, nvr - 1);
          for (uint32_t r = 0; r < nvr; ++r)
            os << std::format("{}{:x}", r ? "," : "", read_vgpr(vb + r, ln));
        }
      });
    }
  }

  if (inst->is_memory_op()) {
    if (inst->data() && inst->data()->tag() == GLOBAL_MEM) {
      auto *d = inst->data_as<VectorMemState>();
      d->issue_pc = active->pc;
    }
    route_memory_inst(inst, *active);
  } else
    delete inst;

  active->pc += inst_size;
}

bool ComputeUnitCore::step() {
  update_wf_states();

  for (auto &wf : wfs_) {
    if (wf->state() == WfState::RUNNING)
      issue_instruction(wf.get());
  }

  ++step_count_;
  if constexpr (util::Logger::group_enabled(util::Logger::GROUP_CP)) {
    if ((step_count_ & 0xFFFFF) == 0) {
      util::Logger::cp([&](auto &os) {
        os << std::format("CU[{}] steps={}M", this->full_path(), step_count_ >> 20);
        for (auto &wf : wfs_) {
          auto st = wf->state();
          if (st == WfState::RUNNING || st == WfState::WAITCNT || st == WfState::BARRIER)
            os << std::format(" wf{}:pc={:#x}:{}", wf->wf_id(), wf->pc,
                              st == WfState::RUNNING   ? "R"
                              : st == WfState::WAITCNT ? "W"
                                                       : "B");
        }
      });
    }
  }

  return has_active_wfs();
}

// Explicit template instantiations for all AMDGPU ISAs and execution modes.
#define ROCJITSU_CU_INSTANTIATE(ISA_TYPE)                                                          \
  template class IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, ISA_TYPE>;                      \
  template class IsaExecComputeUnit<simdojo::ExecMode::CLOCKED, ISA_TYPE>

ROCJITSU_CU_INSTANTIATE(cdna1::Isa);
ROCJITSU_CU_INSTANTIATE(cdna2::Isa);
ROCJITSU_CU_INSTANTIATE(cdna3::Isa);
ROCJITSU_CU_INSTANTIATE(cdna4::Isa);
ROCJITSU_CU_INSTANTIATE(rdna1::Isa);
ROCJITSU_CU_INSTANTIATE(rdna2::Isa);
ROCJITSU_CU_INSTANTIATE(rdna3::Isa);
ROCJITSU_CU_INSTANTIATE(rdna3_5::Isa);
ROCJITSU_CU_INSTANTIATE(rdna4::Isa);
ROCJITSU_CU_INSTANTIATE(cdna5::Isa);

#undef ROCJITSU_CU_INSTANTIATE

} // namespace amdgpu
} // namespace rocjitsu
