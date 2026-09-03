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
#include "rocjitsu/isa/arch/amdgpu/shared/alu_exceptions.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"
#include "rocjitsu/vm/amdgpu/hwreg.h"
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
bool InstructionComputeUnitView::signal_queue_exception(uint32_t queue_id, uint32_t process_id,
                                                        uint64_t status) {
  auto *cp = raw_cu().command_processor();
  return cp && cp->signal_queue_exception(queue_id, process_id, status);
}

uint32_t Wavefront::debug_read_sgpr(uint32_t reg) const {
  return cu_.read_sgpr(sgpr_alloc_.base + reg);
}

uint32_t Wavefront::debug_read_vgpr(uint32_t reg, uint32_t lane) const {
  return cu_.read_vgpr(vgpr_alloc_.base + reg, lane);
}

void Wavefront::debug_write_sgpr(uint32_t reg, uint32_t value) {
  cu_.write_sgpr(sgpr_alloc_.base + reg, value);
}

void Wavefront::debug_write_vgpr(uint32_t reg, uint32_t lane, uint32_t value) {
  cu_.write_vgpr(vgpr_alloc_.base + reg, lane, value);
}

namespace {
constexpr uint32_t kPrivilegedStatusBit = 1u << 5;

bool is_privileged(const Wavefront &wf) { return (wf.status_raw() & kPrivilegedStatusBit) != 0; }

std::string_view instruction_execution_error_name(InstructionExecutionError error) {
  switch (error) {
  case InstructionExecutionError::None:
    return "none";
  case InstructionExecutionError::UnsupportedOperandValue:
    return "unsupported operand value";
  }
  return "unknown instruction execution error";
}

uint32_t pack_barrier_state(uint32_t member_count, uint32_t signal_count,
                            uint32_t allocation_blocks = 0) {
  return 1u | ((member_count & 0x7fu) << 4) | ((signal_count & 0x7fu) << 16) |
         ((allocation_blocks & 0x7u) << 24);
}
} // namespace

template <GpuIsa Isa> void validate_compute_unit_config(const ComputeUnitCore::Config &config) {
  using Limits = IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, Isa>;

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
      wf_size_(wf_size),
      decoder_(config.target == ROCJITSU_CODE_TARGET_INVALID
                   ? Decoder::create(config.arch)
                   : Decoder::create(default_isa_target_registry(), config.target)),
      l2_(l2), l1_scalar_(l2), l1_vector_(l2), lds_(config.lds_size_kb),
      scalar_mem_pipeline_(&l1_scalar_), global_mem_pipeline_(&l1_vector_, l2),
      local_mem_pipeline_() {
  if (!decoder_)
    throw std::runtime_error("Unsupported architecture for ComputeUnit decoder");

  // Enable pool allocation for the hot decode-execute path.
  // Instructions decoded during step() are always deleted before the CU
  // (and its decoder) are destroyed, so pool allocation is safe here.
  decoder_->enable_pool();

  wfs_.resize(config.num_wf_slots);
  sgpr_file_.init(config.num_wf_slots * config.sgprs_per_wf, config.sgprs_per_wf);
  sgpr_block_owners_.resize(config.num_wf_slots);
  if (std::has_single_bit(config.sgprs_per_wf))
    sgpr_block_shift_ = std::countr_zero(config.sgprs_per_wf);

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
  if (config.target != ROCJITSU_CODE_TARGET_INVALID) {
    const IsaTargetRegistry &registry = default_isa_target_registry();
    const IsaTargetDescriptor *target_descriptor = registry.find(config.target);
    const IsaGpuTargetDescription *target_binding = registry.find_gpu_target(config.target);
    if (target_descriptor == nullptr || target_binding == nullptr)
      throw util::ConfigError("unsupported concrete GPU target");
    if (target_descriptor->architecture_id != config.arch)
      throw util::ConfigError("concrete GPU target does not belong to the configured architecture");
    if (!target_descriptor->supports_execution ||
        !target_binding->capabilities.execution_implemented)
      throw util::ConfigError("execution is not implemented for the concrete GPU target");
  }

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
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_CDNA5, cdna5::Isa);
  default:
    break;
  }
#undef ROCJITSU_CU_CASE
  throw std::runtime_error("Unsupported architecture for ComputeUnit");
}

Wavefront *ComputeUnitCore::dispatch_wf(uint32_t wg_id, uint64_t pc, uint32_t num_sgprs,
                                        uint32_t num_vgprs, uint32_t wave_size) {
  std::lock_guard<std::recursive_mutex> wave_state_lock(wave_state_mutex_);
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

  return dispatch_wf_at(static_cast<uint32_t>(slot), wg_id, pc, num_sgprs, num_vgprs, wave_size);
}

Wavefront *ComputeUnitCore::dispatch_wf_at(uint32_t wf_id, uint32_t wg_id, uint64_t pc,
                                           uint32_t num_sgprs, uint32_t num_vgprs,
                                           uint32_t wave_size) {
  assert(wfs_.size() == config_.num_wf_slots && "wavefront slots not properly initialized");
  if (wf_id >= config_.num_wf_slots || !wfs_[wf_id]->is_halted())
    return nullptr;

  auto *wf = wfs_[wf_id].get();
  const uint32_t dispatched_wave_size = wave_size == 0 ? wf->default_wf_size_ : wave_size;
  if ((dispatched_wave_size != 32 && dispatched_wave_size != 64) ||
      dispatched_wave_size > wf->max_wf_size_)
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

  wf->wf_size_ = dispatched_wave_size;
  wf->wg_id_ = wg_id;
  wf->pc = pc;
  wf->sgpr_alloc_ = {static_cast<uint32_t>(sgpr_base), num_sgprs};
  wf->vgpr_alloc_ = {static_cast<uint32_t>(vgpr_base), num_vgprs};
  wf->num_sgprs_ = num_sgprs;
  wf->num_vgprs_ = num_vgprs;
  wf->exec_ = wf->lane_mask();
  wf->vgpr_write_mask_ = wf->lane_mask();
  wf->vcc_ = 0;
  wf->m0_ = 0;
  wf->set_shader_engine_id(shader_engine_id_);
  wf->set_scratch_scoreboard_id(scratch_scoreboard_base_ + wf_id);
  wf->set_status_raw(0);
  wf->set_apertures(shared_aperture_base_, shared_aperture_limit_, private_aperture_base_,
                    private_aperture_limit_);
  wf->state_ = WfState::RUNNING;
  wf->set_ready_cycle(cycle_counter_);
  wf->trace_inst_count_ = 0;

  sgpr_block_owners_[static_cast<uint32_t>(sgpr_base) / config_.sgprs_per_wf] = {
      wf, static_cast<uint32_t>(sgpr_base) + config_.sgprs_per_wf};
  set_vgpr_block_owner(static_cast<uint32_t>(vgpr_base), wf);

  util::Logger::cp([&](auto &os) {
    os << "DISPATCH_WF cu=" << full_path() << " wf=" << wf->wf_id() << " slot=" << wf_id << " pc=0x"
       << std::hex << pc << std::dec << " wg=" << wg_id << " pid=" << wf->process_id();
  });

  schedule_work();
  return wf;
}

size_t ComputeUnitCore::num_wfs() const {
  std::lock_guard<std::recursive_mutex> wave_state_lock(wave_state_mutex_);
  size_t count = 0;
  for (const auto &w : wfs_)
    if (!w->is_halted())
      ++count;
  return count;
}

void ComputeUnitCore::free_wavefront_resources(Wavefront &wf) {
  std::lock_guard<std::recursive_mutex> wave_state_lock(wave_state_mutex_);
  if (wf.sgpr_alloc().count > 0) {
    sgpr_block_owners_[wf.sgpr_alloc().base / config_.sgprs_per_wf] = {};
    sgpr_file_.free(wf.sgpr_alloc().base);
    free_vgprs(wf.vgpr_alloc().base);
  }
  wf.trace_inst_count_ = 0;
  wf.reset();
}

void ComputeUnitCore::flush_wg_completions() {
  // Loops because a notification can retire more work and queue another
  // completion behind it; draining to empty keeps that from waiting for whatever
  // takes the wave-state lock next.
  for (;;) {
    std::vector<std::pair<uint32_t, uint32_t>> ready;
    {
      std::lock_guard<std::recursive_mutex> wave_state_lock(wave_state_mutex_);
      if (pending_wg_completions_.empty())
        return;
      ready.swap(pending_wg_completions_);
    }
    // The lock is released here, so taking hw_queue_mutex_ below cannot invert
    // against the CP's dispatch path.
    if (!cp_)
      return;
    for (const auto &[dispatch_id, wg_id] : ready)
      cp_->notify_wg_complete(dispatch_id, wg_id);
  }
}

void ComputeUnitCore::maybe_reset_lds_alloc() {
  if (!has_active_wfs() && !lds_allocation_pinned())
    reset_lds_alloc();
}

void ComputeUnitCore::begin_workgroup(uint32_t dispatch_id, uint32_t wg_id, uint32_t wf_count,
                                      uint32_t num_named_barriers) {
  // The driver's s_icache_inv rides the launch packet, so it lands once per
  // dispatch, not once per wave: a kernel VA reused by a later dispatch still
  // sees fresh code, while the sibling waves of one dispatch keep filling a
  // shared I$ instead of cold-starting each other.
  if (inst_cache_dispatch_id_ != dispatch_id) {
    inst_cache_.invalidate_all();
    inst_cache_dispatch_id_ = dispatch_id;
  }

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

void ComputeUnitCore::release_wf(uint32_t dispatch_id, uint32_t wg_id,
                                 Wavefront::CpCompletionNotice notice) {
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
    // Queued rather than sent: notify_wg_complete() takes the CP's
    // hw_queue_mutex_, and this runs under the wave-state lock, which the CP
    // takes in the other order when it dispatches. WaveStateGuard delivers it
    // once the lock is dropped.
    if (cp_ && notice == Wavefront::CpCompletionNotice::Send)
      pending_wg_completions_.emplace_back(dispatch_id, wg_id);
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

  const uint8_t route_tag = inst->data()->tag();
  switch (route_tag) {
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

  // Deliberately not gated on debug_active_, unlike the data-side probe below.
  // An unfetchable PC reads back as zeros, and zeros decode to a valid
  // instruction, so an undebugged wave that branches into unmapped memory would
  // otherwise execute zeros forever. Stopping it matters more than the one
  // extra page-table lookup, which is a fraction of the per-issue decode cost.
  if (vmid != 0 && !memory_->is_fetchable(active->pc, vmid)) {
    if (memory_violation_handler_ && memory_violation_handler_(*active, active->pc, false))
      return;
    // Wavefront::halt() is silent, so say why this wave stopped. Without this
    // the wave simply disappears from the run with nothing in the log.
    util::Logger::vm("CU ", this->name(), ": wf", active->wf_id(), " HALT(UnfetchablePc) pc=0x",
                     std::hex, active->pc, std::dec, " vmid=", vmid);
    active->halt();
    return;
  }

  rj_code_binary_inst_t words[4];
  static_assert(sizeof(words) == InstructionCache::kFetchBytes,
                "the I$ fetch width must match the issue window");
  if (debug_active()) {
    // A debugger writes breakpoints straight into code memory with none of the
    // maintenance that invalidates the I$, so bypass it while one is attached.
    for (int i = 0; i < 4; ++i)
      words[i] = memory_->fetch32(active->pc + i * 4, vmid);
  } else {
    // A session that has come and gone may have written over lines cached
    // before it attached, whether or not this wave issued while it was
    // running. Take the invalidation set_debug_active() published.
    sync_inst_cache_debug_epoch();
    inst_cache_.fetch(*memory_, active->pc, vmid, reinterpret_cast<uint8_t *>(words));
  }

  active->trace_inst_count_++;

  util::StringDiagnostic decode_error;
  DecodeResult decoded = decoder_->decode(words, decode_error.emitter());
  if (decoded.failed()) {
    util::Logger::vm("CU ", this->name(), ": wf", active->wf_id(), " HALT(decode rejection) pc=0x",
                     std::hex, active->pc, " words=[0x", words[0], ",0x", words[1], ",0x", words[2],
                     ",0x", words[3], "]", std::dec, " what=", decode_error.message());
    // Under a debugger, surface the undecodable instruction as an illegal-
    // instruction exception (stops the wave at this PC) instead of silently
    // retiring it. Without a debugger this halts as before.
    if (illegal_inst_handler_ && illegal_inst_handler_(*active))
      return;
    active->halt();
    return;
  }
  Instruction *inst = decoded.value().release();

  int inst_size_signed = inst->size();
  assert(inst_size_signed > 0 && "instruction size must be positive");
  auto inst_size = static_cast<uint64_t>(inst_size_signed);
  // The cause classifiers report into this as they run; see alu_exceptions.h.
  active->clear_pending_alu_causes();

  if constexpr (util::Logger::group_enabled(util::Logger::GROUP_VM)) {
    if (active->num_vgprs_ > 0) {
      util::Logger::vm([&](auto &os) {
        uint32_t vb = active->vgpr_alloc().base;
        os << std::format("{} wg[{}] wf[{}] EXECUTE #{} pc={:#x} {} sz={}", full_path(),
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

  // s_trap enters the per-process handler configured by SET_TRAP_HANDLER. The
  // hardware saves the interrupted PC/status in TTMPs and begins fetching at
  // TBA. The handler advances TTMP0:1 for software traps, sends the KFD
  // interrupt message, restores STATUS, and returns through s_rfe_b64.
  if (std::string_view(inst->mnemonic()) == "s_trap") {
    uint32_t trap_id = words[0] & 0xFFu;
    if (!active->in_trap_handler() && trap_handler_resolver_) {
      auto config = trap_handler_resolver_(*active);
      if (config && config->tba != 0) {
        const uint64_t saved_pc = active->pc;
        const uint32_t saved_status = active->status_raw();
        const auto properties = isa_properties(this->arch());
        const auto wave_state_layout = properties.wave_state_layout;
        const bool uses_split_wave_state = wave_state_layout != WaveStateLayout::Legacy;
        active->set_ttmp(0, static_cast<uint32_t>(saved_pc));
        if (uses_split_wave_state) {
          // GFX12 trap entry carries the four-bit trap id in TTMP1[31:28].
          // GFX12.0 has a 48-bit PC and preserves SCHED_MODE in TTMP1[27:26];
          // GFX12.5 expands the PC to 57 bits and enters privileged scheduling.
          const uint32_t pc_hi_mask =
              wave_state_layout == WaveStateLayout::Gfx12_5 ? 0x01FFFFFFu : 0x0000FFFFu;
          const uint32_t sched_mode = wave_state_layout == WaveStateLayout::Gfx12
                                          ? (active->wave_sched_mode_raw() & 0x3u) << 26
                                          : 0u;
          active->set_ttmp(1, (static_cast<uint32_t>(saved_pc >> 32) & pc_hi_mask) | sched_mode |
                                  ((trap_id & 0xFu) << 28));
          const uint32_t debug_enabled = config->debug_enabled ? (1u << 23) : 0u;
          active->set_ttmp(11, (active->ttmp(11) & ~(1u << 23)) | debug_enabled);
          constexpr uint16_t kWholeStatePriv = 4u | (31u << 11);
          uint32_t state_priv = 0;
          const auto state_result = read_hwreg_field(*active, kWholeStatePriv, state_priv);
          assert(state_result == HwregAccessResult::Success);
          (void)state_result;
          active->set_ttmp(12, state_priv);
        } else {
          active->set_ttmp(1, static_cast<uint32_t>(saved_pc >> 32) | (trap_id << 16));
        }
        // Dispatch identity. Which TTMPs carry it is architecture-specific and
        // this must not disagree with what CWSR publishes for the same wave, or
        // rocm-dbgapi correlates the stopped wave to the wrong workgroup.
        //
        // On the profiles where the SPI puts workgroup ids in TTMP6/7/9,
        // init_wavefront_regs() already seeded them at dispatch and trap entry
        // has nothing to add -- zeroing TTMP9 here destroyed one of them. The
        // rest use the gfx9 layout, TTMP8/9/10 = workgroup id x/y/z, which is
        // also what CWSR serializes (cwsr.cpp writes wg_coord into ttmp[8..10]).
        // Writing the flat wg_id() into TTMP8 disagreed with that too.
        if (!properties.uses_ttmp_workgroup_ids) {
          const auto &wg = active->wg_coord();
          active->set_ttmp(8, wg[0]);
          active->set_ttmp(9, wg[1]);
          active->set_ttmp(10, wg[2]);
        }
        if (!uses_split_wave_state) {
          active->set_ttmp(11, ((active->aql_packet_id() & 0x1FFFFFFu) << 6) |
                                   (active->wave_in_group() & 0x3Fu));
          active->set_ttmp(12, saved_status);
          const uint32_t debug_enabled = config->debug_enabled ? (1u << 23) : 0u;
          active->set_ttmp(13, (active->ttmp(13) & ~(1u << 23)) | debug_enabled);
        }
        active->set_ttmp(14, static_cast<uint32_t>(config->tma));
        active->set_ttmp(15, static_cast<uint32_t>(config->tma >> 32));
        active->set_trap_id(trap_id);
        active->set_trap_saved_status(saved_status);
        active->set_trap_saved_exec(active->exec());
        active->set_trap_interrupt_sent(false);
        // A fresh handler entry owns the halt state from here on; a marker left
        // over from a previous stop would attribute this entry's HALT to an
        // s_sendmsghalt that has already been resumed past.
        active->set_self_halted(false);
        active->set_in_trap_handler(true);
        if (uses_split_wave_state)
          active->set_status_raw(saved_status | kPrivilegedStatusBit);
        active->pc = config->tba;
        delete inst;
        return;
      }
    }

    // With no configured TBA, or for a parked s_trap executed by TBA code,
    // retire the instruction without inventing a host-side trap handler.
    active->pc += inst_size;
    delete inst;
    return;
  }

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

  // Sampled before execute so the trap-return test below can be a state
  // transition rather than a per-ISA mnemonic list. See its use.
  const bool was_in_trap_handler = active->in_trap_handler();

  execute_instruction(inst, *active);

  if (active->instruction_execution_failed()) {
    const InstructionExecutionError error = active->instruction_execution_error();
    const std::string failure = std::format("CU {}: wf{} could not execute {} at pc={:#x}: {}",
                                            this->name(), active->wf_id(), inst->mnemonic(),
                                            active->pc, instruction_execution_error_name(error));
    util::Logger::warn(failure);
    if (auto *sim_engine = this->engine())
      sim_engine->request_exit(failure, /*code=*/1);
    active->halt();
    delete inst;
    return;
  }

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

  // Capture debugger probe info before the pipeline consumes the instruction.
  // The checks run after the PC advances (below) so that a wave stopped on a
  // watchpoint or memory fault resumes past the access instead of re-executing
  // it. Gated on debug_active_ so non-debugged runs pay no per-access cost.
  const bool debug_probe = debug_active_.load(std::memory_order_relaxed) && inst->is_memory_op() &&
                           inst->data() &&
                           (inst->data()->tag() == GLOBAL_MEM || inst->data()->tag() == SCALAR_MEM);
  std::vector<uint64_t> dbg_addrs;
  uint32_t dbg_bytes = 0;
  bool dbg_is_write = false;
  bool dbg_is_atomic = false;
  if (debug_probe) {
    if (inst->data()->tag() == SCALAR_MEM) {
      auto &d = *inst->data_as<ScalarMemState>();
      dbg_is_write = !d.is_load;
      dbg_bytes = std::max(1u, d.num_dwords * d.elem_size);
      dbg_addrs.push_back(d.addr);
    } else {
      auto &d = *inst->data_as<VectorMemState>();
      dbg_is_atomic = (d.atomic_op != AtomicOp::NONE);
      dbg_is_write = !d.is_load || dbg_is_atomic;
      // std::max on both arms: VectorMemState::elem_size defaults to 0 and the
      // range check below reports a zero-sized range as *unmapped*, so an
      // atomic whose decoder left elem_size unset would fault on every lane of
      // a perfectly mapped buffer.
      dbg_bytes = std::max(1u, dbg_is_atomic ? d.elem_size : d.num_elems * d.elem_size);
      dbg_addrs.reserve(d.wf_size);
      for (uint32_t lane = 0; lane < d.wf_size; ++lane)
        if (d.lane_mask & (1ULL << lane))
          dbg_addrs.push_back(d.per_lane_addr[lane]);
    }
  }
  // One predicate for the whole access, used both to decide that this
  // instruction faulted and to pick the address reported to the debugger below.
  // The two must agree: if the decision said "faulted" and the report loop then
  // found no faulting address, the access would be dropped with nothing
  // delivered. Checking only the first byte let a multi-byte load, store or
  // atomic that starts near the end of a mapped page run off it unnoticed, so
  // the whole [addr, addr + dbg_bytes) range is validated.
  const uint32_t dbg_vmid = active->process_id();
  auto access_faults = [&](uint64_t addr) {
    const bool shared_address = active->shared_aperture_base() != 0 &&
                                addr >= active->shared_aperture_base() &&
                                addr <= active->shared_aperture_limit();
    return !shared_address && !memory_->is_range_mapped(addr, dbg_bytes, dbg_vmid);
  };
  // Locate the faulting address once. The report loop below resumes from this
  // iterator rather than rescanning: each access_faults() call is a page-table
  // walk per touched page, and dbg_addrs holds one entry per active lane.
  auto first_fault = dbg_addrs.end();
  if (debug_probe && memory_violation_handler_)
    first_fault = std::ranges::find_if(dbg_addrs, access_faults);
  const bool debug_memory_fault = first_fault != dbg_addrs.end();

  // The trap return is whatever instruction just left the handler. Asking the
  // wave rather than matching mnemonics keeps this ISA-agnostic: the spelling
  // differs per ISA (gfx1250 uses s_rfe_i64), and a hard-coded list silently
  // stops calling notify_trap_complete() for any spelling it misses -- KFD is
  // never told the handler returned and the debugger hangs. Only the generated
  // s_rfe body clears the flag.
  const bool trap_return = was_in_trap_handler && !active->in_trap_handler();

  // A faulting access is discarded only if the debugger actually claims the
  // fault. debug_active_ is CU-wide, so the probe also fires for waves of a
  // process nobody is debugging; on_wave_memory_violation() declines those, and
  // dropping the instruction anyway silently lost a store, or left a load's
  // destination registers stale, in an undebugged process.
  //
  // The report has to be made against the resume PC -- a wave the handler
  // serializes must come back past the access, not re-execute it -- so the PC
  // is advanced for the duration of the report and put back if the fault goes
  // unclaimed. Everything downstream then sees exactly the ordering it saw
  // before: the instruction is routed at the issue PC and the PC advances
  // afterwards.
  const uint64_t issue_pc = active->pc;
  bool fault_claimed = false;
  if (debug_memory_fault && !active->debug_halted()) {
    active->pc += inst_size;
    // first_fault is already known to fault, so it is reported without a second
    // range walk; only the lanes after it still have to be tested.
    for (auto it = first_fault; it != dbg_addrs.end(); ++it) {
      if ((it == first_fault || access_faults(*it)) &&
          memory_violation_handler_(*active, *it, dbg_is_write)) {
        fault_claimed = true;
        break;
      }
    }
    // Only undo our own advance. A declining handler is still free to have
    // moved the wave (entering a trap handler, for instance); clobbering that
    // would resume the application with the handler's state half-installed.
    if (!fault_claimed && active->pc == issue_pc + inst_size)
      active->pc = issue_pc;
  }

  if (fault_claimed) {
    delete inst;
    return;
  }
  if (inst->is_memory_op()) {
    if (!inst->data()) {
      // A memory execute path can intentionally reject an invalid complete
      // register operand before constructing pipeline state. Treat that as a
      // fully suppressed instruction: no route callback, wait-counter update,
      // or memory transaction is permitted.
      delete inst;
    } else {
      if (inst->data()->tag() == GLOBAL_MEM) {
        auto *d = inst->data_as<VectorMemState>();
        d->issue_pc = active->pc;
      }
      route_memory_inst(inst, *active);
    }
  } else
    delete inst;

  active->pc += inst_size;

  // Deliver on the causes this instruction raised, not on TRAPSTS changing.
  // TRAPSTS.EXCP is sticky and nothing in the model clears it between
  // instructions, so a rising-edge test went permanently quiet for any cause
  // whose bit was already latched: raise a cause with its MODE.EXCP_EN bit
  // clear, enable it, repeat the operation, and the second occurrence -- the
  // one hardware traps on -- was never reported. The same silence followed a
  // first occurrence the handler declined, which is every occurrence before a
  // debugger attaches.
  // Masks come from alu_exceptions.h, which is also what the classifiers and
  // the generated call sites use. A local copy here would keep checking the
  // old bits if the EXCP set ever widened.
  const uint32_t new_alu_causes = active->pending_alu_causes() & kAluExceptionTrapstsMask;
  const uint32_t enabled_alu_causes = alu_exception_trap_enables(*active);
  if ((new_alu_causes & enabled_alu_causes) != 0 && alu_exception_handler_ &&
      alu_exception_handler_(*active))
    return;

  // s_rfe follows the same target-minus-size convention as other control-flow
  // instructions. Publish the handler-driven stop only after the common PC
  // increment has produced the architectural return PC for CWSR serialization.
  // EXEC is put back by s_rfe itself, for every return and not just this one.
  if (trap_return && active->debug_halted() && active->trap_interrupt_sent())
    notify_trap_complete(*active);

  // Watchpoints, after the access completed and the PC advanced (so the
  // serialized wave resumes at the next instruction). A memory fault is more
  // severe than a watchpoint and wins if both would fire on the same
  // instruction, which it does by returning above before reaching here.
  if (debug_probe && !active->debug_halted() && watchpoint_handler_) {
    for (uint64_t addr : dbg_addrs)
      if (watchpoint_handler_(*active, addr, dbg_bytes, dbg_is_write, dbg_is_atomic))
        break;
  }
}

bool ComputeUnitCore::step() {
  // A wave reaching s_endpgm in this loop retires its workgroup; the guard sends
  // the CP its completion after the lock is released. See WaveStateGuard.
  WaveStateGuard wave_state_lock(*this);
  update_wf_states();

  for (auto &wf : wfs_) {
    if (wf->state() == WfState::RUNNING && !wf->debug_paused()) {
      // Burn down an in-flight S_SLEEP before issuing anything else. A
      // single-step request cancels the remainder instead of spending the
      // debugger's one step on it, which would look like a hung wave.
      if (wf->sleep_cycles() != 0) {
        if (!wf->debug_single_step()) {
          wf->tick_sleep();
          continue;
        }
        wf->set_sleep_cycles(0);
      }
      const bool single_step = wf->debug_single_step();
      issue_instruction(wf.get());
      if (single_step && !wf->in_trap_handler() && !wf->debug_halted() && single_step_handler_)
        single_step_handler_(*wf);
    }
  }

  ++step_count_;
  if constexpr (util::Logger::group_enabled(util::Logger::GROUP_CP)) {
    if ((step_count_ & 0xFFFFF) == 0) {
      util::Logger::cp([&](auto &os) {
        os << std::format("CU[{}] steps={}M", full_path(), step_count_ >> 20);
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

  return has_runnable_wfs();
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
