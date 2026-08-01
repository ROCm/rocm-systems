// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/compute_unit.h"

#include "rocjitsu/vm/amdgpu/command_processor.h"

#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "util/except.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>

namespace rocjitsu {
namespace amdgpu {

namespace {

[[nodiscard]] std::optional<uint32_t> resident_wave_id_capacity(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return 64u;
  case ROCJITSU_CODE_ARCH_GFX1250:
    return 128u;
  default:
    return std::nullopt;
  }
}

} // namespace

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
  free_wf_slot_bits_.assign((config.num_wf_slots + 63u) / 64u, ~uint64_t{0});
  if (!free_wf_slot_bits_.empty() && config.num_wf_slots % 64u != 0)
    free_wf_slot_bits_.back() = (uint64_t{1} << (config.num_wf_slots % 64u)) - 1u;
  free_wf_slot_count_ = config.num_wf_slots;
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
  if (const auto capacity = resident_wave_id_capacity(config.arch);
      capacity && config.num_wf_slots > *capacity) {
    throw std::invalid_argument("ComputeUnit num_wf_slots exceeds the resident-wave hardware ID "
                                "capacity for this architecture");
  }
  // Helper: instantiate the ISA-specific CU for the given execution mode.
#define ROCJITSU_CU_CASE(ARCH_ENUM, ISA_TYPE)                                                      \
  case ARCH_ENUM:                                                                                  \
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
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_GFX1250, gfx1250::Isa);
  default:
    break;
  }
#undef ROCJITSU_CU_CASE
  throw std::runtime_error("Unsupported architecture for ComputeUnit");
}

Wavefront *ComputeUnitCore::dispatch_wf(uint32_t wg_id, uint64_t pc, uint32_t num_sgprs,
                                        uint32_t num_vgprs, uint32_t dispatch_id) {
  assert(wfs_.size() == config_.num_wf_slots && "wavefront slots not properly initialized");
  // Halted wavefronts have already returned their slot and register blocks at
  // s_endpgm. The bitset preserves the historical lowest-slot-first assignment
  // without rescanning every wavefront object on each dispatch.
  size_t slot = config_.num_wf_slots;
  for (size_t word = 0; word < free_wf_slot_bits_.size(); ++word) {
    if (free_wf_slot_bits_[word] != 0) {
      slot = word * 64u + std::countr_zero(free_wf_slot_bits_[word]);
      break;
    }
  }

  // No free slot: fail the dispatch (like the register-allocation failures below)
  // rather than indexing wfs_ out of bounds. The CP normally gates placement on
  // can_accept_workgroup(), but returning nullptr is part of this API's contract
  // and must hold even when a caller dispatches directly to a full CU.
  if (slot >= config_.num_wf_slots)
    return nullptr;
  assert(free_wf_slot_count_ > 0 && "free wavefront slot count is inconsistent");

  int32_t sgpr_base = sgpr_file_.allocate(num_sgprs);
  if (sgpr_base < 0)
    return nullptr;

  int32_t vgpr_base = allocate_vgprs(num_vgprs);
  if (vgpr_base < 0) {
    sgpr_file_.free(static_cast<uint32_t>(sgpr_base));
    return nullptr;
  }

  // RegisterFile::allocate() clears the complete physical block before returning
  // it, including registers above the logical count. Do not clear both blocks a
  // second time here; high-volume dispatches allocate millions of wave contexts.

  // Hardware invalidates scalar-cache state at kernel launch, not between
  // workgroups of one dispatch. Direct standalone waves retain the conservative
  // historical behavior because they have no dispatch identity.
  if (dispatch_id == 0) {
    l1_scalar_.invalidate_all();
  } else if (!scalar_cache_dispatch_prepared_ || scalar_cache_dispatch_id_ != dispatch_id) {
    l1_scalar_.invalidate_all();
    scalar_cache_dispatch_id_ = dispatch_id;
    scalar_cache_dispatch_prepared_ = true;
  }

  auto *wf = wfs_[slot].get();
  wf->wg_id_ = wg_id;
  wf->set_dispatch_id(dispatch_id);
  wf->pc = pc;
  wf->sgpr_alloc_ = {static_cast<uint32_t>(sgpr_base), num_sgprs};
  wf->vgpr_alloc_ = {static_cast<uint32_t>(vgpr_base), num_vgprs};
  wf->num_sgprs_ = num_sgprs;
  wf->num_vgprs_ = num_vgprs;
  wf->exec_ = wf_size_ == 64 ? ~0ULL : (1ULL << wf_size_) - 1;
  wf->vcc_ = 0;
  wf->m0_ = 0;
  wf->set_apertures(shared_aperture_base_, shared_aperture_limit_, private_aperture_base_,
                    private_aperture_limit_);
  wf->state_ = WfState::RUNNING;
  wf->set_ready_cycle(cycle_counter_);
  wf->trace_inst_count_ = 0;
  free_wf_slot_bits_[slot / 64u] &= ~(uint64_t{1} << (slot % 64u));
  --free_wf_slot_count_;

  std::fill(sgpr_to_wave_.begin() + sgpr_base, sgpr_to_wave_.begin() + sgpr_base + num_sgprs, wf);
  fill_vgpr_to_wave(static_cast<uint32_t>(vgpr_base), vgpr_allocation_block_size(), wf);

  util::Logger::cp("DISPATCH_WF cu=", this->full_path(), " wf=", wf->wf_id(), " slot=", slot,
                   " pc=0x", std::hex, pc, std::dec, " wg=", wg_id, " pid=", wf->process_id());

  schedule_work();
  return wf;
}

size_t ComputeUnitCore::num_wfs() const {
  if (free_wf_slot_count_ > wfs_.size())
    throw std::logic_error("free wavefront slot count exceeds capacity");
  return wfs_.size() - free_wf_slot_count_;
}

void ComputeUnitCore::free_wavefront_resources(Wavefront &wf) {
  const bool was_allocated = wf.sgpr_alloc().count > 0;
  if (was_allocated) {
    sgpr_file_.free(wf.sgpr_alloc().base);
    free_vgprs(wf.vgpr_alloc().base);
    const uint32_t slot = wf.wf_id();
    assert((free_wf_slot_bits_[slot / 64u] & (uint64_t{1} << (slot % 64u))) == 0 &&
           "double-free of wavefront slot");
    free_wf_slot_bits_[slot / 64u] |= uint64_t{1} << (slot % 64u);
    ++free_wf_slot_count_;
    assert(free_wf_slot_count_ <= wfs_.size() && "free wavefront slot count overflow");
  }
  wf.trace_inst_count_ = 0;
  wf.reset();
}

void ComputeUnitCore::maybe_reset_lds_alloc() {
  if (!has_active_wfs() && !lds_allocation_pinned())
    reset_lds_alloc();
}

void ComputeUnitCore::release_wf(uint32_t dispatch_id, uint32_t wg_id) {
  auto key = wg_key(dispatch_id, wg_id);
  auto it = active_wgs_.find(key);
  if (it != active_wgs_.end() && --it->second == 0) {
    plugin_group_->onAmdgpuWorkgroupCompleted(dispatch_id, wg_id);
    active_wgs_.erase(it);
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
  for (const auto &w : wfs_)
    if (!w->is_halted() && w->dispatch_id() == dispatch_id && w->wg_id() == wg_id)
      free_wavefront_resources(*w);
  active_wgs_.erase(wg_key(dispatch_id, wg_id));
  maybe_reset_lds_alloc();
}

bool ComputeUnitCore::can_accept_workgroup(uint32_t num_wfs, uint32_t lds_bytes) const {
  if (free_wf_slot_count_ < num_wfs) {
    util::Logger::vm("CU ", this->name(), " can_accept_wg: REJECT free_slots=", free_wf_slot_count_,
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
    if (w->state() != WfState::BARRIER)
      continue;
    uint32_t did = w->dispatch_id();
    uint32_t wg = w->wg_id();
    bool all_at_barrier = true;
    for (auto &w2 : wfs_) {
      if (w2->dispatch_id() == did && w2->wg_id() == wg && w2->state() != WfState::HALTED &&
          w2->state() != WfState::BARRIER) {
        all_at_barrier = false;
        break;
      }
    }
    if (all_at_barrier) {
      std::vector<Wavefront *> barrier_wfs;
      for (auto &w2 : wfs_)
        if (w2->dispatch_id() == did && w2->wg_id() == wg && w2->state() == WfState::BARRIER)
          barrier_wfs.push_back(w2.get());
      plugin_group_->onAmdgpuBarrierResolved(std::span<Wavefront *>(barrier_wfs));
      for (auto *bwf : barrier_wfs) {
        bwf->set_state(WfState::RUNNING);
        bwf->set_ready_cycle(cycle_counter_);
      }
    }
  }
}

void ComputeUnitCore::issue_instruction(Wavefront *active, const FetchedInstruction &words) {
  active->trace_inst_count_++;

  Instruction *inst = nullptr;
  try {
    inst = decoder_->decode(words.data());
    ++decoded_instruction_count_;
  } catch (const util::InvalidInst &e) {
    util::Logger::vm("CU ", this->name(), ": wf", active->wf_id(), " HALT(InvalidInst) pc=0x",
                     std::hex, active->pc, " words=[0x", words[0], ",0x", words[1], ",0x", words[2],
                     ",0x", words[3], "]", std::dec, " what=", e.what());
    active->halt();
    return;
  }
  if (!inst) {
    util::Logger::vm("CU ", this->name(), ": wf", active->wf_id(), " HALT(null decode) pc=0x",
                     std::hex, active->pc, " words=[0x", words[0], ",0x", words[1], ",0x", words[2],
                     ",0x", words[3], "]", std::dec);
    active->halt();
    return;
  }

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
      uint32_t ssrc0_idx = words[0] & 0x7F;
      uint32_t sb = active->sgpr_alloc().base;
      uint64_t target = static_cast<uint64_t>(read_sgpr(sb + ssrc0_idx)) |
                        (static_cast<uint64_t>(read_sgpr(sb + ssrc0_idx + 1)) << 32);
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

  // Functional workloads commonly keep every resident wave at the same PC. Snapshot
  // instruction words once per step for peers in the same address space. Runtime code
  // patching occurs between simulator steps; instruction memory is stable while the
  // waves in one step execute. Decoding remains per wave because execute paths may
  // mutate instruction-owned operand state.
  struct StepFetchEntry {
    bool valid = false;
    uint32_t vmid = 0;
    uint64_t pc = 0;
    FetchedInstruction words{};
  };
  constexpr size_t kStepFetchEntries = 8;
  std::array<StepFetchEntry, kStepFetchEntries> step_fetches{};

  for (auto &wf : wfs_) {
    if (wf->state() != WfState::RUNNING)
      continue;

    const uint32_t vmid = wf->process_id();
    const uint64_t pc = wf->pc;
    const size_t fetch_index =
        ((pc >> 2u) ^ (static_cast<uint64_t>(vmid) * 0x9e3779b9u)) & (kStepFetchEntries - 1u);
    auto &entry = step_fetches[fetch_index];
    if (!entry.valid || entry.vmid != vmid || entry.pc != pc) {
      for (size_t i = 0; i < entry.words.size(); ++i)
        entry.words[i] = memory_->fetch32(pc + i * sizeof(entry.words[i]), vmid);
      ++fetched_instruction_count_;
      entry.valid = true;
      entry.vmid = vmid;
      entry.pc = pc;
    }
    issue_instruction(wf.get(), entry.words);
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
ROCJITSU_CU_INSTANTIATE(gfx1250::Isa);

#undef ROCJITSU_CU_INSTANTIATE

} // namespace amdgpu
} // namespace rocjitsu
