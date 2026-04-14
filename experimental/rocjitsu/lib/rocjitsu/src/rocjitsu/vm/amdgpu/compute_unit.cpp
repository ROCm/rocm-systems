// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/compute_unit.h"

#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "util/debug_print.h"
#include "util/except.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace rocjitsu {
namespace amdgpu {

ComputeUnitCore::ComputeUnitCore(std::string name, const Config &config, GpuMemory *memory,
                                 L2Cache *l2, uint32_t wf_size)
    : simdojo::CompositeComponent(std::move(name)), config_(config), memory_(memory),
      wf_size_(wf_size), decoder_(Decoder::create(config.arch)), l2_(l2), l1_scalar_(l2),
      l1_vector_(l2), lds_(config.lds_size_kb), scalar_mem_pipeline_(&l1_scalar_),
      global_mem_pipeline_(&l1_vector_, l2), local_mem_pipeline_(&lds_) {
  if (!decoder_)
    throw std::runtime_error("Unsupported architecture for ComputeUnit decoder");

  // Enable pool allocation for the hot decode-execute path.
  // Instructions decoded during step() are always deleted before the CU
  // (and its decoder) are destroyed, so pool allocation is safe here.
  decoder_->enable_pool();

  wfs_.resize(config.num_wf_slots);
  sgpr_file_.init(config.num_wf_slots * config.sgprs_per_wf, config.sgprs_per_wf);

  // Completer port: CP sends dispatch activation messages here.
  cpl_ = add_port(std::make_unique<simdojo::Port>("cpl", 0, this, simdojo::PortDirection::IN,
                                                  simdojo::PortProtocol::DISPATCH));
  cpl_->set_handler([this](simdojo::Tick, simdojo::Message *) { activate(); });

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
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_CDNA1, cdna1::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_CDNA2, cdna2::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_CDNA3, cdna3::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_CDNA4, cdna4::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_RDNA1, rdna1::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_RDNA2, rdna2::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_RDNA3, rdna3::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_RDNA3_5, rdna3_5::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_RDNA4, rdna4::Isa);
  default:
    break;
  }
#undef ROCJITSU_CU_CASE
  throw std::runtime_error("Unsupported architecture for ComputeUnit");
}

Wavefront *ComputeUnitCore::dispatch_wf(uint32_t wg_id, uint64_t pc, uint32_t sgprs,
                                        uint32_t vgprs) {
  assert(wfs_.size() == config_.num_wf_slots && "wavefront slots not properly initialized");
  // Free register allocations from previously halted wavefronts before claiming a new slot.
  // Without this, completed wavefronts' SGPR/VGPR blocks remain allocated and each new
  // dispatch wastes a fresh block rather than reusing the freed ones.
  retire_halted_wfs();
  // Find an idle slot.
  size_t slot = config_.num_wf_slots;
  for (size_t i = 0; i < wfs_.size(); ++i) {
    if (wfs_[i]->is_halted()) {
      slot = i;
      break;
    }
  }
  if (slot >= config_.num_wf_slots)
    return nullptr;

  int32_t sgpr_base = sgpr_file_.allocate(sgprs);
  if (sgpr_base < 0)
    return nullptr;

  int32_t vgpr_base = allocate_vgprs(vgprs);
  if (vgpr_base < 0) {
    sgpr_file_.free(static_cast<uint32_t>(sgpr_base));
    return nullptr;
  }

  // Zero the allocated register blocks so reused slots don't inherit stale
  // values from previous kernel runs. Without this, wavefronts reading
  // uninitialized registers (e.g., user SGPRs not set by init_wavefront_regs)
  // see leftover data from the prior occupant.
  std::fill(&sgpr_file_[sgpr_base], &sgpr_file_[sgpr_base] + config_.sgprs_per_wf, 0u);
  std::memset(vgpr_data(static_cast<uint32_t>(vgpr_base)), 0,
              config_.vgprs_per_wf * wf_size_ * sizeof(uint32_t));

  // Invalidate the L1 scalar cache so this wavefront reads fresh kernel
  // arguments from L2/memory rather than stale lines from a prior kernel.
  // On real hardware, the driver issues s_dcache_inv at kernel launch.
  l1_scalar_.invalidate_all();

  auto *wf = wfs_[slot].get();
  wf->wg_id_ = wg_id;
  wf->pc = pc;
  wf->sgpr_alloc_ = {static_cast<uint32_t>(sgpr_base), sgprs};
  wf->vgpr_alloc_ = {static_cast<uint32_t>(vgpr_base), vgprs};
  wf->num_sgprs_ = sgprs;
  wf->num_vgprs_ = vgprs;
  wf->exec_ = wf_size_ == 64 ? ~0ULL : (1ULL << wf_size_) - 1;
  wf->vcc_ = 0;
  wf->m0_ = 0;
  wf->state_ = WfState::RUNNING;
  return wf;
}

size_t ComputeUnitCore::num_wfs() const {
  size_t count = 0;
  for (const auto &w : wfs_)
    if (w->sgpr_alloc().count > 0)
      ++count;
  return count;
}

void ComputeUnitCore::reset_all_wf() {
  for (auto &w : wfs_) {
    if (w->sgpr_alloc().count > 0) {
      sgpr_file_.free(w->sgpr_alloc().base);
      free_vgprs(w->vgpr_alloc().base);
    }
    w->reset();
  }
}

void ComputeUnitCore::retire_halted_wfs() {
  for (auto &w : wfs_) {
    if (w->is_halted() && w->sgpr_alloc().count > 0) {
      if (util::trace::enabled()) {
        static thread_local uint64_t retire_count = 0;
        static thread_local uint32_t max_wg = 0;
        ++retire_count;
        if (w->wg_id() > max_wg)
          max_wg = w->wg_id();
        if (retire_count <= 5 || (retire_count % 40) == 0)
          util::trace::print("CU ", this->name(), ": retire #", retire_count, " wg=", w->wg_id(),
                             " insts=", w->trace_inst_count_, " max_wg_seen=", max_wg);
      }
      sgpr_file_.free(w->sgpr_alloc().base);
      free_vgprs(w->vgpr_alloc().base);
      w->trace_inst_count_ = 0;
      w->reset();
    }
  }
}

bool ComputeUnitCore::can_accept_workgroup(uint32_t num_wfs) const {
  // Count free wavefront slots.
  uint32_t free_slots = 0;
  for (const auto &w : wfs_)
    if (w->is_halted())
      ++free_slots;
  if (free_slots < num_wfs) {
    util::trace::print("CU ", this->name(), " can_accept_wg: REJECT free_slots=", free_slots,
                       " < num_wfs=", num_wfs);
    return false;
  }

  // Check SGPR register blocks.
  uint32_t free_sgpr = sgpr_file_.free_block_count();
  if (free_sgpr < num_wfs) {
    util::trace::print("CU ", this->name(), " can_accept_wg: REJECT free_sgpr=", free_sgpr,
                       " < num_wfs=", num_wfs);
    return false;
  }

  // Check VGPR register blocks.
  uint32_t free_vgpr = free_vgpr_blocks();
  if (free_vgpr < num_wfs) {
    util::trace::print("CU ", this->name(), " can_accept_wg: REJECT free_vgpr=", free_vgpr,
                       " < num_wfs=", num_wfs);
    return false;
  }

  return true;
}

void ComputeUnitCore::tick_pipelines() {
  scalar_mem_pipeline_.tick();
  global_mem_pipeline_.tick();
  local_mem_pipeline_.tick();
}

void ComputeUnitCore::route_memory_inst(Instruction *inst, Wavefront &wf) {
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

void ComputeUnitCore::issue_scalar_mem(uint64_t addr, uint32_t dst_sgpr, uint32_t dword_count,
                                       Mtype /*mtype*/) {
  // Functional mode: synchronous read through L1 scalar cache.
  // Phase D will use mtype to select the correct cache path.
  l1_scalar_.load(addr, dword_count, &sgpr_file_[dst_sgpr]);
}

void ComputeUnitCore::issue_global_mem(const std::array<uint64_t, 64> &addrs, uint64_t lane_mask,
                                       uint32_t dst_vgpr, uint32_t dword_count, Mtype mtype) {
  // Functional mode: synchronous per-lane read through L1 vector cache.
  auto *dst = vgpr_data(dst_vgpr);
  l1_vector_.load(addrs.data(), lane_mask, /*elem_size=*/4, dword_count, dst, mtype,
                  /*non_temporal=*/false);
}

void ComputeUnitCore::issue_local_mem(const std::array<uint64_t, 64> &addrs, uint64_t lane_mask,
                                      uint32_t dst_vgpr, uint32_t dword_count) {
  // Functional mode: synchronous per-lane read from LDS.
  for (uint32_t lane = 0; lane < wf_size_; ++lane) {
    if (!(lane_mask & (1ULL << lane)))
      continue;
    for (uint32_t d = 0; d < dword_count; ++d)
      write_vgpr(dst_vgpr + d, lane, lds_.read32(static_cast<uint32_t>(addrs[lane] + d * 4)));
  }
}

bool ComputeUnitCore::step() {
  tick_pipelines();

  if (!has_active_wfs()) {
    // Final pipeline drain: complete deferred load writebacks for wavefronts
    // that halted on the previous step (after tick_pipelines ran but before
    // the next tick could drain them).
    tick_pipelines();
    return false;
  }

  size_t start = next_wf_;
  Wavefront *active = nullptr;
  for (size_t i = 0; i < wfs_.size(); ++i) {
    size_t idx = (start + i) % wfs_.size();
    if (wfs_[idx]->state() == WfState::RUNNING) {
      active = wfs_[idx].get();
      next_wf_ = (idx + 1) % wfs_.size();
      break;
    }
  }

  if (active == nullptr) {
    // Log wavefront states when no RUNNING wf found.
    if (util::trace::enabled()) {
      static thread_local uint64_t no_run_count = 0;
      if (++no_run_count <= 5) {
        uint32_t n_halt = 0, n_wait = 0, n_bar = 0;
        for (auto &w : wfs_) {
          if (w->state() == WfState::HALTED)
            ++n_halt;
          else if (w->state() == WfState::WAITCNT)
            ++n_wait;
          else if (w->state() == WfState::BARRIER)
            ++n_bar;
        }
        util::trace::print("CU ", this->name(), ": no RUNNING wf. halted=", n_halt,
                           " waitcnt=", n_wait, " barrier=", n_bar,
                           " has_active=", has_active_wfs());
      }
    }
    // Check for barrier resolution: if all non-halted wavefronts in a
    // workgroup are at BARRIER, resume them all to RUNNING.
    for (auto &w : wfs_) {
      if (w->state() != WfState::BARRIER)
        continue;
      uint32_t wg = w->wg_id();
      bool all_at_barrier = true;
      for (auto &w2 : wfs_) {
        if (w2->wg_id() == wg && w2->state() != WfState::HALTED &&
            w2->state() != WfState::BARRIER) {
          all_at_barrier = false;
          break;
        }
      }
      if (all_at_barrier) {
        for (auto &w2 : wfs_)
          if (w2->wg_id() == wg && w2->state() == WfState::BARRIER)
            w2->set_state(WfState::RUNNING);
      }
    }
    retire_halted_wfs();
    return has_active_wfs();
  }

  rj_code_binary_inst_t words[4];
  for (int i = 0; i < 4; ++i)
    words[i] = memory_->fetch32(active->pc + i * 4);

  // Trace: log wf instruction count when it halts (end of program).
  if (util::trace::enabled())
    active->trace_inst_count_++;

  // Trace v4 and instruction words at key PCs in the fill kernel
  if (util::trace::enabled() && active->pc == 0x4d00249258ULL) {
    // At the v_or_b32 (+058): log words and v4 BEFORE execute
    uint32_t vbase = active->vgpr_alloc().base;
    uint32_t v4_0 = read_vgpr(vbase + 4, 0);
    uint32_t v0_0 = read_vgpr(vbase + 0, 0);
    util::trace::print("VOR_BEFORE pc=0x", std::hex, active->pc, " w0=0x", words[0], " w1(lit)=0x",
                       words[1], std::dec, " v4[0]=", v4_0, " v0[0]=", v0_0,
                       " wf=", active->wf_id());
  }
  if (util::trace::enabled() && active->pc == 0x4d00249260ULL) {
    // After v_or (+060): log v4 AFTER execute
    uint32_t vbase = active->vgpr_alloc().base;
    uint32_t v4_0 = read_vgpr(vbase + 4, 0);
    util::trace::print("VOR_AFTER pc=0x", std::hex, active->pc, std::dec, " v4[0]=", v4_0,
                       " wf=", active->wf_id());
  }

  // One-time dump of the fill kernel code (after blit has copied it)
  if (util::trace::enabled() && active->pc == 0x4d00249200ULL) {
    static bool dumped = false;
    if (!dumped) {
      dumped = true;
      util::trace::print("FILL_RUNTIME_CODE at 0x", std::hex, active->pc, std::dec);
      for (int i = 0; i < 128; i += 4) {
        uint32_t w0 = memory_->fetch32(active->pc + i * 4);
        uint32_t w1 = memory_->fetch32(active->pc + i * 4 + 4);
        uint32_t w2 = memory_->fetch32(active->pc + i * 4 + 8);
        uint32_t w3 = memory_->fetch32(active->pc + i * 4 + 12);
        util::trace::print("  +", std::hex, i * 4, ": ", w0, " ", w1, " ", w2, " ", w3, std::dec);
      }
    }
  }

  Instruction *inst = nullptr;
  try {
    inst = decoder_->decode(words);
  } catch (const util::InvalidInst &e) {
    util::trace::print("CU ", this->name(), ": wf", active->wf_id(), " HALT(InvalidInst) pc=0x",
                       std::hex, active->pc, " words=[0x", words[0], ",0x", words[1], ",0x",
                       words[2], ",0x", words[3], "]", std::dec, " what=", e.what());
    active->halt();
    return has_active_wfs();
  }
  if (!inst) {
    util::trace::print("CU ", this->name(), ": wf", active->wf_id(), " HALT(null decode) pc=0x",
                       std::hex, active->pc, " words=[0x", words[0], ",0x", words[1], ",0x",
                       words[2], ",0x", words[3], "]", std::dec);
    active->halt();
    return has_active_wfs();
  }

  int inst_size_signed = inst->size();
  assert(inst_size_signed > 0 && "instruction size must be positive");
  auto inst_size = static_cast<uint64_t>(inst_size_signed);

  // Trace: dump instruction at PCs that produce the zero-fill stores
  if (util::trace::enabled() && (active->pc == 0x4d0024938cULL || active->pc == 0x4d00249304ULL ||
                                 active->pc == 0x4d00249324ULL || active->pc == 0x4d00249358ULL)) {
    util::trace::print("FILL_INST pc=0x", std::hex, active->pc, std::dec,
                       " mnem=", inst->mnemonic(), " exec=0x", std::hex, active->exec(), std::dec,
                       " wf=", active->wf_id());
  }

  execute_instruction(inst, *active);

  if (inst->is_memory_op()) {
    // Tag store with issue PC for debugging.
    if (inst->data() && inst->data()->tag() == GLOBAL_MEM) {
      auto *d = inst->data_as<VectorMemState>();
      d->issue_pc = active->pc;
    }
    route_memory_inst(inst, *active);
  } else
    delete inst;

  // Advance PC past the current instruction. Branch execute() methods are required
  // to account for this by computing: wf.pc = target - inst_size, so the net result
  // after this advance is the correct branch target. Non-taken conditional branches
  // leave wf.pc unchanged, so the +inst_size advance correctly moves to the next
  // instruction.
  active->pc += inst_size;

  return has_active_wfs();
}

// Explicit template instantiations for all 9 ISAs × 2 execution modes.
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

#undef ROCJITSU_CU_INSTANTIATE

} // namespace amdgpu
} // namespace rocjitsu
