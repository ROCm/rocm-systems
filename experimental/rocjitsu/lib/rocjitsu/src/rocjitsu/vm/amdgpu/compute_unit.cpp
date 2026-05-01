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
#include "util/except.h"
#include "util/log.h"

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
  // Free register allocations from previously halted wavefronts before claiming
  // a new slot. This is needed so SGPR/VGPR blocks can be reused. However, we
  // must NOT reset the LDS allocator here — that would zero next_lds_alloc_
  // between WF dispatches of the same WG, causing concurrent WGs to share
  // the same LDS base. The LDS reset is handled separately by the CP.
  retire_halted_wfs_no_lds_reset();
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

void ComputeUnitCore::retire_halted_wfs_no_lds_reset() {
  for (auto &w : wfs_) {
    if (w->is_halted() && w->sgpr_alloc().count > 0) {
      sgpr_file_.free(w->sgpr_alloc().base);
      free_vgprs(w->vgpr_alloc().base);
      w->trace_inst_count_ = 0;
      w->reset();
    }
  }
}

void ComputeUnitCore::retire_halted_wfs() {
  for (auto &w : wfs_) {
    if (w->is_halted() && w->sgpr_alloc().count > 0) {
      util::Logger::vm([&](auto &os) {
        static thread_local uint64_t retire_count = 0;
        static thread_local uint32_t max_wg = 0;
        if (w->wg_id() > max_wg)
          max_wg = w->wg_id();
        if (++retire_count <= 5 || (retire_count % 40) == 0)
          os << std::format("CU {}: retire #{} wg={} insts={} max_wg_seen={}", this->name(),
                            retire_count, w->wg_id(), w->trace_inst_count_, max_wg);
      });
      sgpr_file_.free(w->sgpr_alloc().base);
      free_vgprs(w->vgpr_alloc().base);
      w->trace_inst_count_ = 0;
      w->reset();
    }
  }
  if (!has_active_wfs()) {
    reset_lds_alloc();
  }
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
  plugin_group_->onAmdgpuRouteMemoryInstruction(*inst);
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
    util::Logger::vm([&](auto &os) {
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
        os << std::format("CU {}: no RUNNING wf. halted={} waitcnt={} barrier={} has_active={}",
                          this->name(), n_halt, n_wait, n_bar, has_active_wfs());
      }
    });
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
        plugin_group_->onAmdgpuBarrierResolved(wg);
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

  active->trace_inst_count_++;

  // No instruction-count safety valve — real kernels (Triton flash attention)
  // can legitimately execute hundreds of thousands of instructions per wavefront.
  // Infinite loops are detected via the dispatch logger showing no progress.

  // Trace v4 and instruction words at key PCs in the fill kernel.
  util::Logger::vm([&](auto &os) {
    uint32_t vbase = active->vgpr_alloc().base;
    if (active->pc == 0x4d00249258ULL) {
      os << std::format("VOR_BEFORE pc={:#x} w0={:#x} w1(lit)={:#x} v4[0]={} v0[0]={} wf={}",
                        active->pc, words[0], words[1], read_vgpr(vbase + 4, 0),
                        read_vgpr(vbase + 0, 0), active->wf_id());
    } else if (active->pc == 0x4d00249260ULL) {
      os << std::format("VOR_AFTER pc={:#x} v4[0]={} wf={}", active->pc, read_vgpr(vbase + 4, 0),
                        active->wf_id());
    } else if (active->pc == 0x4d00249200ULL) {
      static bool dumped = false;
      if (!dumped) {
        dumped = true;
        os << std::format("FILL_RUNTIME_CODE at {:#x}", active->pc);
        for (int i = 0; i < 128; i += 4)
          os << std::format(
              "\n[rj log VM]   +{:#x}: {:08x} {:08x} {:08x} {:08x}", i * 4,
              memory_->fetch32(active->pc + i * 4), memory_->fetch32(active->pc + i * 4 + 4),
              memory_->fetch32(active->pc + i * 4 + 8), memory_->fetch32(active->pc + i * 4 + 12));
      }
    }
  });

  Instruction *inst = nullptr;
  try {
    inst = decoder_->decode(words);
  } catch (const util::InvalidInst &e) {
    util::Logger::vm("CU ", this->name(), ": wf", active->wf_id(), " HALT(InvalidInst) pc=0x",
                     std::hex, active->pc, " words=[0x", words[0], ",0x", words[1], ",0x", words[2],
                     ",0x", words[3], "]", std::dec, " what=", e.what());
    active->halt();
    return has_active_wfs();
  }
  if (!inst) {
    util::Logger::vm("CU ", this->name(), ": wf", active->wf_id(), " HALT(null decode) pc=0x",
                     std::hex, active->pc, " words=[0x", words[0], ",0x", words[1], ",0x", words[2],
                     ",0x", words[3], "]", std::dec);
    active->halt();
    return has_active_wfs();
  }

  int inst_size_signed = inst->size();
  assert(inst_size_signed > 0 && "instruction size must be positive");
  auto inst_size = static_cast<uint64_t>(inst_size_signed);

  util::Logger::vm([&](auto &os) {
    if (active->pc == 0x4d0024938cULL || active->pc == 0x4d00249304ULL ||
        active->pc == 0x4d00249324ULL || active->pc == 0x4d00249358ULL)
      os << std::format("FILL_INST pc={:#x} mnem={} exec={:#x} wf={}", active->pc, inst->mnemonic(),
                        active->exec(), active->wf_id());
  });

  // Per-instruction trace: snapshot registers and flags for wf0 (and wf2 first 100).
  if constexpr (util::Logger::group_enabled(util::Logger::GROUP_VM)) {
    if (((active->wf_id() == 0 && active->trace_inst_count_ <= 2000) ||
         (active->wf_id() == 2 && active->trace_inst_count_ <= 100)) &&
        active->num_vgprs_ >= 32) {
      util::Logger::vm([&](auto &os) {
        uint32_t sb = active->sgpr_alloc().base;
        uint32_t vb = active->vgpr_alloc().base;
        os << std::format("{} wg[{}] wf[{}] EXECUTE #{} pc={:#x} {} w={:08x},{:08x}",
                          this->full_path(), active->wg_id(), active->wf_id(),
                          active->trace_inst_count_, active->pc, inst->mnemonic(), words[0],
                          words[1]);
        os << std::format(" s[0:7]={:x},{:x},{:x},{:x},{:x},{:x},{:x},{:x}"
                          " s[8:15]={:x},{:x},{:x},{:x},{:x},{:x},{:x},{:x}",
                          read_sgpr(sb), read_sgpr(sb + 1), read_sgpr(sb + 2), read_sgpr(sb + 3),
                          read_sgpr(sb + 4), read_sgpr(sb + 5), read_sgpr(sb + 6),
                          read_sgpr(sb + 7), read_sgpr(sb + 8), read_sgpr(sb + 9),
                          read_sgpr(sb + 10), read_sgpr(sb + 11), read_sgpr(sb + 12),
                          read_sgpr(sb + 13), read_sgpr(sb + 14), read_sgpr(sb + 15));
        os << std::format(
            " s[16:31]={:x},{:x},{:x},{:x},{:x},{:x},{:x},{:x}"
            ",{:x},{:x},{:x},{:x},{:x},{:x},{:x},{:x}"
            " v[0:7]L0={:x},{:x},{:x},{:x},{:x},{:x},{:x},{:x}"
            " v[8:15]L0={:x},{:x},{:x},{:x},{:x},{:x},{:x},{:x}"
            " scc={} vcc={:x} exec={:x}",
            read_sgpr(sb + 16), read_sgpr(sb + 17), read_sgpr(sb + 18), read_sgpr(sb + 19),
            read_sgpr(sb + 20), read_sgpr(sb + 21), read_sgpr(sb + 22), read_sgpr(sb + 23),
            read_sgpr(sb + 24), read_sgpr(sb + 25), read_sgpr(sb + 26), read_sgpr(sb + 27),
            read_sgpr(sb + 28), read_sgpr(sb + 29), read_sgpr(sb + 30), read_sgpr(sb + 31),
            read_vgpr(vb, 0), read_vgpr(vb + 1, 0), read_vgpr(vb + 2, 0), read_vgpr(vb + 3, 0),
            read_vgpr(vb + 4, 0), read_vgpr(vb + 5, 0), read_vgpr(vb + 6, 0), read_vgpr(vb + 7, 0),
            read_vgpr(vb + 8, 0), read_vgpr(vb + 9, 0), read_vgpr(vb + 10, 0),
            read_vgpr(vb + 11, 0), read_vgpr(vb + 12, 0), read_vgpr(vb + 13, 0),
            read_vgpr(vb + 14, 0), read_vgpr(vb + 15, 0), active->read_scc(), active->vcc(),
            active->exec());
        if (active->num_sgprs_ >= 80)
          os << std::format(
              " s[64:79]={:x},{:x},{:x},{:x},{:x},{:x},{:x},{:x}"
              ",{:x},{:x},{:x},{:x},{:x},{:x},{:x},{:x}",
              read_sgpr(sb + 64), read_sgpr(sb + 65), read_sgpr(sb + 66), read_sgpr(sb + 67),
              read_sgpr(sb + 68), read_sgpr(sb + 69), read_sgpr(sb + 70), read_sgpr(sb + 71),
              read_sgpr(sb + 72), read_sgpr(sb + 73), read_sgpr(sb + 74), read_sgpr(sb + 75),
              read_sgpr(sb + 76), read_sgpr(sb + 77), read_sgpr(sb + 78), read_sgpr(sb + 79));
      });
    }
  }

  plugin_group_->onAmdgpuExecuteInstruction(active->pc, *inst);

  execute_instruction(inst, *active);

  if constexpr (util::Logger::group_enabled(util::Logger::GROUP_VM)) {
    if (active->wf_id() == 0 && active->trace_inst_count_ <= 2000 && active->num_vgprs_ >= 32) {
      util::Logger::vm([&](auto &os) {
        uint32_t sb = active->sgpr_alloc().base;
        uint32_t vb = active->vgpr_alloc().base;
        os << std::format("RESULT  #{}"
                          " s[0:7]={:x},{:x},{:x},{:x},{:x},{:x},{:x},{:x}"
                          " s[8:15]={:x},{:x},{:x},{:x},{:x},{:x},{:x},{:x}",
                          active->trace_inst_count_, read_sgpr(sb), read_sgpr(sb + 1),
                          read_sgpr(sb + 2), read_sgpr(sb + 3), read_sgpr(sb + 4),
                          read_sgpr(sb + 5), read_sgpr(sb + 6), read_sgpr(sb + 7),
                          read_sgpr(sb + 8), read_sgpr(sb + 9), read_sgpr(sb + 10),
                          read_sgpr(sb + 11), read_sgpr(sb + 12), read_sgpr(sb + 13),
                          read_sgpr(sb + 14), read_sgpr(sb + 15));
        os << std::format(
            " s[16:31]={:x},{:x},{:x},{:x},{:x},{:x},{:x},{:x}"
            ",{:x},{:x},{:x},{:x},{:x},{:x},{:x},{:x}"
            " v[0:7]L0={:x},{:x},{:x},{:x},{:x},{:x},{:x},{:x}"
            " v[8:15]L0={:x},{:x},{:x},{:x},{:x},{:x},{:x},{:x}"
            " scc={} vcc={:x} exec={:x}",
            read_sgpr(sb + 16), read_sgpr(sb + 17), read_sgpr(sb + 18), read_sgpr(sb + 19),
            read_sgpr(sb + 20), read_sgpr(sb + 21), read_sgpr(sb + 22), read_sgpr(sb + 23),
            read_sgpr(sb + 24), read_sgpr(sb + 25), read_sgpr(sb + 26), read_sgpr(sb + 27),
            read_sgpr(sb + 28), read_sgpr(sb + 29), read_sgpr(sb + 30), read_sgpr(sb + 31),
            read_vgpr(vb, 0), read_vgpr(vb + 1, 0), read_vgpr(vb + 2, 0), read_vgpr(vb + 3, 0),
            read_vgpr(vb + 4, 0), read_vgpr(vb + 5, 0), read_vgpr(vb + 6, 0), read_vgpr(vb + 7, 0),
            read_vgpr(vb + 8, 0), read_vgpr(vb + 9, 0), read_vgpr(vb + 10, 0),
            read_vgpr(vb + 11, 0), read_vgpr(vb + 12, 0), read_vgpr(vb + 13, 0),
            read_vgpr(vb + 14, 0), read_vgpr(vb + 15, 0), active->read_scc(), active->vcc(),
            active->exec());
      });
    }
  }

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
