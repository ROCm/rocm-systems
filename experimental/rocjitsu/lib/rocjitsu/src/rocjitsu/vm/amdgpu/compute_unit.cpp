// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/compute_unit.h"

#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "util/except.h"

#include <cassert>
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
  switch (config.arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    switch (exec_mode) {
    case simdojo::ExecMode::FUNCTIONAL:
      return std::make_unique<IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, cdna3::Isa>>(
          std::move(name), config, memory, l2);
    case simdojo::ExecMode::CLOCKED:
      return std::make_unique<IsaExecComputeUnit<simdojo::ExecMode::CLOCKED, cdna3::Isa>>(
          std::move(name), config, memory, l2);
    }
    break;
  case ROCJITSU_CODE_ARCH_CDNA4:
    switch (exec_mode) {
    case simdojo::ExecMode::FUNCTIONAL:
      return std::make_unique<IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, cdna4::Isa>>(
          std::move(name), config, memory, l2);
    case simdojo::ExecMode::CLOCKED:
      return std::make_unique<IsaExecComputeUnit<simdojo::ExecMode::CLOCKED, cdna4::Isa>>(
          std::move(name), config, memory, l2);
    }
    break;
  default:
    break;
  }
  throw std::runtime_error("Unsupported architecture for ComputeUnit");
}

Wavefront *ComputeUnitCore::dispatch_wf(uint32_t wg_id, uint64_t pc, uint32_t sgprs,
                                        uint32_t vgprs) {
  assert(wfs_.size() == config_.num_wf_slots && "wavefront slots not properly initialized");
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

void ComputeUnitCore::retire_halted_wfs() {
  for (auto &w : wfs_) {
    if (w->is_halted() && w->sgpr_alloc().count > 0) {
      sgpr_file_.free(w->sgpr_alloc().base);
      free_vgprs(w->vgpr_alloc().base);
      w->reset();
    }
  }
}

void ComputeUnitCore::tick_pipelines() {
  scalar_mem_pipeline_.tick();
  global_mem_pipeline_.tick();
  local_mem_pipeline_.tick();
}

void ComputeUnitCore::route_memory_inst(std::unique_ptr<Instruction> inst, Wavefront &wf) {
  switch (inst->data()->tag()) {
  case SCALAR_MEM:
    scalar_mem_pipeline_.issue(std::move(inst), wf);
    break;
  case LOCAL_MEM:
    local_mem_pipeline_.issue(std::move(inst), wf);
    break;
  case GLOBAL_MEM:
    global_mem_pipeline_.issue(std::move(inst), wf);
    break;
  default:
    break;
  }
}

bool ComputeUnitCore::step() {
  tick_pipelines();

  if (!has_active_wfs())
    return false;

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
    retire_halted_wfs();
    return has_active_wfs();
  }

  rj_code_binary_inst_t words[4];
  for (int i = 0; i < 4; ++i)
    words[i] = memory_->fetch32(active->pc + i * 4);
  std::unique_ptr<Instruction> inst;
  try {
    inst = decoder_->decode(words);
  } catch (const util::InvalidInst &) {
    active->halt();
    return has_active_wfs();
  }
  if (!inst) {
    active->halt();
    return has_active_wfs();
  }

  int inst_size_signed = inst->size();
  assert(inst_size_signed > 0 && "instruction size must be positive");
  auto inst_size = static_cast<uint64_t>(inst_size_signed);

  execute_instruction(inst.get(), *active);

  if (inst->is_memory_op())
    route_memory_inst(std::move(inst), *active);

  // Advance PC past the current instruction. Branch execute() methods are required
  // to account for this by computing: wf.pc = target - inst_size, so the net result
  // after this advance is the correct branch target. Non-taken conditional branches
  // leave wf.pc unchanged, so the +inst_size advance correctly moves to the next
  // instruction.
  active->pc += inst_size;

  return has_active_wfs();
}

template class IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, cdna3::Isa>;
template class IsaExecComputeUnit<simdojo::ExecMode::CLOCKED, cdna3::Isa>;
template class IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, cdna4::Isa>;
template class IsaExecComputeUnit<simdojo::ExecMode::CLOCKED, cdna4::Isa>;

} // namespace amdgpu
} // namespace rocjitsu
