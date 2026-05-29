// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "plugins/cycle_model_plugin.h"

#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_descriptor.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "simdojo/sim/simulation.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#ifndef CYCLE_MODEL_CDNA4_JSON
#define CYCLE_MODEL_CDNA4_JSON "configs/cycle_model/cdna4.json"
#endif

namespace rocjitsu {
namespace amdgpu {

using cycle_model::InstrKind;
using cycle_model::InstrRegSet;
using cycle_model::PendingEvent;
using cycle_model::PendingKind;

CycleModelPlugin::CycleModelPlugin() : ExecutionPlugin("cycle_model") {}
CycleModelPlugin::~CycleModelPlugin() = default;

void CycleModelPlugin::onInit() { enabled_ = std::getenv("RJ_CYCLE") != nullptr; }
void CycleModelPlugin::onShutdown() {}

// --- clock + model resolution ----------------------------------------------

cycle_model::Tick CycleModelPlugin::now_ticks(Wavefront &wf) {
  ComputeUnitCore &cu = wf.cu();
  return cu.engine()->context(cu.partition_id()).current_tick();
}

cycle_model::UarchConfig &CycleModelPlugin::config_for_arch(uint32_t gfx_target_version) {
  auto it = cfg_by_arch_.find(gfx_target_version);
  if (it != cfg_by_arch_.end())
    return it->second;
  // M1: single cdna4 config regardless of arch. Real gfx_target_version-based
  // selection across cdna3/4/rdna lands with the multi-arch config set.
  return cfg_by_arch_
      .emplace(gfx_target_version, cycle_model::load_uarch_config(CYCLE_MODEL_CDNA4_JSON))
      .first->second;
}

CuCycleModel *CycleModelPlugin::cycle_model_for(Wavefront &wf) {
  const ComputeUnitCore *cu = &wf.cu();
  auto it = cu_models_.find(cu);
  if (it != cu_models_.end())
    return it->second;

  CuCycleModel *found = nullptr;
  for (auto &child : wf.cu().children()) {
    if (auto *cm = dynamic_cast<CuCycleModel *>(child.get())) {
      found = cm;
      break;
    }
  }
  if (found && !found->configured())
    found->configure(config_for_arch(/*M1-hardcoded*/ 0));
  cu_models_.emplace(cu, found);

  // Lazily configure the Xcd's shared MemSysCycleModel (idempotent), resolved
  // via CU -> ShaderEngine -> Xcd. Cross-CU L2/HBM contention is modeled there.
  // Group this CU's CuCycleModel under that MemSys for the per-XCD idle reset check.
  MemSysCycleModel *ms = nullptr;
  if (simdojo::Node *se = wf.cu().parent()) {
    if (auto *xcd = dynamic_cast<Xcd *>(se->parent())) {
      for (auto &child : xcd->children()) {
        if (auto *m = dynamic_cast<MemSysCycleModel *>(child.get())) {
          ms = m;
          break;
        }
      }
      if (ms) {
        if (!ms->configured())
          ms->configure(config_for_arch(/*M1-hardcoded*/ 0));
        mem_sys_.emplace(static_cast<const void *>(xcd), ms);
        if (found)
          cus_of_memsys_[ms].push_back(found);
      }
    }
  }
  // A CU the driver wired to a MemSys but whose MemSys we could not resolve/configure
  // would have its MemReqMsgs silently dropped at the MemSys (-> hang). Surface once.
  if (found && !ms) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      std::fprintf(stderr,
                   "[rocjitsu] cycle: could not resolve the shared MemSysCycleModel for a CU "
                   "(CU->ShaderEngine->Xcd walk failed); shared-memory timing will be wrong.\n");
    }
  }
  return found;
}

// A CU is idle when its model has no pending/in-flight work. An unconfigured CU is
// trivially idle. Used to gate per-dispatch resets so an overlapping in-flight dispatch
// is never wiped.
static bool cu_idle(CuCycleModel *cm) { return !cm->configured() || !cm->model().has_work(); }

// --- Instruction -> InstrEvent ---------------------------------------------

InstrKind CycleModelPlugin::kind_of(const Instruction &inst) {
  if (inst.is_mfma())
    return InstrKind::MFMA;
  std::string_view m = inst.mnemonic();
  if (inst.is_memory_op()) {
    if (m.starts_with("ds_"))
      return InstrKind::LDS;
    if (m.starts_with("s_"))
      return InstrKind::SMEM; // s_load / s_buffer
    if (m.starts_with("flat_"))
      return InstrKind::FLAT;
    if (m.starts_with("global_"))
      return InstrKind::GLOBAL;
    return InstrKind::VMEM; // buffer_/tbuffer_/image_
  }
  if (m.starts_with("v_"))
    return InstrKind::VALU;
  if (m.starts_with("s_"))
    return InstrKind::SALU;
  return InstrKind::OTHER;
}

static void add_ref(const RegisterRef &r, InstrRegSet &rs, bool is_read) {
  for (uint16_t w = 0; w < r.width; ++w) {
    uint16_t idx = static_cast<uint16_t>(r.index + w);
    switch (r.cls) {
    case RegClass::VGPR:
      (is_read ? rs.vgprs_read : rs.vgprs_written).push_back(idx);
      break;
    case RegClass::ACC_VGPR:
      (is_read ? rs.vgprs_read : rs.vgprs_written).push_back(static_cast<uint16_t>(idx + 256));
      break; // unified VGPR space
    case RegClass::SGPR:
      (is_read ? rs.sgprs_read : rs.sgprs_written).push_back(idx);
      break;
    case RegClass::VCC:
      (is_read ? rs.reads_vcc : rs.writes_vcc) = true;
      break;
    case RegClass::EXEC:
      (is_read ? rs.reads_exec : rs.writes_exec) = true;
      break;
    case RegClass::SCC:
      (is_read ? rs.reads_scc : rs.writes_scc) = true;
      break;
    case RegClass::M0:
      (is_read ? rs.reads_m0 : rs.writes_m0) = true;
      break;
    default:
      break; // FLAT_SCRATCH/TTMP/PC etc. — not RAW-tracked in M1
    }
  }
}

void CycleModelPlugin::fill_reg_set(const Instruction &inst, InstrRegSet &rs) {
  for (int i = 0; i < inst.num_src_operands(); ++i)
    if (const Operand *op = inst.src_operand(i))
      if (auto r = op->to_register_ref())
        add_ref(*r, rs, /*is_read=*/true);
  for (int i = 0; i < inst.num_dst_operands(); ++i)
    if (const Operand *op = inst.dst_operand(i))
      if (auto r = op->to_register_ref())
        add_ref(*r, rs, /*is_read=*/false);
}

// Memory op -> cycle-domain wait-counter slot. Mirrors rocjitsu's MemoryPipeline
// routing (memory_pipeline.h): vector memory -> VMCNT, scalar + LDS -> LGKMCNT.
cycle_model::WaitCounter CycleModelPlugin::wcnt_slot_of(InstrKind k) {
  switch (k) {
  case InstrKind::SMEM:
  case InstrKind::LDS:
    return cycle_model::WaitCounter::LGKMCNT;
  default:
    return cycle_model::WaitCounter::VMCNT; // VMEM/FLAT/GLOBAL
  }
}

// Pack the functional sim's computed per-lane addresses into the rocjitsu-free
// MemAccess the cycle-model lib coalesces. mem_descriptor() hides the tag
// dispatch + downcast; nullopt (non-memory / unknown tag) leaves the default
// (cached) MemAccess untouched.
static void fill_mem_access(const Instruction &inst, cycle_model::MemAccess &m) {
  auto desc = mem_descriptor(inst);
  if (!desc)
    return;
  m.lane_mask = desc->lane_mask;
  m.elem_bytes = desc->elem_bytes;
  m.mtype = desc->mtype;
  m.non_temporal = desc->non_temporal;
  for (size_t i = 0; i < desc->per_lane_addr.size(); ++i)
    m.lane_addr[i] = desc->per_lane_addr[i];
}

PendingEvent CycleModelPlugin::make_pending_event(const Instruction &inst, Wavefront &wf) {
  PendingEvent e;
  e.metadata_ready = true;
  if (inst.is_waitcnt()) {
    // Post-execute, wf.wait_target() holds this s_waitcnt's decoded thresholds.
    // Copy the funcsim's named fields into the cycle-model slot array.
    e.kind = PendingKind::WaitcntGate;
    const WaitTarget &wt = wf.wait_target();
    auto &t = e.waitcnt.target.t;
    using WC = cycle_model::WaitCounter;
    t[cycle_model::idx(WC::VMCNT)] = wt.vmcnt;
    t[cycle_model::idx(WC::LGKMCNT)] = wt.lgkmcnt;
    t[cycle_model::idx(WC::EXPCNT)] = wt.expcnt;
    t[cycle_model::idx(WC::VSCNT)] = wt.vscnt;
    t[cycle_model::idx(WC::DSCNT)] = wt.dscnt;
    t[cycle_model::idx(WC::KMCNT)] = wt.kmcnt;
  } else if (inst.is_barrier()) {
    e.kind = PendingKind::BarrierGate;
    e.barrier.wg_id = wf.wg_id();
  } else {
    e.kind = PendingKind::Instruction;
    e.instr.mnemonic = std::string(inst.mnemonic());
    e.instr.kind = kind_of(inst);
    fill_reg_set(inst, e.instr.regs);
    if (cycle_model::ArchModel::is_memory(e.instr.kind)) {
      e.instr.wcnt = wcnt_slot_of(e.instr.kind);
      e.instr.mem = std::make_unique<cycle_model::MemAccess>();
      fill_mem_access(inst, *e.instr.mem); // per-lane addresses for cache submodels
    }
  }
  return e;
}

// --- hooks ------------------------------------------------------------------

void CycleModelPlugin::beforeAmdgpuExecuteInstruction(uint64_t, const Instruction &, Wavefront &) {}

void CycleModelPlugin::afterAmdgpuExecuteInstruction(uint64_t, const Instruction &inst,
                                                     Wavefront &wf) {
  if (!enabled_)
    return;
  CuCycleModel *cm = cycle_model_for(wf);
  if (!cm || !cm->configured())
    return;
  CycleWavefrontState *s = state_of(wf);
  if (!s)
    return;
  s->st.pending.push_back(make_pending_event(inst, wf)); // FIFO, program order
  cm->resume_clock(now_ticks(wf));                       // wake the clock if idle
}

void CycleModelPlugin::onAmdgpuRouteMemoryInstruction(const Instruction &, Wavefront &) {}

void CycleModelPlugin::onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &) {}

void CycleModelPlugin::onAmdgpuDispatchExecutionBegin(uint32_t) {
  if (!enabled_)
    return;
  // Reset-per-dispatch (v0): each kernel starts with cold caches / full MSHR pool /
  // idle bandwidth queues, so per-kernel numbers are not polluted by the prior
  // dispatch's warm state. cu_cycle stays monotonic (tick->cycle mapping depends on it).
  // But reset ONLY components that are IDLE here: a still-busy CU/MemSys belongs to an
  // overlapping in-flight dispatch, and wiping its L1/MSHR/rid_owner_ would orphan that
  // dispatch's in_flight requests (-> hang). For the non-overlapping case the prior
  // dispatch has fully drained at this point, so this resets everything as before.
  // On the very first dispatch both maps are empty (no CU resolved yet); a
  // freshly-constructed model is already cold, so resetting nothing is correct.
  for (auto &kv : cu_models_)
    if (kv.second && cu_idle(kv.second))
      kv.second->reset_memory();
  // Shared L2/HBM: reset a MemSys only when every CU feeding it is idle (no overlapping
  // dispatch is using its warmth). Per-XCD scope replaces the old single global ref-count.
  for (auto &kv : cus_of_memsys_) {
    MemSysCycleModel *ms = kv.first;
    if (!ms || !ms->configured())
      continue;
    bool all_idle = true;
    for (CuCycleModel *cm : kv.second)
      if (!cu_idle(cm)) {
        all_idle = false;
        break;
      }
    if (all_idle)
      ms->reset();
  }
}

void CycleModelPlugin::onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) {
  if (!enabled_)
    return;
  // The idle gate at dispatch-begin handles per-kernel reset, and the CuCycleModel clock
  // drains the ArchModel naturally (ticks while has_work()). Read-only counter dump: cu_cycle
  // is monotonic across dispatches (never reset), so we report the absolute CU clock plus the
  // delta since the prior dispatch end. RJ_CYCLE_DUMP=0 silences the per-dispatch line.
  const char *dump_env = std::getenv("RJ_CYCLE_DUMP");
  if (dump_env != nullptr && std::string_view(dump_env) == "0")
    return;
  uint32_t n_cu = 0;
  uint64_t max_cycle = 0, sum_delta = 0, sched_stall = 0, pipe_busy = 0;
  for (auto &kv : cu_models_) {
    CuCycleModel *cm = kv.second;
    if (!cm || !cm->configured())
      continue;
    // Drain the model's pending FIFOs / in-flight tail so cu_cycle reflects all of
    // this dispatch's work. The engine clock only delivers sparse edges to the cu_clk
    // domain (functional sim collapses time), so without this the model under-counts.
    cm->model().drive_to_quiescence_passive();
    cycle_model::CUState &cu = cm->model().cu();
    ++n_cu;
    max_cycle = std::max(max_cycle, cu.cu_cycle);
    uint64_t &prev = last_cu_cycle_[cm];
    sum_delta += (cu.cu_cycle >= prev) ? (cu.cu_cycle - prev) : 0;
    prev = cu.cu_cycle;
    sched_stall += cu.stalls.sched;
    pipe_busy += cu.stalls.pipe_busy;
  }
  if (n_cu == 0)
    return;
  std::fprintf(stderr,
               "[rocjitsu] cycle: dispatch %u end — CUs=%u max_cu_cycle=%llu "
               "delta_this_dispatch=%llu sched_stall=%llu pipe_busy=%llu\n",
               dispatch_id, n_cu, (unsigned long long)max_cycle,
               (unsigned long long)sum_delta, (unsigned long long)sched_stall,
               (unsigned long long)pipe_busy);
}

void CycleModelPlugin::onAmdgpuWorkgroupDispatched(uint32_t, uint32_t, uint32_t vgpr_count,
                                                   uint32_t sgpr_count,
                                                   std::span<Wavefront *> wavefronts) {
  if (!enabled_ || wavefronts.empty())
    return;
  CuCycleModel *cm = cycle_model_for(*wavefronts.front());
  if (!cm || !cm->configured())
    return;
  cycle_model::ArchModel &m = cm->model();
  uint32_t nsimd = m.num_simds(), rr = 0;
  for (Wavefront *wf : wavefronts) {
    auto st = std::make_unique<CycleWavefrontState>();
    st->st.simd_id = (rr++) % nsimd;
    st->st.scoreboard.resize(vgpr_count + 256, sgpr_count); // +256 for AccVGPR space
    m.simd(st->st.simd_id).waves.push_back(&st->st);
    wf->set_plugin_state(slot_index(), std::move(st));
  }
  cm->resume_clock(now_ticks(*wavefronts.front()));
}

void CycleModelPlugin::onAmdgpuWavefrontHalted(Wavefront &wf) {
  if (!enabled_)
    return;
  CycleWavefrontState *st = state_of(wf);
  if (!st)
    return; // wave never dispatched through us
  CuCycleModel *cm = cycle_model_for(wf);
  if (!cm || !cm->configured())
    return;
  // Drive the CU model to quiescence with the full resident wave set still intact, so
  // this wave's (and its co-resident waves') pending FIFOs are actually issued and
  // counted. The simdojo cu_clk domain is starved in the passive LD_PRELOAD path (the
  // functional sim delivers a wave's whole instruction stream and halts it far faster
  // than the engine services clock edges), so without this drive the pending backlog is
  // discarded uncounted when drain_wave_at_halt removes the wave below. The passive
  // variant also force-lands deferred async memory (the MemSys clock is starved too),
  // so waitcnt gates clear and each wave's post-load tail is issued, not dropped.
  cm->model().drive_to_quiescence_passive();
  cm->model().drain_wave_at_halt(st->st);
}

void CycleModelPlugin::onAmdgpuBarrierResolved(std::span<Wavefront *> wavefronts) {
  if (!enabled_ || wavefronts.empty())
    return;
  CuCycleModel *cm = cycle_model_for(*wavefronts.front());
  if (!cm || !cm->configured())
    return;
  for (Wavefront *wf : wavefronts) {
    if (CycleWavefrontState *st = state_of(*wf))
      st->st.barrier_signals++; // positional BarrierGate clears on next step (counter: passive path queues >=2)
  }
  cm->resume_clock(
      now_ticks(*wavefronts.front())); // wake clock to replay now-unblocked barrier gates
}

} // namespace amdgpu
} // namespace rocjitsu
