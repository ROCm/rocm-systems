// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cycle_model_plugin.h
/// @brief Adapter that drives the standalone cycle-model lib from rocjitsu's
/// ExecutionPlugin hook stream. Observational only — it reconstructs cycle counts
/// from the functional hook order; it never alters functional execution. This is
/// the ONLY translation unit that touches both rocjitsu and cycle_model types.
///
/// Declarations track the real PR #6132 ExecutionPlugin (every hook below is an
/// `override` of an existing virtual). Engine time is read per-hook from
/// wf.cu().engine()->context(wf.cu().partition_id()).current_tick() — no
/// interface change is needed; the lazy-tick model advances on that clock.

#pragma once

#include "plugins/cu_cycle_model.h"
#include "plugins/execution_plugin.h"
#include "plugins/mem_sys_cycle_model.h"

#include "cycle_model/ArchModel.h"
#include "cycle_model/CycleWaveState.h"
#include "cycle_model/UarchConfig.h"

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// Adapter-side per-wave state. The standalone cycle-model lib stays free of
/// rocjitsu types, so cycle_model::CycleWaveState does NOT inherit WavefrontState;
/// this thin wrapper bridges it into the Wavefront plugin-state slot (mirrors the
/// race detector's RaceWavefrontState).
struct CycleWavefrontState : WavefrontState {
  cycle_model::CycleWaveState st;
};

class CycleModelPlugin : public ExecutionPlugin {
public:
  CycleModelPlugin();
  ~CycleModelPlugin() override;

  void onInit() override;     // reads RJ_CYCLE gate
  void onShutdown() override; // emits aggregate summary

  // Per-instruction: the model enqueues in `after` (memory metadata is only
  // complete post-execute). `before` is pre-state tracing only; route is an
  // optional debug/consistency signal.
  void beforeAmdgpuExecuteInstruction(uint64_t pc, const Instruction &inst, Wavefront &wf) override;
  void afterAmdgpuExecuteInstruction(uint64_t pc, const Instruction &inst, Wavefront &wf) override;
  void onAmdgpuRouteMemoryInstruction(const Instruction &inst, Wavefront &wf) override;

  // Dispatch lifecycle. PacketProcessed carries per-wf VGPR/SGPR counts (used for
  // scoreboard sizing) but NO ISA string and NO LDS bytes. ExecutionEnd flushes
  // the model to quiescence and emits counters.
  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override;
  void onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) override;
  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) override;

  // Workgroup/wave lifecycle. WorkgroupDispatched installs a CycleWavefrontState
  // in each wave's slot and assigns its SIMD. WavefrontHalted drains any
  // model-domain in-flight memory requests (mem_leak_at_halt).
  void onAmdgpuWorkgroupDispatched(uint32_t wg_id, uint32_t dispatch_id, uint32_t vgpr_count,
                                   uint32_t sgpr_count, std::span<Wavefront *> wavefronts) override;
  void onAmdgpuWavefrontHalted(Wavefront &wf) override;

  // Signals positional barrier gates for the workgroup's waves.
  void onAmdgpuBarrierResolved(std::span<Wavefront *> wavefronts) override;

private:
  CycleWavefrontState *state_of(Wavefront &wf) {
    return static_cast<CycleWavefrontState *>(wf.plugin_state(slot_index()));
  }

  // Resolve (and cache) the CuCycleModel child the driver registered under wf's
  // CU. Returns nullptr if RJ_CYCLE wiring is absent. Configures the shell's
  // ArchModel from the device arch on first resolution.
  CuCycleModel *cycle_model_for(Wavefront &wf);
  cycle_model::UarchConfig &config_for_arch(uint32_t gfx_target_version);

  // Current simdojo engine time (picoseconds) for wf's CU partition.
  cycle_model::Tick now_ticks(Wavefront &wf);

  // Instruction -> InstrEvent / PendingEvent mapping (adapter-only; touches rocjitsu types).
  static cycle_model::InstrKind kind_of(const Instruction &inst);
  static cycle_model::WaitCounter wcnt_slot_of(cycle_model::InstrKind k);
  static void fill_reg_set(const Instruction &inst, cycle_model::InstrRegSet &rs);
  cycle_model::PendingEvent make_pending_event(const Instruction &inst, Wavefront &wf);

  // Cache of resolved CuCycleModel children, keyed on the owning ComputeUnitCore.
  std::unordered_map<const ComputeUnitCore *, CuCycleModel *> cu_models_;
  // One parsed JSON config per arch (gfx_target_version), shared by its CUs.
  std::unordered_map<uint32_t, cycle_model::UarchConfig> cfg_by_arch_;
  // Shared L2/HBM timing component per Xcd, resolved + configured lazily on first
  // dispatch. The values are the set of MemSysCycleModels reset per kernel.
  std::unordered_map<const void *, MemSysCycleModel *> mem_sys_;
  // Each shared MemSys and the CuCycleModels feeding it — for the per-XCD "all CUs
  // idle" shared-reset check (built lazily in cycle_model_for, alongside mem_sys_).
  std::unordered_map<MemSysCycleModel *, std::vector<CuCycleModel *>> cus_of_memsys_;

  // Gated by RJ_CYCLE=1 (read in onInit). When false every hook early-returns.
  bool enabled_ = false;
};

} // namespace amdgpu
} // namespace rocjitsu
