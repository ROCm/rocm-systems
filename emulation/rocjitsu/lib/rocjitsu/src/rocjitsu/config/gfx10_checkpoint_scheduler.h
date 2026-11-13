// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "checkpoint_generated.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "util/diagnostic.h"
#include "util/result.h"

#include <algorithm>
#include <cstdint>

namespace rocjitsu::config {

class Gfx10CheckpointScheduler {
public:
  static util::FailureOr<flatbuffers::Offset<fb::Gfx10WaveScheduling>>
  save_wave(flatbuffers::FlatBufferBuilder &builder, const amdgpu::ComputeUnitCore &cu,
            const amdgpu::Wavefront &wf, util::DiagnosticEmitter emit_error) {
    if (wf.waiting_barrier_bit_ != amdgpu::Wavefront::kNoBarrierWait || wf.named_barrier_id_ ||
        std::ranges::any_of(wf.barrier_complete_, [](bool complete) { return complete; }))
      return emit_error.emit() << "Cannot checkpoint split or named GFX10 barrier state";
    const auto key = amdgpu::wg_key(wf.dispatch_id(), wf.wg_id());
    return fb::CreateGfx10WaveScheduling(
        builder, wf.ready_cycle_, wf.nop_ready_cycle_, wf.sleep_cycles_, 0, 0, 0, 0, 0, false,
        wf.use_system_clock_, wf.state() == amdgpu::WfState::BARRIER, cu.active_wgs_.contains(key),
        cu.barrier_wgs_.contains(key));
  }

  static util::FailureOr<flatbuffers::Offset<fb::Gfx10CuScheduling>>
  save_cu(flatbuffers::FlatBufferBuilder &builder, const amdgpu::ComputeUnitCore &cu,
          util::DiagnosticEmitter emit_error) {
    // Dispatch first registers the expected wave count, then allocates the
    // individual members. A partially populated registration cannot be rebuilt
    // from wave records alone.
    for (const auto &[key, count] : cu.active_wgs_) {
      const auto resident = std::ranges::count_if(cu.wfs_, [&](const auto &wf) {
        return wf && !wf->is_halted() && amdgpu::wg_key(wf->dispatch_id(), wf->wg_id()) == key;
      });
      if (!count || count != static_cast<uint32_t>(resident))
        return emit_error.emit() << "Cannot checkpoint incomplete GFX10 workgroup membership";
      if (count > 1 && !cu.barrier_wgs_.contains(key))
        return emit_error.emit() << "Cannot checkpoint a multiwave GFX10 group without its barrier";
    }
    for (const auto &[key, barriers] : cu.barrier_wgs_) {
      const auto group = cu.active_wgs_.find(key);
      if (group == cu.active_wgs_.end() || barriers.allocated_count != 0 ||
          std::ranges::any_of(barriers.workgroup, [&](const auto &barrier) {
            return barrier.signal_count != 0 || barrier.member_count != group->second;
          }))
        return emit_error.emit() << "Cannot checkpoint split or named GFX10 workgroup barriers";
    }
    return fb::CreateGfx10CuScheduling(builder, cu.cycle_counter_, 0,
                                       cu.model_realtime_timestamp());
  }

  static util::Result restore_cu(const fb::Gfx10CuScheduling &saved, amdgpu::ComputeUnitCore &cu,
                                 util::DiagnosticEmitter emit_error) {
    if (const auto *levels = saved.perf_levels();
        levels && std::ranges::any_of(*levels, [](uint32_t value) { return value != 0; }))
      return emit_error.emit() << "Unsupported GFX10 checkpoint performance counters";
    cu.cycle_counter_ = saved.cycle_count();
    cu.model_realtime_origin_ = saved.model_realtime();
    return util::Result::success();
  }

  static util::Result restore_wave(const fb::Gfx10WaveScheduling &saved,
                                   amdgpu::ComputeUnitCore &cu, amdgpu::Wavefront &wf,
                                   util::DiagnosticEmitter emit_error) {
    if (saved.prefetch_lines() || saved.clause_remaining() || saved.clause_kind() ||
        saved.clause_interval() || saved.clause_run() || saved.owns_clause())
      return emit_error.emit() << "Unsupported GFX10 checkpoint instruction scheduling state";
    const auto key = amdgpu::wg_key(wf.dispatch_id(), wf.wg_id());
    if (saved.multiwave_workgroup() && !saved.registered_workgroup())
      return emit_error.emit() << "Invalid GFX10 wave checkpoint barrier membership";
    if (saved.registered_workgroup()) {
      if (cu.active_wgs_.contains(key) &&
          (!saved.multiwave_workgroup() || !cu.barrier_wgs_.contains(key)))
        return emit_error.emit() << "Inconsistent GFX10 workgroup checkpoint membership";
      const auto count = ++cu.active_wgs_[key];
      if (saved.multiwave_workgroup())
        for (auto &barrier : cu.barrier_wgs_[key].workgroup)
          barrier.member_count = count;
    }

    wf.ready_cycle_ = saved.ready_cycle();
    wf.nop_ready_cycle_ = saved.nop_ready_cycle();
    wf.sleep_cycles_ = saved.sleep_cycles();
    wf.set_system_clock(saved.system_clock());
    if (saved.at_barrier())
      wf.set_state(amdgpu::WfState::BARRIER);
    return util::Result::success();
  }
};

} // namespace rocjitsu::config
