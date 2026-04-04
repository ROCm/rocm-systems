// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/command_processor.h"

#include "simdojo/sim/message.h"
#include "simdojo/sim/simulation.h"
#include "util/debug_print.h"

#include <cassert>
#include <set>
#include <string>

namespace rocjitsu {
namespace amdgpu {

namespace {

/// Initialize a newly dispatched wavefront's registers.
///
/// On gfx9+, the compiler implicitly places the kernarg pointer in s[0:1]
/// regardless of kernel_code_properties flags. The system SGPR
/// (workgroup_id_x) follows at `sbase + num_user_sgprs`.
/// v0 is set to the lane index (workitem_id_x).
void init_wavefront_regs(ComputeUnitCore *cu, Wavefront *wf, const DispatchPacket &pkt,
                         uint32_t global_wg_id) {
  uint32_t sbase = wf->sgpr_alloc().base;

  // User SGPRs: kernarg pointer in s[0:1] (implicit gfx9+ ABI).
  if (pkt.kernarg_addr != 0) {
    cu->write_sgpr(sbase + 0, static_cast<uint32_t>(pkt.kernarg_addr));
    cu->write_sgpr(sbase + 1, static_cast<uint32_t>(pkt.kernarg_addr >> 32));
  }

  // System SGPR: workgroup_id_x after user SGPRs.
  cu->write_sgpr(sbase + pkt.num_user_sgprs, global_wg_id);

  // Workitem ID: v0 = lane index within the wavefront.
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < cu->wf_size(); ++lane)
    cu->write_vgpr(vbase, lane, lane);
}

} // namespace

void CommandProcessor::startup() {
  engine()->register_as_primary();

  doorbell_event_.set_handler(
      [this](simdojo::Tick ts, simdojo::Message *) { handle_doorbell(ts); });

  // If packets were pre-loaded via enqueue(), schedule an initial doorbell.
  // Otherwise, signal immediately that this primary has no work.
  if (dispatched_ < dispatch_queue_.size())
    schedule_event(&doorbell_event_, 0);
  else
    engine()->primary_release();
}

void CommandProcessor::submit(DispatchPacket packet) {
  std::lock_guard<std::mutex> lock(async_mutex_);
  async_submissions_.push_back(std::move(packet));
  // schedule_event_now must be called under async_mutex_ to close the race
  // where the simulation thread could drain async_submissions_ and see
  // quiescence between the push and the doorbell schedule.
  engine()->schedule_event_now(&doorbell_event_);
}

bool CommandProcessor::step() {
  if (dispatched_ >= dispatch_queue_.size())
    return false;

  assert(!cus_.empty() && "command processor has no compute units");

  const DispatchPacket &pkt = dispatch_queue_[dispatched_++];

  // Activate wavefront slots for each workgroup, distributing across CUs.
  // If a CU is full, try all other CUs in round-robin before giving up.
  // If ALL CUs are full, re-enqueue remaining workgroups for retry on next doorbell.
  for (uint32_t wg = 0; wg < pkt.workgroup_count; ++wg) {
    uint32_t global_wg_id = wg + pkt.workgroup_id_offset;
    bool wg_dispatched = true;
    for (uint32_t w = 0; w < pkt.wfs_per_workgroup; ++w) {
      Wavefront *wf = nullptr;
      ComputeUnitCore *chosen_cu = nullptr;

      // Try all CUs in round-robin starting from next_cu_.
      for (size_t attempt = 0; attempt < cus_.size(); ++attempt) {
        size_t cu_idx = (next_cu_ + attempt) % cus_.size();
        ComputeUnitCore *cu = cus_[cu_idx];
        wf = cu->dispatch_wf(wg, pkt.kernel_entry_pc, pkt.sgprs_per_wf, pkt.vgprs_per_wf);
        if (!wf) {
          // CU full or out of registers - retire halted wfs and retry this CU once.
          cu->retire_halted_wfs();
          wf = cu->dispatch_wf(wg, pkt.kernel_entry_pc, pkt.sgprs_per_wf, pkt.vgprs_per_wf);
        }
        if (wf) {
          chosen_cu = cu;
          // Advance round-robin past this CU for the next wavefront.
          next_cu_ = (cu_idx + 1) % cus_.size();
          break;
        }
      }

      if (wf && chosen_cu) {
        init_wavefront_regs(chosen_cu, wf, pkt, global_wg_id);
      } else {
        // ALL CUs are full. Re-enqueue remaining workgroups for retry.
        util::debug::print(__func__, ": all CUs full at wg=", wg, " w=", w, " - re-enqueueing ",
                           pkt.workgroup_count - wg, " remaining workgroups");
        DispatchPacket retry_pkt = pkt;
        retry_pkt.workgroup_count = pkt.workgroup_count - wg;
        retry_pkt.workgroup_id_offset = pkt.workgroup_id_offset + wg;
        retry_queue_.push_back(std::move(retry_pkt));
        wg_dispatched = false;
        break;
      }
    }
    if (!wg_dispatched)
      break;
  }

  return dispatched_ < dispatch_queue_.size();
}

void CommandProcessor::drain_async_submissions() {
  std::lock_guard<std::mutex> lock(async_mutex_);
  for (auto &pkt : async_submissions_)
    dispatch_queue_.push_back(std::move(pkt));
  async_submissions_.clear();
}

void CommandProcessor::check_all_idle() {
  for (auto *cu : cus_)
    if (!cu->is_idle())
      return;

  // If retry queue has pending workgroups, schedule another doorbell to retry.
  if (!retry_queue_.empty()) {
    schedule_event(&doorbell_event_, 1);
    return;
  }

  if (pending_dispatches() == 0)
    engine()->primary_release();
}

void CommandProcessor::handle_doorbell(simdojo::Tick /*timestamp*/) {
  // Drain any externally submitted packets into the dispatch queue.
  drain_async_submissions();

  // Re-enqueue any retry packets from previous doorbell cycles.
  for (auto &rpkt : retry_queue_)
    dispatch_queue_.push_back(std::move(rpkt));
  retry_queue_.clear();

  // Process all pending dispatch packets.
  size_t prev_dispatched = dispatched_;
  while (step()) {
  }

  // Collect CUs that have active wavefronts (received work from dispatch).
  std::set<ComputeUnitCore *> activated_cus;
  if (dispatched_ > prev_dispatched) {
    for (auto *cu : cus_) {
      if (cu->has_active_wfs())
        activated_cus.insert(cu);
    }
  }

  // Activate CUs that have work. Send through dispatch ports so the link's
  // exec_mode governs delivery: FUNCTIONAL = synchronous direct call,
  // CLOCKED = event-based with propagation latency.
  for (size_t i = 0; i < cus_.size(); ++i) {
    if (activated_cus.count(cus_[i]) == 0)
      continue;
    if (dispatch_ports_[i]->link())
      dispatch_ports_[i]->send(std::make_unique<simdojo::Message>(simdojo::MessageHeader{}));
    else
      cus_[i]->activate(); // Fallback if port not yet wired.
  }

  // If retry queue has pending workgroups, schedule another doorbell.
  if (!retry_queue_.empty())
    schedule_event(&doorbell_event_, 1);
  // If nothing was dispatched and no CUs are active, signal done.
  else if (dispatched_ == prev_dispatched)
    check_all_idle();
}

} // namespace amdgpu
} // namespace rocjitsu
