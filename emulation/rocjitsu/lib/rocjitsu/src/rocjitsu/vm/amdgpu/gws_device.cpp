// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gws_device.h"

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/ordered_count.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <unordered_map>

namespace rocjitsu {
namespace amdgpu {

uint32_t GwsDevice::reserve_ordered_ids(uint32_t process_id, uint32_t queue_id, uint32_t count,
                                        uint32_t id_count) {
  std::lock_guard lock(mutex_);
  assert(id_count && id_count <= kComputeWaveIdMask + 1);
  auto &ring = ordered_rings_[process_id][queue_id];
  ring.id_count = id_count;
  const uint32_t base = ring.next_launch;
  ring.next_launch = (base + count) % id_count;
  return base;
}

std::optional<uint32_t> GwsDevice::ordered_count(uint32_t process_id, uint32_t queue_id,
                                                 uint32_t logical_wave, uint32_t address,
                                                 uint32_t value, bool exchange, bool release) {
  std::lock_guard lock(mutex_);
  auto &ring = ordered_rings_[process_id][queue_id];
  if (logical_wave != ring.oldest)
    return std::nullopt;
  auto &gds = gds_.try_emplace(process_id, kGfx10GdsSizeKiB).first->second;
  const uint32_t previous = gds.read32(address);
  gds.write32(address, exchange ? value : previous + value);
  if (release)
    ring.oldest = (ring.oldest + 1) % ring.id_count;
  return previous;
}

void GwsDevice::reset_ordered_ids(uint32_t process_id, uint32_t queue_id) {
  std::lock_guard lock(mutex_);
  if (auto it = ordered_rings_.find(process_id); it != ordered_rings_.end()) {
    it->second.erase(queue_id);
    if (it->second.empty())
      ordered_rings_.erase(it);
  }
}

bool GwsDevice::prepare_register_access(Wavefront &wf, VectorMemState &d, uint8_t offset0,
                                        uint8_t offset1, uint32_t src, uint32_t dst, bool ordered,
                                        bool subtract) {
  const uint32_t m0 = wf.m0();
  const uint32_t index = ordered ? ((m0 >> 16) / 4 + offset0 / 4) : ((offset0 >> 2) & 15);
  const uint8_t op = (offset1 >> 4) & 3;
  if (ordered && ((offset0 & 3) || index >= 16 || (offset1 & 2 && !(offset1 & 1)) || op == 2 ||
                  (op == 1 && (index < 4 || index > 7)) || (op && index >= 8))) {
    wf.report_instruction_execution_error(InstructionExecutionError::InvalidInstructionEncoding);
    return false;
  }
  // RDNA3.5 section 13.4.3 defines one VDST and a B32 return even for ADD
  // to a 64-bit streamout target. The GS register instructions return both words.
  const uint32_t words = ordered || index < 8 ? 1u : 2u;
  const uint64_t exec = ordered || wf.vgpr_alloc().contains(dst, words) ? wf.exec() : 0;
  // An empty ordered wave sends ADD(0) regardless of the encoded operation.
  // Preserve release/done and the walker step, without requiring a WRAP ring.
  if (ordered && !exec)
    offset1 &= ~0x30u;
  else if (ordered && op == 3) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return false;
  }
  const uint32_t operand =
      exec ? RegisterAccess(wf.cu()).read_vgpr(src, std::countr_zero(exec)) : 0;
  d.gds = true;
  d.elem_size = words * sizeof(uint32_t);
  d.num_elems = 1;
  d.is_load = true;
  d.dst_reg_base = dst;
  d.exec_mask = d.lane_mask = streamout_register_lane(exec, ordered);
  d.gds_register.emplace(VectorMemState::GdsRegisterAccess{
      operand, static_cast<uint16_t>(m0 & 0x1fff), static_cast<uint8_t>(index), offset1,
      static_cast<uint8_t>((m0 >> 13) & 7), ordered, subtract});
  return true;
}

void GwsDevice::release_process(uint32_t process_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  gds_.erase(process_id);
  ordered_rings_.erase(process_id);
  resources_.erase(process_id);
  data_shares_.erase(process_id);
  instruction_progress_.erase(process_id);
  auto it = parked_.find(process_id);
  if (it != parked_.end()) {
    parked_total_.fetch_sub(it->second, std::memory_order_release);
    parked_.erase(it);
  }
}

bool GwsDevice::has_gds_data() {
  std::lock_guard lock(mutex_);
  return std::ranges::any_of(
             gds_, [](const auto &entry) { return entry.second.materialized_size_bytes() != 0; }) ||
         std::ranges::any_of(data_shares_, [](const auto &entry) {
           return entry.second.memory.materialized_size_bytes() != 0;
         });
}

void GwsDevice::retire_ordered_dispatch(uint32_t process_id, uint32_t queue_id,
                                        uint32_t dispatch_id) {
  std::lock_guard lock(mutex_);
  const auto share = data_shares_.find(process_id);
  if (share != data_shares_.end())
    share->second.ordered_walkers.erase((uint64_t{queue_id} << 32) | dispatch_id);
}

size_t GwsDevice::tracked_process_count() {
  std::lock_guard<std::mutex> lock(mutex_);
  return resources_.size();
}

bool GwsDevice::has_gds_checkpoint_state() {
  std::lock_guard lock(mutex_);
  if (!ordered_rings_.empty() || std::ranges::any_of(gds_, [](const auto &entry) {
        return entry.second.materialized_size_bytes() != 0;
      }))
    return true;
  return std::ranges::any_of(data_shares_, [](const auto &entry) {
    const auto &share = entry.second;
    return share.memory.materialized_size_bytes() != 0 ||
           std::ranges::any_of(share.streamout, [](uint64_t value) { return value != 0; }) ||
           std::ranges::any_of(share.ordered_walkers,
                               [](const auto &walker) { return walker.second != 0; });
  });
}

void GwsDevice::release_sema_waiter(Resource &res, uint64_t seq) {
  auto it = std::find(res.sema_wait_seqs.begin(), res.sema_wait_seqs.end(), seq);
  if (it != res.sema_wait_seqs.end())
    res.sema_wait_seqs.erase(it);
}

void GwsDevice::park_wave(Wavefront &wf, uint32_t rid, bool is_barrier, uint64_t generation) {
  wf.gws_wait_rid_ = rid;
  wf.gws_wait_is_barrier_ = is_barrier;
  wf.gws_wait_generation_ = generation;
  wf.set_state(WfState::GWS_WAIT);
  ++parked_[wf.process_id()];
  // A semaphore P-waiter joins the resource's arrival-ordered queue and reserves a
  // V/BR credit, so a fresh P (on any CU) cannot steal the release owed to an
  // earlier waiter (see the entitlement note in the class doc).
  if (!is_barrier && rid < kResourceCount) {
    wf.gws_wait_seq_ = sema_seq_next_++;
    resources_[wf.process_id()][rid].sema_wait_seqs.push_back(wf.gws_wait_seq_);
  }
  wf.gws_wait_steps_ = 0;
  wf.gws_wait_progress_ = instruction_progress_[wf.process_id()];
  parked_total_.fetch_add(1, std::memory_order_release);
}

void GwsDevice::wake_wave(ComputeUnitCore &cu, Wavefront &wf) {
  const uint32_t process_id = wf.process_id();
  const uint32_t rid = wf.gws_wait_rid_;
  const bool was_barrier = wf.gws_wait_is_barrier_;
  const uint64_t seq = wf.gws_wait_seq_;
  wf.gws_wait_rid_ = Wavefront::kNoGwsWait;
  wf.gws_wait_is_barrier_ = false;
  wf.set_state(WfState::RUNNING);
  wf.set_ready_cycle(cu.cycle_counter_);
  // Drop the waiter's queued arrival stamp (a sema waiter no longer competes for a
  // reserved credit).
  if (!was_barrier && rid < kResourceCount) {
    auto rit = resources_.find(process_id);
    if (rit != resources_.end())
      release_sema_waiter(rit->second[rid], seq);
  }
  auto it = parked_.find(process_id);
  if (it != parked_.end() && it->second > 0 && --it->second == 0)
    parked_.erase(it);
  parked_total_.fetch_sub(1, std::memory_order_release);
}

uint32_t GwsDevice::poll_local_waiters(ComputeUnitCore &cu) {
  uint32_t woken_count = 0;
  for (auto &slot : cu.wfs_) {
    if (!slot || slot->state() != WfState::GWS_WAIT)
      continue;
    Wavefront &wf = *slot;
    Resource &res = resources_[wf.process_id()][wf.gws_wait_rid_];
    // A barrier release or a semaphore release_all bumps the generation, waking
    // every arrival queued before it regardless of credits.
    if (res.release_gen > wf.gws_wait_generation_) {
      wake_wave(cu, wf);
      ++woken_count;
      continue;
    }
    // A semaphore P consumes one credit left by a V/BR (possibly on another CU),
    // but only on behalf of the oldest queued waiters: this wave may take a credit
    // only when fewer than `counter` older waiters are still parked on the resource
    // (across all CUs). That releases V/BR credits in arrival order, so a later P
    // -- even one polled first because it shares the signaling CU -- cannot jump an
    // earlier waiter parked on another CU (see the entitlement note in the class doc).
    if (!wf.gws_wait_is_barrier_ && res.counter > 0) {
      uint32_t older_waiter_count = 0;
      for (uint64_t queued_seq : res.sema_wait_seqs)
        if (queued_seq < wf.gws_wait_seq_)
          ++older_waiter_count;
      if (older_waiter_count < res.counter) {
        --res.counter;
        wake_wave(cu, wf); // Removes this wave's stamp from sema_wait_seqs.
        ++woken_count;
      }
    }
  }
  return woken_count;
}

void GwsDevice::init(Wavefront &wf, uint32_t rid, uint32_t count) {
  if (rid >= kResourceCount)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  Resource &res = resources_[wf.process_id()][rid];
  // Barrier: (participants - 1) is the programmed counter value. Semaphore reuses
  // the same init to seed the initial credit count.
  res.counter = count;
  res.armed = true;
}

bool GwsDevice::barrier_arrive(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid, uint32_t count) {
  if (rid >= kResourceCount)
    return true;
  std::lock_guard<std::mutex> lock(mutex_);
  Resource &res = resources_[wf.process_id()][rid];
  if (!res.armed)
    return false;
  if (res.counter == 0) {
    // This arrival observes zero: it is the releasing arrival. Bump the release
    // generation so queued arrivals on any CU wake, and reload the counter from
    // this arrival's own value so the next phase may use a different size.
    ++res.release_gen;
    res.counter = count;
    poll_local_waiters(cu); // Same-CU peers wake now; cross-CU peers on their step.
    return true;
  }
  // Positive counter: queue this arrival and park it.
  --res.counter;
  park_wave(wf, rid, /*is_barrier=*/true, res.release_gen);
  return true;
}

void GwsDevice::sema_v(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid) {
  if (rid >= kResourceCount)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  Resource &res = resources_[wf.process_id()][rid];
  ++res.counter;
  // A same-CU waiter consumes the credit immediately; otherwise it persists for a
  // waiter on another CU to consume the next time that CU polls.
  poll_local_waiters(cu);
}

void GwsDevice::sema_p(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid) {
  if (rid >= kResourceCount)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  (void)cu;
  Resource &res = resources_[wf.process_id()][rid];
  // Consume a credit only from the surplus beyond the credits reserved for waiters
  // already parked on this resource (possibly on another CU), so a fresh P cannot
  // steal a V/BR release owed to an earlier waiter (see the entitlement note in the
  // class doc). Otherwise park and join the reserved queue.
  if (res.counter > res.sema_wait_seqs.size()) {
    --res.counter;
    return;
  }
  // No free credit: park until any wave of this process signals the shared
  // resource. A later V/BR leaves a credit this wave's CU consumes on its next
  // poll; a release_all bumps the generation.
  park_wave(wf, rid, /*is_barrier=*/false, res.release_gen);
}

void GwsDevice::sema_br(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid, uint32_t count) {
  if (rid >= kResourceCount || count == 0)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  Resource &res = resources_[wf.process_id()][rid];
  res.counter += count;
  poll_local_waiters(cu); // Same-CU waiters drain credits now; the rest persist.
}

void GwsDevice::sema_release_all(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid) {
  if (rid >= kResourceCount)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  Resource &res = resources_[wf.process_id()][rid];
  // The generation wakes every queued P waiter without consuming a credit.
  // GFX11 keeps any banked credits for future arrivals. Other targets clear them.
  ++res.release_gen;
  if (cu.arch() != ROCJITSU_CODE_ARCH_RDNA3 && cu.arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    res.counter = 0;
  // Retire the released waiters' credit reservations now. The generation bump above
  // wakes every currently queued P-waiter regardless of credits, so their arrival
  // stamps no longer entitle them to a credit; leaving the stamps queued would let a
  // fresh P (or poll_local_waiters) reserve a later V/BR credit on behalf of waves
  // that will instead wake through release_gen without consuming one, so a genuinely
  // creditable P parked afterwards could stall. Each waiter's own CU still performs
  // the wake and state update on its next step (release_sema_waiter there is then a
  // no-op); only the shared reservation bookkeeping is settled here.
  res.sema_wait_seqs.clear();
  poll_local_waiters(cu);
}

void GwsDevice::notify_parked_wave_gone(const Wavefront &wf) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = parked_.find(wf.process_id());
  if (it == parked_.end() || it->second == 0)
    return; // Not accounted as parked (defensive): nothing to reconcile.
  if (--it->second == 0)
    parked_.erase(it);
  parked_total_.fetch_sub(1, std::memory_order_release);
  // A gone semaphore waiter drops its queued arrival stamp (see park_wave).
  if (!wf.gws_wait_is_barrier_ && wf.gws_wait_rid_ < kResourceCount) {
    auto rit = resources_.find(wf.process_id());
    if (rit != resources_.end())
      release_sema_waiter(rit->second[wf.gws_wait_rid_], wf.gws_wait_seq_);
  }
}

void GwsDevice::poll_waits(ComputeUnitCore &cu) {
  // The caller holds this CU's wave lock. CUs with no local waiter need no store lock.
  if (!any_parked() || !std::ranges::any_of(cu.wfs_, [](const auto &wave) {
        return wave && wave->state() == WfState::GWS_WAIT;
      }))
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  poll_local_waiters(cu);
  for (const auto &wave : cu.wfs_) {
    if (!wave || wave->state() != WfState::GWS_WAIT)
      continue;
    const auto progress = instruction_progress_[wave->process_id()];
    if (wave->gws_wait_progress_ != progress) {
      wave->gws_wait_progress_ = progress;
      wave->gws_wait_steps_ = 0;
    }
  }
}

void GwsDevice::note_instruction_progress(uint32_t process_id) {
  if (!any_parked())
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (parked_.contains(process_id))
    ++instruction_progress_[process_id];
}

} // namespace amdgpu
} // namespace rocjitsu
