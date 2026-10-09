// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gws_device.h"

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <unordered_map>

namespace rocjitsu {
namespace amdgpu {

// The authoritative GWS scheduling policy (scope, per-operation behavior, and
// thread-safety) lives in the GwsDevice class documentation in gws_device.h.
// The comments below note only implementation-specific details.
namespace {

// A wave can still execute (and therefore still signal/arrive at a GWS resource)
// unless it is retired or blocked. HALTED is retired; GWS_WAIT is parked here; a
// wave stalled at an s_barrier (BARRIER) can never reach a GWS op either, so it
// counts as blocked for quiescence (breaks the mixed GWS/s_barrier deadlock).
bool can_still_signal(WfState state) {
  switch (state) {
  case WfState::RUNNING:
  case WfState::WAITCNT:
  case WfState::VM_RETRY:
  case WfState::ENDING:
    return true;
  case WfState::HALTED:
  case WfState::BARRIER:
  case WfState::GWS_WAIT:
    return false;
  }
  return false;
}

} // namespace

void GwsDevice::register_compute_unit(ComputeUnitCore *cu) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (std::find(cus_.begin(), cus_.end(), cu) == cus_.end())
    cus_.push_back(cu);
}

void GwsDevice::unregister_compute_unit(ComputeUnitCore *cu) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::erase(cus_, cu);
  cu_runnable_.erase(cu);
  cu_epoch_.erase(cu);
}

void GwsDevice::add_resident(uint32_t dispatch_id, uint32_t count) {
  if (count == 0)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  resident_waves_[dispatch_id] += count;
}

void GwsDevice::remove_resident(uint32_t dispatch_id, uint32_t count) {
  if (count == 0)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = resident_waves_.find(dispatch_id);
  if (it == resident_waves_.end())
    return;
  it->second = it->second > count ? it->second - count : 0;
  if (it->second == 0)
    resident_waves_.erase(it);
}

void GwsDevice::park_wave(Wavefront &wf, uint32_t rid, bool is_barrier, uint64_t generation) {
  wf.gws_wait_rid_ = rid;
  wf.gws_wait_is_barrier_ = is_barrier;
  wf.gws_wait_generation_ = generation;
  wf.set_state(WfState::GWS_WAIT);
  ++parked_[wf.process_id()];
  // 0 -> positive opens a new park epoch, invalidating stale per-CU snapshots so
  // a producer that has not yet published cannot be mistaken for quiescent.
  if (parked_total_.fetch_add(1, std::memory_order_release) == 0)
    ++park_epoch_;
}

void GwsDevice::wake_wave(ComputeUnitCore &cu, Wavefront &wf) {
  const uint32_t process = wf.process_id();
  wf.gws_wait_rid_ = Wavefront::kNoGwsWait;
  wf.gws_wait_is_barrier_ = false;
  wf.set_state(WfState::RUNNING);
  wf.set_ready_cycle(cu.cycle_counter_);
  auto it = parked_.find(process);
  if (it != parked_.end() && it->second > 0 && --it->second == 0)
    parked_.erase(it);
  parked_total_.fetch_sub(1, std::memory_order_release);
}

uint32_t GwsDevice::poll_local_waiters(ComputeUnitCore &cu) {
  uint32_t woke = 0;
  for (auto &slot : cu.wfs_) {
    if (!slot || slot->state() != WfState::GWS_WAIT)
      continue;
    Wavefront &wf = *slot;
    Resource &res = resources_[wf.process_id()][wf.gws_wait_rid_];
    // A barrier release or a semaphore release_all bumps the generation, waking
    // every arrival queued before it regardless of credits.
    if (res.release_gen > wf.gws_wait_generation_) {
      wake_wave(cu, wf);
      ++woke;
      continue;
    }
    // A semaphore P consumes one credit left by a V/BR (possibly on another CU).
    if (!wf.gws_wait_is_barrier_ && res.credits > 0) {
      --res.credits;
      wake_wave(cu, wf);
      ++woke;
    }
  }
  return woke;
}

void GwsDevice::refresh_summary(ComputeUnitCore &cu) {
  std::unordered_map<uint32_t, uint32_t> runnable;
  for (auto &slot : cu.wfs_) {
    if (!slot)
      continue;
    if (can_still_signal(slot->state()))
      ++runnable[slot->process_id()];
  }
  cu_runnable_[&cu] = std::move(runnable);
  cu_epoch_[&cu] = park_epoch_;
}

bool GwsDevice::escape_ready(uint32_t process) const {
  auto parked_it = parked_.find(process);
  if (parked_it == parked_.end() || parked_it->second == 0)
    return false;
  uint64_t total_runnable = 0;
  for (ComputeUnitCore *cu : cus_) {
    // wave_activity_ is a cross-CU atomic; reading it is race-free even while the
    // peer steps. Its low word is the peer's non-halted wave count.
    const uint32_t active =
        static_cast<uint32_t>(cu->wave_activity_.load(std::memory_order_acquire));
    if (active == 0)
      continue; // Idle CU: hosts no waves of any process, contributes nothing.
    auto eit = cu_epoch_.find(cu);
    if (eit == cu_epoch_.end() || eit->second != park_epoch_)
      return false; // An active CU has not published this epoch: picture incomplete.
    auto rit = cu_runnable_.find(cu);
    if (rit == cu_runnable_.end())
      continue;
    auto pit = rit->second.find(process);
    if (pit != rit->second.end())
      total_runnable += pit->second;
  }
  return total_runnable == 0;
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
  res.credits = count;
}

void GwsDevice::barrier_arrive(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid, uint32_t count) {
  if (rid >= kResourceCount)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  auto resident_it = resident_waves_.find(wf.dispatch_id());
  const uint32_t resident = resident_it != resident_waves_.end() ? resident_it->second : 0;
  Resource &res = resources_[wf.process_id()][rid];
  // The still-outstanding arrivals that must occur to release this wave are the
  // current counter (a fresh resource seeds that counter from this arrival's own
  // value). A blocking arrival therefore needs (outstanding + 1) participants to
  // be provably resident in the dispatch; if that set is larger than the resident
  // waves it would reference arrivals that may never happen, so fall back to a
  // non-blocking structural no-op. A releasing arrival (outstanding == 0) never
  // blocks, so it is exempt from the gate. The gate is written as
  // (outstanding >= resident), equivalent to (outstanding + 1 > resident) for
  // integers but without overflow when count is near UINT32_MAX (a huge count
  // takes the fallback, never wraps the sum to zero and parks).
  const uint32_t outstanding = res.armed ? res.counter : count;
  if (outstanding != 0 && outstanding >= resident)
    return;
  if (!res.armed) {
    res.counter = count;
    res.armed = true;
  }
  if (res.counter == 0) {
    // This arrival observes zero: it is the releasing arrival. Bump the release
    // generation so queued arrivals on any CU wake, and reload the counter from
    // this arrival's own value so the next phase may use a different size.
    ++res.release_gen;
    res.counter = count;
    poll_local_waiters(cu); // Same-CU peers wake now; cross-CU peers on their step.
    return;
  }
  // Positive counter: queue this arrival and park it.
  --res.counter;
  park_wave(wf, rid, /*is_barrier=*/true, res.release_gen);
}

void GwsDevice::sema_v(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid) {
  if (rid >= kResourceCount)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  Resource &res = resources_[wf.process_id()][rid];
  ++res.credits;
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
  if (res.credits > 0) {
    --res.credits;
    return;
  }
  // No credit: park until any wave of this process signals the shared resource.
  // A later V/BR leaves a credit this wave's CU consumes on its next poll; a
  // release_all bumps the generation; the quiescence backstop is the last resort.
  park_wave(wf, rid, /*is_barrier=*/false, res.release_gen);
}

void GwsDevice::sema_br(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid, uint32_t count) {
  if (rid >= kResourceCount || count == 0)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  Resource &res = resources_[wf.process_id()][rid];
  res.credits += count;
  poll_local_waiters(cu); // Same-CU waiters drain credits now; the rest persist.
}

void GwsDevice::sema_release_all(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid) {
  if (rid >= kResourceCount)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  Resource &res = resources_[wf.process_id()][rid];
  // Release every currently parked waiter via a generation bump and clear credits
  // so waiters that park afterwards (with the new generation) are not spuriously
  // released by this one.
  ++res.release_gen;
  res.credits = 0;
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
}

void GwsDevice::step_maintenance(ComputeUnitCore &cu) {
  if (!any_parked())
    return; // Fast path: nothing parked anywhere, no store work this step.
  std::lock_guard<std::mutex> lock(mutex_);
  // Apply cross-CU wakeups to this CU's own parked waves (credits/generation left
  // by signals on other CUs), then publish this CU's signalable snapshot so peer
  // escape decisions see any waves this CU just woke as able to run again.
  poll_local_waiters(cu);
  refresh_summary(cu);
  // Genuine-deadlock backstop: release this CU's own parked waves whose process is
  // provably quiescent across every CU. Evaluate each process once.
  std::unordered_map<uint32_t, bool> ready;
  for (auto &slot : cu.wfs_) {
    if (!slot || slot->state() != WfState::GWS_WAIT)
      continue;
    const uint32_t process = slot->process_id();
    auto rit = ready.find(process);
    if (rit == ready.end())
      rit = ready.emplace(process, escape_ready(process)).first;
    if (rit->second)
      wake_wave(cu, *slot);
  }
}

} // namespace amdgpu
} // namespace rocjitsu
