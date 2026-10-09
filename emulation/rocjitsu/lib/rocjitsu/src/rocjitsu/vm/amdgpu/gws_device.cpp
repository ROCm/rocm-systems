// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gws_device.h"

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace rocjitsu {
namespace amdgpu {

// The authoritative GWS scheduling policy (scope, per-operation behavior, and
// synchronization) lives in the GwsDevice class documentation in gws_device.h.
// The comments below note only implementation-specific details.

void GwsDevice::register_compute_unit(ComputeUnitCore *cu) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (std::find(cus_.begin(), cus_.end(), cu) == cus_.end())
    cus_.push_back(cu);
}

void GwsDevice::unregister_compute_unit(ComputeUnitCore *cu) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::erase(cus_, cu);
}

uint32_t GwsDevice::dispatch_resident_waves(uint32_t dispatch_id) const {
  uint32_t resident = 0;
  for (auto *cu : cus_)
    for (const auto &entry : cu->active_wgs_)
      if (static_cast<uint32_t>(entry.first >> 32) == dispatch_id)
        resident += entry.second;
  return resident;
}

uint32_t GwsDevice::release_waiters(uint32_t process_id, uint32_t rid, uint32_t max_wake) {
  uint32_t woke = 0;
  for (auto *cu : cus_) {
    for (auto &w : cu->wfs_) {
      if (woke >= max_wake)
        return woke;
      if (!w || w->state() != WfState::GWS_WAIT)
        continue;
      // GWS state is device-global, so any wave of this process parked on this
      // rid is a candidate -- do not filter on workgroup or dispatch.
      if (w->process_id() == process_id && w->gws_wait_rid_ == rid) {
        w->gws_wait_rid_ = Wavefront::kNoGwsWait;
        w->set_state(WfState::RUNNING);
        w->set_ready_cycle(cu->cycle_counter_);
        ++woke;
      }
    }
  }
  return woke;
}

void GwsDevice::init(Wavefront &wf, uint32_t rid, uint32_t count) {
  if (rid >= kResourceCount)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  auto &res = resources_[wf.process_id()][rid];
  // Barrier: (participants - 1) is the programmed counter value.
  res.counter = count;
  res.armed = true;
  // Semaphore reuses the same init to seed the initial credit count.
  res.credits = count;
}

void GwsDevice::barrier_arrive(Wavefront &wf, uint32_t rid, uint32_t count) {
  if (rid >= kResourceCount)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  const uint32_t resident = dispatch_resident_waves(wf.dispatch_id());
  auto &res = resources_[wf.process_id()][rid];
  // The still-outstanding arrivals that must occur to release this wave are the
  // current counter (a fresh resource seeds that counter from this arrival's own
  // value). A blocking arrival therefore needs (outstanding + 1) participants to
  // be provably resident in the dispatch; if that set is larger than the resident
  // waves it would reference arrivals that may never happen, so fall back to a
  // non-blocking structural no-op. A releasing arrival (outstanding == 0) never
  // blocks, so it is exempt from the gate. The gate is written as
  // (outstanding >= resident), which is equivalent to (outstanding + 1 > resident)
  // for integers but cannot overflow when count is near UINT32_MAX (a huge count
  // must take the fallback, never wrap the sum to zero and park).
  const uint32_t outstanding = res.armed ? res.counter : count;
  if (outstanding != 0 && outstanding >= resident)
    return;
  if (!res.armed) {
    res.counter = count;
    res.armed = true;
  }
  if (res.counter == 0) {
    // This arrival observes zero: it is the releasing arrival. Wake the queued
    // peers (on any CU) and reload the counter from this arrival's own value so
    // the next phase may use a different participant count.
    release_waiters(wf.process_id(), rid, std::numeric_limits<uint32_t>::max());
    res.counter = count;
    return;
  }
  // Positive counter: queue this arrival and park.
  --res.counter;
  wf.gws_wait_rid_ = rid;
  wf.set_state(WfState::GWS_WAIT);
}

void GwsDevice::sema_v(Wavefront &wf, uint32_t rid) {
  if (rid >= kResourceCount)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  auto &res = resources_[wf.process_id()][rid];
  ++res.credits;
  if (release_waiters(wf.process_id(), rid, 1) == 1)
    --res.credits; // the released waiter consumes the credit it was waiting on
}

void GwsDevice::sema_p(Wavefront &wf, uint32_t rid) {
  if (rid >= kResourceCount)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  auto &res = resources_[wf.process_id()][rid];
  if (res.credits > 0) {
    --res.credits;
    return;
  }
  // No credit: park until any wave of this process signals the shared resource.
  // The signal is event-driven (sema_v/br/release_all wake us directly), and the
  // dispatch-wide quiescence backstop releases us only if the whole dispatch
  // deadlocks.
  wf.gws_wait_rid_ = rid;
  wf.set_state(WfState::GWS_WAIT);
}

void GwsDevice::sema_br(Wavefront &wf, uint32_t rid, uint32_t count) {
  if (rid >= kResourceCount || count == 0)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  auto &res = resources_[wf.process_id()][rid];
  res.credits += count;
  uint32_t woke = release_waiters(wf.process_id(), rid, count);
  res.credits -= woke; // released waiters consume credits
}

void GwsDevice::sema_release_all(Wavefront &wf, uint32_t rid) {
  if (rid >= kResourceCount)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  auto &res = resources_[wf.process_id()][rid];
  release_waiters(wf.process_id(), rid, std::numeric_limits<uint32_t>::max());
  res.credits = 0;
}

void GwsDevice::escape_deadlocks() {
  std::lock_guard<std::mutex> lock(mutex_);
  // Dispatches that currently have at least one parked GWS wave somewhere.
  std::unordered_set<uint32_t> parked_dispatches;
  for (auto *cu : cus_)
    for (auto &w : cu->wfs_)
      if (w && w->state() == WfState::GWS_WAIT)
        parked_dispatches.insert(w->dispatch_id());

  for (uint32_t did : parked_dispatches) {
    // Quiescent only if no non-halted wave of this dispatch, on any CU, is still
    // able to run (i.e. every such wave is GWS_WAIT or stalled at an s_barrier).
    // Covering BARRIER also breaks the mixed deadlock where some waves wait at a
    // GWS barrier while their siblings wait at an s_barrier.
    bool quiescent = true;
    for (auto *cu : cus_) {
      for (auto &w : cu->wfs_) {
        if (w && w->dispatch_id() == did && w->state() != WfState::HALTED &&
            w->state() != WfState::GWS_WAIT && w->state() != WfState::BARRIER) {
          quiescent = false;
          break;
        }
      }
      if (!quiescent)
        break;
    }
    if (!quiescent)
      continue;
    for (auto *cu : cus_) {
      for (auto &w : cu->wfs_) {
        if (w && w->dispatch_id() == did && w->state() == WfState::GWS_WAIT) {
          w->gws_wait_rid_ = Wavefront::kNoGwsWait;
          w->set_state(WfState::RUNNING);
          w->set_ready_cycle(cu->cycle_counter_);
        }
      }
    }
  }
}

} // namespace amdgpu
} // namespace rocjitsu
