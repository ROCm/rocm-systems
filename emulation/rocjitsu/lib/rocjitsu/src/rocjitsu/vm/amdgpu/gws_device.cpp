// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gws_device.h"

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/log.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <string>
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
  if (std::find(cus_.begin(), cus_.end(), cu) != cus_.end())
    return;
  cus_.push_back(cu);
  // Reconcile any live GWS state the CU brings with it -- the mirror of
  // unregister_compute_unit's reconcile-out. A CU re-pointed via set_gws_device
  // while it still hosts parked waves (or resident workgroups) must have them
  // accounted in the destination store, or this store's fast-path gate and
  // quiescence scan would understate them: e.g. a later notify_parked_wave_gone
  // for the migrated wave would decrement a count that was never incremented and
  // wrongly cancel an unrelated waiter already parked here. A freshly created CU
  // carries nothing, so this is a no-op at initial wiring.
  //
  // Resource phase state (barrier generation / credits) is intentionally NOT
  // migrated: a live cross-CU rendezvous should re-point only drained CUs. This
  // reconciliation keeps the bookkeeping counts consistent either way.
  bool migrated_parked = false;
  for (auto &slot : cu->wfs_) {
    if (!slot || slot->state() != WfState::GWS_WAIT)
      continue;
    Wavefront &wf = *slot;
    ++parked_[wf.process_id()];
    parked_total_.fetch_add(1, std::memory_order_release);
    migrated_parked = true;
    if (!wf.gws_wait_is_barrier_ && wf.gws_wait_rid_ < kResourceCount) {
      // Re-stamp the waiter into this store's arrival-order sequence space.
      wf.gws_wait_seq_ = sema_seq_next_++;
      resources_[wf.process_id()][wf.gws_wait_rid_].sema_wait_seqs.push_back(wf.gws_wait_seq_);
    }
  }
  for (const auto &entry : cu->active_wgs_) {
    const uint32_t dispatch_id = static_cast<uint32_t>(entry.first >> 32);
    resident_waves_[dispatch_id] += entry.second;
  }
  if (migrated_parked)
    ++summary_epoch_;
}

void GwsDevice::unregister_compute_unit(ComputeUnitCore *cu) {
  std::lock_guard<std::mutex> lock(mutex_);
  // Reconcile this CU's parked/resident contributions before dropping it. A CU
  // destroyed (or re-pointed at another store via set_gws_device) while it still
  // hosts parked or resident GWS waves must not leave inflated bookkeeping behind:
  // a stuck parked_total_ (any_parked() wedged true) or an oversized barrier
  // residency bound on the surviving CUs. During ~ComputeUnitCore this CU's own
  // wfs_/active_wgs_ are still-alive base members and it is on its own thread; we
  // hold mutex_ (which every peer scan also takes), so settling the accounting here
  // is leaf-lock-safe. Waves halted/aborted on the normal path already reconciled
  // via notify_parked_wave_gone/remove_resident, so only waves still parked and
  // workgroups still resident at teardown are settled here (no double count).
  for (auto &slot : cu->wfs_) {
    if (!slot || slot->state() != WfState::GWS_WAIT)
      continue;
    const uint32_t process_id = slot->process_id();
    auto it = parked_.find(process_id);
    // A GWS_WAIT wave not accounted here parked on a different store (e.g. the CU
    // was re-pointed via set_gws_device after parking); it is not this store's to
    // reconcile, so touch neither parked_total_ nor the waiter queue for it.
    if (it == parked_.end() || it->second == 0)
      continue;
    if (--it->second == 0)
      parked_.erase(it);
    parked_total_.fetch_sub(1, std::memory_order_release);
    if (!slot->gws_wait_is_barrier_ && slot->gws_wait_rid_ < kResourceCount) {
      auto rit = resources_.find(process_id);
      if (rit != resources_.end())
        release_sema_waiter(rit->second[slot->gws_wait_rid_], slot->gws_wait_seq_);
    }
  }
  for (const auto &entry : cu->active_wgs_) {
    const uint32_t dispatch_id = static_cast<uint32_t>(entry.first >> 32);
    auto rit = resident_waves_.find(dispatch_id);
    if (rit == resident_waves_.end())
      continue;
    rit->second = rit->second > entry.second ? rit->second - entry.second : 0;
    if (rit->second == 0)
      resident_waves_.erase(rit);
  }
  std::erase(cus_, cu);
  cu_runnable_.erase(cu);
  cu_summary_epoch_.erase(cu);
  // A destroyed/re-pointed CU must not linger as an eligible placement target for
  // any dispatch's admission accounting (its storage is about to disappear), else
  // the quiescence backstop could dereference a dangling CU.
  for (auto &[dispatch_id, adm] : admission_) {
    (void)dispatch_id;
    std::erase(adm.eligible_cus, cu);
  }
  // A removed CU changes the quiescence picture; force peers to republish before
  // any escape decision trusts a snapshot taken while this CU was still listed.
  ++summary_epoch_;
}

void GwsDevice::add_resident(uint32_t process_id, uint32_t dispatch_id, uint32_t count,
                             uint32_t expected_total_waves) {
  if (count == 0 && expected_total_waves == 0)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (count != 0)
    resident_waves_[dispatch_id] += count;
  if (expected_total_waves != 0) {
    // Track the dispatch's participant accounting so the barrier bound and the
    // quiescence scan can reason about waves it will admit later. Every CU/XCD of
    // the dispatch reports the same whole-grid total, so keep the max.
    DispatchAdmission &adm = admission_[dispatch_id];
    adm.process_id = process_id;
    adm.expected = std::max(adm.expected, expected_total_waves);
    adm.admitted += count;
  }
  // New resident waves are potential signalers. Invalidate every published
  // quiescence snapshot so the deadlock backstop cannot treat a peer CU's
  // earlier zero-runnable summary (taken before this work became visible) as
  // proof of quiescence until that CU republishes. The 0 -> positive park bump
  // alone would miss this: workgroup turnover (a parked wave replaced by a
  // runnable producer) or a producer dispatched alongside a parked wave leaves
  // parked_total_ positive, so only this versioning catches the new signaler.
  ++summary_epoch_;
}

void GwsDevice::note_dispatch_placement(uint32_t dispatch_id, uint32_t process_id,
                                        uint32_t expected_total_waves, uint32_t wfs_per_workgroup,
                                        uint32_t lds_bytes, uint32_t scratch_wave_limit,
                                        const std::vector<ComputeUnitCore *> &eligible_cus) {
  std::lock_guard<std::mutex> lock(mutex_);
  DispatchAdmission &adm = admission_[dispatch_id];
  adm.process_id = process_id;
  adm.expected = std::max(adm.expected, expected_total_waves);
  adm.wfs_per_workgroup = wfs_per_workgroup;
  adm.lds_bytes = lds_bytes;
  adm.scratch_wave_limit = scratch_wave_limit;
  // Union the eligible CU set: a fan-out grid keeps one dispatch id across XCDs,
  // so each XCD's command processor contributes the CUs it may place this grid's
  // workgroups on (entry.allows_cu). Dedup keeps repeated publications idempotent.
  for (ComputeUnitCore *cu : eligible_cus)
    if (std::find(adm.eligible_cus.begin(), adm.eligible_cus.end(), cu) == adm.eligible_cus.end())
      adm.eligible_cus.push_back(cu);
  // Placement feasibility may have changed (new eligible CUs); force peers to
  // republish before any escape decision trusts an earlier snapshot.
  ++summary_epoch_;
}

void GwsDevice::remove_resident(uint32_t dispatch_id, uint32_t count) {
  if (count == 0)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = resident_waves_.find(dispatch_id);
  if (it == resident_waves_.end())
    return;
  it->second = it->second > count ? it->second - count : 0;
  if (it->second == 0) {
    resident_waves_.erase(it);
    // A dispatch that has admitted its whole grid and now holds no resident wave
    // is finished; drop its participant accounting so it no longer blocks the
    // quiescence scan. A dispatch still mid-admission can transiently reach zero
    // residents (rocjitsu admits workgroups over time), so keep its accounting
    // until admission completes -- notify_dispatch_gone handles early teardown.
    auto ait = admission_.find(dispatch_id);
    if (ait != admission_.end() && ait->second.admitted >= ait->second.expected)
      admission_.erase(ait);
  }
}

void GwsDevice::remove_admitted(uint32_t dispatch_id, uint32_t count) {
  if (count == 0)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = admission_.find(dispatch_id);
  if (it == admission_.end())
    return; // No accounting tracked (size-unknown dispatch): nothing to un-admit.
  it->second.admitted = it->second.admitted > count ? it->second.admitted - count : 0;
  // A rolled-back producer may have been masking pending work; force peers to
  // republish so the quiescence scan re-evaluates against the corrected count.
  ++summary_epoch_;
}

void GwsDevice::notify_dispatch_gone(uint32_t dispatch_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  // A torn-down dispatch will never admit its remaining grid, so drop its whole-
  // dispatch participant accounting. Residency is a per-CU contribution settled by
  // each CU's own remove_resident as it aborts its share (abort_dispatch is called
  // on every CU), so leave resident_waves_ to that path -- erasing it here would
  // wipe a peer CU's not-yet-aborted contribution and transiently understate its
  // barrier bound.
  admission_.erase(dispatch_id);
  // Removing pending work may permit a previously-blocked escape; force peers to
  // republish before any escape trusts a pre-teardown snapshot.
  ++summary_epoch_;
}

void GwsDevice::release_process(uint32_t process_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  resources_.erase(process_id);
  auto pit = parked_.find(process_id);
  if (pit != parked_.end()) {
    // A well-drained process has no parked waves; keep the global fast-path gate
    // consistent if teardown ever races a straggler.
    if (pit->second != 0)
      parked_total_.fetch_sub(pit->second, std::memory_order_release);
    parked_.erase(pit);
  }
  std::erase_if(admission_,
                [process_id](const auto &e) { return e.second.process_id == process_id; });
  for (auto &per_cu : cu_runnable_)
    per_cu.second.erase(process_id);
  ++summary_epoch_;
}

size_t GwsDevice::tracked_process_count() {
  std::lock_guard<std::mutex> lock(mutex_);
  return resources_.size();
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
  // 0 -> positive opens a fresh snapshot version, invalidating stale per-CU
  // summaries so a producer that has not yet published cannot be mistaken for
  // quiescent at the start of a parking episode.
  if (parked_total_.fetch_add(1, std::memory_order_release) == 0)
    ++summary_epoch_;
}

void GwsDevice::wake_wave(ComputeUnitCore &cu, Wavefront &wf) {
  const uint32_t process = wf.process_id();
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
    auto rit = resources_.find(process);
    if (rit != resources_.end())
      release_sema_waiter(rit->second[rid], seq);
  }
  auto it = parked_.find(process);
  if (it != parked_.end() && it->second > 0 && --it->second == 0)
    parked_.erase(it);
  parked_total_.fetch_sub(1, std::memory_order_release);
  // A woken wave is runnable again (a potential signaler). Advance the snapshot
  // version so a peer escape decision cannot treat a pre-wake zero-runnable
  // summary as quiescent before the owning CU republishes. The waking CU
  // republishes in the same step (refresh_summary runs right after
  // poll_local_waiters), so only peer summaries are invalidated.
  ++summary_epoch_;
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
    // A semaphore P consumes one credit left by a V/BR (possibly on another CU),
    // but only on behalf of the oldest queued waiters: this wave may take a credit
    // only when fewer than `credits` older waiters are still parked on the resource
    // (across all CUs). That releases V/BR credits in arrival order, so a later P
    // -- even one polled first because it shares the signaling CU -- cannot jump an
    // earlier waiter parked on another CU (see the entitlement note in the class doc).
    if (!wf.gws_wait_is_barrier_ && res.credits > 0) {
      uint32_t older = 0;
      for (uint64_t queued_seq : res.sema_wait_seqs)
        if (queued_seq < wf.gws_wait_seq_)
          ++older;
      if (older < res.credits) {
        --res.credits;
        wake_wave(cu, wf); // Removes this wave's stamp from sema_wait_seqs.
        ++woke;
      }
    }
  }
  return woke;
}

void GwsDevice::invalidate_snapshots_for_pending_release() {
  if (parked_total_.load(std::memory_order_acquire) != 0)
    ++summary_epoch_;
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
  cu_summary_epoch_[&cu] = summary_epoch_;
}

bool GwsDevice::escape_ready(uint32_t process_id) const {
  auto parked_it = parked_.find(process_id);
  if (parked_it == parked_.end() || parked_it->second == 0)
    return false;
  // A dispatch of this process that has not yet admitted its whole grid may still
  // produce a signaler (a producer workgroup not placed yet), but only if that
  // producer can still be placed. Mirror CommandProcessor::dispatch_workgroups():
  // the producer is admittable iff some CU the dispatch is allowed to use can
  // currently accept its whole workgroup (slots + registers + LDS + scratch).
  //  * If an eligible CU can accept it, the producer can still be placed and could
  //    signal, so do not escape yet -- bumping the version at its admission cannot
  //    undo an escape that already fired.
  //  * If no eligible CU can accept it, the producer is gated behind the parked
  //    waves being judged -- a genuine deadlock the backstop must break. Blocking
  //    unconditionally would forfeit the termination guarantee (a parked GWS wave
  //    counts as runnable, so the CU would spin on it forever). Fall through.
  // can_accept_workgroup() takes no CU wave lock, so these cross-CU reads stay
  // within the leaf-lock protocol. A dispatch whose placement requirements were
  // never published (standalone/test) cannot be proven admittable and is skipped.
  for (const auto &[dispatch_id, adm] : admission_) {
    (void)dispatch_id;
    if (adm.process_id != process_id || adm.admitted >= adm.expected)
      continue;
    if (adm.wfs_per_workgroup == 0 || adm.eligible_cus.empty())
      continue; // Placement requirements unknown: cannot prove this is admittable.
    for (ComputeUnitCore *cu : adm.eligible_cus) {
      if (cu->can_accept_workgroup(adm.wfs_per_workgroup, adm.lds_bytes, adm.scratch_wave_limit))
        return false; // A pending producer can still be placed and signal.
    }
  }
  uint64_t total_runnable = 0;
  for (ComputeUnitCore *cu : cus_) {
    // has_active_wfs() performs the same acquire load + low-word test on the CU's
    // packed wave_activity_, keeping that representation inside the CU. It is a
    // cross-CU read, race-free even while the peer steps.
    if (!cu->has_active_wfs())
      continue; // Idle CU: hosts no waves of any process, contributes nothing.
    auto eit = cu_summary_epoch_.find(cu);
    if (eit == cu_summary_epoch_.end() || eit->second != summary_epoch_)
      return false; // An active CU has not republished since the latest signaler.
    auto rit = cu_runnable_.find(cu);
    if (rit == cu_runnable_.end())
      continue;
    auto pit = rit->second.find(process_id);
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
  Resource &res = resources_[wf.process_id()][rid];
  // The still-outstanding arrivals that must occur to release this wave are the
  // current counter (a fresh resource seeds that counter from this arrival's own
  // value -- see the first-arrival self-init policy in the class doc). A blocking
  // arrival needs (outstanding + 1) participants co-resident; a releasing arrival
  // (outstanding == 0) never blocks and is exempt from the gate below. Fall back to
  // a non-blocking structural no-op only when that set can never be co-resident.
  // When the dispatch's grid size is known, bound by it: late participants are
  // merely not admitted yet, so the arrival parks and a later arrival (or the
  // backstop) resolves it. Otherwise bound by the currently-resident waves. The
  // test is written as (outstanding >= bound), equivalent to (outstanding + 1 >
  // bound) for integers but overflow-safe for a count near UINT32_MAX (a huge
  // count takes the fallback, never wraps the sum to zero and parks).
  const uint32_t outstanding = res.armed ? res.counter : count;
  if (outstanding != 0) {
    const auto adm = admission_.find(wf.dispatch_id());
    if (adm != admission_.end() && adm->second.expected != 0) {
      if (outstanding >= adm->second.expected)
        return;
    } else {
      const auto resident_it = resident_waves_.find(wf.dispatch_id());
      const uint32_t resident = resident_it != resident_waves_.end() ? resident_it->second : 0;
      if (outstanding >= resident)
        return;
    }
  }
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
    invalidate_snapshots_for_pending_release();
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
  invalidate_snapshots_for_pending_release();
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
  if (res.credits > res.sema_wait_seqs.size()) {
    --res.credits;
    return;
  }
  // No free credit: park until any wave of this process signals the shared
  // resource. A later V/BR leaves a credit this wave's CU consumes on its next
  // poll; a release_all bumps the generation; the quiescence backstop is the last
  // resort.
  park_wave(wf, rid, /*is_barrier=*/false, res.release_gen);
}

void GwsDevice::sema_br(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid, uint32_t count) {
  if (rid >= kResourceCount || count == 0)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  Resource &res = resources_[wf.process_id()][rid];
  res.credits += count;
  poll_local_waiters(cu); // Same-CU waiters drain credits now; the rest persist.
  invalidate_snapshots_for_pending_release();
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
  invalidate_snapshots_for_pending_release();
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

void GwsDevice::invalidate_resource(uint32_t process_id, uint32_t rid) {
  if (rid >= kResourceCount)
    return;
  Resource &res = resources_[process_id][rid];
  res.counter = 0;
  res.credits = 0;
  res.armed = false;
  // Bump the generation so any peer-CU waiter still parked on this resource also
  // abandons it on its next poll rather than observing a stale pre-escape phase.
  ++res.release_gen;
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
    const uint32_t process_id = slot->process_id();
    auto rit = ready.find(process_id);
    if (rit == ready.end())
      rit = ready.emplace(process_id, escape_ready(process_id)).first;
    if (rit->second) {
      // The backstop only reaches here for a process proven quiescent by
      // escape_ready() (no resident wave can still signal and no pending producer
      // can still be placed). Assert the wake-path invariant -- only a still-parked
      // wave is ever abandoned -- so a regression that reaches this branch for a
      // runnable wave is caught in debug builds. (escape_ready() is not re-checked
      // here: waking the first wave of the process bumps summary_epoch_, which would
      // make a re-check spuriously fail for a sibling woken in the same pass.)
      assert(slot->state() == WfState::GWS_WAIT &&
             "GWS deadlock backstop must wake only a still-parked wave");
      report_backstop_escape(cu, *slot, process_id);
      // Abandoning a genuinely deadlocked rendezvous is not a silent wake: drop
      // the resource's unfinished phase so a later dispatch reusing this rid does
      // not inherit a stale counter/credit (recovery policy, see class doc).
      invalidate_resource(process_id, slot->gws_wait_rid_);
      wake_wave(cu, *slot);
    }
  }
}

void GwsDevice::report_backstop_escape(const ComputeUnitCore &cu, const Wavefront &wf,
                                       uint32_t process_id) {
  // Rate-limited recovery diagnostic: abandoning a wait is a real event (a genuine
  // deadlock the model broke to keep the step budget finite), not a silent state
  // change. The backstop can release many waves in one pass, so emit at most one
  // line per interval and fold the rest into a suppressed count. Guarded by mutex_
  // (held by the caller), so the throttle state needs no separate lock.
  const auto now = std::chrono::steady_clock::now();
  const bool first = last_escape_diag_.time_since_epoch().count() == 0;
  if (!first && now - last_escape_diag_ < std::chrono::seconds(1)) {
    ++escape_diag_suppressed_;
    return;
  }
  util::Logger::warn("GWS deadlock backstop released a parked wave: process=", process_id,
                     " resource=", wf.gws_wait_rid_,
                     wf.gws_wait_is_barrier_ ? " (barrier)" : " (semaphore)",
                     " dispatch=", wf.dispatch_id(), " cu=", cu.name(),
                     escape_diag_suppressed_ != 0 ? " [" + std::to_string(escape_diag_suppressed_) +
                                                        " more suppressed since last report]"
                                                  : std::string());
  last_escape_diag_ = now;
  escape_diag_suppressed_ = 0;
}

} // namespace amdgpu
} // namespace rocjitsu
