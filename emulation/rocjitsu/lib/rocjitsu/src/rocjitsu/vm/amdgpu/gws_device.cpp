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
  // Defense-in-depth bookkeeping reconcile, the mirror of unregister_compute_unit's
  // reconcile-out. set_gws_device() now rejects re-pointing a CU that still hosts
  // active waves (a live rendezvous cannot carry its resource phase across stores),
  // so a CU joining here is normally drained and these loops are a no-op. They
  // remain so that if a CU ever did arrive with residual parked/resident state, the
  // fast-path gate and quiescence scan would still count it consistently (otherwise
  // a later notify_parked_wave_gone could decrement a count that was never
  // incremented and wrongly cancel an unrelated waiter already parked here).
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
  // A CU joining this store adds execution slots, so a producer the CP previously
  // could not place may now fit: re-arm any shard recorded Unplaced against the
  // pre-join capacity. (At initial topology wiring no shard exists yet, so this is
  // harmless; it matters only when a drained CU is later re-pointed into the store.)
  ++capacity_epoch_;
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
  // A removed CU changes the quiescence picture; force peers to republish before
  // any escape decision trusts a snapshot taken while this CU was still listed.
  // (Pending-shard keys are opaque CP identities, never CU pointers, so a departing
  // CU leaves no dangling placement reference to scrub -- unlike the former eligible
  // CU set.)
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

void GwsDevice::set_shard_placement(uint32_t process_id, uint32_t dispatch_id,
                                    const void *shard_key, ShardPlacement state) {
  std::lock_guard<std::mutex> lock(mutex_);
  switch (state) {
  case ShardPlacement::Pending: {
    // Accepted but not yet attempted: a future signaler whose placement the CP has
    // not even tried. Record the never-attempted sentinel so escape defers
    // unconditionally until the CP works it.
    DispatchAdmission &adm = admission_[dispatch_id];
    adm.process_id = process_id;
    adm.shard_attempt_epoch[shard_key] = kShardNeverAttempted;
    // Force peers to republish before any escape decision trusts a snapshot taken
    // before this shard was known.
    ++summary_epoch_;
    return;
  }
  case ShardPlacement::Unplaced: {
    // The CP attempted placement and some workgroup did not fit. Record the current
    // capacity version (@ref capacity_epoch_, distinct from the quiescence version):
    // escape defers while this stays current (genuinely stuck -> terminates) and
    // re-arms only if a later event that frees room advances the capacity version
    // before the CP re-attempts against it. Tagging against the capacity version --
    // not the quiescence version -- is deliberate: a wave merely parking advances
    // quiescence without freeing any slot, so a stalled serial dispatch (no sibling
    // to retire and re-trigger placement) must still be allowed to escape.
    DispatchAdmission &adm = admission_[dispatch_id];
    adm.process_id = process_id;
    adm.shard_attempt_epoch[shard_key] = capacity_epoch_;
    return;
  }
  case ShardPlacement::Resolved: {
    // Fully placed or discarded: drop the shard (no-op if the dispatch was already
    // torn down or the shard was never published).
    auto it = admission_.find(dispatch_id);
    if (it == admission_.end())
      return;
    if (it->second.shard_attempt_epoch.erase(shard_key) != 0)
      // Removing a deferring shard may unblock a previously-deferred escape; force
      // peers to republish so the quiescence scan re-evaluates against the cleared
      // state.
      ++summary_epoch_;
    return;
  }
  }
}

void GwsDevice::remove_resident(uint32_t dispatch_id, uint32_t count) {
  if (count == 0)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = resident_waves_.find(dispatch_id);
  if (it == resident_waves_.end())
    return;
  it->second = it->second > count ? it->second - count : 0;
  // A retirement frees CU capacity, so a producer the CP previously could not place
  // may now fit. Advance the capacity version only: any shard recorded Unplaced at an
  // older capacity version is now stale and must defer escape until the CP re-attempts
  // placement against the freed room (see set_shard_placement / escape_ready). A
  // retirement removes a potential signaler rather than introducing one, so it does
  // not need to invalidate the quiescence snapshots.
  ++capacity_epoch_;
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
  // Rolling the wave back also frees the slot it had committed, so re-arm any shard
  // the CP recorded Unplaced against the pre-rollback capacity.
  ++capacity_epoch_;
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
  // Tearing the dispatch down frees whatever slots its waves held, so re-arm any
  // sibling shard the CP recorded Unplaced against the pre-teardown capacity.
  ++capacity_epoch_;
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
  // Reclaiming the process frees every slot its waves held, so re-arm any shard a
  // surviving process's CP recorded Unplaced against the pre-teardown capacity.
  ++capacity_epoch_;
}

size_t GwsDevice::tracked_process_count() {
  std::lock_guard<std::mutex> lock(mutex_);
  return resources_.size();
}

size_t GwsDevice::tracked_dispatch_count() {
  std::lock_guard<std::mutex> lock(mutex_);
  return admission_.size();
}

size_t GwsDevice::tracked_shard_count(uint32_t dispatch_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = admission_.find(dispatch_id);
  return it == admission_.end() ? 0 : it->second.shard_attempt_epoch.size();
}

size_t GwsDevice::tracked_resident_count(uint32_t dispatch_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = resident_waves_.find(dispatch_id);
  return it == resident_waves_.end() ? 0 : it->second;
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
  // A woken wave is runnable again (a potential signaler). Advance the snapshot
  // version so a peer escape decision cannot treat a pre-wake zero-runnable
  // summary as quiescent before the owning CU republishes. The waking CU
  // republishes in the same step (refresh_summary runs right after
  // poll_local_waiters), so only peer summaries are invalidated.
  ++summary_epoch_;
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
    // only when fewer than `credits` older waiters are still parked on the resource
    // (across all CUs). That releases V/BR credits in arrival order, so a later P
    // -- even one polled first because it shares the signaling CU -- cannot jump an
    // earlier waiter parked on another CU (see the entitlement note in the class doc).
    if (!wf.gws_wait_is_barrier_ && res.credits > 0) {
      uint32_t older_waiter_count = 0;
      for (uint64_t queued_seq : res.sema_wait_seqs)
        if (queued_seq < wf.gws_wait_seq_)
          ++older_waiter_count;
      if (older_waiter_count < res.credits) {
        --res.credits;
        wake_wave(cu, wf); // Removes this wave's stamp from sema_wait_seqs.
        ++woken_count;
      }
    }
  }
  return woken_count;
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
  // A dispatch of this process with a shard the command processor has not yet
  // placed may still produce a signaler, so defer. Placement lives in the CP (it
  // alone knows eligible CUs, WGP/PM4 capacity, and it places greedily so a
  // placeable producer is already resident -- see the class doc); the store only
  // reads the CP-published placement records, never re-deriving placement (which
  // would race the owning CU's allocator). A shard defers escape while it is either
  // never-attempted (Pending) or recorded Unplaced at a stale capacity version --
  // the latter means a retirement has freed room since the CP last tried, so it must
  // re-attempt before the wait is abandoned. A dispatch with no shard published is
  // governed by the quiescence scan below.
  for (const auto &[dispatch_id, adm] : admission_) {
    (void)dispatch_id;
    if (adm.process_id != process_id)
      continue;
    for (const auto &[shard_key, attempt_epoch] : adm.shard_attempt_epoch) {
      (void)shard_key;
      if (attempt_epoch != capacity_epoch_)
        return false; // Never-attempted (sentinel) or stale: the CP may still place.
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
  // Seeding semaphore credits is a release a waiter on another CU may consume on a
  // later step (an init can run on a separate dispatch/CU from the P it feeds), so
  // it must invalidate the peers' quiescence snapshots exactly like V/BR/release_all:
  // otherwise a peer could escape a parked waiter through the backstop on the
  // strength of a pre-init zero-runnable summary before the seeded credit wakes it.
  invalidate_snapshots_for_pending_release();
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
