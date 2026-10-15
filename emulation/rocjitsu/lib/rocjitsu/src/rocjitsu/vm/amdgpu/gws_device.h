// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gws_device.h
/// @brief Device-global Global Wave Sync (GWS) resource store shared by CUs.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

class ComputeUnitCore;
class Wavefront;

/// @brief Device-global Global Wave Sync (GWS) resource store shared by CUs.
///
/// This class is the single authoritative description of the GWS scheduling
/// policy; other GWS sites (ComputeUnitCore's forwarding hooks, the
/// implementation file) carry only brief API notes and reference this doc.
///
/// @details Hardware GWS resources are a small fixed set of global registers,
/// allocated per process and shared by every wave on the device regardless of
/// which compute unit or dispatch it belongs to. Unlike @c s_barrier (which
/// synchronizes waves within one workgroup, hence one CU), GWS exists precisely
/// to synchronize waves in *different* workgroups of a grid, which the command
/// processor scatters across many CUs. The resources also persist across
/// dispatches: software can seed a resource with a separate init kernel and
/// consume it from a later kernel (CLR's @c RunGwsInit, @c KFDGWSTest.Semaphore).
///
/// This object models that scope. One resource table is kept per @c process_id
/// (the 6-bit resource id selects within it), shared across every registered CU
/// and never torn down on workgroup or dispatch retirement. A signal/arrival
/// from any wave reaches the same entry and releases a waiter parked by any other
/// wave (any workgroup, any CU, even a later dispatch).
///
/// Thread-safety model (the central design constraint): CUs step concurrently,
/// each under its own wave-state lock, so a signaling CU must never touch another
/// CU's wavefront state. The store mutex is therefore a pure leaf lock guarding
/// only the resource/bookkeeping tables; it never acquires a CU wave lock. All
/// wavefront mutations happen on the owning CU's own thread:
///  * Same-CU wakeups are applied immediately (the signaling wave's CU owns the
///    waiter too), so single-CU rendezvous resolve within one op.
///  * Cross-CU wakeups are published as resource state (credits / a release
///    generation) and applied by the waiter's own CU the next time it runs
///    @ref step_maintenance (driven by update_wf_states each step). This is the
///    only serialization boundary the functional model needs.
/// Global lock order is "any CU wave-state lock -> store mutex_"; there is no
/// inverse edge, so no deadlock.
///
/// Per-operation policy:
///  * Barrier: hardware/LLVM program the resource with (participants - 1); the
///    modeled barrier pseudocode (CDNA/LLVM; the RDNA3/3.5 variant differs -- see
///    the per-ISA modeling notes below) queues an arrival while the counter is
///    positive and, on the arrival that observes zero, releases every queued
///    arrival and reloads the counter from that arrival's own value (so
///    consecutive phases may differ in size). The release bumps a per-resource
///    generation that parked arrivals (on any CU) observe to wake.
///
///    Emulator policy -- first-arrival self-initialization: an arrival at a
///    resource not armed by an explicit @c init seeds the counter from its own
///    operand. Hardware (RDNA3/3.5 Table 63) requires a prior @c init and behavior
///    without one is not validated, so cooperative kernels are expected to @c init
///    first; this default only lets a test, or a phase recovering after the
///    deadlock backstop invalidated its resource, proceed instead of stalling.
///
///    Participant bound (pending-work aware): a blocking arrival must prove the
///    still-required set (counter + 1) can be co-resident before it parks. rocjitsu
///    admits a grid's workgroups over time -- and independently per XCD through
///    drain_fanout_inbox() -- so "currently resident" understates a grid still
///    being admitted. When the dispatch's total size is known (@ref add_resident
///    with a nonzero expected), the bound is that whole-grid size: the arrival
///    parks whenever (counter + 1) <= grid size (late participants are merely not
///    admitted yet), and only a count larger than the entire grid falls back to a
///    non-blocking structural no-op. When the size is unknown (standalone/test) the
///    bound is the currently-resident set, as before; the comparison is
///    overflow-safe for counts near UINT32_MAX. Concurrent dispatches of one
///    process sharing a single barrier resource is not a supported rendezvous
///    (hardware treats a cooperative grid as the sole owner of its barrier); the
///    bound is intentionally dispatch-scoped.
///  * Semaphore: V/BR add credits and @c release_all bumps the generation; P
///    consumes a credit or parks until any wave of the process signals. Credits
///    and the generation are consumed/observed by the waiter's own CU. A credit is
///    reserved for waiters already parked on the resource: a fresh P consumes a
///    credit only when the credit count exceeds the parked-waiter count (@ref
///    Resource::sema_waiters). This stops a P on the signaling CU from stealing a
///    V/BR release owed to a waiter parked on another CU (which applies the wake on
///    its own next step), matching the hardware rule that V/BR release queued waves.
///  * Deadlock-escape (@ref step_maintenance): a parked GWS wave is released only
///    when every non-halted *resident* wave of its process -- across all CUs --
///    is provably unable to signal (blocked in GWS_WAIT or at an s_barrier). This
///    is a genuine-deadlock backstop (the step budget must terminate); it never
///    fires while any resident wave of the process can still run and signal.
///    Quiescence is process-scoped (another dispatch of the same process may
///    still signal) and race-free: each CU publishes a snapshot of its own
///    signalable-wave count, and escape requires every currently-active CU to
///    have republished at the current snapshot version (@ref summary_epoch_), so
///    a runnable producer on another CU is never missed. The version advances on
///    every event that may introduce a signaler -- a wave parking (0 -> positive),
///    a wave waking, or new resident work arriving (@ref add_resident) -- so a
///    stale zero-runnable summary cannot survive workgroup turnover (a parked
///    wave replaced by a runnable producer) or a producer dispatched alongside a
///    parked wave. The guarantee covers waves that are *resident* when the scan
///    runs plus waves the process has not admitted yet, but only while the latter
///    can still be admitted. Placement is the command processor's responsibility,
///    not the store's: CommandProcessor::dispatch_workgroups() alone knows a
///    dispatch's eligible CUs, its WGP paired-LDS/CU-mode reservations, and its PM4
///    scratch pool, and it places greedily -- any workgroup it *can* place it places
///    at once, so a placeable producer immediately becomes a resident, runnable wave
///    (covered by the quiescence scan above). The store therefore never re-derives
///    placement (which would both race the owning CU's allocator and misjudge the
///    WGP/PM4 constraints it cannot see). Instead the command processor publishes
///    each shard's placement state (@ref set_shard_placement), tagged against the
///    capacity version @ref capacity_epoch_ (distinct from the quiescence version;
///    it advances only when free capacity may have *increased* -- a wave retired or
///    rolled back, a dispatch torn down, a process reclaimed, or a CU added):
///     * @c Pending -- accepted but not yet attempted (a fan-out shard still in an
///       inbox, or the owner's own local share registered before fan-out). Always
///       defers: its producer may still be placed once the CP reaches it.
///     * @c Unplaced -- the CP attempted placement and some workgroup did not fit,
///       recorded at the capacity version of that attempt. Defers only while that
///       record is stale (capacity has changed since, so the CP must re-attempt
///       against the freed room before the wait is abandoned); once the CP
///       re-attempts at the current version and still cannot place, the work is
///       genuinely unplaceable behind the parked waves and escape proceeds.
///     * @c Resolved -- fully placed or discarded; drops the shard.
///    Escape is blocked while any shard of the process's dispatches is Pending or
///    has a stale Unplaced record. This re-arms on capacity changes for single-CP
///    grids too: retiring a sibling advances the version, so a parked waiter is not
///    abandoned in the step between that retirement and the CP's next placement pass
///    on the freed slot. Blocking unconditionally on unadmitted work would forfeit
///    termination (a parked GWS wave counts as runnable, so its CU would spin on it
///    forever); tying the Unplaced record to a monotonic capacity version keeps the
///    deadlock case terminating (with no capacity change the record stays current,
///    so escape fires) while deferring across every real re-placement opportunity.
///    Because that version tracks only capacity-freeing events and not quiescence,
///    a wave merely parking does not invalidate the record: a stalled serial
///    dispatch (no sibling to retire and re-trigger placement) still escapes once
///    the CP confirms the work genuinely does not fit, instead of deferring forever.
///    A dispatch with no shard state published (standalone/test use) is governed by
///    the quiescence scan alone.
///
///    Recovery is observable: when the backstop abandons a wait it emits a
///    rate-limited diagnostic (process, resource, wave location) so a genuine
///    deadlock is visible rather than silently papered over (@ref step_maintenance).
///    Recovery policy: abandoning a wait is not a silent state change. The
///    backstop *invalidates* each resource it abandons (counter/credits cleared,
///    disarmed, generation bumped) via @ref invalidate_resource, so a later
///    dispatch reusing that rid begins a fresh phase instead of inheriting the
///    unfinished one.
///
/// Per-ISA modeling notes (no GFX11 hardware was available to validate this
/// review; where ISA manuals disagree the model follows one interpretation and
/// records the discrepancy):
///  * Barrier release ordering: CDNA1-3 and RDNA1/2 test the counter before
///    decrementing; RDNA3/3.5 (Table 63) decrement first, and their
///    "participants - 1" wording would make @c init(1) followed by the first
///    @c barrier(1) release immediately. The model implements the CDNA/LLVM
///    test-then-decrement reading uniformly; the RDNA3/3.5 decrement-first
///    variant is not modeled.
///  * @c release_all credit accounting: CDNA1-3 and RDNA1/2 assign the queued-
///    wave count to the counter (so with no queued waiter it clears to zero),
///    whereas RDNA3/3.5 (Table 63) add the queued-wave count (leaving prior
///    banked credits intact). The model implements the assignment reading
///    (credits cleared); the RDNA3/3.5 additive variant is not modeled.
class GwsDevice {
public:
  /// @brief Number of GWS resources (6-bit resource id space).
  static constexpr uint32_t kResourceCount = 64;

  /// @brief Register a CU so its waves participate in shared wake/quiescence.
  void register_compute_unit(ComputeUnitCore *cu);
  /// @brief Stop routing shared wake/quiescence to a CU being destroyed.
  void unregister_compute_unit(ComputeUnitCore *cu);

  /// @brief Seed a GWS resource's barrier count / semaphore credits.
  void init(Wavefront &wf, uint32_t rid, uint32_t count);
  /// @brief Arrive at a GWS barrier (may park @p wf). See class doc for policy.
  void barrier_arrive(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid, uint32_t count);
  /// @brief Signal (V) a semaphore: add one credit and wake a waiter.
  void sema_v(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid);
  /// @brief Wait (P) on a semaphore: consume a credit, else park @p wf.
  void sema_p(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid);
  /// @brief Bulk-signal (BR): add @p count credits and wake up to that many.
  void sema_br(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid, uint32_t count);
  /// @brief Release every wave parked on a resource and clear its credits.
  void sema_release_all(ComputeUnitCore &cu, Wavefront &wf, uint32_t rid);

  /// @brief Reconcile parked bookkeeping for a wave halted/aborted while parked.
  /// @details A wave freed outside the normal wake path (s_endpgm halt, dispatch
  /// abort) never decremented the parked counters; the CU calls this before the
  /// wave's park state is cleared so the fast-path gate and quiescence stay
  /// accurate. Must be called while @p wf is still in GWS_WAIT.
  void notify_parked_wave_gone(const Wavefront &wf);

  /// @brief Record @p count resident (not-yet-retired) waves of a dispatch, and
  /// (when @p expected_total_waves is nonzero) the dispatch's total participant
  /// count so the barrier bound and quiescence scan can account for waves not yet
  /// admitted (see the pending-work note in the class doc).
  /// @details Maintained by the CU as workgroups begin; @p expected_total_waves is
  /// the whole-grid wave count, reported identically by every CU/XCD of the
  /// dispatch (kept as a max, so repeats and shard order do not matter). Pass zero
  /// for standalone/test use that does not model late admission; the dispatch is
  /// then bounded only by currently-resident waves, as before.
  void add_resident(uint32_t process_id, uint32_t dispatch_id, uint32_t count,
                    uint32_t expected_total_waves);
  /// @brief Placement state a command processor publishes for one shard.
  /// @details See the escape-policy note in the class doc for the full semantics.
  enum class ShardPlacement {
    /// Accepted but not yet attempted for placement (defers unconditionally).
    Pending,
    /// Attempted; some workgroup did not fit (defers only while the record is
    /// stale against the current capacity version @ref summary_epoch_).
    Unplaced,
    /// Fully placed or discarded (drops the shard).
    Resolved,
  };
  /// @brief Publish a shard's placement state so the quiescence backstop defers
  /// while a not-yet-placed producer may still arrive (see @ref escape_ready and
  /// the escape-policy note in the class doc).
  /// @details Placement lives entirely in the command processor, which calls this
  /// as it works a dispatch: @c Pending when it accepts a fan-out shard into its
  /// inbox (or registers its own local share before fan-out), @c Unplaced when a
  /// placement pass could not fit every workgroup (recorded at the current capacity
  /// version so escape re-arms when a sibling retirement frees room), and
  /// @c Resolved once the shard is fully placed or discarded. @p shard_key is an
  /// opaque, never-dereferenced identity for the shard's owning command processor,
  /// so each shard of one dispatch is tracked independently. Argument order matches
  /// @ref add_resident (process then dispatch). A dispatch with no state published
  /// is governed by the quiescence scan alone.
  void set_shard_placement(uint32_t process_id, uint32_t dispatch_id, const void *shard_key,
                           ShardPlacement state);
  /// @brief Drop @p count resident waves of a dispatch (clamped at zero).
  void remove_resident(uint32_t dispatch_id, uint32_t count);
  /// @brief Un-admit @p count waves of a dispatch rolled back before execution.
  /// @details A workgroup committed via @ref add_resident but rolled back (its
  /// clustered peers failed to dispatch, so it never ran and will be re-admitted
  /// on retry) must decrement the cumulative admitted count; otherwise the retry
  /// double-counts and a prematurely-reached @c admitted==expected could stop the
  /// quiescence scan from waiting for genuinely-pending work. Unlike
  /// @ref remove_resident this is NOT called on normal retirement -- an executed
  /// wave stays counted as admitted. Keeps the entry (expected is still the grid
  /// size); @ref notify_dispatch_gone handles whole-dispatch teardown.
  void remove_admitted(uint32_t dispatch_id, uint32_t count);
  /// @brief Forget a dispatch's participant accounting (abort/teardown).
  /// @details Called when a dispatch is torn down before it fully admits+retires
  /// (e.g. abort), so its never-to-arrive participants stop blocking the quiescence
  /// scan. Normal full retirement clears the accounting via @ref remove_resident.
  void notify_dispatch_gone(uint32_t dispatch_id);

  /// @brief Per-step maintenance for the calling CU, on its own thread.
  /// @details Applies cross-CU wakeups to @p cu's own parked waves, refreshes
  /// @p cu's published signalable-wave snapshot, and runs the genuine-deadlock
  /// backstop for @p cu's own parked waves. Cheap lock-free no-op when nothing is
  /// parked anywhere (see @ref any_parked). Mutates only @p cu's wavefronts.
  void step_maintenance(ComputeUnitCore &cu);

  /// @brief True while any wave is parked anywhere (lock-free fast-path gate).
  bool any_parked() const { return parked_total_.load(std::memory_order_acquire) != 0; }

  /// @brief Reclaim a departed process's GWS state after its waves/queues drain.
  /// @details GWS resources persist across dispatches, so they are not freed on
  /// dispatch retirement; a long-running driver would otherwise retain a table per
  /// departed process. The process-teardown path calls this once the process's
  /// waves and queues are gone, so live processes keep their resources while the
  /// departed one's table (and bookkeeping) is released. Preserves the leaf-lock
  /// protocol (takes only @ref mutex_).
  void release_process(uint32_t process_id);

  /// @brief Number of processes with live GWS resource tables (test/diagnostic).
  size_t tracked_process_count();

  /// @brief Number of dispatches with live participant accounting (test/diagnostic).
  /// @details Lets a regression observe that a dropped/aborted dispatch's admission
  /// entry was cleared (so it no longer lingers after teardown).
  size_t tracked_dispatch_count();

  /// @brief Number of unresolved placement shards recorded for @p dispatch_id
  /// (test/diagnostic). Zero if the dispatch has no accounting. Lets a regression
  /// observe shard registration/clearing through the real CP fan-out/drain paths.
  size_t tracked_shard_count(uint32_t dispatch_id);

  /// @brief Resident (not-yet-retired) wave count recorded for @p dispatch_id
  /// (test/diagnostic). Zero if none remain. Lets a lifecycle test confirm that
  /// halting a wave actually retired its residency (via remove_resident keyed by the
  /// wave's own dispatch id) rather than leaving a phantom resident behind until CU
  /// teardown.
  size_t tracked_resident_count(uint32_t dispatch_id);

private:
  /// @brief Per-resource barrier/semaphore state.
  struct Resource {
    /// Live barrier counter: decremented per arrival, reloaded from the releasing
    /// arrival's own value at zero (so consecutive phases may differ in size).
    uint32_t counter = 0;
    uint32_t credits = 0;     ///< Semaphore credits (V/BR add, P consumes).
    bool armed = false;       ///< Whether the barrier counter has been seeded.
    uint64_t release_gen = 0; ///< Bumped by a barrier release / sema release_all.
    /// Arrival-ordered sequence numbers of the semaphore P-waiters currently
    /// parked on this resource (any CU), oldest first. A V/BR credit is reserved
    /// for these already-parked waiters, so a fresh P consumes a credit only when
    /// the credit count exceeds this queue's size, and @ref poll_local_waiters
    /// lets a parked P consume a credit only when fewer than @c credits older
    /// waiters remain -- so a later P (on any CU) cannot jump an earlier parked
    /// waiter (see the entitlement note in the class doc). Barrier waiters wake
    /// via @c release_gen and are not queued here.
    std::vector<uint64_t> sema_wait_seqs;
  };

  /// @brief Per-dispatch participant accounting (see the pending-work note in the
  /// class doc). Lets the barrier bound and the quiescence scan reason about waves
  /// a dispatch will admit later, not only those resident now.
  struct DispatchAdmission {
    uint32_t process_id = 0; ///< Owning process (so the scan can find a process's dispatches).
    uint32_t expected = 0;   ///< Total waves the dispatch will ever admit (grid size).
    uint32_t admitted = 0;   ///< Cumulative waves admitted so far.
    /// Shards (identified by their owning command processor) with workgroups the CP
    /// has not yet placed (@ref set_shard_placement). The value is the capacity
    /// version (@ref capacity_epoch_) at which the CP last attempted placement, or
    /// @ref kShardNeverAttempted for a shard accepted but not yet attempted. While
    /// any entry is unattempted, or attempted at a stale capacity version, the
    /// quiescence backstop defers (the CP may still place a producer, possibly only
    /// after a sibling retirement frees room). The keys are opaque and never
    /// dereferenced (placement lives in the CP, not here).
    std::unordered_map<const void *, uint64_t> shard_attempt_epoch;
  };
  /// @brief Sentinel attempt-version for a shard accepted but not yet attempted.
  static constexpr uint64_t kShardNeverAttempted = ~uint64_t{0};

  /// @brief Drop a parked semaphore P-waiter's queued arrival stamp from @p res
  /// (no-op if absent). Requires @ref mutex_.
  void release_sema_waiter(Resource &res, uint64_t seq);
  /// @brief Park @p wf on @p rid (barrier or semaphore). Requires @ref mutex_.
  void park_wave(Wavefront &wf, uint32_t rid, bool is_barrier, uint64_t generation);
  /// @brief Wake @p wf running on @p cu, clearing its park state. Requires @ref mutex_.
  void wake_wave(ComputeUnitCore &cu, Wavefront &wf);
  /// @brief Wake @p cu's own parked waves whose credit/generation is satisfied.
  /// @returns the number of waves woken. Requires @ref mutex_.
  uint32_t poll_local_waiters(ComputeUnitCore &cu);
  /// @brief Invalidate per-CU quiescence snapshots after a release (credit or
  /// generation bump) that a waiter on another CU may still consume.
  /// @details A release is published as resource state and applied by the waiter's
  /// own CU on a later step, after this CU's @ref poll_local_waiters has woken any
  /// local waiters. If a waiter remains parked anywhere, advance @ref summary_epoch_
  /// so the quiescence backstop cannot treat the signaling CU's post-release
  /// zero-runnable snapshot as proof of quiescence before the peer applies (and
  /// republishes) the wake. Requires @ref mutex_.
  void invalidate_snapshots_for_pending_release();
  /// @brief Refresh @p cu's published signalable-wave snapshot for the current
  /// park epoch. Requires @ref mutex_.
  void refresh_summary(ComputeUnitCore &cu);
  /// @brief Whether process @p process_id is provably quiescent (genuine deadlock).
  /// Requires @ref mutex_ and an up-to-date snapshot from every active CU.
  bool escape_ready(uint32_t process_id) const;
  /// @brief Invalidate a resource abandoned by the deadlock backstop so a later
  /// dispatch cannot inherit its unfinished phase. Requires @ref mutex_.
  void invalidate_resource(uint32_t process_id, uint32_t rid);
  /// @brief Emit a rate-limited recovery diagnostic when the deadlock backstop
  /// abandons a wait (process, resource, wave location). Requires @ref mutex_.
  void report_backstop_escape(const ComputeUnitCore &cu, const Wavefront &wf, uint32_t process_id);

  std::mutex mutex_;
  std::unordered_map<uint32_t, std::array<Resource, kResourceCount>> resources_;
  std::vector<ComputeUnitCore *> cus_;

  /// Resident (not-yet-retired) wave count per dispatch; the legacy barrier bound
  /// when the dispatch's expected size is unknown (see @ref admission_).
  std::unordered_map<uint32_t, uint32_t> resident_waves_;
  /// Per-dispatch participant accounting (expected/admitted), keyed by dispatch id.
  /// Present only for dispatches whose total size was reported (@ref add_resident
  /// with a nonzero expected); the barrier bound and the quiescence scan use it to
  /// account for waves not yet admitted. Cleared on full retirement, dispatch
  /// teardown (@ref notify_dispatch_gone), or process teardown (@ref release_process).
  std::unordered_map<uint32_t, DispatchAdmission> admission_;
  /// Parked (GWS_WAIT) wave count per process; the deadlock-escape precondition.
  std::unordered_map<uint32_t, uint32_t> parked_;
  /// Total parked waves across all processes; lock-free fast-path gate.
  std::atomic<uint32_t> parked_total_{0};
  /// Monotonic version for the per-CU quiescence snapshots below. Bumped whenever a
  /// potential signaler may have appeared -- a 0 -> positive parked transition, a
  /// wave waking (wake_wave), new resident work (add_resident), or a shard being
  /// published/cleared (set_shard_placement Pending/Resolved). escape_ready trusts a
  /// CU's published summary only if it was refreshed at the current version, which
  /// closes the workgroup-turnover gap: a parked wave replaced by (or joined by) a
  /// runnable producer advances the version, so a peer's earlier zero-runnable
  /// summary cannot survive to drive a false escape.
  uint64_t summary_epoch_ = 0;
  /// Monotonic version for shard placement records (@ref DispatchAdmission),
  /// deliberately separate from @ref summary_epoch_. Bumped only when free execution
  /// capacity may have *increased* -- a wave retiring (remove_resident), a committed
  /// wave rolled back (remove_admitted), a dispatch torn down (notify_dispatch_gone),
  /// a process reclaimed (release_process), or a CU joining the store
  /// (register_compute_unit). A shard recorded Unplaced at an older capacity version
  /// is stale and must defer escape until the CP re-attempts placement against the
  /// freed room. Keeping this distinct from the quiescence version is what lets a
  /// stalled serial dispatch escape: a wave merely parking advances quiescence but
  /// frees no slot, so it must not invalidate a legitimate "genuinely unplaceable"
  /// verdict and wedge the backstop forever (nothing would retire to re-trigger
  /// placement).
  uint64_t capacity_epoch_ = 0;
  /// Per-CU published count of signalable (can-still-run) waves, per process,
  /// refreshed each step by the owning CU. Absent/stale CUs block escape.
  std::unordered_map<ComputeUnitCore *, std::unordered_map<uint32_t, uint32_t>> cu_runnable_;
  /// summary_epoch_ at which each CU last published its snapshot.
  std::unordered_map<ComputeUnitCore *, uint64_t> cu_summary_epoch_;
  /// Monotonic arrival stamp handed to each parked semaphore P-waiter so a V/BR
  /// credit is released to the oldest queued waiter first, across CUs (see
  /// @ref Resource::sema_wait_seqs and the entitlement note in the class doc).
  uint64_t sema_seq_next_ = 0;
  /// Rate-limiting state for the deadlock-backstop recovery diagnostic (guarded by
  /// @ref mutex_). The backstop can abandon many waits in one pass; log at most
  /// once per interval and report how many were suppressed since the last line.
  std::chrono::steady_clock::time_point last_escape_diag_{};
  uint64_t escape_diag_suppressed_ = 0;
};

} // namespace amdgpu
} // namespace rocjitsu
