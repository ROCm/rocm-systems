// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gws_device.h
/// @brief Device-global Global Wave Sync (GWS) resource store shared by CUs.

#pragma once

#include <array>
#include <atomic>
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
///    CDNA/RDNA barrier pseudocode queues an arrival while the counter is
///    positive and, on the arrival that observes zero, releases every queued
///    arrival and reloads the counter from that arrival's own value (so
///    consecutive phases may differ in size). A blocking arrival is gated on the
///    outstanding counter: the still-required participant set is (counter + 1),
///    and it parks only when that whole set is provably resident in the dispatch
///    (summed across all CUs via @ref resident_waves_); larger sets fall back to
///    a non-blocking structural no-op. The gate is overflow-safe for counts near
///    UINT32_MAX. The release bumps a per-resource generation that parked
///    arrivals (on any CU) observe to wake. Concurrent dispatches of one process
///    sharing a single barrier resource is not a supported rendezvous (hardware
///    treats a cooperative grid as the sole owner of its barrier); the residency
///    bound is intentionally dispatch-scoped.
///  * Semaphore: V/BR add credits and @c release_all bumps the generation; P
///    consumes a credit or parks until any wave of the process signals. Credits
///    and the generation are consumed/observed by the waiter's own CU.
///  * Deadlock-escape (@ref step_maintenance): a parked GWS wave is released only
///    when every non-halted wave of its process -- across all CUs -- is provably
///    unable to signal (blocked in GWS_WAIT or at an s_barrier). This is a
///    genuine-deadlock backstop (the step budget must terminate); it never fires
///    while any wave of the process can still run and signal. Quiescence is
///    process-scoped (another dispatch of the same process may still signal) and
///    race-free: each CU publishes a snapshot of its own signalable-wave count,
///    and escape requires every currently-active CU to have published in the
///    current park epoch, so a runnable producer on another CU is never missed.
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

  /// @brief Record @p count resident (not-yet-retired) waves of a dispatch.
  /// @details Maintained by the CU as workgroups begin/retire; bounds the GWS
  /// barrier's provable-participant set. See class doc.
  void add_resident(uint32_t dispatch_id, uint32_t count);
  /// @brief Drop @p count resident waves of a dispatch (clamped at zero).
  void remove_resident(uint32_t dispatch_id, uint32_t count);

  /// @brief Per-step maintenance for the calling CU, on its own thread.
  /// @details Applies cross-CU wakeups to @p cu's own parked waves, refreshes
  /// @p cu's published signalable-wave snapshot, and runs the genuine-deadlock
  /// backstop for @p cu's own parked waves. Cheap lock-free no-op when nothing is
  /// parked anywhere (see @ref any_parked). Mutates only @p cu's wavefronts.
  void step_maintenance(ComputeUnitCore &cu);

  /// @brief True while any wave is parked anywhere (lock-free fast-path gate).
  bool any_parked() const { return parked_total_.load(std::memory_order_acquire) != 0; }

private:
  /// @brief Per-resource barrier/semaphore state.
  struct Resource {
    /// Live barrier counter: decremented per arrival, reloaded from the releasing
    /// arrival's own value at zero (so consecutive phases may differ in size).
    uint32_t counter = 0;
    uint32_t credits = 0;     ///< Semaphore credits (V/BR add, P consumes).
    bool armed = false;       ///< Whether the barrier counter has been seeded.
    uint64_t release_gen = 0; ///< Bumped by a barrier release / sema release_all.
  };

  /// @brief Park @p wf on @p rid (barrier or semaphore). Requires @ref mutex_.
  void park_wave(Wavefront &wf, uint32_t rid, bool is_barrier, uint64_t generation);
  /// @brief Wake @p wf running on @p cu, clearing its park state. Requires @ref mutex_.
  void wake_wave(ComputeUnitCore &cu, Wavefront &wf);
  /// @brief Wake @p cu's own parked waves whose credit/generation is satisfied.
  /// @returns the number of waves woken. Requires @ref mutex_.
  uint32_t poll_local_waiters(ComputeUnitCore &cu);
  /// @brief Refresh @p cu's published signalable-wave snapshot for the current
  /// park epoch. Requires @ref mutex_.
  void refresh_summary(ComputeUnitCore &cu);
  /// @brief Whether process @p process is provably quiescent (genuine deadlock).
  /// Requires @ref mutex_ and an up-to-date snapshot from every active CU.
  bool escape_ready(uint32_t process) const;

  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, std::array<Resource, kResourceCount>> resources_;
  std::vector<ComputeUnitCore *> cus_;

  /// Resident (not-yet-retired) wave count per dispatch; the barrier bound.
  std::unordered_map<uint32_t, uint32_t> resident_waves_;
  /// Parked (GWS_WAIT) wave count per process; the deadlock-escape precondition.
  std::unordered_map<uint32_t, uint32_t> parked_;
  /// Total parked waves across all processes; lock-free fast-path gate.
  std::atomic<uint32_t> parked_total_{0};
  /// Park epoch, bumped when parked_total_ transitions 0 -> positive, so stale
  /// cross-epoch snapshots cannot drive an escape decision.
  uint64_t park_epoch_ = 0;
  /// Per-CU published count of signalable (can-still-run) waves, per process,
  /// refreshed each step by the owning CU. Absent/stale CUs block escape.
  std::unordered_map<ComputeUnitCore *, std::unordered_map<uint32_t, uint32_t>> cu_runnable_;
  /// Park epoch at which each CU last published its snapshot.
  std::unordered_map<ComputeUnitCore *, uint64_t> cu_epoch_;
};

} // namespace amdgpu
} // namespace rocjitsu
