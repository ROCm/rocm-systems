// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gws_device.h
/// @brief Device-global Global Wave Sync (GWS) resource store shared by CUs.

#pragma once

#include <array>
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
/// and never torn down on workgroup or dispatch retirement. A mutex serializes
/// all transitions so CUs that step concurrently observe a consistent resource.
/// Wakeups and the deadlock-escape quiescence check both span all registered CUs
/// (a signal/arrival on one CU releases a waiter parked on another; counting
/// only local residents cannot establish dispatch-wide quiescence).
///
/// Cross-CU wavefront state transitions performed by a wakeup are serialized by
/// this object's mutex, which is the synchronization boundary for the functional
/// execution model the emulator uses for GWS.
class GwsDevice {
public:
  /// @brief Number of GWS resources (6-bit resource id space).
  static constexpr uint32_t kResourceCount = 64;

  /// @brief Per-resource barrier/semaphore state.
  struct Resource {
    /// Live barrier counter: decremented per arrival, reloaded from the releasing
    /// arrival's own value at zero (so consecutive phases may differ in size).
    uint32_t counter = 0;
    uint32_t credits = 0; ///< Semaphore credits (V/BR add, P consumes).
    bool armed = false;   ///< Whether the barrier counter has been seeded.
  };

  /// @brief Register a CU so its waves participate in shared wake/quiescence.
  void register_compute_unit(ComputeUnitCore *cu);
  /// @brief Stop routing shared wake/quiescence to a CU being destroyed.
  void unregister_compute_unit(ComputeUnitCore *cu);

  /// @brief Seed a GWS resource's barrier count / semaphore credits.
  void init(Wavefront &wf, uint32_t rid, uint32_t count);

  /// @brief Arrive at a GWS barrier. Parks the wave while the participant set is
  /// provably resident in the dispatch and more arrivals are pending; the final
  /// arrival releases parked peers on any CU. Non-resident/degenerate counts fall
  /// back to a non-blocking structural no-op.
  void barrier_arrive(Wavefront &wf, uint32_t rid, uint32_t count);

  /// @brief Signal (V) a semaphore: add one credit, release one waiter (on any CU).
  void sema_v(Wavefront &wf, uint32_t rid);
  /// @brief Wait (P) on a semaphore: consume a credit, else park until any CU signals.
  void sema_p(Wavefront &wf, uint32_t rid);
  /// @brief Bulk-signal (BR): add @p count credits and release up to that many waiters.
  void sema_br(Wavefront &wf, uint32_t rid, uint32_t count);
  /// @brief Release every wave parked on a resource and clear its credits.
  void sema_release_all(Wavefront &wf, uint32_t rid);

  /// @brief Genuine-deadlock backstop: release parked GWS waves of any dispatch
  /// whose every non-halted wave (across all registered CUs) is already blocked
  /// in GWS_WAIT or at an s_barrier, so no wave can ever reach a signal/arrival.
  void escape_deadlocks();

private:
  /// @brief Total resident (not-yet-retired) waves of a dispatch across all CUs.
  /// Used as the GWS barrier's provable-participant bound. Requires @ref mutex_.
  uint32_t dispatch_resident_waves(uint32_t dispatch_id) const;

  /// @brief Wake up to @p max_wake waves parked on one resource, across every
  /// registered CU. @returns the number of waves released. Requires @ref mutex_.
  uint32_t release_waiters(uint32_t process_id, uint32_t rid, uint32_t max_wake);

  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, std::array<Resource, kResourceCount>> resources_;
  std::vector<ComputeUnitCore *> cus_;
};

} // namespace amdgpu
} // namespace rocjitsu
