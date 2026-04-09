/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @file clr_prof_event_bus.hpp
 * @brief Internal CLR profiling event bus.
 *
 * The EventBus is the single authoritative source of profiling events inside
 * CLR.  It replaces the single-callback report_activity atomic with a
 * multi-subscriber, typed publish/subscribe system.
 *
 * Callers inside CLR use the static emit_* methods directly.  External clients
 * use the C API in clr_prof_interface.h, which forwards to these methods.
 */

#pragma once

#include "platform/clr_prof_interface.h"

#include <array>
#include <atomic>
#include <bitset>
#include <cstdint>
#include <memory>
#include <shared_mutex>

// Forward declare to avoid pulling in the whole command headers here.
namespace amd {
class Command;
}

namespace amd::clr_prof {

/* ── Constants ────────────────────────────────────────────────────────────── */

/// Maximum number of concurrently registered subscribers.
static constexpr size_t kMaxSubscribers = 16;

/// Total number of HIP API IDs (from hip_api_id_t; keep in sync with
/// hip_prof_str.h).  512 covers the current range with room to grow.
static constexpr size_t kMaxApiIds = 512;

/* ── Subscriber slot ─────────────────────────────────────────────────────── */

struct SubscriberSlot {
  clr_prof_callbacks_t cbs{};
  std::bitset<kMaxApiIds> api_filter;  ///< empty == all APIs enabled
  bool                    filter_active{false};
  std::atomic<bool>       active{false};

  /// Returns true when this slot should fire for api_id.
  bool api_enabled(uint32_t api_id) const noexcept {
    if (!filter_active) return true;
    return api_id < kMaxApiIds && api_filter.test(api_id);
  }
};

/* ── EventBus ────────────────────────────────────────────────────────────── */

class EventBus {
 public:
  /// Singleton accessor.
  static EventBus& instance();

  // Non-copyable, non-movable.
  EventBus(const EventBus&) = delete;
  EventBus& operator=(const EventBus&) = delete;

  /* ── Subscription management ──────────────────────────────────────────── */

  /**
   * Register a new subscriber.  Thread-safe.
   * Returns a pointer to the slot (used as the opaque handle), or nullptr
   * if all slots are occupied.
   */
  clr_prof_subscriber_t subscribe(const clr_prof_callbacks_t* cbs,
                                   const clr_prof_api_filter_t* filter);

  /**
   * Unregister a subscriber.  Blocks until in-flight callbacks complete.
   * Thread-safe.
   */
  void unsubscribe(clr_prof_subscriber_t handle);

  /* ── Event emission (called by CLR internals) ─────────────────────────── */

  /**
   * Emit a GPU activity record to all interested subscribers.
   * Called from the HSA signal completion worker thread.
   */
  void emit_gpu(const clr_prof_gpu_record_t& record);

  /**
   * Emit a HIP API enter/exit record to all interested subscribers.
   * Called synchronously from the application thread.
   */
  void emit_api(const clr_prof_api_record_t& record);

  /**
   * Emit a code-object load/unload record.
   */
  void emit_code_object(const clr_prof_code_object_record_t& record);

  /**
   * Emit a queue create/destroy record.
   */
  void emit_queue(const clr_prof_queue_record_t& record);

  /* ── Fast-path query helpers ──────────────────────────────────────────── */

  /// Returns true if at least one subscriber wants GPU activity.
  bool gpu_activity_enabled() const noexcept {
    return gpu_subscriber_count_.load(std::memory_order_acquire) > 0;
  }

  /// Returns true if at least one subscriber wants hip_api callbacks for api_id.
  bool api_enabled(uint32_t api_id) const noexcept;

  /* ── Correlation ID ───────────────────────────────────────────────────── */

  /**
   * Allocate a new CLR-owned correlation ID.
   * Called once per HIP API entry from api_callbacks_spawner_t.
   */
  static uint64_t next_correlation_id() noexcept {
    return correlation_id_counter_.fetch_add(1, std::memory_order_relaxed) + 1;
  }

  /* ── Legacy shim ──────────────────────────────────────────────────────── */

  /**
   * Install a roctracer-style legacy callback.  Used by
   * hipRegisterTracerCallback() for backward compatibility.
   * Installs a special SubscriberSlot that bridges the new typed events
   * back to the old single-function protocol.
   *
   * Passing nullptr removes the legacy subscriber.
   */
  void set_legacy_callback(int (*fn)(activity_domain_t domain, uint32_t op, void* data));

 private:
  EventBus() = default;

  std::array<SubscriberSlot, kMaxSubscribers> slots_{};
  mutable std::shared_mutex                   mutex_;

  /// Counts active subscribers that have a gpu_activity callback.
  std::atomic<int> gpu_subscriber_count_{0};
  /// Per-API-id count of subscribers that want hip_api callbacks.
  std::array<std::atomic<int>, kMaxApiIds> api_subscriber_counts_{};

  /// Monotonically increasing correlation ID counter (CLR-owned).
  static std::atomic<uint64_t> correlation_id_counter_;

  /// Index of the legacy-shim slot (-1 if unused).
  int legacy_slot_idx_{-1};
};

}  // namespace amd::clr_prof
