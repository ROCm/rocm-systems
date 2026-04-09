/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "platform/clr_prof_event_bus.hpp"
#include "platform/clr_prof_interface.h"
#include "platform/prof_protocol.h"

#include <cassert>
#include <cstring>

namespace amd::clr_prof {

/* ── Static members ─────────────────────────────────────────────────────── */

std::atomic<uint64_t> EventBus::correlation_id_counter_{0};

/* ── Singleton ──────────────────────────────────────────────────────────── */

EventBus& EventBus::instance() {
  static EventBus bus;
  return bus;
}

/* ── subscribe ──────────────────────────────────────────────────────────── */

clr_prof_subscriber_t EventBus::subscribe(const clr_prof_callbacks_t*  cbs,
                                           const clr_prof_api_filter_t* filter) {
  if (!cbs || cbs->struct_size < sizeof(clr_prof_callbacks_t)) return nullptr;

  std::unique_lock lock(mutex_);

  // Find a free slot.
  SubscriberSlot* slot = nullptr;
  for (auto& s : slots_) {
    if (!s.active.load(std::memory_order_relaxed)) {
      slot = &s;
      break;
    }
  }
  if (!slot) return nullptr;  // All slots occupied.

  // Copy callbacks.
  slot->cbs = *cbs;

  // Apply API filter if provided.
  slot->api_filter.reset();
  slot->filter_active = false;
  if (filter && filter->struct_size >= sizeof(clr_prof_api_filter_t) && filter->api_ids &&
      filter->count > 0) {
    slot->filter_active = true;
    for (uint32_t i = 0; i < filter->count; ++i) {
      uint32_t id = filter->api_ids[i];
      if (id < kMaxApiIds) slot->api_filter.set(id);
    }
  }

  // Mark active before updating reference counts so emit_* can't race.
  slot->active.store(true, std::memory_order_release);

  // Update fast-path counters.
  if (slot->cbs.gpu_activity) gpu_subscriber_count_.fetch_add(1, std::memory_order_relaxed);

  if (slot->cbs.hip_api) {
    if (!slot->filter_active) {
      // All APIs.
      for (auto& cnt : api_subscriber_counts_) cnt.fetch_add(1, std::memory_order_relaxed);
    } else {
      for (size_t i = 0; i < kMaxApiIds; ++i) {
        if (slot->api_filter.test(i))
          api_subscriber_counts_[i].fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  return reinterpret_cast<clr_prof_subscriber_t>(slot);
}

/* ── unsubscribe ────────────────────────────────────────────────────────── */

void EventBus::unsubscribe(clr_prof_subscriber_t handle) {
  if (!handle) return;
  auto* slot = reinterpret_cast<SubscriberSlot*>(handle);

  std::unique_lock lock(mutex_);

  // Validate the slot belongs to our array.
  if (slot < slots_.data() || slot >= slots_.data() + kMaxSubscribers) return;
  if (!slot->active.load(std::memory_order_relaxed)) return;

  // Decrement fast-path counters before clearing.
  if (slot->cbs.gpu_activity) gpu_subscriber_count_.fetch_sub(1, std::memory_order_relaxed);

  if (slot->cbs.hip_api) {
    if (!slot->filter_active) {
      for (auto& cnt : api_subscriber_counts_) cnt.fetch_sub(1, std::memory_order_relaxed);
    } else {
      for (size_t i = 0; i < kMaxApiIds; ++i) {
        if (slot->api_filter.test(i))
          api_subscriber_counts_[i].fetch_sub(1, std::memory_order_relaxed);
      }
    }
  }

  // Clear the slot.  The write lock ensures no emit_* is mid-iteration.
  slot->active.store(false, std::memory_order_release);
  slot->cbs = {};
  slot->api_filter.reset();
  slot->filter_active = false;

  if (legacy_slot_idx_ >= 0 && &slots_[legacy_slot_idx_] == slot) legacy_slot_idx_ = -1;
}

/* ── api_enabled ────────────────────────────────────────────────────────── */

bool EventBus::api_enabled(uint32_t api_id) const noexcept {
  if (api_id >= kMaxApiIds) return false;
  return api_subscriber_counts_[api_id].load(std::memory_order_acquire) > 0;
}

/* ── emit_gpu ────────────────────────────────────────────────────────────── */

void EventBus::emit_gpu(const clr_prof_gpu_record_t& record) {
  std::shared_lock lock(mutex_);
  for (auto& slot : slots_) {
    if (slot.active.load(std::memory_order_acquire) && slot.cbs.gpu_activity) {
      slot.cbs.gpu_activity(&record, slot.cbs.user_data);
    }
  }
}

/* ── emit_api ────────────────────────────────────────────────────────────── */

void EventBus::emit_api(const clr_prof_api_record_t& record) {
  std::shared_lock lock(mutex_);
  for (auto& slot : slots_) {
    if (slot.active.load(std::memory_order_acquire) && slot.cbs.hip_api &&
        slot.api_enabled(record.api_id)) {
      slot.cbs.hip_api(&record, slot.cbs.user_data);
    }
  }
}

/* ── emit_code_object ────────────────────────────────────────────────────── */

void EventBus::emit_code_object(const clr_prof_code_object_record_t& record) {
  std::shared_lock lock(mutex_);
  for (auto& slot : slots_) {
    if (slot.active.load(std::memory_order_acquire) && slot.cbs.code_object) {
      slot.cbs.code_object(&record, slot.cbs.user_data);
    }
  }
}

/* ── emit_queue ──────────────────────────────────────────────────────────── */

void EventBus::emit_queue(const clr_prof_queue_record_t& record) {
  std::shared_lock lock(mutex_);
  for (auto& slot : slots_) {
    if (slot.active.load(std::memory_order_acquire) && slot.cbs.queue) {
      slot.cbs.queue(&record, slot.cbs.user_data);
    }
  }
}

/* ── Legacy shim ─────────────────────────────────────────────────────────── */

namespace {

// State for the legacy shim subscriber.
struct LegacyShimState {
  int (*fn)(activity_domain_t domain, uint32_t op, void* data){nullptr};

  // Sentinel value used by the old protocol to signal record commitment.
  static void* const kCommitSentinel;

  // Called at api_callbacks_spawner_t construction time via emit_api ENTER.
  // We reconstruct a hip_api_trace_data_t and call fn(ACTIVITY_DOMAIN_HIP_API, ...).
  // The legacy callback expects to return phase_enter/phase_exit function pointers
  // embedded in a hip_api_trace_data_t.  We approximate this by providing
  // pass-through phase functions that call back into the legacy fn.
  // NOTE: This is a best-effort bridge; tools that rely on fine-grained
  // hip_api_trace_data_t fields should migrate to the new interface.
};
void* const LegacyShimState::kCommitSentinel = reinterpret_cast<void*>(uintptr_t{1});

// Thread-local storage for the per-call legacy trace data (mirrors the old
// hip_prof_api.h approach).
thread_local struct {
  bool     enabled{false};
  uint32_t api_id{0};
} tls_legacy_api{};

void legacy_gpu_cb(const clr_prof_gpu_record_t* rec, void* ud) {
  auto* state = static_cast<LegacyShimState*>(ud);
  if (!state->fn) return;

  // Reconstruct an activity_record_t for the legacy callback.
  activity_record_t legacy{};
  legacy.domain         = ACTIVITY_DOMAIN_HIP_OPS;
  legacy.op             = static_cast<activity_op_t>(rec->op);
  legacy.correlation_id = rec->correlation_id;
  legacy.begin_ns       = rec->begin_ns;
  legacy.end_ns         = rec->end_ns;
  legacy.device_id      = rec->device_id;
  legacy.queue_id       = rec->queue_id;
  if (rec->op == CLR_PROF_OP_KERNEL_DISPATCH)
    legacy.kernel_name = rec->kernel_name;
  else
    legacy.bytes = rec->bytes;

  state->fn(ACTIVITY_DOMAIN_HIP_OPS, static_cast<uint32_t>(rec->op), &legacy);
}

void legacy_api_cb(const clr_prof_api_record_t* rec, void* ud) {
  auto* state = static_cast<LegacyShimState*>(ud);
  if (!state->fn) return;
  // The legacy protocol for HIP API uses the report_activity callback with a
  // hip_api_trace_data_t*.  Since the new interface separates enter/exit, we
  // call the legacy fn once with phase embedded in api_data.  Legacy tools
  // that inspect phase_enter/phase_exit pointers must migrate; this shim
  // delivers the minimum expected signal.
  state->fn(ACTIVITY_DOMAIN_HIP_API, rec->api_id, nullptr);
}

}  // namespace

// One static shim state per process — only one legacy callback can be active.
static LegacyShimState g_legacy_shim_state{};

void EventBus::set_legacy_callback(int (*fn)(activity_domain_t domain, uint32_t op, void* data)) {
  std::unique_lock lock(mutex_);

  // Remove existing legacy slot if present.
  if (legacy_slot_idx_ >= 0) {
    auto& old = slots_[legacy_slot_idx_];
    if (old.cbs.gpu_activity)
      gpu_subscriber_count_.fetch_sub(1, std::memory_order_relaxed);
    if (old.cbs.hip_api)
      for (auto& cnt : api_subscriber_counts_) cnt.fetch_sub(1, std::memory_order_relaxed);
    old.active.store(false, std::memory_order_release);
    old.cbs = {};
    legacy_slot_idx_ = -1;
  }

  if (!fn) return;

  // Find a free slot for the legacy shim.
  SubscriberSlot* slot = nullptr;
  int idx = 0;
  for (auto& s : slots_) {
    if (!s.active.load(std::memory_order_relaxed)) {
      slot = &s;
      break;
    }
    ++idx;
  }
  if (!slot) return;  // No room.

  g_legacy_shim_state.fn = fn;

  slot->cbs.struct_size  = sizeof(clr_prof_callbacks_t);
  slot->cbs.gpu_activity = legacy_gpu_cb;
  slot->cbs.hip_api      = legacy_api_cb;
  slot->cbs.user_data    = &g_legacy_shim_state;
  slot->filter_active    = false;
  slot->api_filter.reset();

  slot->active.store(true, std::memory_order_release);
  legacy_slot_idx_ = idx;

  gpu_subscriber_count_.fetch_add(1, std::memory_order_relaxed);
  for (auto& cnt : api_subscriber_counts_) cnt.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace amd::clr_prof
// The public C API entry points (clr_prof_subscribe, clr_prof_unsubscribe,
// clr_prof_get_correlation_id, clr_prof_api_name, clr_prof_gpu_op_name,
// clr_prof_version) are defined in hipamd/src/hip_intercept.cpp so they
// are part of libamdhip64 and can be exported from hip_hcc.map.in.
// rocclr (this file) is a static library; it cannot directly export symbols.
