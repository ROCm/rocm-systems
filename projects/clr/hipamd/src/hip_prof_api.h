/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_SRC_HIP_PROF_API_H
#define HIP_SRC_HIP_PROF_API_H

#include <atomic>
#include <cassert>
#include <ctime>
#include <iostream>
#include <shared_mutex>
#include <utility>

#if defined(__linux__)
#  include <unistd.h>
#  include <sys/syscall.h>
#endif

#include "hip/amd_detail/hip_prof_str.h"
#include "platform/clr_prof_event_bus.hpp"
#include "platform/clr_prof_interface.h"
#include "platform/prof_protocol.h"

namespace {
inline uint64_t clr_prof_clock_ns() noexcept {
  struct timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
         static_cast<uint64_t>(ts.tv_nsec);
}
inline uint32_t clr_prof_tid() noexcept {
#if defined(__linux__)
  return static_cast<uint32_t>(syscall(SYS_gettid));
#else
  return 0;
#endif
}
}  // namespace

// hip_api_trace_data_t is kept for the legacy roctracer path.  When the
// EventBus path is active, it is not used.
struct hip_api_trace_data_t {
  hip_api_data_t api_data;
  uint64_t phase_enter_timestamp;
  uint64_t phase_data;

  void (*phase_enter)(hip_api_id_t operation_id, hip_api_trace_data_t* data);
  void (*phase_exit)(hip_api_id_t operation_id, hip_api_trace_data_t* data);
};

// HIP API callbacks spawner object macro
#define HIP_CB_SPAWNER_OBJECT(operation_id)                                                        \
  api_callbacks_spawner_t<HIP_API_ID_##operation_id> __api_tracer(                                 \
      [=](auto& api_data) { INIT_CB_ARGS_DATA(operation_id, api_data); });

template <hip_api_id_t operation_id> class api_callbacks_spawner_t {
 public:
  template <typename Functor> api_callbacks_spawner_t(Functor init_cb_args_data) {
    static_assert(operation_id >= HIP_API_ID_FIRST && operation_id <= HIP_API_ID_LAST,
                  "invalid HIP_API operation id");

    auto& bus = amd::clr_prof::EventBus::instance();

    // ── New path: EventBus ─────────────────────────────────────────────────
    new_enabled_ = bus.api_enabled(static_cast<uint32_t>(operation_id));
    if (new_enabled_) {
      // Allocate a CLR-owned correlation ID for this call.
      correlation_id_ = amd::clr_prof::EventBus::next_correlation_id();
      // Write it to the TLS slot so GPU commands created during this call
      // pick it up automatically.
      amd::activity_prof::correlation_id = correlation_id_;

      // Populate and emit the ENTER record.
      // api_args will be set below after init_cb_args_data(); we emit after
      // args are populated so subscribers see them on ENTER.
      enter_record_.struct_size    = sizeof(clr_prof_api_record_t);
      enter_record_.phase          = CLR_PROF_PHASE_ENTER;
      enter_record_.api_id         = static_cast<uint32_t>(operation_id);
      enter_record_.correlation_id = correlation_id_;
      enter_record_.timestamp_ns   = clr_prof_clock_ns();
      enter_record_.pid            = static_cast<uint32_t>(getpid());
      enter_record_.tid            = clr_prof_tid();
      enter_record_.api_args       = nullptr;  // set after init below

      init_cb_args_data(new_api_data_);
      enter_record_.api_args = &new_api_data_;
      bus.emit_api(enter_record_);
    }

    // ── Legacy path: roctracer callback ────────────────────────────────────
    if (auto function = amd::activity_prof::report_activity.load(std::memory_order_acquire);
        function &&
        (legacy_enabled_ = function(ACTIVITY_DOMAIN_HIP_API, operation_id, &trace_data_) == 0)) {
      // If the EventBus already set a correlation ID, don't overwrite it;
      // otherwise adopt the roctracer-provided one.
      if (!new_enabled_)
        amd::activity_prof::correlation_id = trace_data_.api_data.correlation_id;

      if (trace_data_.phase_enter != nullptr) {
        if (!new_enabled_) init_cb_args_data(trace_data_.api_data);
        trace_data_.phase_enter(operation_id, &trace_data_);
      }
    }
  }

  ~api_callbacks_spawner_t() {
    auto& bus = amd::clr_prof::EventBus::instance();

    // ── New path: emit EXIT ────────────────────────────────────────────────
    if (new_enabled_) {
      clr_prof_api_record_t exit_rec{};
      exit_rec.struct_size    = sizeof(clr_prof_api_record_t);
      exit_rec.phase          = CLR_PROF_PHASE_EXIT;
      exit_rec.api_id         = static_cast<uint32_t>(operation_id);
      exit_rec.correlation_id = correlation_id_;
      exit_rec.timestamp_ns   = clr_prof_clock_ns();
      exit_rec.pid            = enter_record_.pid;
      exit_rec.tid            = enter_record_.tid;
      exit_rec.api_args       = nullptr;  // args not available at exit
      bus.emit_api(exit_rec);
    }

    // ── Legacy path: roctracer exit callback ───────────────────────────────
    if (legacy_enabled_) {
      if (trace_data_.phase_exit != nullptr) trace_data_.phase_exit(operation_id, &trace_data_);
    }

    // Clear TLS correlation ID when both paths are done.
    if (new_enabled_ || legacy_enabled_) amd::activity_prof::correlation_id = 0;
  }

 private:
  // New-path state.
  bool                  new_enabled_{false};
  uint64_t              correlation_id_{0};
  clr_prof_api_record_t enter_record_{};
  hip_api_data_t        new_api_data_{};

  // Legacy-path state.
  bool legacy_enabled_{false};
  union {
    hip_api_trace_data_t trace_data_;
  };
};

template <> class api_callbacks_spawner_t<HIP_API_ID_NONE> {
 public:
  template <typename Functor> api_callbacks_spawner_t(Functor) {}
};
#endif  // HIP_SRC_HIP_PROF_API_H
