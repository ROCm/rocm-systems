// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file events.cpp
/// @brief KFD event ioctl implementations for the simulated driver.
///
/// @details Implements the EventState methods that model KFD's event
/// lifecycle and the SimulatedDriver ioctl wrappers that delegate to them.

#include "rocjitsu/kmd/linux/simulated_driver.h"
#include "util/log.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

namespace rocjitsu {

namespace {

void write_event_slot(void *page, size_t page_size, uint32_t event_id, uint64_t value) {
  if (!page || event_id >= page_size / sizeof(uint64_t))
    return;
  auto *slots = static_cast<uint64_t *>(page);
  std::atomic_ref<uint64_t>(slots[event_id]).store(value, std::memory_order_release);
}

void reset_event_slot_if_age(void *page, size_t page_size, uint32_t event_id, uint64_t age) {
  if (!page || event_id >= page_size / sizeof(uint64_t))
    return;
  auto *slots = static_cast<uint64_t *>(page);
  uint64_t expected = age;
  std::atomic_ref<uint64_t>(slots[event_id])
      .compare_exchange_strong(expected, KFD_SIGNAL_EVENT_LIMIT, std::memory_order_acq_rel,
                               std::memory_order_acquire);
}

bool event_profile_enabled() {
  static bool enabled = std::getenv("RJ_EVENT_PROFILE") != nullptr;
  return enabled;
}

struct EventProfileCounters {
  std::atomic<uint64_t> wait_calls{0};
  std::atomic<uint64_t> poll_calls{0};
  std::atomic<uint64_t> ready_before_wait{0};
  std::atomic<uint64_t> blocking_waits{0};
  std::atomic<uint64_t> single_event_waits{0};
  std::atomic<uint64_t> multi_event_waits{0};
  std::atomic<uint64_t> single_signal_waits{0};
  std::atomic<uint64_t> single_system_waits{0};
  std::atomic<uint64_t> single_auto_reset_waits{0};
  std::atomic<uint64_t> single_missing_waits{0};
  std::atomic<uint64_t> fast_out_of_range{0};
  std::atomic<uint64_t> fast_invalid{0};
  std::atomic<uint64_t> fast_non_signal{0};
  std::atomic<uint64_t> fast_auto_reset{0};
  std::atomic<uint64_t> fast_not_ready_blocking{0};
  std::atomic<uint64_t> fast_signal_completes{0};
  std::atomic<uint64_t> fast_signal_timeouts{0};
  std::atomic<uint64_t> fast_multi_completes{0};
  std::atomic<uint64_t> fast_multi_timeouts{0};
  std::atomic<uint64_t> fast_multi_not_ready_blocking{0};
  std::atomic<uint64_t> broadcast_interrupts{0};
  std::atomic<uint64_t> broadcast_events{0};
  std::atomic<uint64_t> specific_interrupts{0};
  std::atomic<uint64_t> specific_misses{0};
  std::atomic<uint64_t> complete_results{0};
  std::atomic<uint64_t> timeout_results{0};
  std::atomic<uint64_t> fail_results{0};

  ~EventProfileCounters() {
    if (!event_profile_enabled())
      return;

    std::fprintf(
        stderr,
        "RJ_EVENT_PROFILE wait_calls=%llu poll_calls=%llu ready_before_wait=%llu "
        "blocking_waits=%llu single_event_waits=%llu multi_event_waits=%llu "
        "single_signal_waits=%llu single_system_waits=%llu single_auto_reset_waits=%llu "
        "single_missing_waits=%llu fast_out_of_range=%llu fast_invalid=%llu "
        "fast_non_signal=%llu fast_auto_reset=%llu fast_not_ready_blocking=%llu "
        "fast_signal_completes=%llu fast_signal_timeouts=%llu fast_multi_completes=%llu "
        "fast_multi_timeouts=%llu fast_multi_not_ready_blocking=%llu "
        "broadcast_interrupts=%llu broadcast_events=%llu specific_interrupts=%llu "
        "specific_misses=%llu "
        "complete_results=%llu timeout_results=%llu fail_results=%llu\n",
        static_cast<unsigned long long>(wait_calls.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(poll_calls.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(ready_before_wait.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(blocking_waits.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(single_event_waits.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(multi_event_waits.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(single_signal_waits.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(single_system_waits.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(single_auto_reset_waits.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(single_missing_waits.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(fast_out_of_range.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(fast_invalid.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(fast_non_signal.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(fast_auto_reset.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(fast_not_ready_blocking.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(fast_signal_completes.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(fast_signal_timeouts.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(fast_multi_completes.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(fast_multi_timeouts.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            fast_multi_not_ready_blocking.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(broadcast_interrupts.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(broadcast_events.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(specific_interrupts.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(specific_misses.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(complete_results.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(timeout_results.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(fail_results.load(std::memory_order_relaxed)));
  }
};

EventProfileCounters &event_profile_counters() {
  static EventProfileCounters counters;
  return counters;
}

} // namespace

EventState::~EventState() {
  if (memfd >= 0)
    ::close(memfd);
}

void EventState::adopt_page(void *ptr, size_t size) {
  assert(ptr && "adopt_page called with null pointer");
  assert(size > 0 && "adopt_page called with zero size");
  if (page)
    return;
  page = ptr;
  page_size = size;

  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto &[id, ev] : events_) {
    bool signaled = ev.signaled;
    uint64_t age = ev.event_age;
    if (ev.event_type == 0 && id < fast_events_.size()) {
      uint64_t state = fast_events_[id].state.load(std::memory_order_acquire);
      signaled = (fast_event_flags(state) & kFastEventSignaled) != 0;
      age = fast_event_age(state);
    }
    if (signaled)
      write_event_slot(page, page_size, id, age);
  }
}

/// @brief Signal event(s) from the CP's interrupt callback.
/// @details When event_id is non-zero, signals that specific event. When
///          event_id is zero, broadcasts to all type-0 events — matching real
///          KFD's kfd_signal_event_interrupt(pasid, partial_id=0, valid_id_bits=0).
void EventState::signal_interrupt(uint32_t event_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (event_id == 0) {
    uint64_t signaled_events = 0;
    for (auto &[id, ev] : events_) {
      if (ev.event_type == 0) {
        ++signaled_events;
        ev.signaled = true;
        ev.event_age = 1;
        if (id < fast_events_.size()) {
          auto &fast = fast_events_[id];
          uint8_t flags = kFastEventValid | kFastEventSignal;
          if (ev.auto_reset)
            flags |= kFastEventAutoReset;
          if (ev.signaled)
            flags |= kFastEventSignaled;
          fast.state.store(pack_fast_event_state(ev.event_age, flags), std::memory_order_release);
        }
        write_event_slot(page, page_size, id, ev.event_age);
        util::Logger::cp("SIGNAL_BROADCAST: event_id=", id, " age=", ev.event_age,
                         " waiters=", ev.waiters.size());
        for (auto *cv : ev.waiters)
          cv->notify_one();
      }
    }
    if (event_profile_enabled()) {
      auto &profile = event_profile_counters();
      profile.broadcast_interrupts.fetch_add(1, std::memory_order_relaxed);
      profile.broadcast_events.fetch_add(signaled_events, std::memory_order_relaxed);
    }
    return;
  }
  auto it = events_.find(event_id);
  if (it != events_.end() && it->second.event_type == 0) {
    it->second.signaled = true;
    it->second.event_age = 1;
    if (event_id < fast_events_.size()) {
      auto &fast = fast_events_[event_id];
      uint8_t flags = kFastEventValid | kFastEventSignal;
      if (it->second.auto_reset)
        flags |= kFastEventAutoReset;
      if (it->second.signaled)
        flags |= kFastEventSignaled;
      fast.state.store(pack_fast_event_state(it->second.event_age, flags),
                       std::memory_order_release);
    }
    write_event_slot(page, page_size, event_id, it->second.event_age);
    util::Logger::cp("SIGNAL_INTERRUPT: event_id=", event_id, " age=", it->second.event_age,
                     " waiters=", it->second.waiters.size(), " page=", page ? "valid" : "null");
    for (auto *cv : it->second.waiters)
      cv->notify_one();
    if (event_profile_enabled())
      event_profile_counters().specific_interrupts.fetch_add(1, std::memory_order_relaxed);
  } else {
    util::Logger::cp("SIGNAL_INTERRUPT_MISS: event_id=", event_id,
                     " NOT FOUND or wrong type, events_.size()=", events_.size());
    if (event_profile_enabled())
      event_profile_counters().specific_misses.fetch_add(1, std::memory_order_relaxed);
  }
}

/// @brief Set the closing flag and wake all waiters across all events.
void EventState::notify_closing() {
  std::lock_guard<std::mutex> lock(mutex_);
  closing_.store(true, std::memory_order_release);
  for (auto &[id, ev] : events_) {
    for (auto *cv : ev.waiters)
      cv->notify_one();
  }
}

/// @brief Write KFD_SIGNAL_EVENT_LIMIT to all event page slots.
void EventState::signal_page_shutdown() {
  if (!page)
    return;
  auto *slots = static_cast<uint64_t *>(page);
  size_t count = page_size / sizeof(uint64_t);
  for (size_t i = 0; i < count; ++i)
    std::atomic_ref<uint64_t>(slots[i]).store(KFD_SIGNAL_EVENT_LIMIT, std::memory_order_release);
}

void EventState::reset() { closing_.store(false, std::memory_order_release); }

bool EventState::is_closing() const { return closing_.load(std::memory_order_acquire); }

/// @brief Allocate a new KFD event and return its ID and slot index.
int EventState::create_event(void *arg, uint32_t gpu_id) {
  assert(arg && "create_event called with null arg");
  auto *args = static_cast<kfd_ioctl_create_event_args *>(arg);
  std::lock_guard<std::mutex> lock(mutex_);

  uint32_t max_slots =
      page_size > 0 ? static_cast<uint32_t>(page_size / sizeof(uint64_t)) : KFD_SIGNAL_EVENT_LIMIT;
  if (next_event_id_ >= std::min(max_slots, static_cast<uint32_t>(KFD_SIGNAL_EVENT_LIMIT)))
    return -ENOSPC;

  GpuEvent ev{};
  ev.event_id = next_event_id_++;
  ev.event_type = args->event_type;
  ev.auto_reset = args->auto_reset != 0;
  ev.event_age = 0;

  events_[ev.event_id] = ev;
  auto &fast = fast_events_[ev.event_id];
  uint8_t flags = kFastEventValid;
  if (ev.event_type == 0)
    flags |= kFastEventSignal;
  if (ev.auto_reset)
    flags |= kFastEventAutoReset;
  fast.state.store(pack_fast_event_state(ev.event_age, flags), std::memory_order_release);

  args->event_id = ev.event_id;
  args->event_trigger_data = ev.event_id;
  args->event_slot_index = ev.event_id;
  args->event_page_offset = KFD_MMAP_TYPE_EVENTS | kfd_mmap_gpu_id(gpu_id);

  util::Logger::cp([&](auto &os) {
    os << std::format("CREATE_EVENT: event_id={} type={} auto_reset={} gpu_id={}", ev.event_id,
                      ev.event_type, ev.auto_reset, gpu_id);
  });

  return 0;
}

/// @brief Destroy an event, wake its waiters, and mark its page slot.
int EventState::destroy_event(void *arg) {
  assert(arg && "destroy_event called with null arg");
  auto *args = static_cast<kfd_ioctl_destroy_event_args *>(arg);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = events_.find(args->event_id);
    if (it != events_.end()) {
      for (auto *cv : it->second.waiters)
        cv->notify_one();
      events_.erase(it);
    }
    if (args->event_id < fast_events_.size())
      fast_events_[args->event_id].state.store(0, std::memory_order_release);
  }
  write_event_slot(page, page_size, args->event_id, KFD_SIGNAL_EVENT_LIMIT);
  return 0;
}

/// @brief Signal an event: set signaled flag, increment age, wake waiters.
int EventState::set_event(void *arg) {
  assert(arg && "set_event called with null arg");
  auto *args = static_cast<kfd_ioctl_set_event_args *>(arg);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = events_.find(args->event_id);
  if (it == events_.end()) {
    util::Logger::warn("SET_EVENT_MISS: event_id=", args->event_id,
                       " events_.size()=", events_.size());
    return -EINVAL;
  }
  it->second.signaled = true;
  it->second.event_age = 1;
  if (args->event_id < fast_events_.size()) {
    auto &fast = fast_events_[args->event_id];
    uint8_t flags = kFastEventValid;
    if (it->second.event_type == 0)
      flags |= kFastEventSignal;
    if (it->second.auto_reset)
      flags |= kFastEventAutoReset;
    if (it->second.signaled)
      flags |= kFastEventSignaled;
    fast.state.store(pack_fast_event_state(it->second.event_age, flags), std::memory_order_release);
  }
  write_event_slot(page, page_size, args->event_id, it->second.event_age);
  util::Logger::cp("SET_EVENT: event_id=", args->event_id, " age=", it->second.event_age,
                   " waiters=", it->second.waiters.size());
  for (auto *cv : it->second.waiters)
    cv->notify_one();
  return 0;
}

/// @brief Reset an event's age to 0 (unsignaled).
int EventState::reset_event(void *arg) {
  assert(arg && "reset_event called with null arg");
  auto *args = static_cast<kfd_ioctl_reset_event_args *>(arg);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = events_.find(args->event_id);
  if (it == events_.end())
    return -EINVAL;
  it->second.signaled = false;
  it->second.event_age = 0;
  if (args->event_id < fast_events_.size()) {
    auto &fast = fast_events_[args->event_id];
    uint64_t state = fast.state.load(std::memory_order_acquire);
    uint8_t flags = fast_event_flags(state) & ~kFastEventSignaled;
    fast.state.store(pack_fast_event_state(0, flags), std::memory_order_release);
  }
  write_event_slot(page, page_size, args->event_id, KFD_SIGNAL_EVENT_LIMIT);
  return 0;
}

/// @brief Block until waited events satisfy the predicate, or timeout/close.
int EventState::wait_events(void *arg, uint32_t process_id) {
  assert(arg && "wait_events called with null arg");
  auto *args = static_cast<kfd_ioctl_wait_events_args *>(arg);
  auto *ev_data = reinterpret_cast<kfd_event_data *>(args->events_ptr);
  const bool wait_all = args->wait_for_all != 0;
  util::Logger::cp([&](auto &os) {
    os << "WAIT_EVENTS: pid=" << process_id << " num=" << args->num_events
       << " timeout=" << args->timeout << " wait_all=" << wait_all;
    for (uint32_t i = 0; i < args->num_events && i < 4; ++i)
      os << " ev[" << i << "]=" << ev_data[i].event_id
         << "(age=" << ev_data[i].signal_event_data.last_event_age << ")";
  });

  auto signal_event_age = [&](const GpuEvent &ev) -> uint64_t {
    if (ev.event_type == 0 && ev.event_id < fast_events_.size()) {
      uint64_t state = fast_events_[ev.event_id].state.load(std::memory_order_acquire);
      return fast_event_age(state);
    }
    return ev.event_age;
  };

  auto satisfied = [&](const GpuEvent &ev, const kfd_event_data &ed) -> bool {
    if (ev.event_type == 0) {
      uint64_t caller_age = ed.signal_event_data.last_event_age;
      uint64_t age = signal_event_age(ev);
      return (caller_age != 0) ? (age >= caller_age) : (age > 0);
    }
    return ev.signaled;
  };

  bool is_poll = (args->timeout == 0);

  auto record_wait_call = [&](bool ready_before_wait) {
    if (!event_profile_enabled())
      return;
    auto &profile = event_profile_counters();
    profile.wait_calls.fetch_add(1, std::memory_order_relaxed);
    if (is_poll)
      profile.poll_calls.fetch_add(1, std::memory_order_relaxed);
    if (ready_before_wait)
      profile.ready_before_wait.fetch_add(1, std::memory_order_relaxed);
  };

  auto record_wait_result = [&]() {
    if (!event_profile_enabled())
      return;
    auto &profile = event_profile_counters();
    if (args->wait_result == KFD_IOC_WAIT_RESULT_COMPLETE)
      profile.complete_results.fetch_add(1, std::memory_order_relaxed);
    else if (args->wait_result == KFD_IOC_WAIT_RESULT_TIMEOUT)
      profile.timeout_results.fetch_add(1, std::memory_order_relaxed);
    else
      profile.fail_results.fetch_add(1, std::memory_order_relaxed);
  };

  auto clear_fast_signaled_if_age = [&](uint32_t event_id, uint64_t age) {
    if (event_id >= fast_events_.size())
      return false;
    auto &fast = fast_events_[event_id];
    uint64_t state = fast.state.load(std::memory_order_acquire);
    while (fast_event_age(state) == age) {
      uint8_t flags = fast_event_flags(state) & ~kFastEventSignaled;
      uint64_t desired = pack_fast_event_state(0, flags);
      if (fast.state.compare_exchange_weak(state, desired, std::memory_order_acq_rel,
                                           std::memory_order_acquire))
        return true;
    }
    return false;
  };

  if (args->num_events == 1 && !closing_.load(std::memory_order_acquire)) {
    uint32_t event_id = ev_data[0].event_id;
    if (event_id < fast_events_.size()) {
      auto &fast = fast_events_[event_id];
      uint64_t state = fast.state.load(std::memory_order_acquire);
      uint8_t flags = fast_event_flags(state);
      constexpr uint8_t fast_signal_no_auto = kFastEventValid | kFastEventSignal;
      if ((flags & (kFastEventValid | kFastEventSignal | kFastEventAutoReset)) ==
          fast_signal_no_auto) {
        uint64_t age = fast_event_age(state);
        uint64_t caller_age = ev_data[0].signal_event_data.last_event_age;
        bool ready = caller_age == 0 ? ((flags & kFastEventSignaled) != 0) : age >= caller_age;
        if (ready || is_poll) {
          record_wait_call(ready);
          if (ready) {
            ev_data[0].signal_event_data.last_event_age = age;
            args->wait_result = KFD_IOC_WAIT_RESULT_COMPLETE;
            if (event_profile_enabled())
              event_profile_counters().fast_signal_completes.fetch_add(1,
                                                                       std::memory_order_relaxed);
          } else {
            args->wait_result = KFD_IOC_WAIT_RESULT_TIMEOUT;
            if (event_profile_enabled())
              event_profile_counters().fast_signal_timeouts.fetch_add(1, std::memory_order_relaxed);
          }
          record_wait_result();
          return 0;
        }
        if (event_profile_enabled())
          event_profile_counters().fast_not_ready_blocking.fetch_add(1, std::memory_order_relaxed);
      } else if (event_profile_enabled()) {
        auto &profile = event_profile_counters();
        if ((flags & kFastEventValid) == 0)
          profile.fast_invalid.fetch_add(1, std::memory_order_relaxed);
        else if ((flags & kFastEventSignal) == 0)
          profile.fast_non_signal.fetch_add(1, std::memory_order_relaxed);
        else if ((flags & kFastEventAutoReset) != 0)
          profile.fast_auto_reset.fetch_add(1, std::memory_order_relaxed);
      }
    } else if (event_profile_enabled()) {
      event_profile_counters().fast_out_of_range.fetch_add(1, std::memory_order_relaxed);
    }
  }

  if (args->num_events > 1 && !closing_.load(std::memory_order_acquire)) {
    bool all_fast_signal = true;
    bool all_satisfied = true;
    bool any_satisfied = false;
    constexpr uint8_t fast_signal = kFastEventValid | kFastEventSignal;

    for (uint32_t i = 0; i < args->num_events; ++i) {
      uint32_t event_id = ev_data[i].event_id;
      if (event_id >= fast_events_.size()) {
        if (event_profile_enabled())
          event_profile_counters().fast_out_of_range.fetch_add(1, std::memory_order_relaxed);
        all_fast_signal = false;
        break;
      }

      auto &fast = fast_events_[event_id];
      uint64_t state = fast.state.load(std::memory_order_acquire);
      uint8_t flags = fast_event_flags(state);
      if ((flags & (kFastEventValid | kFastEventSignal)) != fast_signal) {
        if (event_profile_enabled()) {
          auto &profile = event_profile_counters();
          if ((flags & kFastEventValid) == 0)
            profile.fast_invalid.fetch_add(1, std::memory_order_relaxed);
          else if ((flags & kFastEventSignal) == 0)
            profile.fast_non_signal.fetch_add(1, std::memory_order_relaxed);
        }
        all_fast_signal = false;
        break;
      }

      uint64_t caller_age = ev_data[i].signal_event_data.last_event_age;
      if (caller_age == 0 && (flags & kFastEventAutoReset) != 0) {
        if (event_profile_enabled())
          event_profile_counters().fast_auto_reset.fetch_add(1, std::memory_order_relaxed);
        all_fast_signal = false;
        break;
      }

      uint64_t age = fast_event_age(state);
      bool ready = caller_age == 0 ? ((flags & kFastEventSignaled) != 0) : age >= caller_age;
      any_satisfied |= ready;
      all_satisfied &= ready;
    }

    if (all_fast_signal) {
      bool ready = wait_all ? all_satisfied : any_satisfied;
      if (ready || is_poll) {
        if (event_profile_enabled())
          event_profile_counters().multi_event_waits.fetch_add(1, std::memory_order_relaxed);
        record_wait_call(ready);
        if (ready) {
          for (uint32_t i = 0; i < args->num_events; ++i) {
            auto &fast = fast_events_[ev_data[i].event_id];
            uint64_t state = fast.state.load(std::memory_order_acquire);
            uint8_t flags = fast_event_flags(state);
            uint64_t age = fast_event_age(state);
            uint64_t caller_age = ev_data[i].signal_event_data.last_event_age;
            bool event_ready =
                caller_age == 0 ? ((flags & kFastEventSignaled) != 0) : age >= caller_age;
            if (event_ready) {
              ev_data[i].signal_event_data.last_event_age = age;
              if ((flags & kFastEventAutoReset) != 0) {
                clear_fast_signaled_if_age(ev_data[i].event_id, age);
                reset_event_slot_if_age(page, page_size, ev_data[i].event_id, age);
              }
            }
          }
          args->wait_result = KFD_IOC_WAIT_RESULT_COMPLETE;
          if (event_profile_enabled())
            event_profile_counters().fast_multi_completes.fetch_add(1, std::memory_order_relaxed);
        } else {
          args->wait_result = KFD_IOC_WAIT_RESULT_TIMEOUT;
          if (event_profile_enabled())
            event_profile_counters().fast_multi_timeouts.fetch_add(1, std::memory_order_relaxed);
        }
        record_wait_result();
        return 0;
      }
      if (event_profile_enabled())
        event_profile_counters().fast_multi_not_ready_blocking.fetch_add(1,
                                                                         std::memory_order_relaxed);
    }
  }

  std::unique_lock<std::mutex> lock(mutex_);

  if (args->num_events == 1) {
    if (closing_) {
      record_wait_call(true);
      return -EBADF;
    }

    auto it = events_.find(ev_data[0].event_id);
    if (it == events_.end()) {
      if (event_profile_enabled())
        event_profile_counters().single_missing_waits.fetch_add(1, std::memory_order_relaxed);
      record_wait_call(true);
      args->wait_result = KFD_IOC_WAIT_RESULT_FAIL;
      record_wait_result();
      return 0;
    }

    if (event_profile_enabled()) {
      auto &profile = event_profile_counters();
      profile.single_event_waits.fetch_add(1, std::memory_order_relaxed);
      if (it->second.event_type == 0)
        profile.single_signal_waits.fetch_add(1, std::memory_order_relaxed);
      else
        profile.single_system_waits.fetch_add(1, std::memory_order_relaxed);
      if (it->second.auto_reset)
        profile.single_auto_reset_waits.fetch_add(1, std::memory_order_relaxed);
    }

    bool ready = satisfied(it->second, ev_data[0]);
    if (ready || is_poll) {
      record_wait_call(ready);
      if (ready) {
        uint64_t ready_age = 0;
        if (it->second.event_type == 0)
          ready_age = signal_event_age(it->second);
        if (it->second.event_type == 0)
          ev_data[0].signal_event_data.last_event_age = ready_age;
        if (it->second.auto_reset) {
          it->second.signaled = false;
          if (it->second.event_type == 0) {
            clear_fast_signaled_if_age(it->second.event_id, ready_age);
            write_event_slot(page, page_size, it->second.event_id, KFD_SIGNAL_EVENT_LIMIT);
          }
          it->second.event_age = 0;
        }
        args->wait_result = KFD_IOC_WAIT_RESULT_COMPLETE;
      } else {
        args->wait_result = KFD_IOC_WAIT_RESULT_TIMEOUT;
      }
      record_wait_result();
      return 0;
    }
  }

  auto is_ready = [&]() -> bool {
    if (closing_)
      return true;
    bool all_satisfied = true;
    bool any_satisfied = false;
    for (uint32_t i = 0; i < args->num_events; ++i) {
      auto it = events_.find(ev_data[i].event_id);
      if (it == events_.end())
        return true;
      if (satisfied(it->second, ev_data[i]))
        any_satisfied = true;
      else
        all_satisfied = false;
    }
    return wait_all ? all_satisfied : any_satisfied;
  };

  bool ready_before_wait = is_ready();
  record_wait_call(ready_before_wait);
  if (event_profile_enabled() && args->num_events != 1)
    event_profile_counters().multi_event_waits.fetch_add(1, std::memory_order_relaxed);
  if (!is_poll && !ready_before_wait) {
    if (event_profile_enabled())
      event_profile_counters().blocking_waits.fetch_add(1, std::memory_order_relaxed);
    std::condition_variable my_cv;
    for (uint32_t i = 0; i < args->num_events; ++i) {
      auto it = events_.find(ev_data[i].event_id);
      if (it != events_.end())
        it->second.waiters.push_back(&my_cv);
    }

    auto unregister_waiters = [&]() {
      for (uint32_t i = 0; i < args->num_events; ++i) {
        auto it = events_.find(ev_data[i].event_id);
        if (it != events_.end())
          std::erase(it->second.waiters, &my_cv);
      }
    };

    if (args->timeout >= 0xFFFFFFFEu)
      my_cv.wait(lock, is_ready);
    else
      my_cv.wait_for(lock, std::chrono::milliseconds(args->timeout), is_ready);

    unregister_waiters();
  }

  if (closing_)
    return -EBADF;

  bool any_ready = false;
  bool any_destroyed = false;
  bool all_ready = true;
  for (uint32_t i = 0; i < args->num_events; ++i) {
    auto it = events_.find(ev_data[i].event_id);
    if (it == events_.end()) {
      any_destroyed = true;
      all_ready = false;
      continue;
    }
    if (satisfied(it->second, ev_data[i])) {
      any_ready = true;
      uint64_t ready_age = 0;
      if (it->second.event_type == 0)
        ready_age = signal_event_age(it->second);
      if (it->second.event_type == 0)
        ev_data[i].signal_event_data.last_event_age = ready_age;
      if (it->second.auto_reset) {
        it->second.signaled = false;
        if (it->second.event_type == 0) {
          clear_fast_signaled_if_age(it->second.event_id, ready_age);
          write_event_slot(page, page_size, it->second.event_id, KFD_SIGNAL_EVENT_LIMIT);
        }
        it->second.event_age = 0;
      }
    } else {
      all_ready = false;
    }
  }

  if (any_destroyed)
    args->wait_result = KFD_IOC_WAIT_RESULT_FAIL;
  else if (wait_all ? all_ready : any_ready)
    args->wait_result = KFD_IOC_WAIT_RESULT_COMPLETE;
  else
    args->wait_result = KFD_IOC_WAIT_RESULT_TIMEOUT;
  record_wait_result();

  static thread_local uint32_t wait_log_counter = 0;
  if (args->wait_result == KFD_IOC_WAIT_RESULT_COMPLETE) {
    for (uint32_t i = 0; i < args->num_events && i < 4; ++i) {
      auto it = events_.find(ev_data[i].event_id);
      uint64_t age = (it != events_.end()) ? it->second.event_age : 999;
      util::Logger::cp([&](auto &os) {
        os << std::format("WAIT_COMPLETE: pid={} ev={} age={} poll_count={} is_poll={}", process_id,
                          ev_data[i].event_id, age, wait_log_counter, is_poll);
      });
    }
    wait_log_counter = 0;
  } else if (++wait_log_counter % 100 == 1) {
    for (uint32_t i = 0; i < args->num_events && i < 4; ++i) {
      auto it = events_.find(ev_data[i].event_id);
      uint64_t age = (it != events_.end()) ? it->second.event_age : 999;
      uint64_t caller_age = ev_data[i].signal_event_data.last_event_age;
      uint8_t etype = (it != events_.end()) ? it->second.event_type : 255;
      util::Logger::cp([&](auto &os) {
        os << std::format("WAIT_UNSATISFIED: pid={} ev={} age={} caller_age={} type={} result={} "
                          "poll_count={} wait_all={} num_events={} auto_reset={}",
                          process_id, ev_data[i].event_id, age, caller_age, (unsigned)etype,
                          args->wait_result, wait_log_counter, wait_all, args->num_events,
                          (it != events_.end()) ? it->second.auto_reset : false);
      });
    }
  }

  return 0;
}

/// @brief SimulatedDriver wrapper for CREATE_EVENT.
/// @details Resolves the dGPU event page from the allocation table before
///          delegating to EventState.
int SimulatedDriver::create_event_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_create_event_args *>(arg);
  if (args->event_page_offset != 0 && !proc.event_state_.page) {
    uint64_t raw = static_cast<uint64_t>(args->event_page_offset);
    std::lock_guard<std::mutex> alock(proc.alloc_mutex_);
    auto it = proc.allocations_.find(raw >> 12);
    if (it == proc.allocations_.end() || !it->second.host_ptr)
      it = proc.allocations_.find(raw);
    if (it != proc.allocations_.end() && it->second.host_ptr) {
      util::Logger::vm("CREATE_EVENT: adopted event page handle=", it->first, " ptr=0x", std::hex,
                       reinterpret_cast<uintptr_t>(it->second.host_ptr), " size=", std::dec,
                       it->second.size);
      proc.event_state_.adopt_page(it->second.host_ptr, it->second.size);
    } else {
      util::Logger::vm("CREATE_EVENT: event_page_offset=0x", std::hex, raw,
                       " could not find valid allocation");
    }
  }
  return proc.event_state_.create_event(arg, gpu_id());
}

int SimulatedDriver::destroy_event_ioctl(KfdProcess &proc, void *arg) {
  return proc.event_state_.destroy_event(arg);
}

int SimulatedDriver::set_event_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_set_event_args *>(arg);
  util::Logger::cp("SET_EVENT_IOCTL: pid=", proc.process_id(), " event_id=", args->event_id);
  return proc.event_state_.set_event(arg);
}

int SimulatedDriver::reset_event_ioctl(KfdProcess &proc, void *arg) {
  return proc.event_state_.reset_event(arg);
}

int SimulatedDriver::wait_events_ioctl(KfdProcess &proc, void *arg) {
  return proc.event_state_.wait_events(arg, proc.process_id());
}

} // namespace rocjitsu
