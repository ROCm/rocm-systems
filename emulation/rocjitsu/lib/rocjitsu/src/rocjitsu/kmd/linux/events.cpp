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
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <unistd.h>

namespace rocjitsu {

namespace {

void write_event_slot(void *page, size_t page_size, uint32_t event_id, uint64_t value) {
  if (!page || event_id >= page_size / sizeof(uint64_t))
    return;
  auto *slots = static_cast<uint64_t *>(page);
  std::atomic_ref<uint64_t>(slots[event_id]).store(value, std::memory_order_release);
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
}

/// @brief Signal a specific event from the CP's interrupt callback.
void EventState::signal_interrupt(uint32_t event_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = events_.find(event_id);
  if (it != events_.end() && it->second.event_type == 0) {
    it->second.event_age = 1;
    write_event_slot(page, page_size, event_id, 1);
    for (auto *cv : it->second.waiters)
      cv->notify_one();
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

  if (next_event_id_ >= KFD_SIGNAL_EVENT_LIMIT)
    return -ENOSPC;

  GpuEvent ev{};
  ev.event_id = next_event_id_++;
  ev.event_type = args->event_type;
  ev.auto_reset = args->auto_reset != 0;

  events_[ev.event_id] = ev;

  args->event_id = ev.event_id;
  args->event_trigger_data = ev.event_id;
  args->event_slot_index = ev.event_id;
  args->event_page_offset = KFD_MMAP_TYPE_EVENTS | kfd_mmap_gpu_id(gpu_id);

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
  }
  write_event_slot(page, page_size, args->event_id, KFD_SIGNAL_EVENT_LIMIT);
  return 0;
}

/// @brief Signal an event: set age to 1, write event page, wake waiters.
int EventState::set_event(void *arg) {
  assert(arg && "set_event called with null arg");
  auto *args = static_cast<kfd_ioctl_set_event_args *>(arg);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = events_.find(args->event_id);
  if (it == events_.end())
    return -EINVAL;
  it->second.event_age = 1;
  write_event_slot(page, page_size, args->event_id, 1);
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
  it->second.event_age = 0;
  write_event_slot(page, page_size, args->event_id, KFD_SIGNAL_EVENT_LIMIT);
  return 0;
}

/// @brief Block until waited events satisfy the predicate, or timeout/close.
int EventState::wait_events(void *arg) {
  assert(arg && "wait_events called with null arg");
  auto *args = static_cast<kfd_ioctl_wait_events_args *>(arg);
  auto *ev_data = reinterpret_cast<kfd_event_data *>(args->events_ptr);
  const bool wait_all = args->wait_for_all != 0;

  auto satisfied = [](const GpuEvent &ev, const kfd_event_data &ed) -> bool {
    if (ev.event_type == 0) {
      uint64_t caller_age = ed.signal_event_data.last_event_age;
      return (caller_age != 0) ? (ev.event_age >= caller_age) : (ev.event_age > 0);
    }
    return ev.event_age > 0;
  };

  std::condition_variable my_cv;
  std::unique_lock<std::mutex> lock(mutex_);

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

  if (args->timeout == 0) {
    // Poll mode.
  } else if (args->timeout >= 0xFFFFFFFEu) {
    my_cv.wait(lock, is_ready);
  } else {
    my_cv.wait_for(lock, std::chrono::milliseconds(args->timeout), is_ready);
  }

  unregister_waiters();

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
      if (it->second.event_type == 0)
        ev_data[i].signal_event_data.last_event_age = it->second.event_age;
      if (it->second.auto_reset) {
        it->second.event_age = 0;
        if (it->second.event_type == 0)
          write_event_slot(page, page_size, it->second.event_id, KFD_SIGNAL_EVENT_LIMIT);
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

  return 0;
}

/// @brief SimulatedDriver wrapper for CREATE_EVENT.
/// @details Resolves the dGPU event page from the allocation table before
///          delegating to EventState.
int SimulatedDriver::create_event_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_create_event_args *>(arg);
  if (args->event_page_offset != 0) {
    uint64_t handle = static_cast<uint64_t>(args->event_page_offset) >> 12;
    std::lock_guard<std::mutex> alock(alloc_mutex_);
    auto it = allocations_.find(handle);
    if (it != allocations_.end() && it->second.host_ptr) {
      util::Logger::vm("CREATE_EVENT: adopted pre-allocated event page handle=", handle,
                       " host_ptr=0x", std::hex, reinterpret_cast<uintptr_t>(it->second.host_ptr),
                       " size=", std::dec, it->second.size);
      event_state_.adopt_page(it->second.host_ptr, it->second.size);
    }
  }
  return event_state_.create_event(arg, gpu_id_);
}

/// @brief SimulatedDriver wrapper for DESTROY_EVENT.
int SimulatedDriver::destroy_event_ioctl(void *arg) { return event_state_.destroy_event(arg); }

/// @brief SimulatedDriver wrapper for SET_EVENT.
int SimulatedDriver::set_event_ioctl(void *arg) { return event_state_.set_event(arg); }

/// @brief SimulatedDriver wrapper for RESET_EVENT.
int SimulatedDriver::reset_event_ioctl(void *arg) { return event_state_.reset_event(arg); }

/// @brief SimulatedDriver wrapper for WAIT_EVENTS.
int SimulatedDriver::wait_events_ioctl(void *arg) { return event_state_.wait_events(arg); }

} // namespace rocjitsu
