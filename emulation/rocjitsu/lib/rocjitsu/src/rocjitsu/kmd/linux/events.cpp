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

  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto &[id, ev] : events_) {
    if (ev.signaled)
      write_event_slot(page, page_size, id, ev.event_age);
  }
}

/// @brief Signal event(s) from the CP's interrupt callback.
/// @details When event_id is non-zero, signals that specific event. When
///          event_id is zero, broadcasts to all type-0 events — matching real
///          KFD's kfd_signal_event_interrupt(pasid, partial_id=0, valid_id_bits=0).
void EventState::signal_interrupt(uint32_t event_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (event_id == 0) {
    for (auto &[id, ev] : events_) {
      if (ev.event_type == 0) {
        ev.signaled = !ev.auto_reset || ev.waiters.empty();
        if (!(++ev.event_age))
          ev.event_age = 2;
        write_event_slot(page, page_size, id, ev.event_age);
        util::Logger::cp("SIGNAL_BROADCAST: event_id=", id,
                         " age=", ev.event_age,
                         " waiters=", ev.waiters.size());
        for (auto *cv : ev.waiters)
          cv->notify_one();
      }
    }
    return;
  }
  auto it = events_.find(event_id);
  if (it != events_.end() && it->second.event_type == 0) {
    it->second.signaled = !it->second.auto_reset || it->second.waiters.empty();
    if (!(++it->second.event_age))
      it->second.event_age = 2;
    write_event_slot(page, page_size, event_id, it->second.event_age);
    util::Logger::cp("SIGNAL_INTERRUPT: event_id=", event_id,
                     " age=", it->second.event_age,
                     " waiters=", it->second.waiters.size(),
                     " page=", page ? "valid" : "null");
    for (auto *cv : it->second.waiters)
      cv->notify_one();
  } else {
    util::Logger::cp("SIGNAL_INTERRUPT_MISS: event_id=", event_id,
                     " NOT FOUND or wrong type, events_.size()=", events_.size());
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
  ev.event_age = 1;

  events_[ev.event_id] = ev;

  args->event_id = ev.event_id;
  args->event_trigger_data = ev.event_id;
  args->event_slot_index = ev.event_id;
  args->event_page_offset = KFD_MMAP_TYPE_EVENTS | kfd_mmap_gpu_id(gpu_id);

  util::Logger::cp([&](auto &os) {
    os << std::format("CREATE_EVENT: event_id={} type={} auto_reset={} gpu_id={}",
                      ev.event_id, ev.event_type, ev.auto_reset, gpu_id);
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
  it->second.signaled = !it->second.auto_reset || it->second.waiters.empty();
  if (!(++it->second.event_age))
    it->second.event_age = 2;
  write_event_slot(page, page_size, args->event_id, it->second.event_age);
  util::Logger::cp("SET_EVENT: event_id=", args->event_id,
                   " age=", it->second.event_age,
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
    os << "WAIT_EVENTS: pid=" << process_id
       << " num=" << args->num_events << " timeout=" << args->timeout
       << " wait_all=" << wait_all;
    for (uint32_t i = 0; i < args->num_events && i < 4; ++i)
      os << " ev[" << i << "]=" << ev_data[i].event_id
         << "(age=" << ev_data[i].signal_event_data.last_event_age << ")";
  });

  auto satisfied = [](const GpuEvent &ev, const kfd_event_data &ed) -> bool {
    if (ev.event_type == 0) {
      uint64_t caller_age = ed.signal_event_data.last_event_age;
      if (caller_age == 0)
        return ev.signaled;
      return ev.event_age != caller_age;
    }
    return ev.signaled;
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

  bool is_poll = (args->timeout == 0);
  if (is_poll) {
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
        it->second.signaled = false;
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

  static thread_local uint32_t wait_log_counter = 0;
  if (args->wait_result == KFD_IOC_WAIT_RESULT_COMPLETE) {
    for (uint32_t i = 0; i < args->num_events && i < 4; ++i) {
      auto it = events_.find(ev_data[i].event_id);
      uint64_t age = (it != events_.end()) ? it->second.event_age : 999;
      util::Logger::cp([&](auto &os) {
        os << std::format("WAIT_COMPLETE: pid={} ev={} age={} poll_count={} is_poll={}",
                          process_id, ev_data[i].event_id, age, wait_log_counter, is_poll);
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
      util::Logger::vm("CREATE_EVENT: adopted event page handle=", it->first,
                       " ptr=0x", std::hex, reinterpret_cast<uintptr_t>(it->second.host_ptr),
                       " size=", std::dec, it->second.size);
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
