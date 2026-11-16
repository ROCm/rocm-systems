/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>
#include <cinttypes>
#include <cstddef>
#include <chrono>
#include <cerrno>
#include <vector>

#if !defined(WIN32)
#include <sys/eventfd.h>
#include <poll.h>
#include <unistd.h>
#endif

#include "impl/wddm/device.h"
#include "impl/wddm/event.h"

using namespace std::chrono;

namespace wsl {
namespace thunk {

static constexpr uint32_t kWaitTimeout = 6000;

#if defined(WIN32)

struct WaitEntry {
  WaitEntry* next;
  WaitEntry** pnext;
  uint64_t age;
  WaitQueue *q;
  std::atomic_bool* wait_signal;

  bool IsAttached() const { return pnext != nullptr; }
};

struct WaitQueue {
  uint64_t age;
  std::atomic_bool lock;
  std::atomic<WaitQueue*> next;
  WaitEntry* waiters;
  std::atomic_uint64_t cnt;

  WaitQueue(uint64_t age, bool locked, WaitQueue* next)
      : age(age), lock(locked), next(next), waiters(nullptr), cnt(0) {}

  void Lock() { while (lock.exchange(true)) YieldProcessor(); }
  void Unlock() { lock = false; }
  void AddEntry(WaitEntry* ent) {
    ent->next = waiters;
    ent->pnext = &waiters;
    if (waiters) waiters->pnext = &ent->next;
    waiters = ent;
  }
  void RemoveEntry(WaitEntry *ent) {
    if (ent->next) ent->next->pnext = ent->pnext;
    *ent->pnext = ent->next;

    ent->pnext = nullptr;
  }
};

// ROCr's AsyncEventsLoop passes 0xFFFFFFFE (runtime.cpp) and KFD treats 0xFFFFFFFF as
// infinite. Treat anything at or above the lower of the two as "no deadline" so neither
// sentinel can be mistaken for a finite timeout and silently truncated.
static constexpr uint32_t kInfiniteTimeout = 0xFFFFFFFEu;

// ================================================================================================
EventWin::EventWin()
    : os_event_(nullptr),
      is_reference_(false),
      cur_waiting_age_(0),
      cur_age_(1),
      refcnt_(0),
      qhead_(nullptr) {
  EventId = 0;
  memset(&EventData, 0, sizeof(EventData));
  tpwait_obj_ = CreateThreadpoolWait(&EventWin::OnKmdEvent, this, nullptr);
}

// ================================================================================================
EventWin::~EventWin() {
  SetThreadpoolWait(tpwait_obj_, nullptr, nullptr);   // Cancel
  WaitForThreadpoolWaitCallbacks(tpwait_obj_, TRUE);  // drain in-flight, cancel not-started
  assert(refcnt_.load() == 0 && "event destroyed with a wait in flight");

  // Force device 0, since KMD should handle multiple devices.
  WDDMDevice* device = WddmDevice(0);  // Event->EventData.HWData3
  assert(device && "Couldn't obtain a device!");
  if (EventId != 0) {
    if (!device->UnregisterEvent(EventId, os_event_)) {
      pr_err("KMD deregister failed!");
    }
    if (os_event_ != nullptr) {
      CloseHandle(os_event_);
    }
    os_event_ = nullptr;
  }
  CloseThreadpoolWait(tpwait_obj_);
  for (auto* q = qhead_.load(); q;) {
    assert(q->cnt.load() == 0 && "event destroyed with waiters parked");
    auto* n = q->next.load();
    delete q;
    q = n;
  }
}

// ================================================================================================
bool EventWin::Init(const HsaEventDescriptor& event_desc, const wchar_t* pName) {
  if (tpwait_obj_ == nullptr) {
    pr_err("CreateThreadpoolWait failed; event cannot be armed\n");
    return false;
  }

  // Allocate OS specific events to handle HSA event, force device 0 and KMD should handle
  // multiiple devices.
  WDDMDevice* device = WddmDevice(0);  // EventDesc->NodeId
  assert(device && "Couldn't obtain a device!");
  // Allocate OS event
  SECURITY_ATTRIBUTES attributes = {};
  os_event_ = CreateEventW(&attributes, false, false, nullptr);
  if (os_event_ == nullptr) {
    pr_debug("CreateEventW call failed\n");
    return false;
  }

  SetThreadpoolWait(tpwait_obj_, os_event_, nullptr);

  // Register OS event in KMD
  EventId = device->RegisterEvent(event_desc.EventType, os_event_, &EventData.HWData2);
  if (EventId == 0) {
    // KMD ran out of slots or failed
    CloseHandle(os_event_);
    os_event_ = nullptr;
    return false;
  }
  EventData.EventType = event_desc.EventType;
  // ROCR doesn't use HWData1 or HWData3 fields, so save NodeId here...
  EventData.HWData3 = event_desc.NodeId;

  EventData.EventData.SyncVar.SyncVar.UserData = event_desc.SyncVar.SyncVar.UserData;
  EventData.EventData.SyncVar.SyncVarSize = event_desc.SyncVar.SyncVarSize;
  return true;
}

// ================================================================================================
bool EventWin::Set() {
  pr_debug("Set event id %u\n", EventId);
  Signal();
  return true;
}

// ================================================================================================
bool EventWin::Reset() const {
  pr_debug("Reset id %u\n", EventId);
  // Windows function returns non-zero values to indicate success.
  if (ResetEvent(os_event_) == FALSE) {
    return false;
  }
  return true;
}

// ================================================================================================
bool EventWin::Open(EventHandle handle, bool isReference) { return true; }

// ================================================================================================
bool EventWin::Wait(std::chrono::duration<float> timeout) const {
  HsaEvent* self = const_cast<EventWin*>(this);
  uint64_t age = cur_age_.load(std::memory_order_acquire);

  const double msec_f = duration_cast<duration<double, std::milli>>(timeout).count();
  uint32_t msec;
  if (msec_f >= static_cast<double>(kInfiniteTimeout)) {
    msec = kInfiniteTimeout;
  } else {
    msec = static_cast<uint32_t>(msec_f);
  }

  return WaitOnMultipleEvents(&self, &age, 1, true, msec) == HSAKMT_STATUS_SUCCESS;
}

static WaitQueue* kQueuesLockedPtr = (WaitQueue*) 0xdeadbeefdeadbeef;

void EventWin::OnKmdEvent(PTP_CALLBACK_INSTANCE _instance, void* ctx, PTP_WAIT _wait,
                           TP_WAIT_RESULT _wait_result) {
  auto event = reinterpret_cast<EventWin*>(ctx);
  pr_debug("TID %ld Tick %llu OnKmdEvent id %u\n", GetCurrentThreadId(), GetTickCount64(), event->EventId);
  SetThreadpoolWait(event->tpwait_obj_, event->os_event_, nullptr);
  reinterpret_cast<EventWin*>(ctx)->Signal();
}

// ================================================================================================
void EventWin::Signal() {
  std::atomic<WaitQueue*> *pnext;
  WaitQueue *q;
  auto age = cur_age_.fetch_add(1);
  while (refcnt_.load() != 0) YieldProcessor();

 retry:
  for (pnext = &qhead_; (q = pnext->load()); pnext = &q->next) {
    if (q == kQueuesLockedPtr)
      goto retry;
    if (q->age == age)
      break;
  }

  if (!q) {
    return;
  }

  while (q->next.load() != nullptr);
  auto expect = q;
  if (!pnext->compare_exchange_strong(expect, nullptr))
    goto retry;

  int cnt = 0;
  q->Lock();
  for (auto ent = q->waiters; ent; ent = ent->next) {
    ent->wait_signal->store(true);
    WakeByAddressAll(ent->wait_signal);
    cnt++;
  }
  q->Unlock();

  while (q->cnt.load() != 0) YieldProcessor();

  pr_debug("TID %ld Signal age %lld notified %d waiters\n", GetCurrentThreadId(), age, cnt);

  delete q;
}

bool EventWin::AddWaitEntry(WaitEntry* ent) {
  refcnt_.fetch_add(1);
 retry:
  auto q = qhead_.load();

  if (q == kQueuesLockedPtr)
    goto retry;
  if (ent->age != cur_age_.load() || (q && q->age > ent->age)) {
    ent->q = nullptr;
    refcnt_.fetch_sub(1);
    return false;
  }

  if (!q || q->age < ent->age) {
    if (!qhead_.compare_exchange_strong(q, kQueuesLockedPtr))
      goto retry;
    qhead_.store(q = new WaitQueue(ent->age, true, q));
  } else {
    q->Lock();
  }

  q->AddEntry(ent);
  q->Unlock();
  q->cnt.fetch_add(1);

  ent->q = q;
  refcnt_.fetch_sub(1);

  return true;
}

void EventWin::RemoveWaitEntry(WaitEntry* ent) {
  if (ent->q == nullptr)
    return;

  auto q = std::exchange(ent->q, nullptr);
  q->Lock();
  q->RemoveEntry(ent);
  q->Unlock();

  // Signal the ThreadpoolWait callback so that we can free the q.
  q->cnt.fetch_sub(1);
}

HSAKMT_STATUS EventWin::QueryMultipleEvents(HsaEvent* events[], uint64_t event_ages[], uint32_t nr_events,
                                            bool wait_all) {
  if (nr_events == 0)
    return HSAKMT_STATUS_SUCCESS;

  int found = -1;
  bool all = true;
  for (size_t i = 0; i < nr_events; i++) {
    auto cur_age = reinterpret_cast<EventWin*>(events[i])->cur_age_.load();
    if (std::exchange(event_ages[i], cur_age) < cur_age) {
      found = i;
    } else {
      all = false;
    }
  }

  if (found != -1 && wait_all && all) {
    return HSAKMT_STATUS_SUCCESS;
  }

  if (found != -1 && !wait_all) {
    return HSAKMT_STATUS_SUCCESS;
  }

  return HSAKMT_STATUS_WAIT_TIMEOUT;
}

HSAKMT_STATUS EventWin::WaitOnMultipleEvents(HsaEvent* events[], uint64_t event_ages[],
                                             uint32_t nr_events, bool wait_all, uint32_t timeout) {
  if (nr_events == 0)
    return HSAKMT_STATUS_SUCCESS;

  if (!dxg_runtime->disable_wait_timeout_ && timeout > kWaitTimeout) timeout = kWaitTimeout;

  static constexpr int kMaxOnStackEntries = 1024;
  std::unique_ptr<WaitEntry[]> heap_allocated_entries;
  std::unique_ptr<uint64_t[]> heap_allocated_ages;
  uint64_t start_time = GetTickCount64();

  WaitEntry* entries;
  if (nr_events < kMaxOnStackEntries) {
    entries = static_cast<WaitEntry*>(alloca(sizeof(WaitEntry) * nr_events));
  } else {
    heap_allocated_entries.reset(new WaitEntry[nr_events]);
    entries = heap_allocated_entries.get();
  }

  if (event_ages == nullptr) {
    if (nr_events < kMaxOnStackEntries) {
      event_ages = static_cast<uint64_t*>(alloca(sizeof(uint64_t) * nr_events));
    } else {
      heap_allocated_ages.reset(new uint64_t[nr_events]);
      event_ages = heap_allocated_ages.get();
    }
    for (uint32_t i = 0; i < nr_events; i++) {
      event_ages[i] = reinterpret_cast<EventWin*>(events[i])->cur_age_.load();
    }
  }

  HSAKMT_STATUS result;
  while ((result = QueryMultipleEvents(events, event_ages, nr_events, wait_all)) ==
             HSAKMT_STATUS_WAIT_TIMEOUT &&
         timeout > 0) {
    std::fill(entries, entries + nr_events, WaitEntry{});

    std::atomic_bool signal = false;
    bool old = false;
    for (uint32_t i = 0; i < nr_events; i++) {
      entries[i].wait_signal = &signal;
      entries[i].age = event_ages[i];
      reinterpret_cast<EventWin*>(events[i])->AddWaitEntry(&entries[i]);
    }

    uint64_t now = GetTickCount64();
    if (timeout < now - start_time) {
      timeout = 0;
    } else {
      timeout -= now - start_time;
    }
    start_time = now;

    if (timeout > 0 &&
        EventWin::QueryMultipleEvents(events, event_ages, nr_events, wait_all) ==
            HSAKMT_STATUS_WAIT_TIMEOUT) {
      WaitOnAddress(&signal, &old, sizeof(bool), timeout);
    }

    for (uint32_t i = 0; i < nr_events; i++) {
      reinterpret_cast<EventWin*>(events[i])->RemoveWaitEntry(&entries[i]);
      event_ages[i] = entries[i].age;
    }
  }

  return result;
}

#else
// ================================================================================================
EventLnx::EventLnx() : efd_(-1), syncobj_(0) {
  EventId = 0;
  memset(&EventData, 0, sizeof(EventData));
}

// ================================================================================================
EventLnx::~EventLnx() {
  // Force device 0, since KMD should handle multiple devices.
  WDDMDevice* device = WddmDevice(0);  // Event->EventData.HWData3
  assert(device && "Couldn't obtain a device!");
  if (EventId != 0) {
    if (!device->DestroyEvent(EventId, efd_, syncobj_)) {
      pr_err("KMD deregister failed!\n");
    }
    syncobj_ = 0;
    efd_ = -1;
  }
}

// ================================================================================================
bool EventLnx::Init(const HsaEventDescriptor& event_desc, const wchar_t* pName) {
  // Allocate OS specific events to handle HSA event, force device 0 and KMD should handle
  // multiiple devices.
  WDDMDevice* device = WddmDevice(0);  // EventDesc->NodeId
  assert(device && "Couldn't obtain a device!");

  if (!device->CreateEvent(&efd_, event_desc.EventType, &EventId, &EventData.HWData2, &syncobj_)) {
    return false;
  }
  EventData.EventType = event_desc.EventType;
  // ROCR doesn't use HWData1 or HWData3 fields, so save NodeId here...
  EventData.HWData3 = event_desc.NodeId;

  EventData.EventData.SyncVar.SyncVar.UserData = event_desc.SyncVar.SyncVar.UserData;
  EventData.EventData.SyncVar.SyncVarSize = event_desc.SyncVar.SyncVarSize;
  return true;
}

// ================================================================================================
bool EventLnx::Set() const {
  const uint64_t one = 1;
  if (write(efd_, &one, sizeof(one)) != static_cast<ssize_t>(sizeof(one))) {
    pr_err("eventfd write failed!");
    return false;
  }
  return true;
}

// ================================================================================================
bool EventLnx::Reset() const {
  // An eventfd counter stays readable until drained; loop until EAGAIN so a
  // subsequent poll() blocks until the event is signaled again.
  uint64_t value;
  while (read(efd_, &value, sizeof(value)) == static_cast<ssize_t>(sizeof(value))) {
  }
  return true;
}

// ================================================================================================
bool EventLnx::Wait(std::chrono::duration<float> timeout  // max time to wait
) const {
  HsaEvent* self = const_cast<EventLnx*>(this);
  uint64_t event_age = 1;
  HSAKMT_STATUS status = WaitOnMultipleEvents(
      &self, &event_age, 1, true, static_cast<uint32_t>(duration_cast<milliseconds>(timeout).count()));
  return status == HSAKMT_STATUS_SUCCESS;
}

// ================================================================================================
HSAKMT_STATUS EventLnx::WaitOnMultipleEvents(HsaEvent* events[], uint64_t event_ages[],
                                             uint32_t num_elems, bool wait_all, uint32_t msec) {
  if (num_elems == 0) return HSAKMT_STATUS_SUCCESS;

  if (!dxg_runtime->disable_wait_timeout_ && msec > kWaitTimeout) msec = kWaitTimeout;

  std::vector<struct pollfd> pfds(num_elems);
  std::vector<bool> signaled(num_elems, false);
  for (uint32_t i = 0; i < num_elems; i++) {
    pfds[i].fd = reinterpret_cast<EventLnx*>(events[i])->efd_;
    pfds[i].events = POLLIN;
    pfds[i].revents = 0;
  }

  auto done = [&]() -> bool {
    uint32_t count = 0;
    for (uint32_t i = 0; i < num_elems; i++)
      if (signaled[i])
        count++;
    return wait_all ? (count == num_elems) : (count > 0);
  };

  constexpr uint32_t kInfinite = 0xFFFFFFFF;
  const bool infinite = (msec == kInfinite);
  auto deadline = steady_clock::now() + milliseconds(msec);

  while (true) {
    int timeout_ms = -1;
    if (!infinite) {
      auto now = steady_clock::now();
      timeout_ms = now >= deadline
                       ? 0
                       : static_cast<int>(
                             duration_cast<milliseconds>(deadline - now).count());
    }

    for (uint32_t i = 0; i < num_elems; i++)
      pfds[i].revents = 0;

    int ret = poll(pfds.data(), num_elems, timeout_ms);
    if (ret < 0) {
      if (errno == EINTR)
        continue;
      pr_err("poll fail %d\n", errno);
      return HSAKMT_STATUS_WAIT_FAILURE;
    }
    if (ret == 0)
      return done() ? HSAKMT_STATUS_SUCCESS : HSAKMT_STATUS_WAIT_TIMEOUT;

    // An eventfd stays readable until Reset() drains it; a negative fd makes
    // poll() skip it so wait_all blocks on the remaining events instead of spinning.
    for (uint32_t i = 0; i < num_elems; i++) {
      if (pfds[i].revents & (POLLIN | POLLERR | POLLHUP)) {
        signaled[i] = true;
        pfds[i].fd = -1;
      }
    }

    if (done())
      return HSAKMT_STATUS_SUCCESS;
    if (!infinite && steady_clock::now() >= deadline)
      return HSAKMT_STATUS_WAIT_TIMEOUT;
  }
}
#endif

}  // namespace thunk
}  // namespace wsl
