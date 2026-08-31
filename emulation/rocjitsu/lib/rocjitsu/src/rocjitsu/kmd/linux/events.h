// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_EVENTS_H_
#define ROCJITSU_KMD_LINUX_EVENTS_H_

/// @file events.h
/// @brief KFD event subsystem for the simulated driver.
///
/// @details Models the kernel's KFD event infrastructure: event creation,
/// signaling, reset, destruction, and blocking waits. Each event maintains a
/// per-event waiter list matching the kernel's wait_queue model — when an event
/// fires, only threads registered on that specific event are woken.
///
/// Event age semantics follow the real ROCR ↔ KFD protocol: ROCR passes
/// last_event_age=1 on every WAIT_EVENTS call. set_event/interrupt sets
/// event_age to 1; auto-reset clears it to 0. The wait predicate checks
/// event_age >= last_event_age for signal events only.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

/// @brief KFD event subsystem state.
///
/// @details Owns all event-related state for the simulated driver: the event
/// table, event page, and synchronization primitives. The SimulatedKfd
/// delegates event ioctls to this class and accesses the event page and memfd
/// for mmap/munmap operations.
class EventState {
public:
  ~EventState();

  /// @brief Handle KFD CREATE_EVENT ioctl.
  /// @param arg Pointer to kfd_ioctl_create_event_args.
  /// @param gpu_id KFD gpu_id for the event page mmap offset.
  /// @returns 0 on success, -ENOSPC if event limit reached.
  int create_event(void *arg, uint32_t gpu_id);

  /// @brief Handle KFD DESTROY_EVENT ioctl.
  /// @param arg Pointer to kfd_ioctl_destroy_event_args.
  /// @details Wakes all waiters on the destroyed event and writes a sentinel
  ///          to its event page slot.
  /// @returns 0 unconditionally.
  int destroy_event(void *arg);

  /// @brief Handle KFD SET_EVENT ioctl.
  /// @param arg Pointer to kfd_ioctl_set_event_args.
  /// @details Sets event_age to 1, writes to the event page slot, and wakes
  ///          all registered waiters.
  /// @returns 0 on success, -EINVAL if event_id not found.
  int set_event(void *arg);

  /// @brief Handle KFD RESET_EVENT ioctl.
  /// @param arg Pointer to kfd_ioctl_reset_event_args.
  /// @details Clears event_age to 0.
  /// @returns 0 on success, -EINVAL if event_id not found.
  int reset_event(void *arg);

  /// @brief Handle KFD WAIT_EVENTS ioctl.
  /// @param arg Pointer to kfd_ioctl_wait_events_args.
  /// @details Blocks the caller until waited-on signal events satisfy the
  ///          age predicate, respecting wait_for_all (AND vs OR semantics).
  ///          Creates a thread-local CV and registers it with each waited event.
  /// @returns 0 on success, -EBADF if the driver is closing.
  int wait_events(void *arg, uint32_t process_id = 0);

  /// @brief Adopt a pre-allocated event page from the FMM allocator (dGPU path).
  /// @details Idempotent, subsequent calls after the first are no-ops.
  void adopt_page(void *ptr, size_t size);

  /// @brief If @p ptr is the adopted event page, clear page_/page_size_ and return true.
  /// @details Clears the fields under mutex_ so the CP interrupt thread (which
  /// reads page_/page_size_ under the same lock in signal_interrupt) can never race
  /// a concurrent munmap tearing the mapping down.
  [[nodiscard]] bool release_page(void *ptr);

  /// @brief Signal event(s) from the CP's interrupt callback.
  /// @details When event_id is non-zero, signals that specific event. When
  ///          event_id is zero, broadcasts to all type-0 events — matching
  ///          real KFD's kfd_signal_event_interrupt(pasid, 0, 0) broadcast.
  void signal_interrupt(uint32_t event_id);

  /// @brief Wake all event waiters for driver shutdown.
  /// @details Sets the closing flag and notifies every registered waiter
  ///          across all events.
  void notify_closing();

  /// @brief Write sentinel values to all event page slots for shutdown.
  /// @details Writes KFD_SIGNAL_EVENT_LIMIT to every slot, allowing ROCR's
  ///          userspace polling to detect that events will no longer fire. Takes
  ///          mutex_ so it cannot race a concurrent release_page()/munmap tearing
  ///          the mapping down.
  void signal_page_shutdown();

  /// @brief Non-destructively release parked waiters so their caller can unwind.
  /// @details The teardown-probe counterpart of notify_closing(): sets a
  ///          cancellation flag and wakes every registered waiter, so a blocking
  ///          WAIT_EVENTS returns a benign KFD_IOC_WAIT_RESULT_TIMEOUT (rc 0) and
  ///          drops the driver snapshot that would otherwise keep the object alive.
  ///          Mutates NO event, slot, or page state. A spurious TIMEOUT is a
  ///          legal KFD outcome every caller already re-polls on; -EBADF (which
  ///          notify_closing() causes) is not, which is why teardown probing must
  ///          use this and only real teardown may close. Idempotent.
  void begin_wait_cancel();

  /// @brief Reset closing state for driver re-open.
  void reset();

  /// @brief Check if the driver is shutting down.
  bool is_closing() const;

  /// @brief Return the stable event-page backing fd, creating it exactly once.
  /// @details The check-and-create runs under mutex_, so concurrent event-page
  /// mmaps cannot each build a backing and hand different fds to different callers
  /// — which would leave one polling an object that never receives event updates.
  /// @param length Size to pass to @p create_backing.
  /// @param create_backing Builds the backing and returns its fd, or -1 on failure.
  ///        Called at most once per EventState, with mutex_ held.
  /// @returns The backing fd, or -1 if creation failed.
  int ensure_backing(size_t length, const std::function<int(size_t)> &create_backing);

  /// @brief The backing fd, or -1 if none has been created. Takes mutex_.
  [[nodiscard]] int backing_fd() const;

  /// @brief Number of waiters currently registered on @p event_id.
  /// @details Test seam for deterministically observing that a waiter has parked,
  /// so a test need not sleep and hope. Takes mutex_.
  [[nodiscard]] size_t waiter_count(uint32_t event_id) const;

  /// @brief True if an event page is currently adopted.
  /// @details Takes mutex_, so it cannot race a concurrent release_page()/munmap
  /// the way a direct read of the `page_` field can.
  [[nodiscard]] bool has_page() const;

private:
  int memfd_ = -1;       ///< memfd backing the KFD signal event page.
  void *page_ = nullptr; ///< Mapped signal page (libhsakmt polls slots here).
  size_t page_size_ = 0; ///< Size of the mapped event page in bytes.

  /// @brief Internal event representation.
  struct GpuEvent {
    uint32_t event_id = 0;   ///< KFD event ID (1-based, matches slot index).
    uint32_t event_type = 0; ///< HSA event type (0 = signal, others = system).
    bool auto_reset = false; ///< If true, signaled clears after wakeup.
    bool signaled = false;   ///< True when the event has been signaled.
    uint64_t event_age = 1;  ///< Monotonic age counter (starts at 1, matching real KFD).
    std::vector<std::condition_variable *> waiters; ///< Per-event waiter list (kernel wait_queue).
  };

  mutable std::mutex mutex_;                      ///< Protects all mutable event state.
  std::unordered_map<uint32_t, GpuEvent> events_; ///< Event table keyed by event_id.
  uint32_t next_event_id_ = 1;                    ///< Next event ID to allocate.
  std::atomic<bool> closing_{false};              ///< Set by notify_closing() for shutdown.
  /// @brief Set by begin_wait_cancel() to release waiters without closing.
  std::atomic<bool> wait_cancelled_{false};
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_EVENTS_H_
