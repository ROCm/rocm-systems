// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/sdma_queue_scheduler.h"

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/sdma_ring_consumer.h"

#include "util/log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <format>
#include <limits>
#include <stdexcept>
#include <utility>

namespace rocjitsu::amdgpu {
namespace {

using namespace std::chrono_literals;

constexpr std::chrono::microseconds kHostDoorbellPollInterval = 100us;
constexpr std::chrono::microseconds kRetryInterval = 100us;

} // namespace

class SdmaQueueScheduler::QueueRecord {
public:
  QueueRecord(GpuVm &gpu_vm, SdmaQueueConfig config, GpuVmBindingLease retained_address_space,
              SdmaPacketDialect dialect, SdmaPacketCallbacks callbacks,
              SdmaQueueProgressObserver progress_observer)
      : address_space_lease_(std::move(retained_address_space)),
        ring_consumer_(gpu_vm, std::move(config.ring), dialect, std::move(callbacks)),
        process_id_(config.context.process_id), queue_id_(config.context.queue_id),
        engine_id_(config.context.engine_id), doorbell_offset_(config.doorbell_offset),
        doorbell_base_(config.doorbell_base),
        producer_cursor_(ring_consumer_.config().initial_cursor.value_or(0)),
        reported_cursor_(ring_consumer_.config().initial_cursor.value_or(0)),
        last_doorbell_(config.last_doorbell), host_accessible_(config.host_accessible),
        progress_observer_(std::move(progress_observer)) {}

private:
  friend class SdmaQueueScheduler;

  GpuVmBindingLease address_space_lease_;
  SdmaRingConsumer ring_consumer_;
  uint32_t process_id_ = 0;
  uint32_t queue_id_ = 0;
  uint32_t engine_id_ = 0;
  uint32_t doorbell_offset_ = 0;
  void *doorbell_base_ = nullptr;
  uint64_t producer_cursor_ = 0;
  uint64_t reported_cursor_ = 0;
  uint64_t last_doorbell_ = 0;
  std::chrono::steady_clock::time_point retry_deadline_{};
  bool host_accessible_ = false;
  bool enabled_ = true;
  bool ready_ = false;
  bool retry_pending_ = false;
  bool service_active_ = false;
  // Prevent the worker from barging ahead while an update waits for an idle processor.
  uint32_t management_operations_ = 0;
  bool closing_ = false;
  bool terminal_ = false;
  SdmaQueueProgressObserver progress_observer_;
};

SdmaQueueScheduler::SdmaQueueScheduler(GpuVm &gpu_vm, SdmaPacketDialect dialect)
    : SdmaQueueScheduler(gpu_vm, std::make_shared<DeviceCacheCoherence>(), dialect) {}

SdmaQueueScheduler::SdmaQueueScheduler(GpuVm &gpu_vm,
                                       std::shared_ptr<DeviceCacheCoherence> coherence,
                                       SdmaPacketDialect dialect)
    : gpu_vm_(gpu_vm), coherence_(std::move(coherence)), dialect_(dialect) {
  if (!coherence_)
    throw std::invalid_argument("SDMA queue scheduler requires a cache-coherence domain");
}

SdmaQueueScheduler::~SdmaQueueScheduler() { shutdown(); }

std::shared_ptr<SdmaQueueScheduler::QueueRecord>
SdmaQueueScheduler::find_locked(Handle handle) const {
  if (!handle || handle.slot_ >= slots_.size())
    return {};
  const Slot &slot = slots_[handle.slot_];
  if (slot.generation != handle.generation_)
    return {};
  return slot.queue;
}

SdmaQueueScheduler::Handle SdmaQueueScheduler::allocate_locked(std::shared_ptr<QueueRecord> queue) {
  uint32_t slot_index = 0;
  if (free_slots_.empty()) {
    slot_index = static_cast<uint32_t>(slots_.size());
    slots_.push_back({});
  } else {
    slot_index = free_slots_.back();
    free_slots_.pop_back();
  }
  Slot &slot = slots_[slot_index];
  slot.queue = std::move(queue);
  return Handle(slot_index, slot.generation);
}

void SdmaQueueScheduler::ensure_worker_locked() {
  if (worker_.joinable())
    return;
  worker_ = std::jthread([this](std::stop_token stop_token) { worker_loop(stop_token); });
}

SdmaQueueScheduler::Handle SdmaQueueScheduler::attach(SdmaQueueConfig config,
                                                      SdmaPacketCallbacks callbacks,
                                                      SdmaQueueProgressObserver progress_observer) {
  if (!config.ring.address_space || config.ring.ring_bytes == 0)
    return {};

  std::lock_guard lock(mutex_);
  if (shutting_down_)
    return {};
  std::optional<GpuVmBindingLease> address_space_lease =
      gpu_vm_.retain_binding(config.ring.address_space);
  if (!address_space_lease || address_space_lease->info().vmid != config.context.process_id)
    return {};
  std::shared_ptr<QueueRecord> queue =
      std::make_shared<QueueRecord>(gpu_vm_, std::move(config), std::move(*address_space_lease),
                                    dialect_, std::move(callbacks), std::move(progress_observer));
  ensure_worker_locked();
  const Handle handle = allocate_locked(std::move(queue));
  condition_.notify_all();
  return handle;
}

QueueReconfigureStatus SdmaQueueScheduler::update(Handle handle, uint64_t ring_base,
                                                  uint32_t ring_size,
                                                  uint32_t scheduling_percentage) {
  std::unique_lock lock(mutex_);
  std::shared_ptr<QueueRecord> queue = find_locked(handle);
  if (!queue || queue->closing_)
    return QueueReconfigureStatus::Stale;
  // Reserve the next idle interval before dropping the lock in wait().
  ++queue->management_operations_;
  condition_.wait(lock, [&queue]() { return queue->closing_ || !queue->service_active_; });

  QueueReconfigureStatus status = QueueReconfigureStatus::Stale;
  if (!queue->closing_) {
    if (ring_base == 0) {
      if (queue->ring_consumer_.in_flight()) {
        status = QueueReconfigureStatus::Busy;
      } else {
        queue->enabled_ = false;
        status = QueueReconfigureStatus::Disabled;
      }
    } else if (ring_size == 0) {
      status = QueueReconfigureStatus::Invalid;
    } else {
      SdmaRingConfig config = queue->ring_consumer_.config();
      const bool ring_changed = config.ring_base != ring_base || config.ring_bytes != ring_size;
      if (ring_changed) {
        config.ring_base = ring_base;
        config.ring_bytes = ring_size;
        config.initial_cursor.reset();
        status = queue->ring_consumer_.reconfigure(std::move(config))
                     ? QueueReconfigureStatus::Applied
                     : QueueReconfigureStatus::Busy;
      } else {
        status = QueueReconfigureStatus::Applied;
      }
      if (status == QueueReconfigureStatus::Applied && scheduling_percentage == 0 &&
          queue->ring_consumer_.in_flight()) {
        status = QueueReconfigureStatus::Busy;
      }
      if (status == QueueReconfigureStatus::Applied) {
        queue->enabled_ = scheduling_percentage != 0;
        if (queue->enabled_ &&
            (queue->retry_pending_ || queue->producer_cursor_ != queue->ring_consumer_.cursor()))
          queue->ready_ = true;
        if (!queue->enabled_)
          status = QueueReconfigureStatus::Disabled;
      }
    }
  }

  --queue->management_operations_;
  condition_.notify_all();
  return status;
}

QueueSubmissionStatus SdmaQueueScheduler::notify(Handle handle, uint64_t producer_cursor) {
  std::lock_guard lock(mutex_);
  std::shared_ptr<QueueRecord> queue = find_locked(handle);
  if (!queue || queue->closing_ || queue->terminal_)
    return QueueSubmissionStatus::Faulted;

  queue->producer_cursor_ = std::max(queue->producer_cursor_, producer_cursor);
  queue->last_doorbell_ = producer_cursor;
  if (queue->enabled_) {
    queue->ready_ = true;
    queue->retry_pending_ = false;
  }
  condition_.notify_all();
  return QueueSubmissionStatus::Accepted;
}

QueuePrepareCloseStatus SdmaQueueScheduler::prepare_detach(Handle handle) noexcept {
  return detach(handle, false);
}

bool SdmaQueueScheduler::detach(Handle handle) noexcept {
  return detach(handle, true) == QueuePrepareCloseStatus::Ready;
}

QueuePrepareCloseStatus SdmaQueueScheduler::detach(Handle handle, bool force) noexcept {
  std::shared_ptr<QueueRecord> queue;
  {
    std::unique_lock lock(mutex_);
    queue = find_locked(handle);
    if (!queue)
      return QueuePrepareCloseStatus::Faulted;
    queue->closing_ = true;
    queue->ready_ = false;
    queue->retry_pending_ = false;
    condition_.notify_all();
    condition_.wait(lock, [&queue]() { return !queue->service_active_; });

    if (!force && queue->ring_consumer_.in_flight()) {
      const QueuePrepareCloseStatus status =
          queue->ring_consumer_.terminal() && queue->ring_consumer_.publication_pending()
              ? QueuePrepareCloseStatus::Faulted
              : QueuePrepareCloseStatus::Busy;
      queue->closing_ = false;
      if (status == QueuePrepareCloseStatus::Busy && queue->enabled_ && !queue->terminal_) {
        queue->retry_pending_ = true;
        queue->retry_deadline_ = std::chrono::steady_clock::now();
      }
      condition_.notify_all();
      return status;
    }

    Slot &slot = slots_[handle.slot_];
    slot.queue.reset();
    ++slot.generation;
    if (slot.generation == 0)
      ++slot.generation;
    free_slots_.push_back(handle.slot_);
    condition_.notify_all();
  }
  return QueuePrepareCloseStatus::Ready;
}

void SdmaQueueScheduler::set_process_doorbell_base(uint32_t process_id, void *base) {
  std::lock_guard lock(mutex_);
  for (const Slot &slot : slots_) {
    if (!slot.queue || slot.queue->process_id_ != process_id || !slot.queue->host_accessible_)
      continue;
    slot.queue->doorbell_base_ = base;
  }
  condition_.notify_all();
}

bool SdmaQueueScheduler::set_packet_dialect(SdmaPacketDialect dialect) {
  std::lock_guard lock(mutex_);
  const bool has_queue =
      std::ranges::any_of(slots_, [](const Slot &slot) { return static_cast<bool>(slot.queue); });
  if (has_queue)
    return false;
  dialect_ = dialect;
  return true;
}

void SdmaQueueScheduler::shutdown() noexcept {
  {
    std::jthread worker;
    {
      std::lock_guard lock(mutex_);
      if (shutting_down_ && !worker_.joinable())
        return;
      shutting_down_ = true;
      for (Slot &slot : slots_)
        if (slot.queue)
          slot.queue->closing_ = true;
      if (worker_.joinable()) {
        worker_.request_stop();
        worker = std::move(worker_);
      }
      condition_.notify_all();
    }
  }
  {
    std::lock_guard lock(mutex_);
    slots_.clear();
    free_slots_.clear();
    condition_.notify_all();
  }
}

std::size_t SdmaQueueScheduler::active_queues() const {
  std::lock_guard lock(mutex_);
  return static_cast<std::size_t>(std::ranges::count_if(
      slots_, [](const Slot &slot) { return static_cast<bool>(slot.queue); }));
}

bool SdmaQueueScheduler::contains(Handle handle) const {
  std::lock_guard lock(mutex_);
  return static_cast<bool>(find_locked(handle));
}

SdmaPacketDialect SdmaQueueScheduler::packet_dialect() const {
  std::lock_guard lock(mutex_);
  return dialect_;
}

void SdmaQueueScheduler::scan_host_doorbells_locked() {
  for (Slot &slot : slots_) {
    std::shared_ptr<QueueRecord> &queue = slot.queue;
    if (!queue || queue->closing_ || queue->terminal_ || !queue->enabled_ ||
        !queue->host_accessible_ || queue->doorbell_base_ == nullptr)
      continue;
    const uint64_t value =
        std::atomic_ref<uint64_t>(
            *reinterpret_cast<uint64_t *>(static_cast<std::byte *>(queue->doorbell_base_) +
                                          queue->doorbell_offset_))
            .load(std::memory_order_acquire);
    if (value == std::numeric_limits<uint64_t>::max() || value == queue->last_doorbell_)
      continue;
    queue->last_doorbell_ = value;
    if (value > queue->producer_cursor_) {
      queue->producer_cursor_ = value;
      queue->ready_ = true;
    }
  }
}

std::shared_ptr<SdmaQueueScheduler::QueueRecord> SdmaQueueScheduler::select_queue_locked() {
  if (slots_.empty())
    return {};
  const std::size_t start = next_slot_ % slots_.size();
  const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  for (std::size_t offset = 0; offset < slots_.size(); ++offset) {
    const std::size_t slot_index = (start + offset) % slots_.size();
    std::shared_ptr<QueueRecord> queue = slots_[slot_index].queue;
    if (!queue || queue->closing_ || queue->terminal_ || !queue->enabled_ ||
        queue->service_active_ || queue->management_operations_ != 0)
      continue;
    if (!queue->ready_ && !(queue->retry_pending_ && queue->retry_deadline_ <= now))
      continue;
    queue->ready_ = false;
    queue->retry_pending_ = false;
    queue->service_active_ = true;
    next_slot_ = (slot_index + 1) % slots_.size();
    return queue;
  }
  return {};
}

std::optional<std::chrono::steady_clock::time_point>
SdmaQueueScheduler::next_wake_deadline_locked() const {
  std::optional<std::chrono::steady_clock::time_point> deadline;
  const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  for (const Slot &slot : slots_) {
    const std::shared_ptr<QueueRecord> &queue = slot.queue;
    if (!queue || queue->closing_ || queue->terminal_ || !queue->enabled_)
      continue;
    if (queue->retry_pending_ && (!deadline || queue->retry_deadline_ < *deadline))
      deadline = queue->retry_deadline_;
    if (queue->host_accessible_ && queue->doorbell_base_ != nullptr) {
      const std::chrono::steady_clock::time_point poll_deadline = now + kHostDoorbellPollInterval;
      if (!deadline || poll_deadline < *deadline)
        deadline = poll_deadline;
    }
  }
  return deadline;
}

void SdmaQueueScheduler::worker_loop(std::stop_token stop_token) {
  while (!stop_token.stop_requested()) {
    std::shared_ptr<QueueRecord> queue;
    uint64_t producer = 0;
    {
      std::unique_lock lock(mutex_);
      scan_host_doorbells_locked();
      queue = select_queue_locked();
      if (!queue) {
        // shutdown() requests stop while holding mutex_ and then notifies this
        // condition. Recheck under the same mutex before waiting so the worker
        // cannot miss that notification between the loop condition and wait().
        if (stop_token.stop_requested())
          break;
        const std::optional<std::chrono::steady_clock::time_point> deadline =
            next_wake_deadline_locked();
        if (deadline)
          condition_.wait_until(lock, *deadline);
        else
          condition_.wait(lock);
        continue;
      }
      producer = queue->producer_cursor_;
    }

    SdmaRingStatus outcome = SdmaRingStatus::Faulted;
    try {
      outcome = queue->ring_consumer_.service(producer, kPacketsPerTurn);
    } catch (const std::exception &error) {
      util::Logger::warn(std::format("SDMA{} queue {}:{} stopped after exception: {}",
                                     queue->engine_id_, queue->process_id_, queue->queue_id_,
                                     error.what()));
      outcome = SdmaRingStatus::Faulted;
    } catch (...) {
      util::Logger::warn(std::format("SDMA{} queue {}:{} stopped after unknown exception",
                                     queue->engine_id_, queue->process_id_, queue->queue_id_));
      outcome = SdmaRingStatus::Faulted;
    }

    const bool terminal_outcome =
        outcome == SdmaRingStatus::Faulted || outcome == SdmaRingStatus::Malformed;
    const uint64_t consumer_cursor = queue->ring_consumer_.cursor();
    if (terminal_outcome) {
      // Publish terminal state before an observer can wake a producer. The
      // callback remains inside this service operation (service_active_ stays
      // true) but outside mutex_ so detach waits for it without risking
      // callback re-entrancy under the scheduler lock.
      std::lock_guard lock(mutex_);
      queue->terminal_ = true;
    }
    if (queue->progress_observer_ &&
        (consumer_cursor != queue->reported_cursor_ || terminal_outcome)) {
      try {
        queue->progress_observer_(
            {.consumer_cursor = consumer_cursor, .terminal = terminal_outcome});
      } catch (const std::exception &error) {
        util::Logger::warn(std::format("SDMA{} queue {}:{} progress observer failed: {}",
                                       queue->engine_id_, queue->process_id_, queue->queue_id_,
                                       error.what()));
      } catch (...) {
        util::Logger::warn(
            std::format("SDMA{} queue {}:{} progress observer failed with unknown exception",
                        queue->engine_id_, queue->process_id_, queue->queue_id_));
      }
      queue->reported_cursor_ = consumer_cursor;
    }

    {
      std::lock_guard lock(mutex_);
      queue->service_active_ = false;
      if (!queue->closing_) {
        switch (outcome) {
        case SdmaRingStatus::Idle:
          if (queue->producer_cursor_ != producer)
            queue->ready_ = true;
          break;
        case SdmaRingStatus::Runnable:
          queue->ready_ = true;
          break;
        case SdmaRingStatus::Blocked:
          queue->retry_pending_ = true;
          queue->retry_deadline_ = std::chrono::steady_clock::now() + kRetryInterval;
          break;
        case SdmaRingStatus::Faulted:
        case SdmaRingStatus::Malformed:
          // Published before the progress observer above.
          break;
        }
      }
      condition_.notify_all();
    }
  }

  std::lock_guard lock(mutex_);
  for (Slot &slot : slots_)
    if (slot.queue)
      slot.queue->service_active_ = false;
  condition_.notify_all();
}

} // namespace rocjitsu::amdgpu
