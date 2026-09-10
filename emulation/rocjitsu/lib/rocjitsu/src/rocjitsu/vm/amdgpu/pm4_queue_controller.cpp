// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pm4_queue_controller.h"

#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {

Pm4QueueController::QueueRecord::QueueRecord(GpuVm &gpu_vm, Pm4QueueConfig config,
                                             GpuVmBindingLease retained_address_space)
    : address_space_lease(std::move(retained_address_space)),
      packet_processor(std::move(config.packet_callbacks)),
      ring_consumer(packet_processor, gpu_vm, config.address_space, config.ring_base,
                    config.ring_size_bytes, config.consumer_pointer_address,
                    config.initial_consumer_cursor) {}

Pm4QueueController::RegistrationId Pm4QueueController::attach(Pm4QueueConfig config) {
  if (!config.address_space || config.ring_base == 0 || config.consumer_pointer_address == 0 ||
      config.ring_size_bytes == 0 || (config.ring_base % sizeof(uint32_t)) != 0 ||
      (config.consumer_pointer_address % sizeof(uint32_t)) != 0 ||
      (config.ring_size_bytes % sizeof(uint32_t)) != 0) {
    return 0;
  }

  std::optional<GpuVmBindingLease> address_space_lease =
      gpu_vm_.retain_binding(config.address_space);
  if (!address_space_lease)
    return 0;
  auto queue =
      std::make_shared<QueueRecord>(gpu_vm_, std::move(config), std::move(*address_space_lease));
  std::lock_guard lock(mutex_);
  RegistrationId registration_id = next_registration_id_++;
  if (next_registration_id_ == 0)
    next_registration_id_ = 1;
  while (registration_id == 0 || queues_.contains(registration_id)) {
    registration_id = next_registration_id_++;
    if (next_registration_id_ == 0)
      next_registration_id_ = 1;
  }
  queues_.emplace(registration_id, std::move(queue));
  return registration_id;
}

QueuePrepareCloseStatus
Pm4QueueController::prepare_detach(RegistrationId registration_id) noexcept {
  return detach(registration_id, false);
}

bool Pm4QueueController::detach(RegistrationId registration_id) noexcept {
  return detach(registration_id, true) == QueuePrepareCloseStatus::Ready;
}

QueuePrepareCloseStatus Pm4QueueController::detach(RegistrationId registration_id,
                                                   bool force) noexcept {
  std::shared_ptr<QueueRecord> queue;
  std::unique_lock lock(mutex_);
  const auto found = queues_.find(registration_id);
  if (found == queues_.end() || found->second->closing)
    return QueuePrepareCloseStatus::Faulted;
  queue = found->second;
  queue->closing = true;
  queue->ready = false;
  condition_.wait(lock, [&queue]() { return !queue->service_active; });
  if (!force) {
    const QueuePrepareCloseStatus status = queue->ring_consumer.prepare_close();
    if (status != QueuePrepareCloseStatus::Ready) {
      queue->closing = false;
      queue->ready = !queue->terminal;
      condition_.notify_all();
      return status;
    }
  } else {
    queue->ring_consumer.reset();
  }
  queues_.erase(registration_id);
  return QueuePrepareCloseStatus::Ready;
}

QueueReconfigureStatus Pm4QueueController::update(RegistrationId registration_id,
                                                  const QueueReconfigureRequest &request) {
  std::unique_lock lock(mutex_);
  const auto found = queues_.find(registration_id);
  if (found == queues_.end())
    return QueueReconfigureStatus::Stale;
  const std::shared_ptr<QueueRecord> queue_record = found->second;
  if (queue_record->closing)
    return QueueReconfigureStatus::Stale;
  ++queue_record->management_operations;
  condition_.wait(
      lock, [&queue_record]() { return queue_record->closing || !queue_record->service_active; });
  if (queue_record->closing) {
    --queue_record->management_operations;
    condition_.notify_all();
    return QueueReconfigureStatus::Stale;
  }
  QueueRecord &queue = *queue_record;
  if (queue.ring_consumer.in_flight()) {
    --queue.management_operations;
    condition_.notify_all();
    return QueueReconfigureStatus::Busy;
  }
  QueueReconfigureStatus status = QueueReconfigureStatus::Applied;
  if (request.ring_base_address == 0 || request.ring_size_bytes == 0 ||
      request.scheduling_percentage == 0) {
    queue.enabled = false;
    queue.ready = false;
    queue.terminal = false;
    queue.ring_consumer.reconfigure(request.ring_base_address, request.ring_size_bytes);
    status = QueueReconfigureStatus::Disabled;
  } else if ((request.ring_base_address % sizeof(uint32_t)) != 0 ||
             (request.ring_size_bytes % sizeof(uint32_t)) != 0) {
    status = QueueReconfigureStatus::Invalid;
  } else {
    queue.enabled = true;
    queue.ready = false;
    queue.terminal = false;
    queue.ring_consumer.reconfigure(request.ring_base_address, request.ring_size_bytes);
  }
  --queue.management_operations;
  condition_.notify_all();
  return status;
}

QueueSubmissionStatus Pm4QueueController::notify(RegistrationId registration_id,
                                                 uint64_t producer_cursor) {
  std::lock_guard lock(mutex_);
  const auto found = queues_.find(registration_id);
  if (found == queues_.end() || found->second->closing || !found->second->enabled ||
      found->second->terminal)
    return QueueSubmissionStatus::Faulted;
  found->second->producer_cursor = std::max(found->second->producer_cursor, producer_cursor);
  found->second->ready = true;
  return QueueSubmissionStatus::Accepted;
}

QueueSubmissionStatus Pm4QueueController::service_queue(QueueRecord &queue,
                                                        uint64_t producer_cursor) {
  const Pm4RingResult result = queue.ring_consumer.consume(producer_cursor, kPacketsPerTurn);
  if (result.status == Pm4RingStatus::Blocked)
    return QueueSubmissionStatus::Retry;
  if (result.status != Pm4RingStatus::Complete)
    return QueueSubmissionStatus::Faulted;
  return QueueSubmissionStatus::Accepted;
}

bool Pm4QueueController::service() {
  std::vector<RegistrationId> registrations;
  {
    std::lock_guard lock(mutex_);
    registrations.reserve(queues_.size());
    for (const auto &[registration_id, queue] : queues_) {
      (void)queue;
      registrations.push_back(registration_id);
    }
  }

  bool retry_pending = false;
  for (const RegistrationId registration_id : registrations) {
    std::shared_ptr<QueueRecord> queue;
    uint64_t producer_cursor = 0;
    {
      std::lock_guard lock(mutex_);
      const auto found = queues_.find(registration_id);
      if (found == queues_.end())
        continue;
      queue = found->second;
      if (!queue->ready || queue->terminal || !queue->enabled || queue->closing ||
          queue->service_active || queue->management_operations != 0) {
        continue;
      }
      queue->ready = false;
      queue->service_active = true;
      producer_cursor = queue->producer_cursor;
    }

    QueueSubmissionStatus status = QueueSubmissionStatus::Faulted;
    try {
      status = service_queue(*queue, producer_cursor);
    } catch (...) {
      status = QueueSubmissionStatus::Faulted;
    }

    {
      std::lock_guard lock(mutex_);
      queue->service_active = false;
      if (!queue->closing) {
        if (status == QueueSubmissionStatus::Retry) {
          queue->ready = true;
          retry_pending = true;
        } else if (status == QueueSubmissionStatus::Faulted) {
          queue->terminal = true;
          queue->ready = false;
        } else {
          queue->ready = queue->ring_consumer.consumer_cursor() != queue->producer_cursor;
          retry_pending = retry_pending || queue->ready;
        }
      }
      condition_.notify_all();
    }
  }
  return retry_pending;
}

void Pm4QueueController::close_all() noexcept {
  std::unordered_map<RegistrationId, std::shared_ptr<QueueRecord>> queues;
  {
    std::unique_lock lock(mutex_);
    for (auto &[registration_id, queue] : queues_) {
      (void)registration_id;
      queue->closing = true;
      queue->ready = false;
    }
    condition_.wait(lock, [this]() {
      return std::ranges::none_of(queues_,
                                  [](const auto &entry) { return entry.second->service_active; });
    });
    for (auto &[registration_id, queue] : queues_) {
      (void)registration_id;
      queue->ring_consumer.reset();
    }
    queues.swap(queues_);
  }
}

std::size_t Pm4QueueController::active_queues() const {
  std::lock_guard lock(mutex_);
  return queues_.size();
}

} // namespace rocjitsu::amdgpu
