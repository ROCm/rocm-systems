// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/queue_service.h"

#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include <cassert>
#include <exception>
#include <memory>
#include <utility>

namespace rocjitsu::amdgpu {

QueueService::OperationLease::OperationLease(QueueService &service,
                                             std::shared_ptr<QueueRecord> queue)
    : service_(&service), queue_(std::move(queue)) {}

QueueService::OperationLease::OperationLease(OperationLease &&other) noexcept
    : service_(std::exchange(other.service_, nullptr)), queue_(std::move(other.queue_)) {}

QueueService::OperationLease &
QueueService::OperationLease::operator=(OperationLease &&other) noexcept {
  if (this == &other)
    return *this;
  release();
  service_ = std::exchange(other.service_, nullptr);
  queue_ = std::move(other.queue_);
  return *this;
}

QueueService::OperationLease::~OperationLease() { release(); }

void QueueService::OperationLease::release() {
  if (service_ == nullptr)
    return;
  service_->release_operation(queue_);
  service_ = nullptr;
  queue_.reset();
}

QueueHandle QueueService::allocate_locked(std::shared_ptr<QueueRecord> queue) {
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
  return {.slot = slot_index, .generation = slot.generation};
}

std::shared_ptr<QueueService::QueueRecord> QueueService::find_locked(QueueHandle handle) const {
  if (!handle || handle.slot >= slots_.size())
    return {};
  const Slot &slot = slots_[handle.slot];
  if (slot.generation != handle.generation || !slot.queue || slot.queue->state != QueueState::Open)
    return {};
  return slot.queue;
}

QueueService::OperationLease QueueService::acquire_operation(QueueHandle handle) {
  std::lock_guard lock(mutex_);
  std::shared_ptr<QueueRecord> queue = find_locked(handle);
  if (!queue)
    return {};
  ++queue->active_operations;
  return OperationLease(*this, std::move(queue));
}

void QueueService::release_operation(const std::shared_ptr<QueueRecord> &queue) {
  std::lock_guard lock(mutex_);
  assert(queue->active_operations != 0);
  --queue->active_operations;
  reset_cv_.notify_all();
}

std::shared_ptr<QueueService::QueueRecord> QueueService::begin_close_locked(QueueHandle handle) {
  std::shared_ptr<QueueRecord> queue = find_locked(handle);
  if (!queue)
    return {};
  queue->state = QueueState::Closing;
  ++active_closes_;

  Slot &slot = slots_[handle.slot];
  slot.queue.reset();
  ++slot.generation;
  if (slot.generation == 0)
    ++slot.generation;
  free_slots_.push_back(handle.slot);
  return queue;
}

bool QueueService::close(std::shared_ptr<QueueRecord> queue) {
  queue->backend->detach(queue->queue_id, queue->process_id);
  const bool released = gpu_vm_.release_queue(queue->address_space);
  {
    std::lock_guard lock(mutex_);
    assert(queue->state == QueueState::Closing);
    assert(active_closes_ != 0);
    queue->state = QueueState::Dead;
    --active_closes_;
    reset_cv_.notify_all();
  }
  return released;
}

void QueueService::finish_create() {
  std::lock_guard lock(mutex_);
  assert(active_creates_ != 0);
  --active_creates_;
  reset_cv_.notify_all();
}

QueueHandle QueueService::create(const QueueCreateInfo &info) {
  if (!info.backend || !info.address_space)
    return {};
  uint64_t admission_epoch = 0;
  {
    std::lock_guard lock(mutex_);
    if (!admission_open_)
      return {};
    admission_epoch = admission_epoch_;
    ++active_creates_;
  }
  const std::optional<AddressSpaceInfo> address_space =
      gpu_vm_.retain_queue_address_space(info.address_space);
  if (!address_space) {
    finish_create();
    return {};
  }
  // A queue has one authoritative address-space identity.  The numeric VMID is
  // retained only because the legacy CP API still uses it for diagnostics and
  // compatibility routing; never let a caller pin one handle and execute
  // through another VMID.
  if (info.process_id != address_space->vmid) {
    (void)gpu_vm_.release_queue(info.address_space);
    finish_create();
    return {};
  }

  try {
    info.backend->attach(info);
  } catch (...) {
    info.backend->rollback_attach(info);
    (void)gpu_vm_.release_queue(info.address_space);
    finish_create();
    throw;
  }

  std::shared_ptr<QueueRecord> queue;
  try {
    queue = std::make_shared<QueueRecord>(QueueRecord{
        .address_space = info.address_space,
        .backend = info.backend,
        .process_id = info.process_id,
        .queue_id = info.queue_id,
    });
  } catch (...) {
    info.backend->detach(info.queue_id, info.process_id);
    (void)gpu_vm_.release_queue(info.address_space);
    finish_create();
    throw;
  }

  QueueHandle handle;
  std::exception_ptr allocation_error;
  bool create_finished = false;
  {
    std::lock_guard lock(mutex_);
    if (admission_open_ && admission_epoch_ == admission_epoch) {
      try {
        handle = allocate_locked(queue);
      } catch (...) {
        allocation_error = std::current_exception();
      }
      --active_creates_;
      reset_cv_.notify_all();
      create_finished = true;
    }
  }
  if (handle)
    return handle;
  info.backend->detach(info.queue_id, info.process_id);
  (void)gpu_vm_.release_queue(info.address_space);
  if (!create_finished)
    finish_create();
  if (allocation_error)
    std::rethrow_exception(allocation_error);
  return {};
}

bool QueueService::update(QueueHandle handle, uint64_t ring_base_va, uint32_t ring_size,
                          uint32_t queue_percentage) {
  OperationLease queue = acquire_operation(handle);
  if (!queue)
    return false;
  queue->backend->update(queue->queue_id, queue->process_id, ring_base_va, ring_size,
                         queue_percentage);
  return true;
}

bool QueueService::notify_doorbell(QueueHandle handle, uint64_t value) {
  return static_cast<bool>(notify_doorbell_result(handle, value));
}

QueueDoorbellResult QueueService::notify_doorbell_result(QueueHandle handle, uint64_t value) {
  OperationLease queue = acquire_operation(handle);
  if (!queue)
    return {};
  return {.found = true,
          .disposition =
              queue->backend->notify_doorbell(queue->queue_id, queue->process_id, value)};
}

bool QueueService::destroy(QueueHandle handle) {
  std::shared_ptr<QueueRecord> queue;
  {
    std::unique_lock lock(mutex_);
    queue = begin_close_locked(handle);
    if (!queue)
      return false;
    reset_cv_.wait(lock, [&queue]() { return queue->active_operations == 0; });
  }
  return close(std::move(queue));
}

bool QueueService::reset() {
  std::vector<std::shared_ptr<QueueRecord>> queues;
  {
    std::unique_lock lock(mutex_);
    reset_cv_.wait(lock, [this]() { return !reset_in_progress_; });
    reset_in_progress_ = true;
    admission_open_ = false;
    ++admission_epoch_;
    if (admission_epoch_ == 0)
      ++admission_epoch_;
    reset_cv_.wait(lock, [this]() { return active_creates_ == 0; });
    for (uint32_t index = 0; index < slots_.size(); ++index) {
      Slot &slot = slots_[index];
      if (!slot.queue)
        continue;
      slot.queue->state = QueueState::Closing;
      ++active_closes_;
      queues.push_back(slot.queue);
      slot.queue.reset();
      ++slot.generation;
      if (slot.generation == 0)
        ++slot.generation;
      free_slots_.push_back(index);
    }
    reset_cv_.wait(lock, [&queues]() {
      for (const std::shared_ptr<QueueRecord> &queue : queues)
        if (queue->active_operations != 0)
          return false;
      return true;
    });
  }
  for (std::shared_ptr<QueueRecord> &queue : queues)
    (void)close(std::move(queue));
  {
    std::unique_lock lock(mutex_);
    reset_cv_.wait(lock, [this]() { return active_closes_ == 0; });
  }
  const bool vm_reset = gpu_vm_.reset();
  {
    std::lock_guard lock(mutex_);
    if (vm_reset) {
      ++reset_epoch_;
      if (reset_epoch_ == 0)
        ++reset_epoch_;
    }
    admission_open_ = true;
    reset_in_progress_ = false;
    reset_cv_.notify_all();
  }
  return vm_reset;
}

bool QueueService::contains(QueueHandle handle) const {
  std::lock_guard lock(mutex_);
  return find_locked(handle) != nullptr;
}

std::size_t QueueService::active_queues() const {
  std::lock_guard lock(mutex_);
  std::size_t count = 0;
  for (const Slot &slot : slots_)
    count += slot.queue != nullptr;
  return count;
}

uint64_t QueueService::reset_epoch() const {
  std::lock_guard lock(mutex_);
  return reset_epoch_;
}

bool QueueService::accepting_creates_for_test() const {
  std::lock_guard lock(mutex_);
  return admission_open_;
}

} // namespace rocjitsu::amdgpu
