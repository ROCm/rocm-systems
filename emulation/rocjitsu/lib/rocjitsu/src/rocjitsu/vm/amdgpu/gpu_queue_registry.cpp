// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"

#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <new>
#include <utility>

namespace rocjitsu::amdgpu {

GpuQueueRegistry::OperationLease::OperationLease(GpuQueueRegistry &registry,
                                                 std::shared_ptr<QueueRecord> queue)
    : registry_(&registry), queue_(std::move(queue)) {}

GpuQueueRegistry::OperationLease::OperationLease(OperationLease &&other) noexcept
    : registry_(std::exchange(other.registry_, nullptr)), queue_(std::move(other.queue_)) {}

GpuQueueRegistry::OperationLease &
GpuQueueRegistry::OperationLease::operator=(OperationLease &&other) noexcept {
  if (this == &other)
    return *this;
  release();
  registry_ = std::exchange(other.registry_, nullptr);
  queue_ = std::move(other.queue_);
  return *this;
}

GpuQueueRegistry::OperationLease::~OperationLease() { release(); }

void GpuQueueRegistry::OperationLease::release() {
  if (registry_ == nullptr)
    return;
  registry_->release_operation(queue_);
  registry_ = nullptr;
  queue_.reset();
}

GpuQueueRegistry::RegistrationTransaction::~RegistrationTransaction() {
  registry_.finish_registration();
}

GpuQueueRegistry::QueueRecord::~QueueRecord() { close_binding(); }

void GpuQueueRegistry::QueueRecord::close_binding() noexcept { binding.reset(); }

GpuQueueRegistry::~GpuQueueRegistry() { close_all(false); }

GpuQueueRegistry::RegistrationRollback::~RegistrationRollback() {
  if (!committed_)
    (void)gpu_vm_.release_queue(address_space_);
}

void GpuQueueRegistry::close_all(bool reopen_registration) {
  std::vector<std::shared_ptr<QueueRecord>> queues;
  {
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [this]() { return !close_all_in_progress_; });
    close_all_in_progress_ = true;
    admission_open_ = false;
    ++admission_epoch_;
    if (admission_epoch_ == 0)
      ++admission_epoch_;
    condition_.wait(lock, [this]() { return active_registrations_ == 0 && active_closes_ == 0; });
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
    condition_.wait(lock, [&queues]() {
      return std::ranges::all_of(queues, [](const std::shared_ptr<QueueRecord> &queue) {
        return queue->active_operations == 0;
      });
    });
  }
  for (std::shared_ptr<QueueRecord> &queue : queues)
    (void)close(std::move(queue));
  {
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [this]() { return active_closes_ == 0; });
    ++lifecycle_epoch_;
    if (lifecycle_epoch_ == 0)
      ++lifecycle_epoch_;
    admission_open_ = reopen_registration;
    close_all_in_progress_ = false;
    condition_.notify_all();
  }
}

QueueHandle GpuQueueRegistry::allocate_locked(std::shared_ptr<QueueRecord> queue) {
  if (fail_next_registry_allocation_) {
    fail_next_registry_allocation_ = false;
    throw std::bad_alloc();
  }
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

std::shared_ptr<GpuQueueRegistry::QueueRecord>
GpuQueueRegistry::find_locked(QueueHandle handle) const {
  if (!handle || handle.slot >= slots_.size())
    return {};
  const Slot &slot = slots_[handle.slot];
  if (slot.generation != handle.generation || !slot.queue || slot.queue->state != QueueState::Open)
    return {};
  return slot.queue;
}

GpuQueueRegistry::OperationLease GpuQueueRegistry::acquire_operation(QueueHandle handle) {
  std::lock_guard lock(mutex_);
  std::shared_ptr<QueueRecord> queue = find_locked(handle);
  if (!queue)
    return {};
  ++queue->active_operations;
  return OperationLease(*this, std::move(queue));
}

void GpuQueueRegistry::release_operation(const std::shared_ptr<QueueRecord> &queue) {
  std::lock_guard lock(mutex_);
  assert(queue->active_operations != 0);
  --queue->active_operations;
  condition_.notify_all();
}

void GpuQueueRegistry::close(std::shared_ptr<QueueRecord> queue) {
  queue->close_binding();
  const bool released = gpu_vm_.release_queue(queue->identity.address_space);
  assert(released);
  (void)released;
  {
    std::lock_guard lock(mutex_);
    assert(queue->state == QueueState::Closing);
    assert(active_closes_ != 0);
    queue->state = QueueState::Dead;
    --active_closes_;
    condition_.notify_all();
  }
}

void GpuQueueRegistry::finish_registration() {
  std::lock_guard lock(mutex_);
  assert(active_registrations_ != 0);
  --active_registrations_;
  condition_.notify_all();
}

QueueHandle GpuQueueRegistry::register_queue(const QueueRegistrationRequest &request) {
  if (!request.binding_factory || !request.identity.address_space)
    return {};
  uint64_t admission_epoch = 0;
  {
    std::lock_guard lock(mutex_);
    if (!admission_open_)
      return {};
    admission_epoch = admission_epoch_;
    ++active_registrations_;
  }
  RegistrationTransaction transaction(*this);
  const std::optional<AddressSpaceInfo> address_space =
      gpu_vm_.retain_queue_address_space(request.identity.address_space);
  if (!address_space)
    return {};
  // A queue has one authoritative address-space identity.  The numeric VMID is
  // retained only because the legacy CP API still uses it for diagnostics and
  // compatibility routing; never let a caller pin one handle and execute
  // through another VMID.
  if (request.identity.process_id != address_space->vmid) {
    (void)gpu_vm_.release_queue(request.identity.address_space);
    return {};
  }

  RegistrationRollback rollback(gpu_vm_, request.identity.address_space);
  QueueBindingCreateResult bound = request.binding_factory->create_binding(request);
  if (!bound)
    return {};
  std::shared_ptr<QueueRecord> queue;
  try {
    queue = std::make_shared<QueueRecord>(request.identity);
  } catch (...) {
    bound.binding.reset();
    throw;
  }
  queue->binding = std::move(bound.binding);

  QueueHandle handle;
  {
    std::lock_guard lock(mutex_);
    if (admission_open_ && admission_epoch_ == admission_epoch)
      handle = allocate_locked(queue);
  }
  if (handle) {
    rollback.commit();
    return handle;
  }
  return {};
}

QueueReconfigureResult GpuQueueRegistry::reconfigure_queue(QueueHandle handle,
                                                           QueueReconfigureRequest request) {
  OperationLease queue = acquire_operation(handle);
  if (!queue)
    return {};
  return {.found = true, .status = queue->binding->reconfigure(request)};
}

QueueSubmissionResult GpuQueueRegistry::submit_producer(QueueHandle handle,
                                                        uint64_t producer_value) {
  OperationLease queue = acquire_operation(handle);
  if (!queue)
    return {};
  return {.found = true, .status = queue->binding->submit_producer(producer_value)};
}

QueueCloseResult GpuQueueRegistry::unregister_queue(QueueHandle handle, QueueCloseMode mode) {
  std::shared_ptr<QueueRecord> queue;
  {
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [this]() { return !close_all_in_progress_; });
    queue = find_locked(handle);
    if (!queue)
      return {};
    queue->state = QueueState::Closing;
    ++active_closes_;
    condition_.wait(lock, [&queue]() { return queue->active_operations == 0; });
  }

  const QueuePrepareCloseStatus prepared = mode == QueueCloseMode::ForceCancel
                                               ? QueuePrepareCloseStatus::Ready
                                               : queue->binding->prepare_close();
  if (prepared != QueuePrepareCloseStatus::Ready) {
    std::lock_guard lock(mutex_);
    queue->state = QueueState::Open;
    --active_closes_;
    condition_.notify_all();
    return {.found = true,
            .status = prepared == QueuePrepareCloseStatus::Busy ? QueueCloseStatus::Busy
                                                                : QueueCloseStatus::Faulted};
  }

  {
    std::lock_guard lock(mutex_);
    Slot &slot = slots_[handle.slot];
    assert(slot.generation == handle.generation);
    assert(slot.queue == queue);
    slot.queue.reset();
    ++slot.generation;
    if (slot.generation == 0)
      ++slot.generation;
    free_slots_.push_back(handle.slot);
  }
  close(std::move(queue));
  return {.found = true, .status = QueueCloseStatus::Closed};
}

void GpuQueueRegistry::close_all() { close_all(true); }

bool GpuQueueRegistry::contains(QueueHandle handle) const {
  std::lock_guard lock(mutex_);
  return find_locked(handle) != nullptr;
}

std::size_t GpuQueueRegistry::active_queues() const {
  std::lock_guard lock(mutex_);
  std::size_t count = 0;
  for (const Slot &slot : slots_)
    count += slot.queue != nullptr;
  return count;
}

uint64_t GpuQueueRegistry::lifecycle_epoch() const {
  std::lock_guard lock(mutex_);
  return lifecycle_epoch_;
}

bool GpuQueueRegistry::accepting_registrations_for_test() const {
  std::lock_guard lock(mutex_);
  return admission_open_;
}

} // namespace rocjitsu::amdgpu
