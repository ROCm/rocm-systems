// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/drm_retirement_owner.h"

#include "rocjitsu/kmd/linux/drm_file_state.h"
#include "rocjitsu/kmd/linux/linux_kfd.h"

#include <cassert>
#include <exception>
#include <utility>

namespace rocjitsu {

DrmRetirementOwner::Batch::Batch() = default;
DrmRetirementOwner::Batch::~Batch() { retire(); }
DrmRetirementOwner::Batch::Batch(Batch &&other) noexcept = default;

DrmRetirementOwner::Batch &DrmRetirementOwner::Batch::operator=(Batch &&other) noexcept {
  if (this != &other) {
    retire();
    backend_ = std::move(other.backend_);
    cleanup_ = std::move(other.cleanup_);
    files_ = std::move(other.files_);
  }
  return *this;
}

void DrmRetirementOwner::Batch::retire() noexcept {
  while (files_) {
    auto file = std::move(files_);
    files_ = std::move(file->retirement_next_);
    (*cleanup_)(*file, *backend_);
  }
  cleanup_ = {};
  // The last pin may synchronously enter the runtime deleter and finish draining
  // this owner. All files and callbacks from this batch must be gone first.
  backend_.reset();
}

DrmRetirementOwner::DrmRetirementOwner(Cleanup cleanup)
    : cleanup_(std::make_shared<const Cleanup>(std::move(cleanup))) {
  assert(*cleanup_);
}

DrmRetirementOwner::~DrmRetirementOwner() {
  // Deleting queued files here would bypass cleanup and could dereference GPU
  // services after their runtime has gone. Every owner must first be drained.
  if (pending_)
    std::terminate();
}

void DrmRetirementOwner::bind_backend(std::weak_ptr<LinuxKfd> backend) {
  std::lock_guard lock(mutex_);
  assert(phase_ == Phase::Running && !pending_ && backend_.expired());
  backend_ = std::move(backend);
}

void DrmRetirementOwner::enqueue(std::unique_ptr<DrmFileState> file) noexcept {
  if (!file)
    return;
  std::lock_guard lock(mutex_);
  // Closing still accepts releases from engine/process teardown. Closed means
  // all producers have stopped and the runtime may already be gone.
  if (phase_ == Phase::Closed)
    std::terminate();
  assert(!file->retirement_next_);
  auto *tail = file.get();
  if (pending_tail_)
    pending_tail_->retirement_next_ = std::move(file);
  else
    pending_ = std::move(file);
  pending_tail_ = tail;
}

DrmRetirementOwner::Batch DrmRetirementOwner::take_batch() {
  Batch batch;
  {
    std::lock_guard lock(mutex_);
    if (phase_ != Phase::Running || !pending_)
      return batch;
    // Sharing the callback cannot allocate, unlike copying std::function. The
    // result lives outside this lock, including when its final pin is released.
    batch.cleanup_ = cleanup_;
    batch.backend_ = backend_.lock();
    if (batch.backend_) {
      batch.files_ = std::move(pending_);
      pending_tail_ = nullptr;
    }
  }
  return batch;
}

void DrmRetirementOwner::drain() {
  while (auto batch = take_batch()) {
    // Each batch retires before obtaining the next one. Retirement can enqueue
    // more files, and the final backend pin can itself start runtime teardown.
  }
}

void DrmRetirementOwner::begin_teardown() {
  std::lock_guard lock(mutex_);
  assert(phase_ == Phase::Running);
  phase_ = Phase::Closing;
  // The runtime deleter retains this owner. Release the weak control-block
  // reference so the deleter's capture does not keep its own block allocated.
  backend_.reset();
}

void DrmRetirementOwner::finish_teardown(LinuxKfd &backend) {
  std::unique_ptr<DrmFileState> pending;
  for (;;) {
    {
      std::lock_guard lock(mutex_);
      assert(phase_ == Phase::Closing);
      if (!pending_) {
        phase_ = Phase::Closed;
        return;
      }
      pending = std::move(pending_);
      pending_tail_ = nullptr;
    }
    while (pending) {
      auto file = std::move(pending);
      pending = std::move(file->retirement_next_);
      (*cleanup_)(*file, backend);
    }
  }
}

} // namespace rocjitsu
