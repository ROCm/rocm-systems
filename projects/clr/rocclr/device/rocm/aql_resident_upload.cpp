// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "aql_resident_upload.hpp"
#include "rocdevice.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace amd::roc::aql_resident {
UploadPool::UploadPool(Device& device) : owner_(device), device_(&device) {}

void UploadPool::destroy(Block* block) {
  if (block->base) device_->hostFree(block->base, block->capacity);
  delete block;
}

UploadPool::~UploadPool() {
  // A live lease retains the pool, so no concurrent recycling is possible.
  while (free_) {
    Block* block = free_;
    free_ = block->next;
    destroy(block);
  }
}

UploadPool::Lease::~Lease() {
  if (block_) pool_->recycle(block_);
}

void UploadPool::recycle(Block* block) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (block->capacity <= kCacheLimit - cachedBytes_) {
      block->next = free_;
      free_ = block;
      cachedBytes_ += block->capacity;
      return;
    }
  }
  // No allocation or container growth on the completion/recycle path.
  destroy(block);
}

std::unique_ptr<UploadPool::Lease> UploadPool::acquire(size_t bytes) {
  if (!bytes || bytes > std::numeric_limits<size_t>::max() - 63) return {};
  const size_t capacity = (bytes + 63) & ~size_t(63);
  // Allocate the lease before removing a block from the cache; allocation
  // failure cannot orphan a GPU-visible allocation.
  auto lease = std::unique_ptr<Lease>(new Lease(shared_from_this()));
  {
    std::lock_guard<std::mutex> lock(mutex_);
    Block** best = nullptr;
    for (Block** entry = &free_; *entry; entry = &(*entry)->next) {
      if ((*entry)->capacity >= capacity &&
          (!best || (*entry)->capacity < (*best)->capacity)) best = entry;
    }
    if (best) {
      lease->block_ = *best;
      *best = (*best)->next;
      cachedBytes_ -= lease->block_->capacity;
    }
  }
  const bool reused = lease->block_ != nullptr;
  if (!reused) {
    lease->block_ = new Block;
    lease->block_->capacity = capacity;
    lease->block_->base = device_->hostAlloc(capacity, 64, amd::Device::MemorySegment::kKernArg);
    if (!lease->block_->base) {
      delete lease->block_;
      lease->block_ = nullptr;
      return {};
    }
  }
  if (std::getenv("HIP_AQL_IB_AUDIT")) {
    std::fprintf(stderr, "HIP_RESIDENT_UPLOAD bytes=%zu capacity=%zu reused=%d\n",
                 bytes, lease->block_->capacity, reused);
  }
  return lease;
}
}  // namespace amd::roc::aql_resident
