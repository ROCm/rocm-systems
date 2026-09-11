// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "device/device.hpp"
#include <memory>
#include <mutex>

namespace amd::roc { class Device; }
namespace amd::roc::aql_resident {
// Completion-owned leases: only the root signal's resource release makes a
// submitted allocation reusable. The cache limit bounds idle, not in-flight,
// memory. Each physical-queue program variant owns its own pool initially.
class UploadPool final : public std::enable_shared_from_this<UploadPool> {
  struct Block {
    void* base = nullptr;
    size_t capacity = 0;
    Block* next = nullptr;
  };
 public:
  class Lease final {
   public:
    ~Lease();
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    void* base() const { return block_->base; }
   private:
    friend class UploadPool;
    explicit Lease(std::shared_ptr<UploadPool> pool) : pool_(std::move(pool)) {}
    std::shared_ptr<UploadPool> pool_;
    Block* block_ = nullptr;
  };
  explicit UploadPool(Device& device);
  ~UploadPool();
  std::unique_ptr<Lease> acquire(size_t bytes);
 private:
  void recycle(Block* block);
  void destroy(Block* block);
  amd::SharedReference<amd::Device> owner_;
  Device* device_;
  std::mutex mutex_;
  Block* free_ = nullptr;
  size_t cachedBytes_ = 0;
  static constexpr size_t kCacheLimit = 4 * 1024 * 1024;
};
}  // namespace amd::roc::aql_resident
