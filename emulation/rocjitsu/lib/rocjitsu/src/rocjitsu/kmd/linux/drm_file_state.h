// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/kmd/linux/drm_vm.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {
class LinuxKfd;
class RemoteDriver;

/// @brief Backend identity for a DRM file.
/// @details Simulated local files use a weak route; guest/remote files preserve
/// their counted operational lease.
class DrmBackendLease {
public:
  DrmBackendLease() = default;
  DrmBackendLease(std::shared_ptr<LinuxKfd> local, std::function<void()> release)
      : local_(std::move(local)), release_(std::move(release)) {}
  DrmBackendLease(std::shared_ptr<RemoteDriver> remote, std::function<void()> release)
      : remote_(std::move(remote)), release_(std::move(release)) {}
  DrmBackendLease(std::weak_ptr<LinuxKfd> local, std::function<void()> release)
      : weak_local_(std::move(local)), weak_local_mode_(true), release_(std::move(release)) {}
  DrmBackendLease(const DrmBackendLease &) = delete;
  DrmBackendLease &operator=(const DrmBackendLease &) = delete;
  DrmBackendLease(DrmBackendLease &&other) noexcept { *this = std::move(other); }
  DrmBackendLease &operator=(DrmBackendLease &&other) noexcept {
    if (this != &other) {
      release();
      local_ = std::move(other.local_);
      weak_local_ = std::move(other.weak_local_);
      remote_ = std::move(other.remote_);
      weak_local_mode_ = std::exchange(other.weak_local_mode_, false);
      release_ = std::exchange(other.release_, {});
    }
    return *this;
  }
  ~DrmBackendLease() { release(); }
  std::shared_ptr<LinuxKfd> local() const { return weak_local_mode_ ? weak_local_.lock() : local_; }
  const std::shared_ptr<RemoteDriver> &remote() const { return remote_; }
  bool is_weak_local() const { return weak_local_mode_; }
  explicit operator bool() const { return local() || remote_; }
  // Local simulator external accounting ends at last descriptor/reservation,
  // independently of the acquired file's shared lifetime.
  std::function<void()> take_external_release() { return std::exchange(release_, {}); }
  void release() {
    auto callback = take_external_release();
    local_.reset();
    weak_local_.reset();
    remote_.reset();
    if (callback)
      callback();
  }

private:
  std::shared_ptr<LinuxKfd> local_;
  std::weak_ptr<LinuxKfd> weak_local_;
  std::shared_ptr<RemoteDriver> remote_;
  bool weak_local_mode_ = false;
  std::function<void()> release_;
};

struct SyncobjFence {
  mutable std::atomic<bool> signaled{false};
  std::atomic<bool> failed{false};
  // Merged sync files retain immutable snapshots of the original fences.
  std::vector<std::shared_ptr<SyncobjFence>> dependencies;
  bool is_signaled() const {
    if (signaled.load(std::memory_order_acquire))
      return true;
    // Merges and timeline transfers form a DAG. Visit each payload once,
    // without recursive calls or repeated walks through shared predecessors.
    std::vector<const SyncobjFence *> pending{this};
    std::unordered_set<const SyncobjFence *> visited;
    while (!pending.empty()) {
      const auto *fence = pending.back();
      pending.pop_back();
      if (fence->signaled.load(std::memory_order_acquire) || !visited.insert(fence).second)
        continue;
      if (fence->dependencies.empty())
        return false;
      // Check the newly appended work before revisiting its predecessors.
      for (auto it = fence->dependencies.rbegin(); it != fence->dependencies.rend(); ++it)
        pending.push_back(it->get());
    }
    // Successful completion is permanent; subsequent queries need not revisit
    // an immutable subtree whose leaves have all completed successfully.
    for (const auto *fence : visited)
      fence->signaled.store(true, std::memory_order_release);
    return true;
  }
  bool is_failed() const {
    std::vector<const SyncobjFence *> pending{this};
    std::unordered_set<const SyncobjFence *> visited;
    while (!pending.empty()) {
      const auto *fence = pending.back();
      pending.pop_back();
      if (fence->signaled.load(std::memory_order_acquire) || !visited.insert(fence).second)
        continue;
      if (fence->failed.load(std::memory_order_acquire))
        return true;
      for (const auto &dependency : fence->dependencies)
        pending.push_back(dependency.get());
    }
    return false;
  }
};

struct SyncobjEntry {
  bool has_fence = false;
  bool signaled = false;
  uint64_t submitted_point = 0;
  uint64_t signaled_point = 0;
  std::shared_ptr<SyncobjFence> binary_fence;
  std::map<uint64_t, std::shared_ptr<SyncobjFence>> timeline_fences;
};

struct DrmFileState {
  uint64_t id = 0;
  uint32_t render_minor = 128;
  // Local simulated files cannot strongly own their runtime: KFD is owned
  // by that runtime and can retain this file. Application calls pin the route
  // temporarily, while external descriptor accounting keeps it operational.
  DrmBackendLease backend_lease;
  std::shared_ptr<DrmVm> vm;
  // Descriptor and synchronous reservation references, guarded by fd_mutex_.
  // Shared KFD/submission ownership keeps resources alive independently.
  size_t open_fds = 0;
  uint32_t next_syncobj_handle = 1;
  std::unordered_map<uint32_t, std::shared_ptr<SyncobjEntry>> syncobj_entries;
  // Timeline activity on one DRM file must not wake waiters belonging to every
  // other open DRM file in the process. Waiters sleep on this generation with
  // futex(2), which preserves EINTR instead of transparently restarting after a
  // signal as std::condition_variable does.
  std::atomic<uint32_t> syncobj_generation{0};
  bool imported_sync_fence = false; // Wait on the cross-file notification generation.
  uint32_t next_context_id = 1;
  std::unordered_set<uint32_t> contexts;
  bool vmid_reserved = false;
  std::mutex submission_mutex;
  /// Submission identity and retained fences for one context/engine/ring namespace.
  struct Pm4Queue {
    uint64_t key = 0;           ///< SimulatedKfd queue identity, retained until release.
    uint64_t next_sequence = 1; ///< CS sequence local to this namespace.
    std::map<uint64_t, std::shared_ptr<SyncobjFence>> fences; ///< Bounded WAIT_CS history.
  };
  std::unordered_map<uint64_t, Pm4Queue> pm4_queues; ///< Context/engine/ring namespaces.

private:
  friend class DrmRetirementOwner;
  // Linked only after the last shared owner releases this file. Retirement
  // detaches each successor before deletion so long queues are not recursive.
  std::unique_ptr<DrmFileState> retirement_next_;
};

} // namespace rocjitsu
