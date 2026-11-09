// SPDX-License-Identifier: NCSA
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace wsl {
namespace thunk {

// Keep replaced scratch allocations until the last submission that could
// reference them completes.
template <class Memory>
class ScratchRetirement {
 public:
  static bool Completed(uint64_t fence, uint64_t target, uint64_t submitted) {
    // WDDM writes all ones when a queue is invalidated by a GPU timeout.
    return fence != std::numeric_limits<uint64_t>::max() &&
           fence <= submitted && fence >= target;
  }

  bool Retire(Memory* memory, uint64_t target) {
    if (!memory) return false;
    // Add an empty entry first, so allocation failure leaves caller ownership intact.
    try {
      entries_.emplace_back();
    } catch (const std::bad_alloc&) {
      return false;
    } catch (const std::length_error&) {
      return false;
    }
    entries_.back().memory.reset(memory);
    entries_.back().target = target;
    return true;
  }

  void Reap(uint64_t fence, uint64_t submitted) {
    if (fence == std::numeric_limits<uint64_t>::max() || fence > submitted) return;
    size_t kept = 0;
    for (size_t i = 0; i < entries_.size(); ++i) {
      Entry& entry = entries_[i];
      if (Completed(fence, entry.target, submitted)) {
        entry = {};
      } else if (kept != i) {
        entries_[kept++] = std::move(entry);
      } else {
        ++kept;
      }
    }
    entries_.resize(kept);
  }

  // Called only after the WDDM queue has been destroyed, matching the
  // existing teardown order for the queue's currently installed scratch.
  void ReleaseAfterQueueDestroy() {
    entries_.clear();
  }

  // If WDDM cannot confirm queue destruction, keep its allocations alive.
  // Their ownership is deliberately left to process teardown.
  void AbandonAfterFailedQueueDestroy() {
    for (Entry& entry : entries_) entry.memory.release();
    entries_.clear();
  }

  size_t Count() const { return entries_.size(); }

 private:
  struct Entry {
    std::unique_ptr<Memory> memory;
    uint64_t target = 0;
  };
  std::vector<Entry> entries_;
};

}  // namespace thunk
}  // namespace wsl
