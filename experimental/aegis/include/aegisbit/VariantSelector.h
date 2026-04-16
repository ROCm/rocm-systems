//===-- aegisbit/VariantSelector.h - Per-kernel round-robin variant pick --===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Phase 4a of the Instrumentation Replay 5a plan: a single-responsibility
/// pure helper that picks which patched-ELF variant to dispatch next for a
/// given kernel name.
///
/// The selection policy is a thread-safe round-robin per kernel name: every
/// call to `pick` returns `Counters[Name]++ % VariantCount`.  Policy is
/// internal so `TracingEngine` need not know about atomics or locking, and
/// can later be swapped for e.g. a plateau-aware scheme without touching
/// the dispatch hot path.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_VARIANT_SELECTOR_H
#define AEGISBIT_VARIANT_SELECTOR_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace aegisbit {

/// Thread-safe round-robin variant picker.
///
/// `pick(Name, N)` returns values `0, 1, ..., N-1, 0, 1, ...` across
/// successive calls for the same `Name`.  Different kernel names have
/// independent counters.  With `N <= 1` it always returns `0`, which
/// makes the single-variant (no-replay) path free of atomic contention.
class VariantSelector {
public:
  VariantSelector() = default;

  // Non-copyable / non-movable — `std::atomic` cannot be moved, and we
  // hand out references into the internal counter map.
  VariantSelector(const VariantSelector &) = delete;
  VariantSelector &operator=(const VariantSelector &) = delete;

  /// Pick the next variant index to dispatch for `KernelName`.
  /// \param KernelName  Stable kernel identifier (same string used by the
  ///                    bundle cache so the counter lines up).
  /// \param VariantCount Number of variants currently bundled for the
  ///                    kernel.  Must be >= 1 in practice; the `<=1` fast
  ///                    path returns 0 without touching the map.
  uint32_t pick(const std::string &KernelName, uint32_t VariantCount) {
    if (VariantCount <= 1)
      return 0;
    auto &Ctr = counter(KernelName);
    return Ctr.fetch_add(1, std::memory_order_relaxed) % VariantCount;
  }

  /// Reset the counter for a specific kernel name (restarts round-robin
  /// at variant 0 on the next `pick`).  Primarily used by tests.
  void resetKernel(const std::string &KernelName) {
    std::lock_guard<std::mutex> Lock(MapMutex);
    auto It = Counters.find(KernelName);
    if (It != Counters.end())
      It->second->store(0, std::memory_order_relaxed);
  }

  /// Reset all counters.  Primarily used by tests.
  void resetAll() {
    std::lock_guard<std::mutex> Lock(MapMutex);
    for (auto &[_, P] : Counters)
      P->store(0, std::memory_order_relaxed);
  }

private:
  std::atomic<uint32_t> &counter(const std::string &KernelName) {
    std::lock_guard<std::mutex> Lock(MapMutex);
    auto It = Counters.find(KernelName);
    if (It == Counters.end())
      It = Counters.emplace(KernelName,
                            std::make_unique<std::atomic<uint32_t>>(0)).first;
    return *It->second;
  }

  mutable std::mutex MapMutex;
  // `unique_ptr<atomic>` because `std::atomic` is neither copyable nor
  // movable, and `unordered_map` insert cannot move values in-place for
  // that type.  Pointers are stable once inserted.
  std::unordered_map<std::string, std::unique_ptr<std::atomic<uint32_t>>>
      Counters;
};

} // namespace aegisbit

#endif // AEGISBIT_VARIANT_SELECTOR_H
